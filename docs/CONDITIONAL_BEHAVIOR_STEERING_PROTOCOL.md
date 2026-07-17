# 隐藏状态条件行为引导协议

## 动机

Layer 24 行为轨迹向量在两个 test development audit 切片上的合并压缩率只有
0.87%。随后对同一向量采用 512、768、1024 token 延迟注入，在独立 train
validation 上分别得到 -3.03%、2.15%、-0.54%，没有候选通过。

完整 paired 报告进一步表明：768 阈值的 2.15% 仅来自 30 题中的 2 题缩短，
其余 28 题 token 数完全相同；10% 截尾压缩率为 0。因此它不是可推广的弱效果，
而是两个尾部样本造成的均值变化。

这排除了“仅凭已生成长度决定是否注入”的开环门控。新的最后一项单向量实验改为
直接读取模型当前隐藏状态，只在状态仍位于 verbose 一侧时加正向向量。

## 定义

行为向量保持不变：

```text
v = unit(mean(concise trajectory - verbose trajectory))
```

提取 metadata 已记录沿该向量的两类均值：

```text
mu_verbose = -112.1697
mu_concise =  -47.2516
```

对 `alpha in {0.25, 0.5, 0.75}` 定义门控阈值：

```text
tau(alpha) = mu_verbose + alpha * (mu_concise - mu_verbose)
```

对应阈值约为 `-95.940`、`-79.711`、`-63.481`。生成时在 Layer 24 计算：

```text
p_t = dot(h_t, v)

if p_t < tau:
    h_t <- h_t + gamma * v
else:
    h_t <- h_t
```

由于 `v` 是 target-minus-source 且 `mu_concise > mu_verbose`，门控和干预使用同一
有向坐标：只检测 verbose 状态并向 concise 方向移动。它不使用负 gamma、不反转
向量、不截断生成，也不读取答案正确性。prompt prefill 不属于行为轨迹分类域，
因此只从 cached response generation 开始判断。

## 数据隔离

- 行为向量：GSM8K train 的 40 个合格 pair（尝试过的 64 行全部排除）；
- 旧层/强度 validation：30 行；
- 延迟门控 validation：seed 20260717 的 30 行；
- 本实验：seed 20260718 的新 30 行，与上述集合均无重叠；
- test 100--299 已用于失败诊断，不再算最终测试；
- 只有本实验通过后，才冻结配置并使用未查看的 test 300--399。

## AutoDL 命令

基线只计算一次，三个 alpha 共 120 次生成。Layer 24 和 gamma=0.5 已冻结，不再
同时搜索其他层或强度。

```bash
export OMP_NUM_THREADS=8
export MODEL=/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python select_causal_conciseness_vector.py \
  --model_name "$MODEL" \
  --local_data_path datasets/gsm8k/train.jsonl \
  --vector_dir vectors/gsm8k_behavior_trajectory_train40 \
  --file_prefix qwen7b_behavior_trajectory \
  --layer_indices 24 \
  --candidate_gammas 0.5 \
  --candidate_start_tokens 0 \
  --intervention_mode conditional_additive \
  --candidate_projection_alphas 0.25,0.5,0.75 \
  --validation_samples 30 \
  --validation_seed 20260718 \
  --injection_scope all_tokens \
  --min_compression 0.05 \
  --max_accuracy_drop 0.04 \
  --max_repetition_increase 0.04 \
  --max_corruption_rate 0 \
  --max_length_capped_increase 0.04 \
  --min_trimmed_compression 0.02 \
  --min_pairwise_win_margin 0 \
  --trim_fraction 0.1 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --attn_impl sdpa \
  --save_details all \
  --output_report results/conditional_behavior_trajectory_trainval30.json \
  --output_vector_path vectors/qwen7b_causally_selected_conditional_behavior_trajectory.pt
```

每个候选额外报告 `gate`，表示被判为 verbose 并实际注入的位置比例。如果三个候选
均为 `null_result`，则停止对这个单一行为方向继续增加门控或调参，并将结论记录为：
该方向可区分两类 teacher-forced 表示，但不足以对自由生成提供可泛化因果控制。
