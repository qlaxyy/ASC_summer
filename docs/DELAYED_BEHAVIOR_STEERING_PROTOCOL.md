# 延迟行为轨迹引导协议

## 失败复现结论

固定使用 Layer 24、`gamma=0.5`、`all_tokens` 正向相加，在两个互不重叠的
GSM8K test 切片上表现相反：

| test 行 | baseline tokens | steering tokens | 压缩率 | baseline acc | steering acc |
|---|---:|---:|---:|---:|---:|
| 100--199 | 754.63 | 694.85 | 7.92% | 90% | 89% |
| 200--299 | 668.59 | 715.92 | -7.08% | 92% | 93% |
| 合并 200 题 | 711.61 | 705.39 | 0.87% | 91% | 91% |

合并后的 paired bootstrap 95% 压缩率区间约为 `[-6.84%, 7.94%]`，不能证明
稳定压缩。向量重提取的 Layer 24 diagnostics 与原结果一致，因此这不是环境或
向量重提取错误。

逐题审计显示，固定立即注入只对 baseline 最长的四分位数平均有效：该组从
1432.8 tokens 降至 1300.1（9.3%）；其余三个四分位数分别增长 11.0%、3.5%
和 9.0%。这说明当前问题不是方向符号，而是无条件、从第一个 token 开始干预
所有题目。

由于 100--299 行已参与失败诊断和方法设计，它们从现在起属于 development
audit，不再作为新方法的最终测试集。后续配置只能在 GSM8K train validation
选择；冻结后使用尚未查看的 test 300--399 做一次正式检验。

## 延迟注入定义

底层方向和干预公式不变：

```text
v = unit(mean(concise trajectory - verbose trajectory))
h_t' = h_t + gamma * v, gamma > 0
```

只增加时间门控：前 `T` 个生成 token 不注入，生成长度达到 `T` 后才在每个
cached generation step 注入。`T=0` 严格保持原有行为。

```text
if generated_tokens_seen >= T:
    h_t' = h_t + gamma * v
else:
    h_t' = h_t
```

它不截断输出、不修改提示词，也不根据答案正确性做事后选择；仍然是同一个激活
引导向量。门控只使用推理过程中已经生成的 token 数，因此部署时可直接执行。

## 新的 train validation 筛选

冻结 Layer 24 与 `gamma=0.5`，只比较三个延迟阈值。基线只计算一次，总成本为
`30 * (1 + 3) = 120` 次生成。`validation_seed=20260717` 产生的 30 行与提取
轨迹的 64 行、此前层/强度筛选的 30 行均无重叠。

候选除平均压缩率至少 5% 外，还必须满足：去掉 paired token savings 两端各
10% 后仍至少压缩 2%，并且缩短的题数不少于增长的题数。这两项约束专门防止
少数极端长输出再次制造虚假的平均压缩。

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
  --candidate_start_tokens 512,768,1024 \
  --validation_samples 30 \
  --validation_seed 20260717 \
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
  --output_report results/delayed_behavior_trajectory_trainval30.json \
  --output_vector_path vectors/qwen7b_causally_selected_delayed_behavior_trajectory.pt
```

若结果是 `null_result`，则延迟门控也没有通过当前因果标准，停止该候选而不进入
test。若结果是 `selected`，先审计逐题分布并冻结 `start_after`；不得根据 test
300--399 再修改阈值或 gamma。
