# 行为轨迹简洁推理向量协议

## 为什么更换提取对象

上一轮向量比较的是提示末尾表示：

```text
h(concise instruction prompt)[-1] - h(base/verbose instruction prompt)[-1]
```

它能稳定区分提示，但 held-out 生成实验表明，正向加入后没有稳定压缩。因此新方法不再把“识别到简洁指令”直接等同于“进入简洁推理轨迹”。

新的底层方法仍是单个激活引导向量：

```text
v = unit(mean(concise target trajectory - verbose source trajectory))
h' = h + gamma * v, gamma > 0
```

变化仅在于：向量从目标模型实际生成的正确推理轨迹中提取。

## 数据与质量约束

`generate_behavior_trajectory_pairs.py` 在 GSM8K train 随机抽题，并用同一个目标模型分别生成：

- `paper_cot_concise_prefix`；
- `paper_cot_verbose_prefix`。

两次生成从同一个 batch RNG 状态开始。只有同时满足以下条件的 pair 才用于提取：

- concise 与 verbose 的答案都正确；
- 两边都没有达到生成长度上限；
- 两边都没有明显乱码或模板循环；
- concise 至少比 verbose 短 30%；
- concise 不是过短的空答案。

程序最多尝试 120 道题，获得 60 个合格 pair 后提前停止。所有尝试和拒绝原因都保存在同一个 JSON。

所有尝试过的 train 行（包括未通过过滤的行）都会写入后续向量 metadata 的 `causal_validation_exclusion_indices`，因果筛选器不会再次把它们当作 validation，避免数据重用。

## 相对进度对齐

短链和长链无法逐 token 对齐。`extract_behavior_trajectory_vectors.py` 将每条 response 均分为 8 个相对进度区间：

```text
0%-12.5%, 12.5%-25%, ..., 87.5%-100%
```

在每个区间内平均 block-output residual，再计算同一相对阶段的：

```text
concise activation - verbose activation
```

随后先对 8 个阶段取平均，再对所有题目取平均并单位归一化。这样不会只比较答案末端，也不会因为 verbose token 更多而给予长链更高权重。

## AutoDL 第一步：生成干净行为 pair

```bash
export OMP_NUM_THREADS=8

python generate_behavior_trajectory_pairs.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_path datasets/gsm8k/train.jsonl \
  --output_path pairs/gsm8k_behavior_trajectory_pairs_seed20260716.json \
  --target_accepted_pairs 60 \
  --max_candidate_samples 120 \
  --min_pair_compression 0.30 \
  --batch_size 8 \
  --max_new_tokens_concise 2048 \
  --max_new_tokens_verbose 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --attn_impl sdpa
```

先查看最后的 `accepted / attempted`。JSON 中的 `pairs` 包含全部推理链、答案提取、token 数和拒绝原因。

## AutoDL 第二步：一次前向提取多层向量

只有第一步至少获得 30 个合格 pair 才运行：

```bash
python extract_behavior_trajectory_vectors.py \
  --pairs_path pairs/gsm8k_behavior_trajectory_pairs_seed20260716.json \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --output_dir vectors/gsm8k_behavior_trajectory_seed20260716 \
  --file_prefix qwen7b_behavior_trajectory \
  --layer_indices 8,12,16,20,24 \
  --num_relative_bins 8 \
  --min_pairs 30 \
  --activation_batch_size 2 \
  --max_input_tokens 8192 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --device_map auto
```

五个层在同一组 concise/verbose teacher-forced 前向中同时捕获，不会把模型前向成本乘以五。

行为向量的推荐注入范围是 `all_tokens`。在本项目现有 hook 中，这个历史命名具体表示：prefill 阶段只注入最后一个 prompt token，随后注入每个生成 token；它不会改写问题 prompt 的所有位置。这样与 response trajectory 的提取位置一致。

第二步只产生表示诊断，不直接宣称压缩有效。把终端的 `Layer diagnostics` 发回本地后，再选三个层做 held-out 正 gamma 因果筛选，以控制 GPU 成本。

## 解释边界

- 不允许通过乘以 `-1` 或负 gamma 改写方向；
- 不把提示词缩短、强制截断或微调当作向量效果；
- 如果行为轨迹向量仍然没有正向压缩，结论应是该候选向量族在当前模型上无效，而不是继续扩大 gamma 直到出现偶然结果。
