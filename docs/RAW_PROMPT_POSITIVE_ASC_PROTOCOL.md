# Raw Prompt 正向 ASC 生存测试

## 研究问题

作者公开工程路线在 `raw problem + author vector + subtraction` 下可能压缩，但它与论文
`short-minus-long` 后正向相加的公式冲突。本测试不翻转向量、不使用负 gamma，也不训练模型：

```text
v = mean_i(h_L(raw_question_i + concise_CoT_i)[-1]
           - h_L(raw_question_i + verbose_CoT_i)[-1])

h_L(x_t) <- h_L(x_t) + gamma * v, gamma > 0
```

提取和注入都使用同一个 Transformer block 的 `block_output`。推理输入仅为 raw question，
因此共享问题前缀与作者 README 的推理入口一致。

这仍是 ASC endpoint mean-vector 方法，不是 ActAdd、instruction steering、behavior trajectory
bin vector 或 LoReFT。与论文的差异是：40 对 GSM8K 轨迹来自同一个目标模型在 concise/verbose
提示下的正确输出，而不是 GPT-4o concise 对；因此应称为“理论公式一致的自提取扩展复现”。

## 第一步：重新提取 raw endpoint 向量

这里只做已有文本的前向计算，不更新模型参数：

输入 JSON 保存了64个尝试样本；提取器只读取其中
`selected_for_extraction=true` 的40个合格 pair，不会重新混入被质量规则拒绝的样本。

```bash
MODEL=/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python extract_steering_vector.py \
  --model_name "$MODEL" \
  --pairs_path pairs/gsm8k_behavior_trajectory_pairs_train40.json \
  --output_vector_path vectors/gsm8k_raw_endpoint_train40/qwen7b_raw_endpoint_short_minus_long_layer20.pt \
  --vector_method asc_endpoint_raw \
  --direction short_minus_long \
  --layer_index 20 \
  --activation_site block_output \
  --pool_last_n_tokens 1 \
  --max_input_tokens 8192 \
  --activation_batch_size 2 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --device_map auto
```

生成的 metadata 必须显示：

```text
direction=short_minus_long
recommended_injection_sign=add
matching_prompt_mode=raw
matching_injection_site=block_output
recommended_injection_scope=all_tokens
recommended_vector_normalization=none
positive_gamma_only=true
```

## 第二步：20 题低成本生存测试

保持刚才作者 raw 路线的题目、greedy decoding 和 gamma，只替换为自提取正向向量：

```bash
python eval_asc_paper.py \
  --model_name "$MODEL" \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/test.jsonl \
  --start_index 0 \
  --limit 20 \
  --prompt_mode raw \
  --candidate_gammas 0,0.27 \
  --steering_vector_path vectors/gsm8k_raw_endpoint_train40/qwen7b_raw_endpoint_short_minus_long_layer20.pt \
  --layer_index 20 \
  --injection_sign add \
  --injection_site block_output \
  --injection_scope all_tokens \
  --injection_token_count 1 \
  --vector_normalization none \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0 \
  --top_p 1 \
  --repetition_penalty 1.1 \
  --paired_batch_seeds \
  --seed 42 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --num_gpus 1 \
  --save_details all \
  --save_failures \
  --per_gamma_output_dir results/raw_positive_asc_pilot20 \
  --output_path results/raw_positive_asc_pilot20_summary.json
```

## 解释边界

- 只比较本次 `gamma=0` 与 `gamma=0.27`，不引用旧 baseline；
- 20 题只用于排除明显无效，不用于论文结论；
- 若正向 gamma 增加 token、产生乱码/复读或明显降准确率，停止该路线；
- 若压缩达到约 10% 且准确率不下降超过 1/20，再用新的开发切片复验；
- 不因失败改成负 gamma，因为那会重新回到理论与工程不一致的问题。

## 实际结果与终止结论

20 道 GSM8K 低成本生存测试得到：

| gamma | Accuracy | Avg tokens | 相对 gamma=0 | Repeat | Corrupt | Answer review |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16/20 (80%) | 733.4 | baseline | 0% | 0% | 0% |
| +0.27 | 16/20 (80%) | 736.2 | -0.38% compression（即增长） | 0% | 0% | 0% |

正向干预只增加约 2.8 个 token，准确率和异常指标不变。这不是有效压缩，也不是答案解析、
乱码或复读造成的假象。该路线记为 `null_result`，不继续搜索更大 gamma、负 gamma 或其他
test 切片。

提示模式冻结为：

- `raw`：作者 README/公开 `generate.py` 对齐的主要工程复现提示模式；
- `paper_cot`：保留为论文文字中“standard CoT prompting”的历史扩展对照；
- 不跨提示模式共用 baseline 或拼接结果；
- 作者向量在 raw+subtract 下的压缩只能报告为理论符号冲突的工程复现信号；
- raw 自提取 short-minus-long 向量的正向结果为零，说明改成 raw prompt 没有救活理论一致的
  单一 endpoint mean vector。
