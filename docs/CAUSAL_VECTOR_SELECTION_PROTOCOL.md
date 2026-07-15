# 简洁推理引导向量：因果筛选协议

## 目的

底层方法保持为激活引导向量，不改成微调、输出截断或提示词压缩：

```text
v = unit(mean(h(concise target) - h(source)))
h' = h + gamma * v, gamma > 0
```

新的步骤只回答一个工程问题：哪个层和哪个正强度在未参与向量提取的题目上，确实能把冗长推理引向更简洁的推理，同时不明显损害答案质量？

## 借鉴点

- CAA：对候选层做生成端干预实验，而不是仅凭表示空间的 SNR 排层。
- ITI：在独立验证数据上选择干预位置和强度，再冻结配置。
- Refusal direction：同时使用 addition、输出质量与异常指标筛选方向，不能仅凭向量余弦或投影变化宣称因果有效。
- LLM-Steer-Instruct：继续使用 instruction target-minus-source、block output、全序列正向加法的单向量结构。

项目内论文和官方代码索引见 `reference/README.md`。

## 数据隔离

`select_causal_conciseness_vector.py` 会读取每个候选向量的 metadata，把 `selected_row_indices`（已用于提取向量的 GSM8K train 行）全部排除，然后从其余 train 行中固定随机抽取验证题。它不会读取 GSM8K test。

因此数据被分为：

1. GSM8K train extraction：提取表示方向；
2. GSM8K train validation：选择层和正 gamma；
3. GSM8K test：只做一次冻结配置后的正式报告。

## 通过条件

默认候选必须同时满足：

- 平均 token 至少减少 5%；
- 相对同一批次 `gamma=0` 的准确率下降不超过 4%；
- 明显重复率增加不超过 4%；
- 明显乱码率为 0；
- 达到 `max_new_tokens` 的比例增加不超过 4%。

如果没有候选通过，程序输出 `status: null_result`，且不会创建“有效向量”。这是一项有效的负结果，不能通过反转向量或改用负 gamma 来包装成理论一致。

## AutoDL：低成本第一轮

下面命令保持此前正式评测的 `paper_cot / temperature=0.7 / top_p=0.9 / repetition_penalty=1.1 / max_new_tokens=4096`，避免 `gamma=0` 因参数变化而不可比：

```bash
export OMP_NUM_THREADS=8

python select_causal_conciseness_vector.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local_data_path datasets/gsm8k/train.jsonl \
  --vector_dir vectors/instruction_length_projection_gsm8k_train100 \
  --file_prefix qwen7b_instruction_length_projection \
  --layer_indices 16,20,24 \
  --candidate_gammas 0.5,1.0 \
  --validation_samples 30 \
  --validation_seed 314159 \
  --min_compression 0.05 \
  --max_accuracy_drop 0.04 \
  --max_repetition_increase 0.04 \
  --max_corruption_rate 0 \
  --max_length_capped_increase 0.04 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --attn_impl sdpa \
  --save_details all \
  --output_report results/causal_conciseness_trainval30.json \
  --output_vector_path vectors/qwen7b_causally_selected_conciseness.pt
```

基线只运行一次。候选为 3 层乘 2 个强度，因此总计是 `30 * (1 + 3*2) = 210` 次生成。不要在 `--candidate_gammas` 中加入 0；程序会自动先计算并配对复用基线 RNG。

## 正式测试

只有筛选报告为 `status: selected` 时才运行。先从报告的 `selected.layer_index` 和 `selected.gamma` 读取两个值，替换下方的 `<LAYER>` 与 `<GAMMA>`：

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/test.jsonl \
  --limit 100 \
  --candidate_gammas 0,<GAMMA> \
  --steering_vector_path vectors/qwen7b_causally_selected_conciseness.pt \
  --layer_index <LAYER> \
  --injection_sign add \
  --injection_site block_output \
  --injection_scope sequence_all \
  --injection_token_count 1 \
  --vector_normalization unit_l2 \
  --intervention_mode additive \
  --prompt_mode paper_cot \
  --paired_batch_seeds \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --attn_impl sdpa \
  --save_details all \
  --per_gamma_output_dir results/causal_conciseness_gsm8k_test100
```

完整输出和答案提取结果都保存在 JSON 的 `detailed_results`，100 道题放在一起，不单独重排错误题。
