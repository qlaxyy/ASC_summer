# ASC Result Summary

## Reading Notes

- `CoT` 表示不使用 ASC 的普通 chain-of-thought baseline。
- `Author vector` 表示作者原始代码/配置中提供的 steering vector。
- `Self-extracted vector` 表示本项目重新生成 long/short CoT pairs 后提取的新 steering vector。
- `Token compression` 按同模型、同数据集的 CoT baseline 计算：

```text
Token compression = (CoT avg tokens - ASC avg tokens) / CoT avg tokens
```

## Main Findings

- 下方各模型表格是早期 exploratory sweep 的历史记录，不是当前因果有效性结论。
- DeepSeek-R1-Distill-Qwen-7B/GSM8K 的严格复现中，单一行为轨迹向量在两个
  100 题切片上分别压缩 7.92% 和增长 7.08%；合并压缩仅 0.87%，准确率同为
  91%，bootstrap 95% 区间跨过 0。
- 在两个新的 GSM8K train validation split 上，长度延迟门控与隐藏状态条件门控
  均为 `null_result`；门控实际触发但没有稳健压缩。
- 因此不能继续声称该自提取向量在 DeepSeek-R1-Distill-Qwen-7B 上“稳定压缩”。
  Qwen3、MATH 和作者向量结果尚未按同等协议复核，当前保留为待审计信号。

## Rigorous causal audit: Qwen-7B / GSM8K

| Stage | Split | Result |
|---|---|---|
| Immediate positive-add | test 100--199 | 7.92% compression, accuracy 90% → 89% |
| Immediate positive-add | test 200--299 | -7.08% compression, accuracy 92% → 93% |
| Combined | 200 development-audit questions | 0.87% compression, both 91% |
| Delayed 512/768/1024 | fresh train validation 30 | all `null_result` |
| State gate alpha .25/.50/.75 | fresh train validation 30 | -7.36%/-10.41%/-10.48% raw compression |

结论：当前单一平均方向可以区分 teacher-forced concise/verbose 表示，但不足以对
自由生成产生可泛化的简洁性因果控制。完整证据见
`docs/SINGLE_VECTOR_CAUSAL_AUDIT.md`。

## Qwen3-8B / GSM8K

官方 thinking-mode 口径：

```text
thinking=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
```

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 95.00% | 1908.30 | 0.0% |
| Self-extracted vector | 0.20 | 97.50% | 1669.20 | 12.5% |
| Self-extracted vector | 0.29 | 95.50% | 1612.00 | 15.5% |
| Self-extracted vector | 0.38 | 96.00% | 1628.50 | 14.7% |
| Self-extracted vector | 0.46 | 97.00% | 1559.10 | 18.3% |
| Self-extracted vector | 0.55 | 96.00% | 1521.00 | 20.3% |
| Self-extracted vector | 0.65 | 96.00% | 1467.00 | 23.1% |
| Self-extracted vector | 0.75 | 95.00% | 1513.90 | 20.7% |
| Self-extracted vector | 0.90 | 94.50% | 1470.90 | 22.9% |

## DeepSeek-R1-Distill-Qwen-7B / GSM8K

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 91.25% | 716.12 | 0.0% |
| Author vector | 0.10 | 92.25% | 632.54 | 11.7% |
| Author vector | 0.20 | 88.50% | 548.65 | 23.4% |
| Author vector | 0.24 | 90.00% | 534.64 | 25.3% |
| Self-extracted vector | 0.30 | 90.50% | 499.41 | 30.3% |
| Self-extracted vector | 0.35 | 86.75% | 480.99 | 32.8% |
| Self-extracted vector | 0.40 | 86.50% | 485.53 | 32.2% |

## DeepSeek-R1-Distill-Qwen-7B / MATH

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 86.00% | 2351.38 | 0.0% |
| Author vector | 0.27 | 88.00% | 1774.40 | 24.5% |
| Self-extracted vector | 0.20 | 86.00% | 1623.40 | 31.0% |
| Self-extracted vector | 0.30 | 88.00% | 1639.63 | 30.3% |

## DeepSeek-R1-Distill-Llama-8B / GSM8K

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 90.00% | 853.58 | 0.0% |
| Author vector | 0.20 | 85.20% | 700.02 | 18.0% |

## DeepSeek-R1-Distill-Llama-8B / MATH

| Vector source | gamma | Accuracy | Avg tokens | Token compression |
|---|---:|---:|---:|---:|
| CoT | CoT | 88.00% | 2457.16 | 0.0% |
| Author vector | 0.20 | 92.00% | 2281.86 | 7.1% |
| Author vector | 0.47 | 86.00% | 1810.80 | 26.3% |
