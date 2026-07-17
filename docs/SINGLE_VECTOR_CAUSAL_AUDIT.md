# 单一简洁性引导向量：因果审计终结报告

## 审计对象

模型为 DeepSeek-R1-Distill-Qwen-7B，任务为 GSM8K。向量从 40 个同时正确、无
乱码/复读/截断且 concise 至少缩短 30% 的 train 行为轨迹 pair 中提取：

```text
v = unit(mean_pair(mean_relative_bin(
    h(concise trajectory) - h(verbose trajectory)
)))
```

方向始终保持 `target-minus-source`，生成端只使用 `gamma > 0` 正向相加；没有用
整体反转或负 gamma 包装符号。Layer 24 表示诊断为：

```text
pair agreement = 90.0%
SNR            = 1.381
resultant      = 0.578
bin cosine min = 0.467
```

这些值证明该方向能区分配对的 teacher-forced 表示，但不能单独证明生成端因果效果。

## 1. 固定正向注入的跨切片复现

冻结 Layer 24、gamma=0.5、block output、unit L2、正向 additive 与 paired batch
RNG，在两个不重叠的 GSM8K test development audit 切片上运行：

| test 行 | baseline | steering | token 变化 | baseline acc | steering acc |
|---|---:|---:|---:|---:|---:|
| 100--199 | 754.63 | 694.85 | 7.92% 压缩 | 90% | 89% |
| 200--299 | 668.59 | 715.92 | 7.08% 增长 | 92% | 93% |
| 合并 | 711.61 | 705.39 | 0.87% 压缩 | 91% | 91% |

合并 200 题的 paired bootstrap 压缩率 95% 区间为约 `[-6.84%, 7.94%]`，不能
拒绝零效果。缩短/增长/不变题数为 77/64/59。baseline 最长四分位平均缩短
9.3%，其余三个四分位分别增长 11.0%、3.5%、9.0%，说明效果高度异质。

## 2. 长度延迟门控

为避免干扰短推理，冻结向量、层和 gamma，只在生成超过 T 个 token 后注入。
使用全新的 GSM8K train validation 30 题（seed 20260717）：

| T | Accuracy | Avg tokens | Raw compression | Trimmed compression | shorter/longer/same |
|---:|---:|---:|---:|---:|---:|
| baseline | 29/30 | 657.5 | — | — | — |
| 512 | 29/30 | 677.4 | -3.03% | 0.00% | 2/3/25 |
| 768 | 29/30 | 643.3 | 2.15% | 0.00% | 2/0/28 |
| 1024 | 29/30 | 661.1 | -0.54% | 0.00% | 0/1/29 |

768 的均值改善只来自 2 道题，其余 28 道不变。三个候选均未达到 5% raw 和
2% 双侧 10% 截尾压缩标准，结果为 `null_result`。

## 3. 隐藏状态条件门控

长度不是可靠状态指标，因此最后检验同一向量是否可以同时作为 verbose detector
与正向 intervention。metadata 中：

```text
mu_verbose = -112.1697
mu_concise =  -47.2516
```

仅当 `dot(h_t, v)` 低于两均值之间的阈值时，执行 `h_t <- h_t + 0.5v`。使用另
一个全新 train validation 30 题（seed 20260718）：

| alpha | gate rate | Accuracy | Avg tokens | Raw compression | Trimmed compression |
|---:|---:|---:|---:|---:|---:|
| baseline | — | 30/30 | 584.3 | — | — |
| 0.25 | 18.67% | 30/30 | 627.3 | -7.36% | 0.16% |
| 0.50 | 21.79% | 28/30 | 645.1 | -10.41% | -0.33% |
| 0.75 | 57.60% | 29/30 | 645.5 | -10.48% | 0.68% |

门控在三个阈值下均实际触发，排除 hook 未工作。raw mean 中存在个别显著增长链，
但截尾后仍约为零效果；三个候选均为 `null_result`。

## 结论边界

在当前模型、数据、层与提取协议下，证据支持：

1. 单一平均方向与 concise/verbose teacher-forced 表示相关；
2. 这种表示相关性没有转化为稳定的自由生成长度控制；
3. 立即、长度延迟和状态条件三种正向注入均未通过独立因果筛选；
4. 不应再对同一方向搜索符号、gamma、起始长度或投影阈值。

证据不支持把 ASC 或所有 activation steering 一概判为错误；它否定的是本项目当前
“单一平均行为方向可以稳定压缩 Qwen-7B/GSM8K 自由推理”的具体假设。

test 100--299 已参与方法诊断，不能再当最终测试。test 300--399 保持未查看，只有
在提出并完全通过 train validation 的新方法后才能使用一次。

## 推荐的下一研究阶段

下一阶段如继续保持 activation steering，应转向低秩/多向量控制，而不是继续修饰
当前单向量：

- 保留 8 个相对推理阶段的差分，不再提前平均成一个方向；
- 使用独立 train 激活训练线性 probe/router 判断当前推理阶段与冗长状态；
- 根据 probe 选择对应的 phase-specific positive vector；
- 在新 train validation 上同时要求 raw、trimmed、逐题胜负和准确率约束；
- 失败时同样输出 `null_result`，通过后才使用保留的 test 300--399。

这仍属于引导向量方法，但研究假设从“全局一维简洁性轴”升级为“简洁行为位于一个
随推理阶段变化的低秩子空间”。它是新的方法分支，需单独设计和授权，不应与当前
单向量结果混合报告。
