# RepE PCA 独立复现协议

## 为什么先复现 reading，而不是直接继续压缩 CoT

当前 ASC 实验的空结果不能推出“所有 activation steering 都无效”。它只否定了当前模型、
提示、提取和注入协议下的固定 conciseness vector。要验证更基础的命题，应按以下层次分开：

1. **Representation reading**：隐藏状态中是否存在能在未见样本上区分目标行为的线性方向；
2. **Causal control**：正反向干预是否使行为按预注册方向变化；
3. **目标任务迁移**：该方向能否压缩推理，同时保持答案正确。

PCA 只能直接支持第 1 层。即使 PCA 图上分离明显，也不能单独证明第 2、3 层。

## 论文与代码核对结果

目标工作是 Zou et al., *Representation Engineering: A Top-Down Approach to AI
Transparency*（arXiv:2310.01405），本地论文与官方代码快照分别位于：

- `reference/papers/representation_engineering_2310.01405.pdf`
- `reference/representation-engineering-source/`

论文 Appendix C.1 的 LAT-PCA 流程为：

1. 构造成对 stimulus；
2. 读取 decoder 最后一个 prompt token 的隐藏状态；
3. 对每对隐藏状态作差并逐行 L2 归一化；
4. 对差分拟合 PCA，取第一主成分；
5. 使用训练标签确定 PCA 主轴的符号；
6. 在未见样本上比较沿该方向的投影。

需要特别记录两个边界：

- PCA 主轴天然有正负号不确定性，论文也没有把 PCA 原始符号解释成语义方向；
- 论文说差分先 L2 归一化，但官方 `repe/rep_readers.py` 快照直接对 raw differences
  做 PCA。因此脚本默认 `l2` 对齐论文，并保留 `none` 供代码对照。

论文 Table 2 也表明固定 reading vector 的因果控制效果远弱于逐输入 Contrast Vector：
LLaMA-2-7B-Chat 的 TruthfulQA MC1 从 31.0% 提升到 34.1%，而 Contrast Vector 达到
47.9%。所以不能把该论文概括为“固定 PCA 向量普遍具有强控制能力”。

## 本项目的第一阶段验证

`reproduce_repe_pca_reading.py` 使用官方 RepE 代码中的 6 对 TruthfulQA primer 提取方向，
并从 TruthfulQA MC1 随机但确定性地划分：

- 30 题 validation：只用于从预注册候选层中选层；
- 100 题 test：只在选层后报告一次；
- 200 组随机方向：每组都同样用 primer 定符号、用 validation 选层，再测 test；
- bootstrap 95% CI：显示 100 题统计不确定性。

这是对 RepE 方法在 DeepSeek-R1-Distill-Qwen-7B 上的**适配复现**，不是原论文
LLaMA-2 权重和完整 817 题设置的逐数值复现。

## AutoDL 命令

```bash
cd /root/autodl-tmp/ASC_summer
git pull

export MODEL=/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export OMP_NUM_THREADS=8

python reproduce_repe_pca_reading.py \
  --model_name "$MODEL" \
  --layer_indices 8,12,16,20,24 \
  --num_validation 30 \
  --num_test 100 \
  --difference_normalization l2 \
  --activation_batch_size 8 \
  --max_input_tokens 512 \
  --num_random_directions 200 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --device_map auto \
  --output_path results/repe_pca_reading_truthfulqa.json
```

该脚本只做前向传播，不训练模型，也不生成长 CoT。首次运行会从 Hugging Face 下载
TruthfulQA 数据集；模型仍使用服务器已有路径。

## 预注册判据

不要以“比 50% 高”作为成功线，因为 MC1 每题选项数不同。第一阶段通过必须同时满足：

1. 层完全由 30 题 validation 选择；
2. 该层 100 题 test accuracy 高于 200 个随机方向的 95th percentile；
3. empirical `p < 0.05`；
4. bootstrap CI 与随机方向均值有实质分离；
5. 不因为看到 test 结果而更换 seed、层集合或归一化方式。

若通过，只能写“该模型存在可泛化的 truthfulness linear reading direction”。下一步才复现
正负 causal control；若不通过，先检查模型/模板迁移和 paper-vs-code normalization，不触碰
ASC 的 token 压缩结论。
