# LoReFT 简洁推理干预协议

## 1. 为什么停止继续调单向量

现有审计已经否定了当前模型上的具体假设：一个 `concise - verbose` 平均方向虽然能分离
teacher-forced 表征，却不能在跨切片测试中稳定缩短自由生成。即时、延迟和隐藏状态门控三类
固定方向干预都没有通过独立因果筛选。因此本阶段不再搜索同一向量的符号、gamma、起始长度或
投影阈值。

CAA、ActAdd、RepE 和 ITI 都提供了重要依据，但其核心控制量仍是一个或若干固定均值方向。
下一步改用 **Low-rank Linear Subspace ReFT（LoReFT）**：它不假设“简洁性由一根全局向量
表达”，而是在冻结基础模型的前提下学习一个低秩表示子空间内的干预。

## 2. 论文依据与边界

主要依据是 Wu et al., *ReFT: Representation Finetuning for Language Models*, NeurIPS 2024：

- 正式论文：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html>
- 官方代码：<https://github.com/stanfordnlp/pyreft>
- 本地论文：`reference/papers/reft_neurips2024.pdf`
- 本地官方代码快照：`reference/pyreft-source/`，上游 commit
  `dafd0995a366d7b47160a337dcc388eda7431821`（2025-02-06）

论文定义的干预为：

```text
Phi(h) = h + R^T (W h + b - R h)
```

其中 `R` 的行张成秩为 `r` 的正交子空间，`W` 和 `b` 给出该子空间中的目标值；基础语言模型
参数全部冻结，只训练 `{R, W, b}`。训练生成任务时，论文使用输出 token 的 teacher-forced
交叉熵；推理时只在选定层和选定提示词位置应用干预。

选择它的现实理由：

- NeurIPS 2024 正式论文，公式、消融、训练目标和复现脚本完整；
- 官方仓库约有 1.6k stars、132 forks、454 commits（2026-07 查询快照），在表示干预子领域
  有较高采用度；
- 论文覆盖长文本指令微调，并用约 0.0039% 参数取得强 Alpaca-Eval 结果；
- 它直接放宽了本项目已经失败的“一维固定方向”假设。

必须同时保留以下负面事实：

- 官方仓库最后一次提交是 2025-02-06；社区采用度高，但当前维护频率不能称为非常活跃；
- 原论文在算术推理上低于 LoRA，作者明确报告了这一点；
- 论文没有验证 DeepSeek-R1 的 CoT token 压缩，因此本项目实现仍是待证伪的新适配；
- LoReFT 需要训练少量参数，不再是 training-free steering vector；更准确的名称是“低秩表示
  干预/representation steering”。

## 3. 本项目实现与官方实现的对应

`loreft_utils.py` 直接实现论文公式，并参考官方
`reference/pyreft-source/pyreft/interventions.py::LoreftIntervention`：

- `R^T` 使用列正交的 `[hidden_size, rank]` 参数化；
- 每个选定层有独立 `{R, W, b}`，同一层的提示词位置共享参数；
- 干预点固定为 transformer `block_output`；
- 训练和推理只修改 `fN+lN` 指定的提示词位置；
- 生成出的 token 不持续注入；
- `scale=1` 是论文公式，`scale=0` 是原模型 baseline；不把 scale 当作新 gamma 网格搜索。

没有直接把 `pyreft` 加入 `requirements.txt`，因为当前官方包要求 `accelerate>=0.29.1`，而项目
为复现旧环境固定在 `accelerate==0.26.0`。本实现只覆盖当前实验需要的最小 LoReFT 子集，避免
再次破坏可用环境。

## 4. 数据隔离

首轮只做低成本 pilot：

```text
pairs/gsm8k_behavior_trajectory_pairs_train40.json
    32 条训练
     8 条验证
```

固定随机种子 `20260719`。输入提示词不是 pair 中带“Be concise”的提示，而是与测试完全相同的
`paper_cot`：

```text
Question: {problem}
Let's think step by step.
```

监督目标为已通过正确性、截断、重复和乱码检查的 `concise_output`。这样 LoReFT 必须通过隐藏
表示干预学会简洁输出，不能依赖测试时不存在的显式 concise instruction。

40 条数据具有明显选择偏差，只足以判断方法是否完全无效，不能支撑论文主结果。若 pilot 有效，
应扩大到至少 200 条独立正确简洁轨迹，并重新冻结训练/验证划分。

GSM8K test 0--299 已参与历史开发诊断；300--399 继续保持未触碰，只能在方法通过开发门槛后
运行一次。

## 5. AutoDL 命令

先确认代码版本与环境：

```bash
cd /root/autodl-tmp/ASC_summer
git pull
conda activate ASC01
export OMP_NUM_THREADS=8

python - <<'PY'
import torch, transformers, accelerate
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
PY
```

训练首轮 adapter（单 GPU，基础模型冻结）：

```bash
MODEL=/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

python train_loreft_conciseness.py \
  --model_name "$MODEL" \
  --pairs_path pairs/gsm8k_behavior_trajectory_pairs_train40.json \
  --output_path vectors/loreft_gsm8k_concise_train40.pt \
  --layer_indices 12,16,20,24 \
  --rank 4 \
  --positions f5+l5 \
  --epochs 8 \
  --learning_rate 9e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 2048 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --device_map single \
  --seed 20260719
```

若显存不足，先把 `--max_seq_length` 降为 `1536`。脚本会跳过超长轨迹并打印 row index；不要
偷偷截断输出，因为右截断可能删掉最终答案。若可用训练条目少于 24 条，应停止并补数据。

只在已使用过的开发切片做 30 题生存测试：

```bash
python eval_asc_paper.py \
  --model_name "$MODEL" \
  --dataset gsm8k \
  --start_index 0 \
  --limit 30 \
  --loreft_adapter_path vectors/loreft_gsm8k_concise_train40.pt \
  --candidate_gammas 0,1 \
  --prompt_mode paper_cot \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --max_new_tokens 4096 \
  --batch_size 8 \
  --paired_batch_seeds \
  --seed 20260720 \
  --dtype bfloat16 \
  --attn_impl sdpa \
  --num_gpus 1 \
  --per_gamma_output_dir results/loreft_pilot_test0_29
```

注意：这里命令行仍沿用评估器的 `--candidate_gammas` 名称，但在提供
`--loreft_adapter_path` 后其含义是 LoReFT scale。默认也正是 `0,1`。

只有 pilot 同时满足以下条件，才运行最终未触碰切片：

```text
token compression >= 10%
accuracy drop <= 2 percentage points
repeat/corrupt 均不增加到不可接受水平
```

最终命令只把切片改为：

```bash
  --start_index 300 \
  --limit 100 \
  --per_gamma_output_dir results/loreft_final_test300_399
```

若 pilot 压缩低于 5% 或准确率下降超过 3 个百分点，直接记录 null result，不搜索更多 scale、
层或 test 切片。5%--10% 属于不确定结果，应先扩大 train 数据并在新的 train-validation 上复验。
