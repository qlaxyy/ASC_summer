# 本地—GitHub—AutoDL 协作流程

GitHub 是代码、实验配置和摘要结果的唯一共享来源；AutoDL 仅负责 GPU 推理。

## 每次代码更新

1. 本地修改后运行轻量检查并提交。
2. 推送到 GitHub 的 `main` 或功能分支。
3. AutoDL 在开始实验前拉取指定提交。

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git log -1 --oneline
python -m compileall -q .
```

只有在 `git log -1` 显示预期提交后，才运行 GPU 实验。不要在存在未提交改动时执行 `git pull`。

## AutoDL 环境准备

```bash
python -m pip install -r requirements.txt
# 如需 FlashAttention，请按当前 CUDA、PyTorch、Python 版本安装匹配 wheel。
```

模型权重、原始数据、私钥和 API key 不提交 Git。数据可通过 `download_datasets.py` 准备；
向量在 AutoDL 生成并保留在 `vectors/`。它们默认被 `.gitignore` 忽略。

## 运行实验

每次实验应记录：commit hash、模型完整路径、数据集/样本 ID、prompt、layer、向量文件与 SHA256、
gamma 列表、随机种子、软件版本和完整命令。建议将终端输出保存为日志：

```bash
mkdir -p logs
python eval_asc_paper.py ... 2>&1 | tee logs/<experiment-name>.log
```

`results/` 中的逐样本 JSON 默认不提交。将可复现的配置、汇总指标和结论补充到
`results/RESULT_SUMMARY.md`，然后只提交 Markdown 和必要的小型 manifest。

## 回传结果

```bash
git add results/RESULT_SUMMARY.md docs/ README.md
git commit -m "results: add qwen7b gsm8k paired gamma sweep"
git push origin main
```

若结果文件过大，请上传到外部存储，并在结果摘要中记录链接、SHA256、生成 commit 和文件结构；
不要通过 Git 提交模型权重、完整逐样本 CoT 或密钥。

## 分支与提交约定

- `main`：可运行且经过基础检查的版本。
- `feat/<topic>`：较大功能开发，例如 `feat/paired-evaluation`。
- `fix/<topic>`：独立修复，例如 `fix/steering-vector-sign`。
- 提交格式：`type: imperative summary`，如 `fix: align steering vector injection sign`。
