# Activation-steering literature and code index

本目录保存 ASC 后续开发需要反复核对的论文与作者代码。新增源码采用“可审计快照”而不是嵌套 Git 仓库：每个快照都固定到下载时的上游 commit，并排除了模型权重、预计算向量和大体积实验输出。这样可以随主项目经 GitHub 中转，又不会把仓库膨胀到数百 MB。

## 本轮新增的核心资料

热度数据是 2026-07-15 查询 GitHub API 得到的快照，只用于说明代码采用范围，之后会变化。

| 工作 | 发表情况与选择理由 | 本地论文 | 本地代码 | 官方代码快照 |
|---|---|---|---|---|
| Contrastive Activation Addition (CAA) | ACL 2024 Long Paper；Outstanding Paper Award 与 SAC Award；240 stars / 66 forks | [`papers/caa_acl2024.pdf`](papers/caa_acl2024.pdf) | [`caa-source/`](caa-source/) | [`nrimsky/CAA`](https://github.com/nrimsky/CAA) @ `5dabbbd9a0bca5f25e174501e959de378806aa48` |
| Representation Engineering (RepE) | 奠定 reading/control pipeline 的高影响开源框架；约 1,011 stars / 132 forks。它是 arXiv 工作，不把它误写成正式会议论文 | [`papers/representation_engineering_2310.01405.pdf`](papers/representation_engineering_2310.01405.pdf) | [`representation-engineering-source/`](representation-engineering-source/) | [`andyzoujm/representation-engineering`](https://github.com/andyzoujm/representation-engineering) @ `5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e` |
| Inference-Time Intervention (ITI) | NeurIPS 2023 Main Conference；用少量 head、验证集 probe 与尺度校准做稀疏干预；581 stars / 53 forks | [`papers/iti_neurips2023.pdf`](papers/iti_neurips2023.pdf) | [`honest-llama-source/`](honest-llama-source/) | [`likenneth/honest_llama`](https://github.com/likenneth/honest_llama) @ `2c6b2179be7b5aa8f0a171688cf9e01b812ca327` |
| Refusal Direction | NeurIPS 2024 Main Conference；13 个模型上的单方向结果，并用 addition、ablation 和 KL 共同筛方向；420 stars / 113 forks | [`papers/refusal_direction_neurips2024.pdf`](papers/refusal_direction_neurips2024.pdf) | [`refusal-direction-source/`](refusal-direction-source/) | [`andyrdt/refusal_direction`](https://github.com/andyrdt/refusal_direction) @ `9d852fae1a9121c78b29142de733cb1340770cc3` |
| Instruction-Following Activation Steering | ICLR 2025；本项目最直接的长度控制参考，包含 conciseness/verbosity 与 length constraints；95 stars / 15 forks | [`papers/instruction_following_steering_iclr2025.pdf`](papers/instruction_following_steering_iclr2025.pdf) | [`llm-steer-instruct-source/`](llm-steer-instruct-source/) | [`microsoft/llm-steer-instruct`](https://github.com/microsoft/llm-steer-instruct) @ `9dac937ef6fc3e483b1efc13863deeb03ec38dbe` |

CAA 上游仓库约 1.2 GB，主要是向量和实验产物，因此本地只保留根目录核心 Python、notebook 和 `utils/`。Refusal Direction 只保留 pipeline 源码。Microsoft 快照保留 Python 与 Hydra 配置，不含 assets、data 和生成结果。其余两个源码仓库体积较小，保留完整工作树（不含 `.git`）。

本目录此前已有：

- ActAdd 论文 [`2308.10248v5.pdf`](2308.10248v5.pdf) 与作者实现 [`activation_additions_hf-main/`](activation_additions_hf-main/)；
- Conditional Activation Steering (CAST) 论文 [`2409.05907v3.pdf`](2409.05907v3.pdf) 与 IBM 实现 [`activation-steering-main/`](activation-steering-main/)；
- Burns et al. 的 latent knowledge 论文 [`2205.05124v1.pdf`](2205.05124v1.pdf)。

## 论文公式与实际代码的共同结论

### 1. “正向相加”不等于方向天然正确

对 mean-difference 方法，方向由样本语义定义：

- CAA：`positive behavior - negative behavior`，见 `caa-source/generate_vectors.py`；
- ITI 的 center-of-mass 版本：`truthful - false`，见 `honest-llama-source/legacy/llama_utils.py`；
- Refusal Direction：`harmful prompt activation - harmless prompt activation`，见 `refusal-direction-source/pipeline/submodules/generate_directions.py`；
- Microsoft instruction steering：`with instruction - base query`，见 `llm-steer-instruct-source/length/evaluate.py`；
- RepE 的 cluster mean：`positive - negative`；PCA 本身有符号不确定性，所以代码另外用训练标签确定 `direction_signs`，见 `representation-engineering-source/repe/rep_readers.py`。

这支持 ASC 论文的 `concise - verbose` 公式在符号定义上是清楚的，但**表征差为正并不保证正向干预会产生目标因果效果**。方向的因果含义必须由未参与提取的样本验证，不能靠把整个向量翻转后重新命名来建立。

### 2. 表征分离度不能替代因果层选择

几个成熟实现都把“提取候选方向”和“选择可用干预”分开：

- CAA 在 held-out questions 上扫描全部层以及正负 multiplier；
- ITI 用 validation probe accuracy 排 head，只干预 top-K heads，并按方向投影标准差缩放；
- Refusal Direction 对每个 `(source position, source layer)` 候选同时做 direction ablation、positive addition 和 harmless KL，过滤后才选方向；
- Microsoft 在 held-out validation 上选 layer/weight，并将 perplexity 或质量检查纳入选择。

因此当前 `agreement/SNR/resultant/phrase_cos_min` 只能排除明显坏的表征候选。它们全为正、甚至达到 100% agreement，也不能说明该层会缩短生成。

### 3. 注入位置和持续时间是方法定义的一部分

- CAA 论文与实现：在用户 prompt 结束后，对后续所有 token position 注入；
- Microsoft：单层 `resid_post`，所有 sequence positions；生成循环每一步都挂 hook；
- ITI：选择 attention heads，在每个自回归预测步骤修改最后 token；
- Refusal Direction：addition hook 修改目标 block 输入，生成时持续生效；ablation 更强，作用于所有层的 residual/attention/MLP；
- RepE：明确支持指定 token positions 或整个张量，不能把这些配置混为同一种实验。

这也说明 ActAdd 的 prompt-only 序列注入、ASC 的 decoding-step 注入、Microsoft 的 sequence-all 注入不能仅凭“都是加向量”直接比较。

### 4. 不同论文的 coefficient/gamma 数值不可横向比较

尺度约定不同：Microsoft 使用单位方向；CAA 官方评估使用跨 behavior/layer 归一化向量；ITI 再乘该方向上的 activation standard deviation；Refusal Direction 的 addition 候选直接使用 mean difference，而 ablation 会单位化；ActAdd 还保留 prompt 激活差的原始尺度。因此某篇论文使用 `20` 或更大 coefficient，不代表 ASC 的 raw vector 或另一个 unit vector 也应使用同一数值。

### 5. gamma 越大、输出越短不是 activation steering 的普遍定律

Microsoft 在 Phi-3 的完整长度协议中观测到 `0, 5, 10, 20, 40` 的递增 conciseness 趋势；这是特定模型、向量、层、单位化和生成协议下的经验结果。论文也报告了 suboptimal steering 导致乱码和重复，并指出不完整目标会选择出 over-steering 的 layer/weight。CAA 同样用正负系数与 held-out effect size 选层，而不是假定单调性。

所以当前 DeepSeek-R1-Distill-Qwen-7B 上不单调且弱的 token 变化不能通过继续放大 gamma 来“修好”；它是该候选方向缺乏稳定压缩因果性的证据。

## 对当前 ASC 工程的直接判断

当前 `extract_instruction_conciseness_layers.py` + `eval_asc_paper.py --injection_scope sequence_all` 已经对齐 Microsoft 长度方法的核心结构：

1. 同一 base query 的 `with concise instruction - without instruction`；
2. 相同 final prompt token 的 residual difference；
3. mean direction 的 L2 单位化；
4. 单层 block output / residual-post；
5. prompt 和每个生成步骤持续正向相加；
6. greedy 模式可用于确定性诊断。

因此本轮对照没有发现一个足以解释负结果的简单符号或 hook 错误。`projection_match` 的负结果也说明只控制单个标量投影不足以传输“简洁生成”行为。

下一项有理论和代码先例的改进应是**因果候选筛选**，而不是继续构造更多只看分离度的向量：

1. 从 train 提取多个 `(layer, prompt position/contrast)` 候选；
2. 在很小的 held-out train-validation 子集上，对每个候选测正向 addition 的 token delta、accuracy delta、repeat/乱码和 KL/perplexity；
3. 只保留“确实压缩、准确率在容差内、质量不过阈值”的候选；
4. 最终只在 GSM8K test 做一次报告，避免用 test 选择层和强度；
5. 若所有候选均被过滤，应报告该模型上的 null result，而不是反转符号或扩大搜索直到偶然成功。

这一路径结合了 CAA 的 held-out layer sweep、Refusal Direction 的 addition/ablation/KL 过滤，以及 Microsoft 的 length-specific objective，比继续依据 representation SNR 选 layer 更有依据。

## PDF 完整性

下载后已检查 `%PDF` 文件头、页数、文本提取，并渲染检查首页和方法页。SHA-256：

```text
caa_acl2024.pdf                             3ACE1FB9AAA326318C2E452CC1109557A9847AD57EDAC6AA18AD457711BCEE16
representation_engineering_2310.01405.pdf   5CD54799F795433BD6652231E0D113DEAD887F21C798148619FC0CE6F0FD589F
iti_neurips2023.pdf                         3E7BCBFBD328E66B756CA8104427ACE197423E98E5A3D750CC0A924F71EB9A00
refusal_direction_neurips2024.pdf           00B84FCE7160C8CEC3AD5C02DD434292EC386E4B97A5D0E53C03FA40A0854CC6
instruction_following_steering_iclr2025.pdf 477D48FD28C5120F1049DB675010A5E1CB14A486E15BC2DF81D834742457CBFC
```
