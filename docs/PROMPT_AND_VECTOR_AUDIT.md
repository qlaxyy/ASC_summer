# ASC 提示词与向量符号审计

## 向量符号

论文定义：

```text
v = h(question + concise_CoT) - h(question + verbose_CoT)
h <- h + gamma * v
```

作者公开代码 `ASC-main/extract_steering_vector.py` 的 `act1` 来自 short CoT，`act2` 来自
long CoT，因此文件中的计算同样是 `short - long`。但作者的 `generate.py` 在推理时执行
`h <- h - gamma*v`。这与论文公式相反，不能仅凭发布向量在减法下注入有效，就断言发布
向量实际是 `long - short`；发布向量的生成过程缺少足够的 provenance。

另外，作者提取代码读取 `hidden_states[layer_index]`，而注入代码 hook
`model.model.layers[layer_index]`。Hugging Face decoder 的 `hidden_states[0]` 是 embedding，
所以这两个位置通常相差一个 transformer block。当前项目统一在同一个 block 的输出上提取
和注入，不沿用该偏移。

为避免继续混用，`eval_asc_paper.py` 对非零 gamma 要求显式传入：

- `--injection_sign add`：论文方向 `short_minus_long`；
- `--injection_sign subtract`：作者发布向量，或显式提取的 `long_minus_short`。

## 作者代码中的提示词

### concise CoT

作者 `generate_short_cots.py` 实际配置为 `gpt-4o-mini`（README 写 GPT-4o），使用 system：

```text
You are an expert competition mathematician. When you give a solution, express it
primarily in formal math notation with minimal surrounding English. Return the final
answer in a boxed format.
```

user：

```text
Solve the following problem step by step. Answer almost entirely in math notation;
keep English words to the bare minimum.

Problem:
{problem}
```

### verbose CoT

作者目标模型的生成输入为：

```text
{problem} Let's think step by step.\n
```

参数为 temperature=0.3、repetition_penalty=1.2、max_new_tokens=4096。pipeline 默认返回
“输入提示词 + 新生成文本”的完整字符串。

### 激活提取

作者 short 输入为 `raw_problem + short_answer`。long 输入为
`raw_problem + long_generated_text`；而 `long_generated_text` 已包含一次带 step-by-step 的
问题，因此 long 侧实际上重复了问题。两个字符串也没有显式分隔符。

### 作者推理示例

`generate.py` 不自动添加任何 CoT 提示词，直接编码 `--problem` 的值。README 示例传入的是
原始问题，因此作者公开的向量生成提示词与推理示例并不完全一致。

## 当前项目中的提示词

DeepSeek-R1-Distill-Qwen/Llama 的 long CoT 生成和默认评测 `paper_cot` 均使用：

```text
Question: {problem}
Let's think step by step.
```

激活提取时，short/long 两侧都使用相同的已保存 `long_prompt` 前缀：

```text
long_prompt + short_cot
long_prompt + long_cot
```

因此当前项目在“long 生成—激活提取—评测”三处的目标模型提示词是一致的，但与作者脚本
相比多了 `Question:` 和换行。short CoT 当前由 DeepSeek API 生成，其提示词也与作者的
GPT-4o-mini 提示词不同；这应标记为扩展复现，而不是逐字复现。

Qwen3 使用 chat template，thinking mode 下内容为：

```text
Question: {problem}
Please reason step by step, and put your final answer within \boxed{}.
```

## 编码检查

`pairs/qwen7b_math_train_deepseek_pairs_100_checked.json` 当前实际包含 80 条记录。按 Unicode
码点检查，problem/prompt 均为 ASCII，CoT 中的非 ASCII 字符是 π、√、上下标、乘号等正常
数学字符，没有 replacement character、CJK 异常字符或控制字符。Windows 终端曾显示的
`鈧/螖/虏` 是控制台解码错误，不应据此删除样本。重复推理暂不作为自动剔除条件。

## 2026-07 prompt-contrast protocol update

`asc_endpoint` still represents the original complete-answer endpoint method.
Its negative causal sign is recorded as a paper/code discrepancy and is not
hidden by renaming a flipped vector.

`actadd_prompt` constructs a matched concise prompt and the exact `paper_cot`
step-by-step prompt from each problem. It reads the last prompt-token
`block_input` state before generation, computes concise minus step-by-step, and
adds the unchanged vector only during the initial prompt forward pass. Metadata
enforces positive gamma, additive injection, `block_input`, `prompt_only`, and
`paper_cot` unless a diagnostic mismatch override is explicitly requested.
