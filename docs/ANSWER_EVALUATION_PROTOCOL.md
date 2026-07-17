# 答案提取、判分与存档协议

> 当前解析器版本：`2.0.1`。本协议适用于 `eval_asc_paper.py` 产生的所有逐题结果。

## 1. 已确认的问题

历史判分链路存在三类问题：

1. 旧解析器会漏掉没有 `\boxed{}`、但在 Markdown 加粗等式或 LaTeX 货币表达式中明确给出
   的最终答案。两个已确认的假阴性为：
   - `Total downloads = ... = **366**` 被提取为 `60`；
   - `\$40 + \$24 = \$64` 被提取为 `24`。
2. `--use_our_eval` 原本会改变判分路径。未开启时，代码可能遍历多个候选，只要较早的某个
   候选碰巧等于 ground truth 就判对，即使模型后面已经更正为另一个最终答案。这是潜在
   假阳性。现在该参数只为兼容旧命令保留，两种设置使用完全相同的逻辑。
3. 两边已经是明确的普通数字且不等时，旧逻辑仍尝试加载 SymPy/ANTLR。这会产生不必要的
   `antlr4` 警告，并受到环境中 SymPy/gmpy2 二进制兼容性的影响。现在普通数值比较不再进入
   符号解析。

## 2. 当前选择规则

程序只选择一个最终答案进行判分，不使用 ground truth 反向挑选候选。优先级为：

1. 最后一个 `\boxed{...}`；
2. 最后一个明确的 `Final Answer` / `Answer` 标记候选；
3. 程序代码块的显式 output；
4. 最终结论、总计等式、答案列表；
5. 尾部等式或结论中的数值；
6. 最后才使用“文本尾部最后一个数字”的兼容性猜测。

如果存在多个明确答案，例如先写 `\boxed{42}`、后更正为 `\boxed{41}`，选择最后一个 `41`。
即使 ground truth 是 42，也必须判错，不能从历史候选中挑一个正确值。

每个新结果都保存 `answer_extraction`：

```json
{
  "parser_version": "2.0.1",
  "answer": "366",
  "candidates": ["366"],
  "source": "post_think_final_phrase",
  "confidence": "high",
  "selection_policy": "single_selected_candidate",
  "requires_review": false,
  "warnings": []
}
```

`boxed/marked/program/final phrase` 通常为高置信；尾部等式为中置信；没有明确答案标记、只能猜
最后数字时为低置信并进入人工复核队列。多个显式候选仍会完整记录，但按“最后一个显式答案”
确定性选择，不会因为普通的逐步推理而把大量样本都送去人工检查。

## 3. 等价判断

判分顺序为：

1. 规范化后的精确字符串相等；
2. assignment/list/union 等结构化比较；
3. 普通整数、小数、逗号、百分号和简单分数的数值比较；
4. 只有两侧不是普通标量时，才尝试 SymPy 代数等价。

当前 GSM8K ground truth 是规范化后的单个数值，因此绝大多数样本不会依赖 SymPy。MATH 的
集合、多根、根式和一般代数表达式仍比 GSM8K 更复杂；如果下一阶段重新使用 MATH，应另做一轮
针对 MATH 官方 evaluator 的专项对照，不能仅凭 GSM8K 测试宣称解析器完整。

## 4. 原始输出存档

`eval_asc_paper.py` 的 `--save_details` 默认值为 `all`。每个样本保存：

- 完整 `question` 和完整 `model_output`（原始 CoT）；
- `pred_answer`、`gt_answer`、`correct`；
- `answer_extraction` 的版本、来源、候选、置信度与警告；
- `answer_review_required`，并在汇总中报告 `answer_review_required_rate`；
- token、截断、复读和乱码标记。

不要把 `--save_details none` 用于正式结果。正式 JSON 生成后不原地改判分；旧文件视为原始
生成档案，修复解析器后生成独立审计 sidecar。

## 5. 重审旧结果

无需重新运行 GPU，也无需把全部思维链逐条发给人工阅读：

```bash
python audit_saved_answers.py \
  result_a.json result_b.json \
  --output_dir results/answer_audits
```

审计脚本会递归找到逐题记录，重新提取和判分，并只把以下样本放入 `review_queue`：

- 新旧 prediction 或 correctness 发生变化；
- 只能低置信猜测答案或完全找不到答案；
- 输出被截断或检测到乱码。

它不会覆盖源 JSON。sidecar 文件名同时包含解析器版本和源文件 SHA-256：

```text
<source>.answer_audit.s2.v2_0_1.<source_hash_12>.json
```

因此源文件任何变化都会生成不同文件名，旧审计不会被静默覆盖。

## 6. 当前存档的审计结果

使用解析器 2.0.1 重审当前工作区的 300 条逐题记录：

| 源结果 | 记录数 | 正确数变化 | 判定变化 | 人工复核队列 |
|---|---:|---:|---:|---:|
| concise prompt control | 30 | 26 → 27 | 1 | 2 |
| verbose prompt control | 30 | 25 → 26 | 1 | 2 |
| conditional behavior steering | 120 | 117 → 117 | 0 | 1 |
| delayed behavior steering | 120 | 116 → 116 | 0 | 0 |

两个判定变化正是上面已确认的 `366` 和 `64` 假阴性；此前
[`PROMPT_LENGTH_CONTROL_ANALYSIS.md`](PROMPT_LENGTH_CONTROL_ANALYSIS.md) 已按人工核对后的
27/30 与 26/30 报告。conditional/delayed 两组没有发现判定漂移。

concise 复核队列中的另一个样本明确写出“Melanie started with **18**”，解析答案与 ground
truth 均为 18，只因缺少标准 Final Answer/boxed 标记而被保守标为低置信；它不是新的误判。
verbose 队列另有一个达到 4096-token 上限且判错的样本；conditional 队列另有一个达到上限
但已明确给出正确答案 18 的样本。二者因为 `length_capped` 被保守保留供人工确认，判定本身
没有因解析器版本发生变化。

## 7. 版本管理与回退

- 每次修改提取/比较行为都必须更新 `ANSWER_PARSER_VERSION`；
- 先加入导致问题的真实输出回归测试，再修改解析器；
- 答案逻辑使用独立 Git 提交，不与模型方法或实验结果混合；
- 回退使用 `git revert <answer-eval-commit>`，不使用会丢失其他工作的 `git reset --hard`；
- 旧 GPU 结果 JSON 与旧审计 sidecar 均保留，新的解析版本生成新的 sidecar。

当前真实误判样本以及“较早候选正确、最后答案错误”的假阳性反例已固化在
`tests/test_answer_utils.py`；不可变 sidecar 行为在 `tests/test_answer_audit.py` 中覆盖。
