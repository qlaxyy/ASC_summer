# Paired Prompt Length Control Analysis

## Scope

This analysis compares the 30-example GSM8K runs produced with:

- `paper_cot_concise_prefix`: `Be extremely concise.`
- `paper_cot_verbose_prefix`: `Be extremely detailed.`

The question order is identical after removing the condition-specific first
prompt line. All evaluation settings are identical except `prompt_mode` and
`output_path`, and `deterministic_batch_seeds` pairs sampler states by batch.

## Length results

| metric | concise | verbose |
|---|---:|---:|
| mean tokens | 536.1 | 1178.97 |
| median tokens | 444.0 | 1027.5 |
| first quartile | 369.25 | 633.75 |
| third quartile | 564.0 | 1461.25 |
| maximum | 2335 | 4096 |
| length-capped outputs | 0 | 1 |
| detected repetition artifacts | 0 | 0 |

The concise prompt saved 642.87 tokens on average, a 54.53% reduction relative
to the verbose prompt. The median paired reduction was 51.95%. Concise was
shorter on 28 of 30 questions (two-sided exact sign-test `p=8.68e-7`). A paired
bootstrap estimated a 95% interval of 457.7 to 835.0 saved tokens for the mean
difference.

The result is not caused by the single 4096-token verbose truncation. Excluding
that paired example leaves 29 questions with means of 474.07 concise versus
1078.38 verbose, a 56.04% reduction.

## Correctness and evaluator correction

The original files reported 26/30 concise and 25/30 verbose. Manual inspection
found two false negatives in `answer_utils.py`:

- concise example 10 ended with `... = **366**`, but the parser returned `60`;
- verbose example 5 ended with `... = \$64`, but the parser returned `24`.

Both model answers are correct. After correcting extraction of Markdown-bold
and LaTeX-currency right-hand sides, the accuracies are:

| outcome | count |
|---|---:|
| both correct | 24 |
| concise only correct | 3 |
| verbose only correct | 2 |
| both wrong | 1 |

Thus the corrected totals are 27/30 (90.0%) concise and 26/30 (86.67%) verbose.
The one-question difference is not enough to claim an accuracy improvement,
but it rules out a large accuracy penalty in this control sample.

The remaining incorrect cases reflect model reasoning errors rather than answer
extraction errors. They include an arithmetic error (`794` instead of `694`),
an hourly-rate misreading, an age-sign error, and disagreement over whether
breaking even counts as "starts earning money."

## Interpretation

DeepSeek-R1-Distill-Qwen-7B exhibits a strong, paired behavioral response to the
concise-versus-verbose instruction under the ASC sampling configuration. The
failed positive-gamma activation experiments therefore cannot be attributed to
the model ignoring length instructions. The failure occurs when a distributed
prompt effect is reduced to one mean residual direction and applied through
unconstrained additive steering.

The next justified intervention is projection matching: estimate the concise
prompt's scalar projection on the learned unit direction, then interpolate the
inference residual toward that measured target. This should be implemented and
validated separately from additive ASC gamma sweeps.
