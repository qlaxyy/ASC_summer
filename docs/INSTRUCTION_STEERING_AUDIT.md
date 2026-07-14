# Conciseness Steering: Literature and Implementation Audit

## Why the latest gamma sweep is not a dose-response result

The 50-example GSM8K sweep produced:

| gamma | accuracy | average tokens | change from gamma 0 |
|---:|---:|---:|---:|
| 0.0 | 86% | 673.5 | baseline |
| 0.5 | 92% | 784.0 | +16.4% |
| 1.0 | 90% | 727.2 | +8.0% |
| 2.0 | 86% | 655.7 | -2.64% |

This is not a stable monotonic compression curve. The 2.64% reduction at gamma
2 is too small to support a practical compression claim, especially on only 50
sampled generations.

A linear intervention does not imply a monotonic output length. Only the hidden
state at the intervention is linear in the coefficient. The remaining model is
nonlinear, autoregressive sampling is stochastic, and EOS is a discrete event.
Monotonicity therefore has to be demonstrated empirically for a particular
vector, layer, injection protocol, and decoding configuration.

## What the reference methods actually implement

### Activation Addition (ActAdd)

[Turner et al.](https://arxiv.org/abs/2308.10248) preserve a token-by-hidden
activation sequence, align it with the front of the user prompt, and intervene
only during the prompt forward pass. Their official code does not replace that
sequence with one vector repeated across positions. The paper explicitly says
the coefficient cannot be increased indefinitely and reports eventual loss of
competence. ActAdd does not establish a theorem that output length must vary
monotonically with its coefficient.

### Contrastive Activation Addition (CAA)

[Panickssery et al.](https://arxiv.org/abs/2312.06681) average residual-stream
differences across contrastive examples. Their evaluation adds the resulting
direction at generated token positions, sweeps both positive and negative
coefficients, and normalizes vectors across behaviors/layers to make coefficient
comparisons more consistent.

### Instruction steering for output length

[Stolfo et al.](https://arxiv.org/abs/2410.12877) are the closest published
method to the current objective. Their official implementation:

1. pairs each base query with a version containing a concise/length instruction;
2. extracts the last input-token residual and averages instruction-minus-base;
3. divides the mean direction by its L2 norm;
4. adds the unit direction at a single selected layer across all sequence
   positions during generation;
5. selects the intervention layer on held-out validation prompts; and
6. uses greedy decoding for the reported length-weight curves.

For Phi-3, the paper reports progressively shorter outputs at weights
`0, 5, 10, 20, 40`. This is an empirical result under that complete protocol,
not a general property of activation addition. The authors also report some
nonsensical or repetitive failures from suboptimal steering.

### ASC

The ASC paper defines a unit direction and injects it at every decoding step.
Its Figure 5 describes continued compression as gamma increases on one
DeepSeek-Qwen/MATH500 experiment, followed by accuracy degradation. The public
code does not faithfully implement every part of the paper: notably, the saved
mean direction is not unit-normalized and the released sign conflicts with the
paper formula. Consequently, that reported trend cannot validate a different
prompt-only vector implementation.

## Audit of `actadd_prompt_paper_aligned`

The current method has a valid target-minus-source causal interpretation, but
its combined design is not directly supported by any one reference method:

- it averages the last eight target/source positions into one vector;
- it repeats that same vector at eight prompt positions;
- it keeps the raw, unnormalized vector scale;
- it uses the ASC layer 20 without a layer search; and
- it stops steering after prompt prefill.

It is therefore best treated as a documented negative/weak experimental result,
not as a faithful reproduction of ActAdd or instruction steering.

## New `instruction_conciseness` protocol

The new path follows the length-instruction implementation more closely while
preserving the established paper CoT baseline:

```text
source(q) = Question: q
            Let's think step by step.

target(q) = Reasoning instruction: <one concise instruction>
            Question: q
            Let's think step by step.
```

The concise instruction is assigned round-robin from seven semantically
equivalent phrasings (for example, "Be extremely concise" and "Use only the
essential mathematical steps"). This follows the reference implementation's
use of varied length instructions more closely and reduces the risk that the
saved vector primarily represents one fixed sentence.

The final token is identical. At each candidate layer:

```text
d_l = mean_i(h_l(target_i)[-1] - h_l(source_i)[-1])
v_l = d_l / ||d_l||_2
```

Evaluation uses positive addition at block output across all prompt and generated
positions:

```text
h_l[:, :, :] <- h_l[:, :, :] + weight * v_l
```

The first diagnostic extracts layers `8,12,16,20,24` simultaneously. Its
representation ranking is only a filter; it cannot replace causal generation
tests. In addition to agreement, SNR, and resultant ratio, it reports
`phrase_cos_min`: the minimum cosine between each phrasing subgroup's mean
direction and the overall mean direction. A positive value for every phrasing
is evidence that the common direction survives wording changes; it is not proof
that the direction causally shortens generated answers.

The initial fixed-phrase run ranked layers 16 and 12 above layer 20, but its
100% agreement and high resultant ratios can include fixed-instruction lexical
content. Treat that run as preliminary and use the multi-phrasing extraction
below before selecting layers for generation.

## Low-cost layer extraction

```bash
python extract_instruction_conciseness_layers.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --problems_path datasets/gsm8k/train.jsonl \
  --output_dir vectors/instruction_conciseness_gsm8k_train100_multiphrase \
  --file_prefix qwen7b_instruction_conciseness \
  --layer_indices 8,12,16,20,24 \
  --num_samples 100 \
  --activation_batch_size 8 \
  --max_input_tokens 8192 \
  --device_map auto \
  --dtype bfloat16 \
  --attn_impl sdpa
```

This performs 100 target and 100 source prompt forwards while capturing all five
layers together. It does not generate answers. Use the printed ranking to select
two candidate layers for a small greedy causal test before returning to the ASC
sampling configuration.

## Generic greedy causal test after choosing a layer

Replace `{LAYER}` with a selected layer. Unit-normalized weights are deliberately
on the same order as the published length-instruction sweep.

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/test.jsonl \
  --limit 30 \
  --prompt_mode paper_cot \
  --candidate_gammas 0,5,10,20 \
  --steering_vector_path vectors/instruction_conciseness_gsm8k_train100_multiphrase/qwen7b_instruction_conciseness_layer{LAYER}.pt \
  --layer_index {LAYER} \
  --injection_sign add \
  --injection_site block_output \
  --injection_scope sequence_all \
  --injection_token_count 1 \
  --vector_normalization unit_l2 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0 \
  --top_p 1 \
  --repetition_penalty 1.0 \
  --seed 42 \
  --attn_impl sdpa \
  --save_details all \
  --use_our_eval \
  --per_gamma_output_dir results/instruction_conciseness_layer{LAYER}_greedy30
```

Only if a layer shows a coherent length trend without accuracy/output collapse
should it be confirmed on 100 examples using the paper sampling settings.

## Matched length-axis contrast after the base-contrast result

The multi-phrasing `concise_vs_base` vector at layer 8 shortened greedy output
at weight 2, but did not transfer to the ASC sampling configuration. On 30
paired-seed GSM8K examples, weights `1,2,3` changed average tokens from the
664.0 baseline to `812.9,850.3,695.1`; none compressed. This rejects that vector
as a robust compression intervention.

Cross-phrasing consistency alone was insufficient because every target prompt
contained a `Reasoning instruction:` prefix while every source prompt lacked
one. The common representation may therefore encode the presence of an extra
instruction rather than its conciseness semantics.

The matched contrast keeps the wrapper, problem, CoT prompt, and final token on
both sides:

```text
target(q) = Reasoning instruction: <concise phrasing>
            Question: q
            Let's think step by step.

source(q) = Reasoning instruction: <matched verbose phrasing>
            Question: q
            Let's think step by step.
```

It saves the explicitly oriented unit direction
`unit(mean(h(concise) - h(verbose)))`. Positive addition therefore retains the
intended concise semantic sign without relabeling or post-hoc vector reversal.

```bash
python extract_instruction_conciseness_layers.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --problems_path datasets/gsm8k/train.jsonl \
  --output_dir vectors/instruction_length_axis_gsm8k_train100 \
  --file_prefix qwen7b_instruction_length_axis \
  --contrast_mode concise_vs_verbose \
  --layer_indices 8,12,16,20,24 \
  --num_samples 100 \
  --activation_batch_size 8 \
  --max_input_tokens 8192 \
  --device_map auto \
  --dtype bfloat16 \
  --attn_impl sdpa
```
