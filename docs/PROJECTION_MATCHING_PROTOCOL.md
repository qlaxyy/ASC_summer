# Projection-Matched Conciseness Steering

## Motivation

The paired prompt control shows that DeepSeek-R1-Distill-Qwen-7B responds
strongly to explicit length instructions: the concise condition averaged 536.1
tokens and the verbose condition 1178.97 tokens on the same 30 GSM8K questions.
However, positive additive steering with either concise-minus-base or
concise-minus-verbose mean vectors increased sampled output length. The prompt
effect exists, but unconstrained `h <- h + gamma*v` does not reproduce it.

The official `microsoft/llm-steer-instruct` repository includes an `adjust_rs`
intervention that controls the scalar residual projection rather than adding an
arbitrary fixed displacement. The new mode adapts that idea while exposing a
continuous interpolation coefficient.

## Definition

For each layer, extraction still defines the explicitly oriented unit vector:

```text
v = unit(mean_i(h_concise_i - h_verbose_i))
```

It now also records the measured concise target:

```text
p_concise = mean_i(h_concise_i dot v)
```

At inference, for every selected residual `h`, projection matching applies:

```text
p       = h dot v
p_new   = p + alpha * (p_concise - p)
h_new   = h + alpha * (p_concise - p) * v
```

- `alpha=0` is the untouched model.
- `alpha=1` exactly matches the measured concise scalar projection (up to model
  dtype precision).
- `0 < alpha < 1` interpolates toward that target.

This is not a sign reversal and not a relabeled additive gamma. The vector is
still concise-minus-verbose; positive alpha always moves the scalar projection
toward the measured concise state.

## Safety and reproducibility constraints

`projection_match` requires all of the following:

- vector sidecar metadata containing `projection_target`;
- `--vector_normalization unit_l2`;
- `--injection_sign add`;
- extraction and injection at the same site;
- alpha values in `[0,1]` unless an explicitly labeled diagnostic override is
  requested.

Historical vectors must be re-extracted because their sidecars do not contain
the absolute concise projection target. Existing additive evaluation remains
unchanged under the default `--intervention_mode additive`.

## Cloud verification

After pulling the commit, first run the hook unit tests. They require no model
load or generation:

```bash
python -m unittest tests.test_projection_matching -v
```

Then regenerate the layer vectors and projection metadata:

```bash
python extract_instruction_conciseness_layers.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --problems_path datasets/gsm8k/train.jsonl \
  --output_dir vectors/instruction_length_projection_gsm8k_train100 \
  --file_prefix qwen7b_instruction_length_projection \
  --contrast_mode concise_vs_verbose \
  --layer_indices 8,12,16,20,24 \
  --num_samples 100 \
  --activation_batch_size 8 \
  --max_input_tokens 8192 \
  --device_map auto \
  --dtype bfloat16 \
  --attn_impl sdpa
```

The first causal screen uses layer 20, paper sampling, paired batch RNG states,
and four interpolation values:

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/test.jsonl \
  --limit 30 \
  --prompt_mode paper_cot \
  --candidate_gammas 0,0.25,0.5,1 \
  --steering_vector_path vectors/instruction_length_projection_gsm8k_train100/qwen7b_instruction_length_projection_layer20.pt \
  --layer_index 20 \
  --intervention_mode projection_match \
  --injection_sign add \
  --injection_site block_output \
  --injection_scope sequence_all \
  --injection_token_count 1 \
  --vector_normalization unit_l2 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --paired_batch_seeds \
  --seed 42 \
  --attn_impl sdpa \
  --save_details all \
  --use_our_eval \
  --per_gamma_output_dir results/projection_match_layer20_paper30
```

The CLI retains the historical name `candidate_gammas`; in projection mode the
printed gamma value is mathematically `alpha`, not an additive vector weight.

## Observed result and status

The layer-20 projection-matching screen was run on 30 GSM8K examples with ASC
sampling and paired batch RNG states:

| alpha | accuracy | average tokens | change from alpha 0 |
|---:|---:|---:|---:|
| 0.00 | 80.00% | 664.0 | baseline |
| 0.10 | 90.00% | 768.3 | +15.7% |
| 0.25 | 86.67% | 873.9 | +31.6% |
| 0.50 | 86.67% | 698.3 | +5.2% |

Every nonzero interpolation increased output length. The apparent accuracy
improvements were accompanied by longer reasoning and are not compression
gains. This rejects scalar projection matching as a compression intervention
for this model and decoding protocol. Do not escalate to `alpha=1`, change the
sign, or search additional layers merely to obtain a favorable sample.

The implementation remains in the repository as a documented negative result
and reusable diagnostic. It must not be presented as a successful ASC variant.
