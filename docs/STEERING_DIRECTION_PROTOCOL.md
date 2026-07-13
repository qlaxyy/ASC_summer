# Steering Direction Protocol

## Why the ASC endpoint direction remains a reproduction baseline

The ASC paper defines

```text
v_endpoint = mean_i(h(short_answer_i) - h(long_answer_i))
h <- h + gamma*v_endpoint, gamma > 0.
```

The released extraction code computes `short - long`, but the released
generation code subtracts the result. Our independent vectors align strongly
with the author vector, and two signed-gamma GSM8K experiments consistently
showed that negative gamma shortened generation while positive gamma lengthened
it. Therefore, reversing the vector must not be presented as validation of the
paper formula.

`asc_endpoint` remains available to reproduce and document this discrepancy. A
causally flipped vector, if used for an engineering diagnostic, is a modified
method rather than confirmation of the original theory.

## `actadd_prompt`: matched prompt contrast

The new method tests the target-minus-source addition rule without using answer
endpoints. For every training problem `q_i`, it constructs two prompts with the
same question prefix:

```text
P_short(q_i) = Question: {q_i}
               Let's solve this briefly and directly, using concise
               mathematical reasoning without repeated verification or
               unnecessary prose.

P_long(q_i)  = Question: {q_i}
               Let's think step by step.
```

The long prompt is exactly the evaluator's `paper_cot` prompt. No generated
short or long answer is included. At the input of transformer block `L`, the
last prompt-token representations define

```text
v_prompt = mean_i(
    h_pre_L(P_short(q_i))[-1]
    - h_pre_L(P_long(q_i))[-1]
).
```

Evaluation applies only the declared target-minus-source direction:

```text
h_pre_L(P_long(q))[-1] <- h_pre_L(P_long(q))[-1] + gamma*v_prompt,
gamma >= 0.
```

The intervention runs only on the initial prompt forward pass. Cached
single-token generation steps are not modified. This is an ASC-compatible,
last-token adaptation of ActAdd: it follows ActAdd's target-minus-source,
positive-addition, block-input, prompt-time contract while retaining a single
dataset-level vector and the existing ASC evaluator.

The vector metadata requires all of the following:

- `direction=short_minus_long`;
- `activation_site=block_input`;
- `recommended_injection_sign=add`;
- `recommended_injection_scope=prompt_only`;
- `matching_prompt_mode=paper_cot`;
- nonnegative gamma values only.

The evaluator rejects a mismatch by default. It also refuses to causally flip
an `actadd_prompt` vector with `--orient_vector_output_path`, because doing so
would conceal a negative result rather than test the stated theory.

## Extraction command

```bash
python extract_steering_vector.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --pairs_path pairs/qwen7b_math_train_deepseek_pairs_100_checked.json \
  --output_vector_path vectors/qwen7b_actadd_prompt_short_minus_long_pre_layer20.pt \
  --vector_method actadd_prompt \
  --layer_index 20 \
  --direction short_minus_long \
  --activation_site block_input \
  --max_input_tokens 8192 \
  --activation_batch_size 4 \
  --device_map auto \
  --dtype bfloat16 \
  --attn_impl sdpa
```

Although this command reads the checked pair JSON, `actadd_prompt` uses only
each row's problem text. The saved CoTs do not enter the vector.

## Development evaluation on GSM8K train

Use only nonnegative gamma values. Do not pass
`--orient_vector_output_path`.

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/train.jsonl \
  --limit 50 \
  --prompt_mode paper_cot \
  --candidate_gammas 0,0.02,0.05,0.1,0.2 \
  --steering_vector_path vectors/qwen7b_actadd_prompt_short_minus_long_pre_layer20.pt \
  --layer_index 20 \
  --injection_sign add \
  --injection_site block_input \
  --injection_scope prompt_only \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --attn_impl sdpa \
  --save_details none \
  --per_gamma_output_dir results/qwen7b_gsm8k_train_actadd_prompt
```

Choose gamma on the train/development results, freeze it, and evaluate the
unchanged configuration once on an unused test split. If positive addition does
not compress generation, report that outcome; do not select a negative sign and
rename the vector.

## Interpretation

The two methods answer different questions:

| Method | Representation contrast | Injection | Claim tested |
|---|---|---|---|
| `asc_endpoint` | complete short-answer endpoint minus complete long-answer endpoint | legacy all-token or explicit diagnostic scope | reproduction of released ASC |
| `actadd_prompt` | concise prompt minus step-by-step prompt before generation | positive add, block input, prompt only | matched target-minus-source causal test |

Success of `actadd_prompt` would show that the endpoint mismatch caused the ASC
sign reversal. Failure would be stronger evidence that this model/layer does not
support the proposed positive short-direction intervention.

## `actadd_prompt_aligned`: shared-question multi-token contrast

The first prompt-contrast experiment did not produce stable compression on the
100-example GSM8K test run. One important confound was that its final compared
tokens were different words: the concise prompt ended in `prose.` while the
detailed prompt ended in `step.`. A last-token activation difference can
therefore encode lexical identity more strongly than reasoning style.

`actadd_prompt_aligned` keeps the target-minus-source sign but removes that
confound. It places the style instruction before the question:

```text
P_short(q) = Reasoning instruction: Be concise and direct. ...
             Question: {q}

P_long(q)  = Reasoning instruction: Work through the solution carefully ...
             Question: {q}
```

The final `N=8` token IDs are required to be identical for every pair. They are
the same question suffix under two different style contexts. At block input,
the method extracts one vector per training question:

```text
v_i = mean(h_pre_L(P_short(q_i))[-8:])
      - mean(h_pre_L(P_long(q_i))[-8:])
v = mean_i(v_i)
```

During evaluation, the same positive vector is added to the corresponding last
eight prompt positions:

```text
h_pre_L(P_long(q))[-8:] <- h_pre_L(P_long(q))[-8:] + gamma*v
```

No answer or chain of thought is used during extraction. The vector source is
GSM8K train questions; GSM8K test is used only for evaluation. The intervention
still runs only during the initial prompt forward pass, not on cached generation
tokens. This remains an ActAdd-style dataset-level adaptation rather than the
original ActAdd sequence-vector implementation.

### Extract the aligned vector from GSM8K train

The extraction requires only 200 short prompt forwards and 200 long prompt
forwards. It does not generate 400 answers.

```bash
python extract_steering_vector.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --problems_path datasets/gsm8k/train.jsonl \
  --num_samples 200 \
  --output_vector_path vectors/qwen7b_actadd_prompt_aligned_gsm8k_train200_layer20.pt \
  --vector_method actadd_prompt_aligned \
  --pool_last_n_tokens 8 \
  --layer_index 20 \
  --direction short_minus_long \
  --activation_site block_input \
  --max_input_tokens 8192 \
  --activation_batch_size 8 \
  --device_map auto \
  --dtype bfloat16 \
  --attn_impl sdpa
```

The command prints `suffix match:100.00%`. Any lower value is treated as an
error and no vector is saved.

### Low-cost GSM8K test sweep

This sweep keeps complete outputs for all 100 questions in each gamma JSON.
`--paired_batch_seeds` gives corresponding batches the same sampler seed at
every gamma, reducing Monte Carlo noise in the comparison.

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/test.jsonl \
  --limit 100 \
  --prompt_mode actadd_aligned_long \
  --candidate_gammas 0,0.05,0.1,0.2 \
  --steering_vector_path vectors/qwen7b_actadd_prompt_aligned_gsm8k_train200_layer20.pt \
  --layer_index 20 \
  --injection_sign add \
  --injection_site block_input \
  --injection_scope prompt_only \
  --injection_token_count 8 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --seed 42 \
  --paired_batch_seeds \
  --attn_impl sdpa \
  --save_details all \
  --use_our_eval \
  --per_gamma_output_dir results/qwen7b_actadd_aligned_gsm8k_test100
```

Gamma zero in this sweep is the only valid baseline for the aligned method,
because its prompt is intentionally different from the earlier `paper_cot`
prompt. Do not compare its token count directly with a baseline generated using
a different prompt or seed protocol.

### Preliminary aligned-prompt result and baseline confound

The first 100-example run of this method produced:

| gamma | accuracy | average tokens | change from its own baseline |
|---:|---:|---:|---:|
| 0.0 | 89% | 1458.6 | baseline |
| 0.2 | 89% | 1406.8 | -51.8 (-3.55%) |

This is a small positive-direction compression signal with unchanged aggregate
accuracy, but it is not comparable with the earlier roughly 746-token baseline.
Two protocol changes affected gamma zero: `actadd_aligned_long` explicitly asks
the model to work carefully and step by step before presenting the question,
and the original implementation of `--paired_batch_seeds` reset the seed before
each batch. The former strongly encourages longer reasoning.

## `actadd_prompt_paper_aligned`: unchanged paper baseline

This follow-up keeps the original evaluator prompt byte-for-byte unchanged:

```text
P_source(q) = Question: {q}
              Let's think step by step.
```

Only the target prompt receives a concise prefix:

```text
P_target(q) = Reasoning instruction: Be concise and direct. ...
              Question: {q}
              Let's think step by step.
```

Thus both prompts still have identical final eight token IDs, but gamma zero
uses the original `paper_cot` prompt. Positive addition continues to test the
unchanged target-minus-source formula. This method is the appropriate one for
comparison with the earlier paper-prompt experiments.

The paired sampler protocol is also corrected. Gamma zero now follows the
ordinary continuous seeded RNG trajectory and records the state before each
batch. Every nonzero gamma replays those recorded states. Pairing therefore no
longer changes the baseline output merely by resetting its seed per batch.

### Extract from GSM8K train

```bash
python extract_steering_vector.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --problems_path datasets/gsm8k/train.jsonl \
  --num_samples 200 \
  --output_vector_path vectors/qwen7b_actadd_paper_aligned_gsm8k_train200_layer20.pt \
  --vector_method actadd_prompt_paper_aligned \
  --pool_last_n_tokens 8 \
  --layer_index 20 \
  --direction short_minus_long \
  --activation_site block_input \
  --max_input_tokens 8192 \
  --activation_batch_size 8 \
  --device_map auto \
  --dtype bfloat16 \
  --attn_impl sdpa
```

### Two-point GSM8K test

```bash
python eval_asc_paper.py \
  --model_name /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset gsm8k \
  --local_data_path datasets/gsm8k/test.jsonl \
  --limit 100 \
  --prompt_mode paper_cot \
  --candidate_gammas 0,0.2 \
  --steering_vector_path vectors/qwen7b_actadd_paper_aligned_gsm8k_train200_layer20.pt \
  --layer_index 20 \
  --injection_sign add \
  --injection_site block_input \
  --injection_scope prompt_only \
  --injection_token_count 8 \
  --batch_size 8 \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --seed 42 \
  --paired_batch_seeds \
  --attn_impl sdpa \
  --save_details all \
  --use_our_eval \
  --per_gamma_output_dir results/qwen7b_actadd_paper_aligned_gsm8k_test100
```

The metadata rejects `actadd_aligned_long` for this vector and requires
`paper_cot`, preventing another accidental baseline change.
