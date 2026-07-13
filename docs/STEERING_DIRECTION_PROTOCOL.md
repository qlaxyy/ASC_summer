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
