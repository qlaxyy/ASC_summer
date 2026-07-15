"""Generate clean concise/verbose reasoning pairs from the target model.

Both outputs are sampled from the same model with paired RNG states.  A pair is
selected for trajectory extraction only when both answers are correct, neither
output is capped or obviously corrupted/repetitive, and the concise response is
materially shorter than the verbose response.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


def assess_pair_quality(
    concise_correct: bool,
    verbose_correct: bool,
    concise_tokens: int,
    verbose_tokens: int,
    concise_capped: bool,
    verbose_capped: bool,
    concise_repetition: bool,
    verbose_repetition: bool,
    concise_corruption: bool,
    verbose_corruption: bool,
    min_pair_compression: float,
    min_concise_tokens: int,
) -> dict[str, Any]:
    failures: list[str] = []
    compression = 1.0 - concise_tokens / verbose_tokens if verbose_tokens > 0 else -1.0
    if not concise_correct:
        failures.append("concise_answer_incorrect")
    if not verbose_correct:
        failures.append("verbose_answer_incorrect")
    if concise_tokens < min_concise_tokens:
        failures.append("concise_too_short")
    if compression < min_pair_compression:
        failures.append("insufficient_pair_compression")
    if concise_capped:
        failures.append("concise_length_capped")
    if verbose_capped:
        failures.append("verbose_length_capped")
    if concise_repetition:
        failures.append("concise_repetition_artifact")
    if verbose_repetition:
        failures.append("verbose_repetition_artifact")
    if concise_corruption:
        failures.append("concise_corruption_artifact")
    if verbose_corruption:
        failures.append("verbose_corruption_artifact")
    return {
        "pair_compression_fraction": compression,
        "quality_eligible": not failures,
        "rejection_reasons": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate target-model concise/verbose GSM8K train trajectories and "
            "automatically retain clean, correct, materially shorter pairs."
        )
    )
    parser.add_argument(
        "--model_name",
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--dataset_path", default="datasets/gsm8k/train.jsonl")
    parser.add_argument(
        "--output_path", default="pairs/gsm8k_behavior_trajectory_pairs.json"
    )
    parser.add_argument("--target_accepted_pairs", type=int, default=60)
    parser.add_argument("--max_candidate_samples", type=int, default=120)
    parser.add_argument("--min_pair_compression", type=float, default=0.30)
    parser.add_argument("--min_concise_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens_concise", type=int, default=2048)
    parser.add_argument("--max_new_tokens_verbose", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--attn_impl", default="sdpa")
    parser.add_argument("--num_gpus", default="auto", choices=["auto", "1", "2"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--max_memory", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_accepted_pairs <= 0:
        raise ValueError("--target_accepted_pairs must be positive.")
    if args.max_candidate_samples < args.target_accepted_pairs:
        raise ValueError(
            "--max_candidate_samples must be at least --target_accepted_pairs."
        )
    if not 0 < args.min_pair_compression < 1:
        raise ValueError("--min_pair_compression must be between 0 and 1.")
    if args.min_concise_tokens <= 0:
        raise ValueError("--min_concise_tokens must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_new_tokens_concise <= 0 or args.max_new_tokens_verbose <= 0:
        raise ValueError("Both max-new-token limits must be positive.")


def generate_batch_with_token_ids(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    input_device: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float,
) -> tuple[list[str], list[list[int]], list[int]]:
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    ).to(input_device)
    prompt_width = int(inputs["input_ids"].shape[1])
    kwargs: dict[str, Any] = dict(inputs)
    do_sample = temperature > 0
    kwargs.update(
        {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "repetition_penalty": repetition_penalty,
        }
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            kwargs["top_k"] = top_k
        if min_p is not None:
            kwargs["min_p"] = min_p
    with torch.no_grad():
        try:
            outputs = model.generate(**kwargs)
        except TypeError as exc:
            if "min_p" in kwargs and "min_p" in str(exc):
                kwargs.pop("min_p", None)
                outputs = model.generate(**kwargs)
            else:
                raise

    generated = outputs[:, prompt_width:].detach().cpu()
    token_ids: list[list[int]] = []
    for row in generated:
        ids = [
            int(token) for token in row.tolist() if int(token) != tokenizer.pad_token_id
        ]
        token_ids.append(ids)
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    counts = [len(ids) for ids in token_ids]
    return texts, token_ids, counts


def main() -> None:
    args = parse_args()
    validate_args(args)

    import random

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from answer_utils import extract_ground_truth
    from eval_asc_paper import build_paper_prompt
    from eval_asc_paper import capture_rng_state
    from eval_asc_paper import get_input_device
    from eval_asc_paper import has_obvious_corruption_artifact
    from eval_asc_paper import has_repetition_artifact
    from eval_asc_paper import parse_max_memory
    from eval_asc_paper import parse_prediction
    from eval_asc_paper import read_jsonl
    from eval_asc_paper import resolve_device_map
    from eval_asc_paper import restore_rng_state
    from eval_asc_paper import set_seed
    from eval_asc_paper import summarize_device_map
    from eval_asc_paper import torch_dtype_from_arg
    from eval_asc_paper import write_json_atomic

    output_path = Path(args.output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; choose a new --output_path."
        )
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    rows = list(read_jsonl(dataset_path))
    if args.max_candidate_samples > len(rows):
        raise ValueError("--max_candidate_samples exceeds dataset size.")
    candidate_indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(candidate_indices)
    candidate_indices = candidate_indices[: args.max_candidate_samples]

    args.resolved_device_map = resolve_device_map(args)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_arg(args.dtype),
        "attn_implementation": args.attn_impl,
        "trust_remote_code": True,
    }
    if args.resolved_device_map is not None:
        model_kwargs["device_map"] = args.resolved_device_map
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).eval()
    if hasattr(model, "hf_device_map"):
        print(f"hf_device_map: {summarize_device_map(model.hf_device_map)}")
    input_device = get_input_device(model)

    report: dict[str, Any] = {
        "status": "running",
        "method": "paired_target_model_behavior_trajectory_generation",
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "resolved_device_map"
        },
        "dataset_path": str(dataset_path),
        "source_row_count": len(rows),
        "candidate_row_indices": candidate_indices,
        "attempted_row_indices": [],
        "selected_row_indices": [],
        "pairs": [],
        "created_at_unix": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_count = 0
    pbar = tqdm(
        range(0, len(candidate_indices), args.batch_size),
        desc="Generating behavior pairs",
    )
    for batch_number, offset in enumerate(pbar):
        if accepted_count >= args.target_accepted_pairs:
            break
        batch_indices = candidate_indices[offset : offset + args.batch_size]
        batch_rows = [rows[index] for index in batch_indices]
        questions = [str(row["question"]) for row in batch_rows]
        ground_truths = [extract_ground_truth("gsm8k", row) for row in batch_rows]
        concise_prompts = [
            build_paper_prompt(question, "paper_cot_concise_prefix", tokenizer)
            for question in questions
        ]
        verbose_prompts = [
            build_paper_prompt(question, "paper_cot_verbose_prefix", tokenizer)
            for question in questions
        ]

        set_seed(args.seed + batch_number)
        paired_state = capture_rng_state()
        concise_outputs, concise_ids, concise_counts = generate_batch_with_token_ids(
            model,
            tokenizer,
            concise_prompts,
            input_device,
            args.max_new_tokens_concise,
            args.temperature,
            args.top_p,
            args.top_k,
            args.min_p,
            args.repetition_penalty,
        )
        restore_rng_state(paired_state)
        verbose_outputs, verbose_ids, verbose_counts = generate_batch_with_token_ids(
            model,
            tokenizer,
            verbose_prompts,
            input_device,
            args.max_new_tokens_verbose,
            args.temperature,
            args.top_p,
            args.top_k,
            args.min_p,
            args.repetition_penalty,
        )

        for position, source_index in enumerate(batch_indices):
            concise_pred, concise_correct = parse_prediction(
                concise_outputs[position], ground_truths[position], "gsm8k", False
            )
            verbose_pred, verbose_correct = parse_prediction(
                verbose_outputs[position], ground_truths[position], "gsm8k", False
            )
            concise_capped = concise_counts[position] >= args.max_new_tokens_concise
            verbose_capped = verbose_counts[position] >= args.max_new_tokens_verbose
            concise_repetition = has_repetition_artifact(concise_outputs[position])
            verbose_repetition = has_repetition_artifact(verbose_outputs[position])
            concise_corruption = has_obvious_corruption_artifact(
                concise_outputs[position]
            )
            verbose_corruption = has_obvious_corruption_artifact(
                verbose_outputs[position]
            )
            quality = assess_pair_quality(
                concise_correct=concise_correct,
                verbose_correct=verbose_correct,
                concise_tokens=concise_counts[position],
                verbose_tokens=verbose_counts[position],
                concise_capped=concise_capped,
                verbose_capped=verbose_capped,
                concise_repetition=concise_repetition,
                verbose_repetition=verbose_repetition,
                concise_corruption=concise_corruption,
                verbose_corruption=verbose_corruption,
                min_pair_compression=args.min_pair_compression,
                min_concise_tokens=args.min_concise_tokens,
            )
            selected = (
                quality["quality_eligible"]
                and accepted_count < args.target_accepted_pairs
            )
            if selected:
                accepted_count += 1
                report["selected_row_indices"].append(source_index)
            report["attempted_row_indices"].append(source_index)
            report["pairs"].append(
                {
                    "source_row_index": source_index,
                    "question": questions[position],
                    "gt_answer": ground_truths[position],
                    "concise_prompt": concise_prompts[position],
                    "verbose_prompt": verbose_prompts[position],
                    "concise_output": concise_outputs[position],
                    "verbose_output": verbose_outputs[position],
                    "concise_output_token_ids": concise_ids[position],
                    "verbose_output_token_ids": verbose_ids[position],
                    "concise_tokens": concise_counts[position],
                    "verbose_tokens": verbose_counts[position],
                    "concise_pred_answer": concise_pred,
                    "verbose_pred_answer": verbose_pred,
                    "concise_correct": concise_correct,
                    "verbose_correct": verbose_correct,
                    "concise_length_capped": concise_capped,
                    "verbose_length_capped": verbose_capped,
                    "concise_repetition_artifact": concise_repetition,
                    "verbose_repetition_artifact": verbose_repetition,
                    "concise_corruption_artifact": concise_corruption,
                    "verbose_corruption_artifact": verbose_corruption,
                    **quality,
                    "selected_for_extraction": selected,
                }
            )

        report["accepted_pair_count"] = accepted_count
        report["attempted_pair_count"] = len(report["attempted_row_indices"])
        write_json_atomic(output_path, report)
        pbar.set_postfix(
            accepted=f"{accepted_count}/{args.target_accepted_pairs}",
            attempted=len(report["attempted_row_indices"]),
        )

    report["status"] = (
        "target_reached"
        if accepted_count >= args.target_accepted_pairs
        else "candidate_budget_exhausted"
    )
    report["finished_at_unix"] = time.time()
    write_json_atomic(output_path, report)
    print("RESULT")
    print(f"  status:    {report['status']}")
    print(f"  accepted:  {accepted_count}/{args.target_accepted_pairs}")
    print(f"  attempted: {len(report['attempted_row_indices'])}")
    print(f"  output:    {output_path}")
    if accepted_count < args.target_accepted_pairs:
        print(
            "  [warn] Fewer clean pairs than requested. Inspect rejection_reasons "
            "before deciding whether to generate a second shard."
        )


if __name__ == "__main__":
    main()
