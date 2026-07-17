"""Select a causally effective conciseness steering vector on held-out train data.

The representation direction is never flipped: every candidate must already be
target-minus-source and is applied as ``h <- h + gamma * v`` with gamma > 0.
This script selects the layer, positive strength, and optional generated-token
start threshold whose outputs show robust held-out compression without
unacceptable quality regressions.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any


def parse_int_list(text: str, label: str = "integer") -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"The {label} list cannot be empty.")
    if any(value < 0 for value in values):
        raise ValueError(f"{label.capitalize()} values cannot be negative.")
    return list(dict.fromkeys(values))


def parse_positive_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("The gamma list cannot be empty.")
    if any(value <= 0 for value in values):
        raise ValueError(
            "Causal conciseness selection accepts only positive nonzero gammas; "
            "gamma=0 is evaluated once automatically as the baseline."
        )
    return list(dict.fromkeys(values))


def choose_held_out_rows(
    row_count: int,
    excluded_indices: set[int],
    sample_count: int,
    seed: int,
) -> list[int]:
    available = [index for index in range(row_count) if index not in excluded_indices]
    if sample_count <= 0:
        raise ValueError("--validation_samples must be positive.")
    if len(available) < sample_count:
        raise ValueError(
            f"Only {len(available)} rows remain after excluding extraction rows; "
            f"cannot draw {sample_count} held-out samples."
        )
    return random.Random(seed).sample(available, sample_count)


def assess_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    min_compression: float,
    max_accuracy_drop: float,
    max_repetition_increase: float,
    max_corruption_rate: float,
    max_length_capped_increase: float,
    min_trimmed_compression: float = 0.0,
    min_pairwise_win_margin: float = -1.0,
    trim_fraction: float = 0.1,
) -> dict[str, Any]:
    baseline_tokens = float(baseline["avg_tokens"])
    if baseline_tokens <= 0:
        raise ValueError("Baseline average tokens must be positive.")
    compression = 1.0 - float(candidate["avg_tokens"]) / baseline_tokens
    accuracy_drop = float(baseline["accuracy"]) - float(candidate["accuracy"])
    repetition_increase = float(candidate["repetition_artifact_rate"]) - float(
        baseline["repetition_artifact_rate"]
    )
    length_capped_increase = float(candidate["length_capped_rate"]) - float(
        baseline["length_capped_rate"]
    )
    corruption_rate = float(candidate["corruption_artifact_rate"])

    paired_diagnostics: dict[str, Any] = {}
    baseline_details = baseline.get("detailed_results") or []
    candidate_details = candidate.get("detailed_results") or []
    robust_checks_requested = (
        min_trimmed_compression > 0 or min_pairwise_win_margin > -1
    )
    if robust_checks_requested:
        if len(baseline_details) != len(candidate_details) or not baseline_details:
            raise ValueError(
                "Robust causal selection requires paired detailed_results. "
                "Use --save_details all."
            )
        if not 0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5).")
        paired_rows = []
        for baseline_row, candidate_row in zip(
            baseline_details, candidate_details, strict=True
        ):
            if baseline_row.get("question") != candidate_row.get("question"):
                raise ValueError("Baseline/candidate detailed results are not paired.")
            baseline_length = int(baseline_row["tokens"])
            candidate_length = int(candidate_row["tokens"])
            paired_rows.append(
                (baseline_length - candidate_length, baseline_length)
            )
        shorter_count = sum(saving > 0 for saving, _ in paired_rows)
        longer_count = sum(saving < 0 for saving, _ in paired_rows)
        same_count = len(paired_rows) - shorter_count - longer_count
        win_margin = (shorter_count - longer_count) / len(paired_rows)
        trim_count = math.floor(len(paired_rows) * trim_fraction)
        ranked = sorted(paired_rows, key=lambda row: row[0])
        retained = ranked[trim_count : len(ranked) - trim_count or None]
        retained_baseline_mean = sum(row[1] for row in retained) / len(retained)
        retained_saving_mean = sum(row[0] for row in retained) / len(retained)
        trimmed_compression = retained_saving_mean / retained_baseline_mean
        paired_diagnostics = {
            "paired_shorter_count": shorter_count,
            "paired_longer_count": longer_count,
            "paired_same_count": same_count,
            "pairwise_win_margin": win_margin,
            "trim_fraction": trim_fraction,
            "trimmed_compression_fraction": trimmed_compression,
        }

    failures: list[str] = []
    epsilon = 1e-12
    if compression + epsilon < min_compression:
        failures.append("insufficient_compression")
    if accuracy_drop > max_accuracy_drop + epsilon:
        failures.append("accuracy_drop")
    if repetition_increase > max_repetition_increase + epsilon:
        failures.append("repetition_increase")
    if corruption_rate > max_corruption_rate + epsilon:
        failures.append("corruption_artifact")
    if length_capped_increase > max_length_capped_increase + epsilon:
        failures.append("length_capped_increase")
    if paired_diagnostics:
        if (
            paired_diagnostics["trimmed_compression_fraction"] + epsilon
            < min_trimmed_compression
        ):
            failures.append("insufficient_trimmed_compression")
        if (
            paired_diagnostics["pairwise_win_margin"] + epsilon
            < min_pairwise_win_margin
        ):
            failures.append("negative_pairwise_win_margin")

    return {
        "compression_fraction": compression,
        "accuracy_drop": accuracy_drop,
        "repetition_increase": repetition_increase,
        "corruption_rate": corruption_rate,
        "length_capped_increase": length_capped_increase,
        "eligible": not failures,
        "rejection_reasons": failures,
        **paired_diagnostics,
    }


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["compression_fraction"],
            row["accuracy"],
            -row["corruption_artifact_rate"],
            -row["repetition_artifact_rate"],
            -row["gamma"],
            row.get("injection_start_generated_token", 0),
        ),
    )


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"detailed_results", "failure_cases", "run_metrics"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one positive-add conciseness intervention (including an "
            "optional delayed start) using held-out GSM8K train generations; "
            "the GSM8K test set is never read."
        )
    )
    parser.add_argument(
        "--model_name",
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--local_data_path", default="datasets/gsm8k/train.jsonl")
    parser.add_argument("--vector_dir", required=True)
    parser.add_argument("--file_prefix", required=True)
    parser.add_argument("--layer_indices", default="16,20,24")
    parser.add_argument("--candidate_gammas", default="0.5,1.0")
    parser.add_argument(
        "--candidate_start_tokens",
        default="0",
        help=(
            "Comma-separated generated-token counts to leave unsteered before "
            "activating each positive-gamma candidate. Zero preserves immediate "
            "steering. Delayed candidates require --injection_scope all_tokens."
        ),
    )
    parser.add_argument("--validation_samples", type=int, default=30)
    parser.add_argument("--validation_seed", type=int, default=314159)
    parser.add_argument(
        "--injection_scope",
        choices=["sequence_all", "all_tokens"],
        default="sequence_all",
        help=(
            "sequence_all steers every prompt/generation position. all_tokens "
            "uses the evaluator's legacy name for final-prompt plus generation "
            "positions, which is appropriate for response-trajectory vectors."
        ),
    )
    parser.add_argument("--min_compression", type=float, default=0.05)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.04)
    parser.add_argument("--max_repetition_increase", type=float, default=0.04)
    parser.add_argument("--max_corruption_rate", type=float, default=0.0)
    parser.add_argument("--max_length_capped_increase", type=float, default=0.04)
    parser.add_argument("--min_trimmed_compression", type=float, default=0.02)
    parser.add_argument("--min_pairwise_win_margin", type=float, default=0.0)
    parser.add_argument("--trim_fraction", type=float, default=0.1)
    parser.add_argument(
        "--output_report",
        default="results/causal_conciseness_selection.json",
    )
    parser.add_argument(
        "--output_vector_path",
        default="vectors/qwen7b_causally_selected_conciseness.pt",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--prompt_mode", default="paper_cot", choices=["paper_cot"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_impl", default="sdpa")
    parser.add_argument("--num_gpus", default="auto", choices=["auto", "1", "2"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--max_memory", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--save_details",
        default="all",
        choices=["all", "none"],
        help="Save every held-out chain in the single selection report.",
    )
    return parser.parse_args()


def validate_thresholds(args: argparse.Namespace) -> None:
    for name in (
        "min_compression",
        "max_accuracy_drop",
        "max_repetition_increase",
        "max_corruption_rate",
        "max_length_capped_increase",
        "min_trimmed_compression",
    ):
        value = getattr(args, name)
        if value < 0:
            raise ValueError(f"--{name} cannot be negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if not -1 <= args.min_pairwise_win_margin <= 1:
        raise ValueError("--min_pairwise_win_margin must be in [-1, 1].")
    if not 0 <= args.trim_fraction < 0.5:
        raise ValueError("--trim_fraction must be in [0, 0.5).")
    if (
        (args.min_trimmed_compression > 0 or args.min_pairwise_win_margin > -1)
        and args.save_details != "all"
    ):
        raise ValueError(
            "Robust paired selection requires --save_details all."
        )


def main() -> None:
    args = parse_args()

    # Pure selection helpers remain importable in the CPU-only local checkout.
    # The model runtime is loaded only when this AutoDL entrypoint actually runs.
    import torch
    import torch.backends.cudnn as cudnn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from answer_utils import extract_ground_truth
    from eval_asc_paper import evaluate_gamma
    from eval_asc_paper import get_input_device
    from eval_asc_paper import load_and_validate_vector_metadata
    from eval_asc_paper import load_steering_vector
    from eval_asc_paper import parse_max_memory
    from eval_asc_paper import read_jsonl
    from eval_asc_paper import resolve_device_map
    from eval_asc_paper import set_seed
    from eval_asc_paper import summarize_device_map
    from eval_asc_paper import torch_dtype_from_arg
    from eval_asc_paper import write_json_atomic

    validate_thresholds(args)
    layers = parse_int_list(args.layer_indices, "layer")
    gammas = parse_positive_float_list(args.candidate_gammas)
    start_tokens = parse_int_list(args.candidate_start_tokens, "start-token")
    if any(value >= args.max_new_tokens for value in start_tokens):
        raise ValueError(
            "Every delayed start token must be smaller than --max_new_tokens."
        )
    if any(start_tokens) and args.injection_scope != "all_tokens":
        raise ValueError(
            "Delayed candidates require --injection_scope all_tokens."
        )
    vector_dir = Path(args.vector_dir)
    if Path(args.output_vector_path).exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing selected vector: "
            f"{args.output_vector_path}. Choose a new --output_vector_path."
        )

    candidate_specs: list[dict[str, Any]] = []
    excluded_indices: set[int] = set()
    source_row_counts: set[int] = set()
    for layer_index in layers:
        vector_path = vector_dir / f"{args.file_prefix}_layer{layer_index}.pt"
        if not vector_path.exists():
            raise FileNotFoundError(f"Missing candidate vector: {vector_path}")
        metadata = load_and_validate_vector_metadata(
            str(vector_path),
            injection_sign="add",
            injection_site="block_output",
            injection_scope=args.injection_scope,
            injection_token_count=1,
            vector_normalization="unit_l2",
            intervention_mode="additive",
            prompt_mode=args.prompt_mode,
            allow_mismatch=False,
        )
        if metadata is None:
            raise FileNotFoundError(
                f"Candidate metadata is required: {vector_path}.metadata.json"
            )
        direction = str(metadata.get("direction", ""))
        direction_semantics = metadata.get("direction_semantics")
        if (
            "target_minus_source" not in direction
            and direction_semantics != "target_minus_source"
        ):
            raise ValueError(
                f"{vector_path} is not declared target-minus-source: {direction!r}"
            )
        if metadata.get("positive_gamma_only") is not True:
            raise ValueError(f"{vector_path} is not marked positive_gamma_only.")
        metadata_layer = int(metadata.get("layer_index", layer_index))
        if metadata_layer != layer_index:
            raise ValueError(
                f"Layer mismatch for {vector_path}: metadata says {metadata_layer}."
            )
        validation_exclusions = metadata.get(
            "causal_validation_exclusion_indices",
            metadata["selected_row_indices"],
        )
        excluded_indices.update(int(index) for index in validation_exclusions)
        source_row_counts.add(int(metadata["source_row_count"]))
        candidate_specs.append(
            {
                "layer_index": layer_index,
                "vector_path": str(vector_path),
                "metadata": metadata,
            }
        )

    rows = list(read_jsonl(args.local_data_path))
    if len(source_row_counts) != 1 or next(iter(source_row_counts)) != len(rows):
        raise ValueError(
            "Candidate metadata source_row_count does not match the held-out "
            f"training file ({len(rows)} rows)."
        )
    validation_indices = choose_held_out_rows(
        len(rows), excluded_indices, args.validation_samples, args.validation_seed
    )
    samples = [
        {
            "question": rows[index]["question"],
            "gt_answer": extract_ground_truth("gsm8k", rows[index]),
        }
        for index in validation_indices
    ]

    # evaluate_gamma consumes this stable evaluator interface.
    args.dataset = "gsm8k"
    args.num_runs = 1
    args.deterministic_batch_seeds = False
    args.paired_batch_seeds = True
    args.qwen3_enable_thinking = False
    args.qwen3_add_no_think_tag = False
    args.use_our_eval = False
    args.save_failures = False
    args.injection_sign = "add"
    args.injection_site = "block_output"
    args.injection_token_count = 1
    args.injection_start_generated_token = 0
    args.vector_normalization = "unit_l2"
    args.intervention_mode = "additive"
    args.resolved_projection_target = None
    args.resolved_device_map = resolve_device_map(args)

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    cudnn.benchmark = True

    print("Causal conciseness-vector selection")
    print(f"  held-out source: {args.local_data_path}")
    print(f"  train rows excluded from validation: {len(excluded_indices)}")
    print(f"  validation samples: {len(samples)}")
    print(f"  layers: {layers}")
    print(f"  positive gammas: {gammas}")
    print(f"  generated-token start candidates: {start_tokens}")
    print(
        "  generation protocol: "
        f"paper_cot, temperature={args.temperature}, top_p={args.top_p}, "
        f"repetition_penalty={args.repetition_penalty}"
    )
    print(f"  intervention: block_output / {args.injection_scope} / positive additive")

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
        print(f"  hf_device_map: {summarize_device_map(model.hf_device_map)}")
    input_device = get_input_device(model)
    paired_rng_states: dict[tuple[int, int], dict[str, Any]] = {}

    criteria = {
        "min_compression": args.min_compression,
        "max_accuracy_drop": args.max_accuracy_drop,
        "max_repetition_increase": args.max_repetition_increase,
        "max_corruption_rate": args.max_corruption_rate,
        "max_length_capped_increase": args.max_length_capped_increase,
        "min_trimmed_compression": args.min_trimmed_compression,
        "min_pairwise_win_margin": args.min_pairwise_win_margin,
        "trim_fraction": args.trim_fraction,
    }
    report: dict[str, Any] = {
        "status": "running",
        "method": "held_out_causal_layer_gamma_and_start_selection",
        "invariant": (
            "v=concise_target-source; h<-h+gamma*v; gamma>0; "
            f"scope={args.injection_scope}; delayed_start_candidates={start_tokens}"
        ),
        "model_name": args.model_name,
        "data": {
            "dataset": "gsm8k_train_validation",
            "path": args.local_data_path,
            "row_count": len(rows),
            "extraction_indices_excluded": sorted(excluded_indices),
            "validation_exclusion_indices": sorted(excluded_indices),
            "validation_indices": validation_indices,
            "validation_seed": args.validation_seed,
        },
        "generation": {
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "paired_batch_rng": True,
        },
        "criteria": criteria,
        "baseline": None,
        "candidates": [],
        "selected": None,
        "created_at_unix": time.time(),
    }

    print("[baseline] gamma=0 (computed once)")
    baseline = evaluate_gamma(
        gamma=0.0,
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        args=args,
        input_device=input_device,
        steering_vec_cpu=None,
        layer_index=layers[0],
        paired_rng_states=paired_rng_states,
    )
    report["baseline"] = baseline
    write_json_atomic(args.output_report, report)
    print(
        f"  acc={baseline['accuracy']:.4f}, tokens={baseline['avg_tokens']:.1f}, "
        f"repeat={baseline['repetition_artifact_rate']:.2%}, "
        f"corrupt={baseline['corruption_artifact_rate']:.2%}"
    )

    for spec in candidate_specs:
        vector = load_steering_vector(spec["vector_path"])
        vector_norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(vector_norm) or vector_norm <= 0:
            raise ValueError(f"Invalid vector norm for {spec['vector_path']}")
        vector = vector / vector_norm
        for start_token in start_tokens:
            args.injection_start_generated_token = start_token
            for gamma in gammas:
                print(
                    f"[candidate] layer={spec['layer_index']}, gamma={gamma:.6g}, "
                    f"start_after={start_token}"
                )
                metrics = evaluate_gamma(
                    gamma=gamma,
                    model=model,
                    tokenizer=tokenizer,
                    samples=samples,
                    args=args,
                    input_device=input_device,
                    steering_vec_cpu=vector,
                    layer_index=spec["layer_index"],
                    paired_rng_states=paired_rng_states,
                )
                assessment = assess_candidate(baseline, metrics, **criteria)
                candidate = {
                    **metrics,
                    **assessment,
                    "layer_index": spec["layer_index"],
                    "vector_path": spec["vector_path"],
                    "injection_start_generated_token": start_token,
                }
                report["candidates"].append(candidate)
                write_json_atomic(args.output_report, report)
                verdict = "PASS" if candidate["eligible"] else "reject"
                print(
                    f"  {verdict}: acc={candidate['accuracy']:.4f}, "
                    f"tokens={candidate['avg_tokens']:.1f}, "
                    f"compression={candidate['compression_fraction']:.2%}, "
                    f"reasons={candidate['rejection_reasons']}"
                )

    selected = select_best_candidate(report["candidates"])
    if selected is None:
        report["status"] = "null_result"
        report["conclusion"] = (
            "No positive-add target-minus-source candidate met all causal quality "
            "criteria. No selected vector was written."
        )
        write_json_atomic(args.output_report, report)
        print("RESULT: no candidate passed; refusing to label a vector effective.")
        print(f"  report: {args.output_report}")
        return

    selected_vector = load_steering_vector(selected["vector_path"])
    selected_vector = selected_vector / torch.linalg.vector_norm(selected_vector)
    output_path = Path(args.output_vector_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected_vector.cpu(), output_path)
    source_metadata = next(
        spec["metadata"]
        for spec in candidate_specs
        if spec["vector_path"] == selected["vector_path"]
    )
    selected_summary = compact_metrics(selected)
    output_metadata = {
        **source_metadata,
        "vector_path": str(output_path),
        "vector_type": "causally_selected_conciseness",
        "source_vector_path": selected["vector_path"],
        "causal_selection_report": args.output_report,
        "selected_layer_index": selected["layer_index"],
        "recommended_positive_gamma": selected["gamma"],
        "recommended_injection_sign": "add",
        "matching_injection_site": "block_output",
        "recommended_injection_scope": args.injection_scope,
        "recommended_injection_token_count": 1,
        "recommended_injection_start_generated_token": selected[
            "injection_start_generated_token"
        ],
        "recommended_vector_normalization": "unit_l2",
        "recommended_intervention_mode": "additive",
        "positive_gamma_only": True,
        "causal_validation_dataset": "gsm8k_train_validation",
        "causal_validation_indices": validation_indices,
        "causal_validation_excluded_extraction_indices": sorted(excluded_indices),
        "causal_validation_exclusion_indices": sorted(excluded_indices),
        "causal_selection_criteria": criteria,
        "causal_baseline": compact_metrics(baseline),
        "causal_selected_metrics": selected_summary,
        "saved_vector_is_unit_l2": True,
        "created_at_unix": time.time(),
    }
    write_json_atomic(str(output_path) + ".metadata.json", output_metadata)
    report["status"] = "selected"
    report["selected"] = selected_summary
    report["output_vector_path"] = str(output_path)
    write_json_atomic(args.output_report, report)

    print("RESULT: selected a held-out causal candidate")
    print(
        f"  layer={selected['layer_index']}, gamma={selected['gamma']:.6g}, "
        f"start_after={selected['injection_start_generated_token']}, "
        f"compression={selected['compression_fraction']:.2%}, "
        f"accuracy_drop={selected['accuracy_drop']:.2%}"
    )
    print(f"  vector: {output_path}")
    print(f"  report: {args.output_report}")


if __name__ == "__main__":
    main()
