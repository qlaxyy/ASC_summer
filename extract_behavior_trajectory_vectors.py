"""Extract multi-layer concise-minus-verbose behavioral trajectory vectors.

Selected target-model generations are teacher-forced back through the model.
Response-token activations are pooled into equal relative-progress bins so a
short and a long trajectory can be compared without endpoint or token alignment.
The saved vector is always the positive concise-target minus verbose-source mean.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def parse_layer_indices(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--layer_indices must contain at least one layer.")
    if any(value < 0 for value in values):
        raise ValueError("Layer indices cannot be negative.")
    return list(dict.fromkeys(values))


def relative_bin_bounds(token_count: int, num_bins: int) -> list[tuple[int, int]]:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if token_count < num_bins:
        raise ValueError(
            f"A {token_count}-token response cannot be split into {num_bins} bins."
        )
    bounds = []
    for index in range(num_bins):
        start = math.floor(index * token_count / num_bins)
        end = math.floor((index + 1) * token_count / num_bins)
        if end <= start:
            raise RuntimeError("Relative binning produced an empty bin.")
        bounds.append((start, end))
    return bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force clean concise/verbose target-model trajectories and "
            "save a unit target-minus-source vector at each requested layer."
        )
    )
    parser.add_argument(
        "--pairs_path", default="pairs/gsm8k_behavior_trajectory_pairs.json"
    )
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--output_dir", default="vectors/gsm8k_behavior_trajectory_train"
    )
    parser.add_argument("--file_prefix", default="qwen7b_behavior_trajectory")
    parser.add_argument("--layer_indices", default="8,12,16,20,24")
    parser.add_argument("--num_relative_bins", type=int, default=8)
    parser.add_argument("--min_pairs", type=int, default=30)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--activation_batch_size", type=int, default=2)
    parser.add_argument("--attn_impl", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def prepare_teacher_forced_sequences(
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    kind: str,
    num_bins: int,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    if kind not in {"concise", "verbose"}:
        raise ValueError(f"Unknown trajectory kind: {kind}")
    sequences: list[dict[str, Any]] = []
    for row in pairs:
        prompt = str(row[f"{kind}_prompt"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = [int(token) for token in row[f"{kind}_output_token_ids"]]
        if len(output_ids) < num_bins:
            raise ValueError(
                f"Row {row['source_row_index']} has only {len(output_ids)} "
                f"{kind} response tokens for {num_bins} bins."
            )
        combined = prompt_ids + output_ids
        if len(combined) > max_input_tokens:
            raise ValueError(
                f"Row {row['source_row_index']} {kind} sequence has "
                f"{len(combined)} tokens, above --max_input_tokens={max_input_tokens}."
            )
        sequences.append(
            {
                "input_ids": combined,
                "response_start": len(prompt_ids),
                "response_end": len(combined),
                "source_row_index": int(row["source_row_index"]),
            }
        )
    return sequences


def extract_relative_pooled_activations(
    model: Any,
    tokenizer: Any,
    sequences: list[dict[str, Any]],
    input_device: Any,
    layer_indices: list[int],
    num_bins: int,
    batch_size: int,
    description: str,
) -> dict[int, Any]:
    import torch
    from tqdm import tqdm

    from asc_steering_utils import first_tensor
    from asc_steering_utils import get_transformer_layers

    layers = get_transformer_layers(model)
    for layer_index in layer_indices:
        if layer_index >= len(layers):
            raise IndexError(
                f"layer_index={layer_index} is out of range for {len(layers)} layers."
            )
    collected: dict[int, list[Any]] = {layer: [] for layer in layer_indices}
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        for offset in tqdm(range(0, len(sequences), batch_size), desc=description):
            batch = sequences[offset : offset + batch_size]
            max_length = max(len(row["input_ids"]) for row in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                pad_token_id,
                dtype=torch.long,
                device=input_device,
            )
            attention_mask = torch.zeros(
                (len(batch), max_length),
                dtype=torch.long,
                device=input_device,
            )
            response_spans: list[tuple[int, int]] = []
            for batch_index, row in enumerate(batch):
                ids = torch.tensor(
                    row["input_ids"], dtype=torch.long, device=input_device
                )
                left_padding = max_length - len(row["input_ids"])
                input_ids[batch_index, left_padding:] = ids
                attention_mask[batch_index, left_padding:] = 1
                response_spans.append(
                    (
                        left_padding + int(row["response_start"]),
                        left_padding + int(row["response_end"]),
                    )
                )

            captured: dict[int, Any] = {}
            handles = []
            for layer_index in layer_indices:

                def capture_hook(
                    _module: Any,
                    _inputs: Any,
                    output: Any,
                    *,
                    current_layer: int = layer_index,
                ) -> None:
                    hidden = first_tensor(output)
                    pooled_rows = []
                    for batch_index, (response_start, response_end) in enumerate(
                        response_spans
                    ):
                        response = hidden[batch_index, response_start:response_end, :]
                        bins = [
                            response[start:end, :].mean(dim=0)
                            for start, end in relative_bin_bounds(
                                int(response.shape[0]), num_bins
                            )
                        ]
                        pooled_rows.append(torch.stack(bins, dim=0))
                    captured[current_layer] = (
                        torch.stack(pooled_rows, dim=0).detach().float().cpu()
                    )

                handles.append(layers[layer_index].register_forward_hook(capture_hook))
            try:
                base_model = getattr(model, "model", None)
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                if base_model is not None:
                    base_model(**model_inputs)
                else:
                    model(**model_inputs)
            finally:
                for handle in handles:
                    handle.remove()

            missing = set(layer_indices) - set(captured)
            if missing:
                raise RuntimeError(f"Hooks did not capture layers: {sorted(missing)}")
            for layer_index in layer_indices:
                collected[layer_index].append(captured[layer_index])

    return {
        layer_index: torch.cat(layer_batches, dim=0)
        for layer_index, layer_batches in collected.items()
    }


def main() -> None:
    args = parse_args()
    if args.num_relative_bins <= 0:
        raise ValueError("--num_relative_bins must be positive.")
    if args.min_pairs <= 0:
        raise ValueError("--min_pairs must be positive.")
    if args.activation_batch_size <= 0:
        raise ValueError("--activation_batch_size must be positive.")
    layer_indices = parse_layer_indices(args.layer_indices)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import asc_steering_utils as utils

    pairs_path = Path(args.pairs_path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair report not found: {pairs_path}")
    pair_report = json.loads(pairs_path.read_text(encoding="utf-8"))
    selected_pairs = [
        row for row in pair_report["pairs"] if row.get("selected_for_extraction")
    ]
    if len(selected_pairs) < args.min_pairs:
        raise ValueError(
            f"Only {len(selected_pairs)} selected pairs; --min_pairs={args.min_pairs}."
        )
    selected_indices = [int(row["source_row_index"]) for row in selected_pairs]
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("Selected pair source_row_index values must be unique.")
    exclusion_indices = sorted(
        {int(index) for index in pair_report["attempted_row_indices"]}
    )
    model_name = args.model_name or pair_report["config"]["model_name"]
    output_dir = Path(args.output_dir)
    diagnostics_path = output_dir / f"{args.file_prefix}_layer_diagnostics.json"
    vector_outputs = [
        output_dir / f"{args.file_prefix}_layer{layer_index}.pt"
        for layer_index in layer_indices
    ]
    planned_outputs = (
        vector_outputs
        + [Path(str(path) + ".metadata.json") for path in vector_outputs]
        + [diagnostics_path]
    )
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing trajectory outputs: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    print("Behavioral trajectory vector extraction")
    print(f"  pairs:       {len(selected_pairs)}")
    print(f"  bins:        {args.num_relative_bins}")
    print(f"  layers:      {layer_indices}")
    print(f"  model:       {model_name}")
    print(f"  output dir:  {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    concise_sequences = prepare_teacher_forced_sequences(
        tokenizer,
        selected_pairs,
        "concise",
        args.num_relative_bins,
        args.max_input_tokens,
    )
    verbose_sequences = prepare_teacher_forced_sequences(
        tokenizer,
        selected_pairs,
        "verbose",
        args.num_relative_bins,
        args.max_input_tokens,
    )

    dtype = (
        "auto"
        if args.dtype == "auto"
        else {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_impl,
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    max_memory = utils.parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
    input_device = utils.get_input_device(model)
    if hasattr(model, "hf_device_map"):
        print(f"  hf_device_map: {utils.summarize_device_map(model.hf_device_map)}")

    concise_activations = extract_relative_pooled_activations(
        model,
        tokenizer,
        concise_sequences,
        input_device,
        layer_indices,
        args.num_relative_bins,
        args.activation_batch_size,
        "Concise trajectories",
    )
    verbose_activations = extract_relative_pooled_activations(
        model,
        tokenizer,
        verbose_sequences,
        input_device,
        layer_indices,
        args.num_relative_bins,
        args.activation_batch_size,
        "Verbose trajectories",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = time.time()
    layer_reports: list[dict[str, Any]] = []
    concise_token_mean = sum(row["concise_tokens"] for row in selected_pairs) / len(
        selected_pairs
    )
    verbose_token_mean = sum(row["verbose_tokens"] for row in selected_pairs) / len(
        selected_pairs
    )

    for layer_index in layer_indices:
        target = concise_activations[layer_index]
        source = verbose_activations[layer_index]
        bin_differences = target - source
        pair_differences = bin_differences.mean(dim=1)
        raw_mean = pair_differences.mean(dim=0)
        raw_norm = torch.linalg.vector_norm(raw_mean)
        if not torch.isfinite(raw_norm) or raw_norm <= 0:
            raise ValueError(f"Invalid mean vector norm at layer {layer_index}.")
        unit_vector = raw_mean / raw_norm
        pair_norms = torch.linalg.vector_norm(pair_differences, dim=1)
        margins = pair_differences @ unit_vector
        margin_std = margins.std(unbiased=False).clamp_min(1e-12)
        target_projections = target @ unit_vector
        source_projections = source @ unit_vector

        bin_cosines: list[float] = []
        for bin_index in range(args.num_relative_bins):
            bin_mean = bin_differences[:, bin_index, :].mean(dim=0)
            bin_norm = torch.linalg.vector_norm(bin_mean).clamp_min(1e-12)
            bin_cosines.append(float((bin_mean @ unit_vector / bin_norm).item()))

        vector_path = output_dir / f"{args.file_prefix}_layer{layer_index}.pt"
        torch.save(unit_vector.cpu(), vector_path)
        report = {
            "layer_index": layer_index,
            "vector_path": str(vector_path),
            "num_pairs": len(selected_pairs),
            "num_relative_bins": args.num_relative_bins,
            "hidden_size": int(unit_vector.shape[0]),
            "raw_mean_vector_norm": float(raw_norm.item()),
            "mean_pair_vector_norm": float(pair_norms.mean().item()),
            "mean_resultant_ratio": float(
                (raw_norm / pair_norms.mean().clamp_min(1e-12)).item()
            ),
            "orientation_pair_agreement": float((margins > 0).float().mean().item()),
            "mean_projection_margin": float(margins.mean().item()),
            "projection_margin_std": float(margins.std(unbiased=False).item()),
            "projection_signal_to_noise": float((margins.mean() / margin_std).item()),
            "relative_bin_direction_cosines": bin_cosines,
            "relative_bin_cosine_min": min(bin_cosines),
            "relative_bin_cosine_mean": sum(bin_cosines) / len(bin_cosines),
            "concise_projection_mean": float(target_projections.mean().item()),
            "verbose_projection_mean": float(source_projections.mean().item()),
        }
        metadata = {
            **report,
            "model_name": model_name,
            "vector_type": "behavior_trajectory_conciseness",
            "vector_method": "behavior_trajectory_relative_progress_bins",
            "direction": "concise_target_trajectory_minus_verbose_source_trajectory",
            "direction_semantics": "target_minus_source",
            "formula": (
                "unit(mean_pair(mean_relative_bin("
                "h(concise_trajectory)-h(verbose_trajectory))))"
            ),
            "activation_site": "block_output",
            "matching_injection_site": "block_output",
            "recommended_injection_sign": "add",
            "recommended_injection_scope": "all_tokens",
            "recommended_injection_token_count": 1,
            "recommended_vector_normalization": "unit_l2",
            "supported_intervention_modes": [
                "additive",
                "conditional_additive",
            ],
            "recommended_intervention_mode": "additive",
            "matching_prompt_mode": "paper_cot",
            "positive_gamma_only": True,
            "contains_generated_answers": True,
            "trajectory_alignment": "equal_relative_progress_bins",
            "trajectory_target_prompt_mode": "paper_cot_concise_prefix",
            "trajectory_source_prompt_mode": "paper_cot_verbose_prefix",
            "selected_row_indices": selected_indices,
            "causal_validation_exclusion_indices": exclusion_indices,
            "source_path": pair_report["dataset_path"],
            "source_row_count": int(pair_report["source_row_count"]),
            "pairs_path": str(pairs_path),
            "mean_concise_tokens": concise_token_mean,
            "mean_verbose_tokens": verbose_token_mean,
            "mean_pair_compression": 1.0 - concise_token_mean / verbose_token_mean,
            "saved_vector_is_unit_l2": True,
            "created_at_unix": created_at,
        }
        utils.write_json(str(vector_path) + ".metadata.json", metadata)
        layer_reports.append(report)

    ranking = sorted(
        layer_reports,
        key=lambda row: (
            row["relative_bin_cosine_min"],
            row["projection_signal_to_noise"],
            row["mean_resultant_ratio"],
        ),
        reverse=True,
    )
    utils.write_json(
        diagnostics_path,
        {
            "method": "behavior_trajectory_relative_progress_bins",
            "direction_semantics": "target_minus_source",
            "pairs_path": str(pairs_path),
            "num_pairs": len(selected_pairs),
            "num_relative_bins": args.num_relative_bins,
            "mean_concise_tokens": concise_token_mean,
            "mean_verbose_tokens": verbose_token_mean,
            "mean_pair_compression": 1.0 - concise_token_mean / verbose_token_mean,
            "ranking": ranking,
            "created_at_unix": created_at,
        },
    )

    print("Layer diagnostics (representation ranking only)")
    for row in ranking:
        print(
            f"  layer={row['layer_index']:>2} | "
            f"agreement={row['orientation_pair_agreement']:.2%} | "
            f"SNR={row['projection_signal_to_noise']:.3f} | "
            f"resultant={row['mean_resultant_ratio']:.3f} | "
            f"bin_cos_min={row['relative_bin_cosine_min']:.3f}"
        )
    print(f"  report: {diagnostics_path}")


if __name__ == "__main__":
    main()
