"""Extract literature-aligned conciseness instruction vectors at many layers.

This follows the difference-in-means instruction-steering recipe:

1. Pair each base prompt with a version carrying a concise instruction.
2. Read the final shared prompt-token residual at several block outputs.
3. Average target-minus-base differences independently at every layer.
4. Save a unit-L2 vector per layer for continuous positive steering.

All requested layers are captured during the same two sets of prompt forwards,
so a layer diagnostic does not multiply model inference cost by layer count.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import asc_steering_utils as utils


CONCISE_INSTRUCTION_PHRASINGS = (
    "Be extremely concise.",
    "Be extremely brief.",
    "Keep the reasoning extremely short.",
    "Use only the essential mathematical steps.",
    "Avoid repeated verification and unnecessary prose.",
    "Give the shortest sufficient reasoning.",
    "Use minimal mathematical reasoning.",
)

VERBOSE_INSTRUCTION_PHRASINGS = (
    "Be extremely detailed.",
    "Be extremely thorough.",
    "Make the reasoning extensive and fully elaborated.",
    "Show every relevant mathematical step.",
    "Include repeated verification and extensive explanatory prose.",
    "Give fully elaborated reasoning.",
    "Use extensive mathematical reasoning.",
)


def torch_dtype_from_arg(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_layer_indices(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--layer_indices must contain at least one layer.")
    if any(value < 0 for value in values):
        raise ValueError("--layer_indices cannot contain negative values.")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unit conciseness-instruction vectors at multiple layers."
    )
    parser.add_argument(
        "--model_name",
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--problems_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_prefix", default="qwen7b_instruction_conciseness")
    parser.add_argument(
        "--contrast_mode",
        choices=["concise_vs_base", "concise_vs_verbose"],
        default="concise_vs_base",
        help=(
            "Use the published instruction-vs-base contrast, or a matched "
            "concise-vs-verbose contrast that cancels the generic instruction prefix."
        ),
    )
    parser.add_argument("--layer_indices", default="8,12,16,20,24")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--activation_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn_impl",
        default="sdpa",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    return parser.parse_args()


def build_prompt_pair(
    problem: str,
    phrasing_index: int,
    contrast_mode: str,
) -> tuple[str, str]:
    base = utils.ACTADD_LONG_PROMPT_TEMPLATE.format(problem=problem)
    instruction = CONCISE_INSTRUCTION_PHRASINGS[phrasing_index]
    target = f"Reasoning instruction: {instruction}\n{base}"
    if contrast_mode == "concise_vs_base":
        source = base
    elif contrast_mode == "concise_vs_verbose":
        verbose_instruction = VERBOSE_INSTRUCTION_PHRASINGS[phrasing_index]
        source = f"Reasoning instruction: {verbose_instruction}\n{base}"
    else:
        raise ValueError(f"Unsupported contrast mode: {contrast_mode}")
    return target, source


@torch.no_grad()
def extract_multi_layer_last_token_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    input_device: torch.device,
    layer_indices: list[int],
    max_input_tokens: int,
    batch_size: int,
    description: str,
) -> dict[int, torch.Tensor]:
    layers = utils.get_transformer_layers(model)
    if max(layer_indices) >= len(layers):
        raise IndexError(
            f"Requested layer {max(layer_indices)}, but model has {len(layers)} layers."
        )

    activations: dict[int, list[torch.Tensor]] = {
        layer_index: [] for layer_index in layer_indices
    }
    old_truncation_side = tokenizer.truncation_side
    old_padding_side = tokenizer.padding_side
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    try:
        batches = list(utils.iter_batches(texts, batch_size))
        for batch in tqdm(batches, desc=description):
            captured: dict[int, torch.Tensor] = {}
            handles = []

            for layer_index in layer_indices:

                def capture_hook(
                    _module: Any,
                    _hook_inputs: Any,
                    output: Any,
                    *,
                    current_layer: int = layer_index,
                ) -> None:
                    captured[current_layer] = (
                        utils.first_tensor(output)[:, -1, :].detach().float().cpu()
                    )

                handles.append(layers[layer_index].register_forward_hook(capture_hook))

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                return_token_type_ids=False,
            ).to(input_device)
            try:
                base_model = getattr(model, "model", None)
                if base_model is not None:
                    base_model(**inputs, use_cache=False)
                else:
                    model(**inputs, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            missing = set(layer_indices) - set(captured)
            if missing:
                raise RuntimeError(f"Hooks did not capture layers: {sorted(missing)}")
            for layer_index in layer_indices:
                activations[layer_index].append(captured[layer_index])
    finally:
        tokenizer.truncation_side = old_truncation_side
        tokenizer.padding_side = old_padding_side

    return {
        layer_index: torch.cat(layer_batches, dim=0)
        for layer_index, layer_batches in activations.items()
    }


def main() -> None:
    args = parse_args()
    utils.set_seed(args.seed)
    args.dtype = torch_dtype_from_arg(args.dtype)
    layer_indices = parse_layer_indices(args.layer_indices)

    source_path = Path(args.problems_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Problems file not found: {source_path}")
    rows = (
        utils.read_jsonl(source_path)
        if source_path.suffix.lower() == ".jsonl"
        else utils.read_json(source_path)
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("--problems_path must contain non-empty problem rows.")

    source_row_count = len(rows)
    selected_indices = list(range(source_row_count))
    if args.num_samples == 0 or args.num_samples < -1:
        raise ValueError("--num_samples must be positive or -1.")
    if 0 < args.num_samples < source_row_count:
        selected_indices = random.Random(args.seed).sample(
            selected_indices, args.num_samples
        )
        rows = [rows[index] for index in selected_indices]

    active_phrasings = CONCISE_INSTRUCTION_PHRASINGS[
        : min(len(rows), len(CONCISE_INSTRUCTION_PHRASINGS))
    ]
    phrasing_indices = [index % len(active_phrasings) for index in range(len(rows))]
    prompt_pairs = [
        build_prompt_pair(
            utils.problem_from_row(row),
            phrasing_index,
            args.contrast_mode,
        )
        for row, phrasing_index in zip(rows, phrasing_indices)
    ]
    target_prompts = [pair[0] for pair in prompt_pairs]
    source_prompts = [pair[1] for pair in prompt_pairs]

    model, tokenizer, input_device = utils.load_model_and_tokenizer(args)
    suffix_matches = []
    for target, source in prompt_pairs:
        target_ids = tokenizer(target)["input_ids"]
        source_ids = tokenizer(source)["input_ids"]
        suffix_matches.append(
            bool(target_ids and source_ids and target_ids[-1] == source_ids[-1])
        )
    suffix_agreement = sum(suffix_matches) / len(suffix_matches)
    if suffix_agreement < 1.0:
        raise ValueError(
            "Instruction steering requires an identical final token for target "
            f"and source prompts; agreement={suffix_agreement:.2%}."
        )

    print("Instruction-conciseness layer extraction")
    print(f"  model:        {args.model_name}")
    print(f"  source:       {source_path}")
    print(f"  contrast:     {args.contrast_mode}")
    print(f"  samples:      {len(rows)}")
    print(f"  layers:       {layer_indices}")
    print(f"  final-token agreement: {suffix_agreement:.2%}")
    print(f"  input device: {input_device}")

    target_activations = extract_multi_layer_last_token_activations(
        model,
        tokenizer,
        target_prompts,
        input_device,
        layer_indices,
        args.max_input_tokens,
        args.activation_batch_size,
        "Target concise prompts",
    )
    source_activations = extract_multi_layer_last_token_activations(
        model,
        tokenizer,
        source_prompts,
        input_device,
        layer_indices,
        args.max_input_tokens,
        args.activation_batch_size,
        "Source contrast prompts",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_reports = []
    created_at = time.time()

    for layer_index in layer_indices:
        differences = target_activations[layer_index] - source_activations[layer_index]
        raw_mean = differences.mean(dim=0)
        raw_mean_norm = raw_mean.norm()
        if float(raw_mean_norm.item()) <= 1e-12:
            raise RuntimeError(f"Layer {layer_index} produced a near-zero mean vector.")
        unit_vector = raw_mean / raw_mean_norm
        pair_norms = differences.norm(dim=1)
        margins = differences @ unit_vector
        margin_std = margins.std(unbiased=False).clamp_min(1e-12)
        phrasing_cosines = []
        for phrasing_index in range(len(active_phrasings)):
            mask = torch.tensor(
                [index == phrasing_index for index in phrasing_indices],
                dtype=torch.bool,
            )
            phrasing_mean = differences[mask].mean(dim=0)
            phrasing_mean = phrasing_mean / phrasing_mean.norm().clamp_min(1e-12)
            phrasing_cosines.append(float((phrasing_mean @ unit_vector).item()))
        report = {
            "layer_index": layer_index,
            "num_vectors": int(differences.shape[0]),
            "hidden_size": int(differences.shape[1]),
            "raw_mean_vector_norm": float(raw_mean_norm.item()),
            "mean_pair_vector_norm": float(pair_norms.mean().item()),
            "mean_resultant_ratio": float(
                (raw_mean_norm / pair_norms.mean().clamp_min(1e-12)).item()
            ),
            "orientation_pair_agreement": float((margins > 0).float().mean().item()),
            "mean_projection_margin": float(margins.mean().item()),
            "projection_margin_std": float(margin_std.item()),
            "projection_signal_to_noise": float((margins.mean() / margin_std).item()),
            "phrasing_direction_cosine_mean": float(
                sum(phrasing_cosines) / len(phrasing_cosines)
            ),
            "phrasing_direction_cosine_min": float(min(phrasing_cosines)),
            "phrasing_direction_cosines": phrasing_cosines,
        }
        vector_path = output_dir / f"{args.file_prefix}_layer{layer_index}.pt"
        torch.save(unit_vector.cpu(), vector_path)
        report["vector_path"] = str(vector_path)

        metadata = {
            **report,
            "model_name": args.model_name,
            "vector_method": f"instruction_conciseness_{args.contrast_mode}",
            "direction": f"{args.contrast_mode}_target_minus_source",
            "formula": "unit(mean_i(h(target_i)[-1] - h(source_i)[-1]))",
            "activation_site": "block_output",
            "matching_injection_site": "block_output",
            "recommended_injection_sign": "add",
            "recommended_injection_scope": "sequence_all",
            "recommended_injection_token_count": 1,
            "recommended_vector_normalization": "unit_l2",
            "matching_prompt_mode": "paper_cot",
            "positive_gamma_only": True,
            "representation_token": "identical_final_prompt_token",
            "contains_generated_answers": False,
            "paper_cot_baseline_unchanged": True,
            "source_path": str(source_path),
            "source_row_count": source_row_count,
            "selected_row_indices": selected_indices,
            "sample_seed": args.seed,
            "target_prompt_example": target_prompts[0],
            "source_prompt_example": source_prompts[0],
            "concise_instruction_phrasings": list(active_phrasings),
            "verbose_instruction_phrasings": (
                list(VERBOSE_INSTRUCTION_PHRASINGS[: len(active_phrasings)])
                if args.contrast_mode == "concise_vs_verbose"
                else []
            ),
            "concise_instruction_phrasing_indices": phrasing_indices,
            "final_token_agreement": suffix_agreement,
            "saved_vector_is_unit_l2": True,
            "created_at_unix": created_at,
        }
        utils.write_json(str(vector_path) + ".metadata.json", metadata)
        layer_reports.append(report)

    ranking = sorted(
        layer_reports,
        key=lambda row: (
            row["phrasing_direction_cosine_min"],
            row["projection_signal_to_noise"],
            row["mean_resultant_ratio"],
        ),
        reverse=True,
    )
    summary = {
        "method": f"instruction_conciseness_{args.contrast_mode}",
        "model_name": args.model_name,
        "layers": layer_reports,
        "diagnostic_ranking": [row["layer_index"] for row in ranking],
        "ranking_warning": (
            "Representation separation is not causal validation. Generation must "
            "still test the leading layers on held-out prompts."
        ),
    }
    summary_path = output_dir / f"{args.file_prefix}_layer_diagnostics.json"
    utils.write_json(summary_path, summary)

    print("\nLayer diagnostics (representation ranking only)")
    for row in ranking:
        print(
            f"  layer={row['layer_index']:>2} | "
            f"agreement={row['orientation_pair_agreement']:.2%} | "
            f"SNR={row['projection_signal_to_noise']:.3f} | "
            f"resultant={row['mean_resultant_ratio']:.3f} | "
            f"phrase_cos_min={row['phrasing_direction_cosine_min']:.3f}"
        )
    print(f"  report: {summary_path}")


if __name__ == "__main__":
    main()
