"""Extract ASC endpoint or ActAdd prompt-contrast steering vectors.

Endpoint extraction reads manually filtered answer pairs. Prompt-contrast
extraction can instead read unlabeled problem rows and never generates answers.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch

import asc_steering_utils as utils


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ASC endpoint or ActAdd prompt-contrast vectors."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--pairs_path",
        type=str,
        help="Checked JSON pair list used by asc_endpoint or actadd_prompt.",
    )
    inputs.add_argument(
        "--problems_path",
        type=str,
        help=(
            "JSONL/JSON problem rows. actadd_prompt_aligned should use an "
            "unlabeled training split such as datasets/gsm8k/train.jsonl."
        ),
    )
    parser.add_argument("--output_vector_path", type=str, required=True)
    parser.add_argument("--layer_index", type=str, default="auto")
    parser.add_argument(
        "--vector_method",
        choices=[
            "asc_endpoint",
            "asc_endpoint_raw",
            "actadd_prompt",
            "actadd_prompt_aligned",
            "actadd_prompt_paper_aligned",
        ],
        default="asc_endpoint",
        help=(
            "asc_endpoint compares complete short/long answers after the saved "
            "long_prompt. asc_endpoint_raw instead uses the unchanged raw problem "
            "as the shared prefix, matching README-style raw inference. "
            "actadd_prompt constructs matched concise/step-by-step prompts from "
            "each problem. actadd_prompt_aligned places the style instruction "
            "before a shared question suffix and pools shared question tokens. "
            "actadd_prompt_paper_aligned keeps paper_cot exactly unchanged as "
            "the long/source prompt and prepends only the concise instruction."
        ),
    )
    parser.add_argument(
        "--pool_last_n_tokens",
        type=int,
        default=None,
        help=(
            "Average the final N activation positions per contrast prompt. "
            "Defaults to 8 for actadd_prompt_aligned and 1 otherwise."
        ),
    )
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--activation_batch_size", type=int, default=4)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help=(
            "Randomly select this many input rows with --seed. -1 uses all rows."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["short_minus_long", "long_minus_short"],
        default="short_minus_long",
    )
    parser.add_argument(
        "--activation_site",
        choices=["block_input", "block_output"],
        default="block_input",
        help=(
            "Residual-stream location to read. block_input matches ActAdd/CAST "
            "pre-hook implementations and must be paired with the same "
            "--injection_site during evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utils.set_seed(args.seed)
    args.dtype = torch_dtype_from_arg(args.dtype)
    if args.pool_last_n_tokens is None:
        args.pool_last_n_tokens = (
            8
            if args.vector_method
            in {"actadd_prompt_aligned", "actadd_prompt_paper_aligned"}
            else 1
        )
    if args.pool_last_n_tokens <= 0:
        raise ValueError("--pool_last_n_tokens must be positive.")

    source_path = Path(args.pairs_path or args.problems_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    source_payload = (
        utils.read_jsonl(source_path)
        if source_path.suffix.lower() == ".jsonl"
        else utils.read_json(source_path)
    )
    source_manifest = None
    source_container_row_count = None
    if isinstance(source_payload, dict) and isinstance(source_payload.get("pairs"), list):
        source_manifest = source_payload
        container_pairs = source_payload["pairs"]
        source_container_row_count = len(container_pairs)
        has_selection_flags = any(
            "selected_for_extraction" in row for row in container_pairs
        )
        pairs = (
            [row for row in container_pairs if row.get("selected_for_extraction")]
            if has_selection_flags
            else container_pairs
        )
    else:
        pairs = source_payload
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(
            "The input path must contain a non-empty row list or a report with pairs[]."
        )
    source_row_count = len(pairs)
    selected_indices = list(range(source_row_count))
    if 0 < args.num_samples < source_row_count:
        rng = random.Random(args.seed)
        selected_indices = rng.sample(selected_indices, args.num_samples)
        pairs = [pairs[index] for index in selected_indices]
    elif args.num_samples == 0:
        raise ValueError("--num_samples must be positive or -1.")
    is_actadd = args.vector_method.startswith("actadd_prompt")
    is_aligned = args.vector_method == "actadd_prompt_aligned"
    is_paper_aligned = args.vector_method == "actadd_prompt_paper_aligned"
    has_shared_suffix = is_aligned or is_paper_aligned
    if (
        args.vector_method in {"asc_endpoint", "asc_endpoint_raw"}
        and args.pairs_path is None
    ):
        raise ValueError(
            f"{args.vector_method} requires --pairs_path with checked CoT pairs."
        )
    if has_shared_suffix and args.problems_path is None:
        raise ValueError(
            "Aligned ActAdd prompt methods require --problems_path so their "
            "provenance is an explicit unlabeled training split."
        )
    if is_actadd:
        if args.direction != "short_minus_long":
            raise ValueError(
                "ActAdd prompt methods test target-minus-source and therefore "
                "require --direction short_minus_long."
            )
        if args.activation_site != "block_input":
            raise ValueError(
                "ActAdd prompt methods require --activation_site block_input to match "
                "the ActAdd residual-stream intervention site."
            )

    model, tokenizer, input_device = utils.load_model_and_tokenizer(args)
    layer_index = utils.resolve_layer_index(model, args.model_name, args.layer_index)

    shared_suffix_agreement = None
    if has_shared_suffix:
        agreements = []
        for row in pairs:
            short_prompt, long_prompt = utils.pair_prompts_for_activation(
                row, args.vector_method
            )
            short_ids = tokenizer(short_prompt)["input_ids"]
            long_ids = tokenizer(long_prompt)["input_ids"]
            if min(len(short_ids), len(long_ids)) < args.pool_last_n_tokens:
                raise ValueError(
                    "A contrast prompt is shorter than --pool_last_n_tokens="
                    f"{args.pool_last_n_tokens}."
                )
            agreements.append(
                short_ids[-args.pool_last_n_tokens :]
                == long_ids[-args.pool_last_n_tokens :]
            )
        shared_suffix_agreement = sum(agreements) / len(agreements)
        if shared_suffix_agreement < 1.0:
            raise ValueError(
                "Aligned ActAdd prompts require identical final pooled token IDs "
                f"on every pair; agreement={shared_suffix_agreement:.2%}."
            )

    print("Extracting steering vectors")
    print(f"  model:       {args.model_name}")
    print(f"  input:       {source_path}")
    print(f"  samples:     {len(pairs)}")
    print(f"  layer:       {layer_index}")
    print(f"  method:      {args.vector_method}")
    print(f"  direction:   {args.direction}")
    print(f"  site:        {args.activation_site}")
    print(f"  pooled tokens:{args.pool_last_n_tokens}")
    if shared_suffix_agreement is not None:
        print(f"  suffix match:{shared_suffix_agreement:.2%}")
    is_raw_endpoint = args.vector_method == "asc_endpoint_raw"
    if is_paper_aligned:
        text_mode = "concise prefix vs unchanged paper_cot with shared suffix"
        metadata_text_mode = "concise_prefix_vs_unchanged_paper_cot_shared_suffix"
        positive_text = "concise_instruction_plus_unchanged_paper_cot"
        negative_text = "unchanged_paper_cot_prompt"
        matching_prompt_mode = "paper_cot"
    elif is_aligned:
        text_mode = "aligned style instruction + shared question suffix"
        metadata_text_mode = "aligned_style_instruction_shared_question_suffix"
        positive_text = "aligned_concise_instruction_plus_question"
        negative_text = "aligned_detailed_instruction_plus_question"
        matching_prompt_mode = "actadd_aligned_long"
    elif args.vector_method == "actadd_prompt":
        text_mode = "matched concise prompt vs paper_cot prompt (no generated answers)"
        metadata_text_mode = "short_prompt_vs_long_prompt_last_token"
        positive_text = "matched_concise_prompt"
        negative_text = "paper_cot_prompt"
        matching_prompt_mode = "paper_cot"
    elif is_raw_endpoint:
        text_mode = "raw problem + short/long cot"
        metadata_text_mode = "raw_problem+cot_last_token"
        positive_text = "raw_problem+short_cot"
        negative_text = "raw_problem+long_cot"
        matching_prompt_mode = "raw"
    else:
        text_mode = "long_prompt + short/long cot"
        metadata_text_mode = "long_prompt+cot_last_token"
        positive_text = "long_prompt+short_cot"
        negative_text = "long_prompt+long_cot"
        matching_prompt_mode = None
    print(f"  text mode:   {text_mode}")
    print(f"  input device:{input_device}")
    if hasattr(model, "hf_device_map"):
        print(f"  hf_device_map summary: {utils.summarize_device_map(model.hf_device_map)}")

    vectors = utils.extract_vectors(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        input_device=input_device,
        layer_index=layer_index,
        max_input_tokens=args.max_input_tokens,
        activation_batch_size=args.activation_batch_size,
        direction=args.direction,
        activation_site=args.activation_site,
        vector_method=args.vector_method,
        pool_last_n_tokens=args.pool_last_n_tokens,
    )

    output_vector = Path(args.output_vector_path)
    output_vector.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, output_vector)

    norms = vectors.norm(dim=1)
    mean_vector = vectors.mean(dim=0)
    mean_vector_norm = mean_vector.norm().clamp_min(1e-12)
    pair_projection_margins = vectors @ (mean_vector / mean_vector_norm)
    first_pair = pairs[0]
    actadd_short_example, actadd_long_example = (
        utils.pair_prompts_for_activation(first_pair, args.vector_method)
        if is_actadd
        else (None, None)
    )
    metadata = {
        "model_name": args.model_name,
        "layer_index": layer_index,
        "vector_method": args.vector_method,
        "direction": args.direction,
        "activation_site": args.activation_site,
        "matching_injection_site": args.activation_site,
        "paper_direction": "short_minus_long",
        "recommended_injection_sign": (
            "add" if args.direction == "short_minus_long" else "subtract"
        ),
        "recommended_injection_scope": (
            "prompt_only" if is_actadd else "all_tokens"
        ),
        "recommended_injection_token_count": (
            args.pool_last_n_tokens if is_actadd else 1
        ),
        "recommended_vector_normalization": "none",
        "formula": (
            "v = h(short) - h(long); h <- h + gamma*v"
            if args.direction == "short_minus_long"
            else "v = h(long) - h(short); h <- h - gamma*v"
        ),
        "extraction_formula": (
            "v_i = mean_last_N(h_pre_L(P_short(q_i))) - "
            "mean_last_N(h_pre_L(P_long(q_i)))"
            if is_actadd
            else (
                "v_i = h_L(raw_problem+short_cot)[-1] - "
                "h_L(raw_problem+long_cot)[-1]"
                if is_raw_endpoint
                else "v_i = h_L(long_prompt+short_cot)[-1] - "
                "h_L(long_prompt+long_cot)[-1]"
            )
        ),
        "injection_formula": (
            "h_pre_L(P_long(q))[-N:] <- h_pre_L(P_long(q))[-N:] + gamma*mean(v_i)"
            if is_actadd
            else "h_L[-1] <- h_L[-1] + sign*gamma*mean(v_i)"
        ),
        "text_mode": metadata_text_mode,
        "contains_generated_answers": not is_actadd,
        "positive_text": positive_text,
        "negative_text": negative_text,
        "representation_token": (
            f"mean_last_{args.pool_last_n_tokens}_prompt_tokens"
            if is_actadd
            else "last_answer_token"
        ),
        "intervention_token": (
            f"last_{args.pool_last_n_tokens}_prompt_tokens"
            if is_actadd
            else "last_prompt_token"
        ),
        "matching_prompt_mode": matching_prompt_mode,
        "paper_cot_baseline_unchanged": is_paper_aligned,
        "positive_gamma_only": is_actadd or is_raw_endpoint,
        "pool_last_n_tokens": args.pool_last_n_tokens,
        "shared_suffix_token_agreement": shared_suffix_agreement,
        "num_vectors": int(vectors.shape[0]),
        "hidden_size": int(vectors.shape[1]),
        "vector_norm_mean": float(norms.mean().item()),
        "vector_norm_median": float(norms.median().item()),
        "vector_norm_min": float(norms.min().item()),
        "vector_norm_max": float(norms.max().item()),
        "mean_vector_norm": float(mean_vector.norm().item()),
        "orientation_pair_agreement": float(
            (pair_projection_margins > 0).float().mean().item()
        ),
        "mean_pair_projection_margin": float(
            pair_projection_margins.mean().item()
        ),
        "source_path": str(source_path),
        "source_row_count": source_row_count,
        "source_container_row_count": source_container_row_count,
        "source_manifest_method": (
            source_manifest.get("method") if source_manifest is not None else None
        ),
        "source_pair_row_indices": [
            row.get("source_row_index")
            for row in pairs
            if row.get("source_row_index") is not None
        ],
        "selected_row_indices": selected_indices,
        "sample_selection": (
            "seeded_random_without_replacement"
            if len(pairs) < source_row_count
            else "all_rows"
        ),
        "sample_seed": args.seed,
        "pairs_path": str(source_path) if args.pairs_path else None,
        "problems_path": str(source_path) if args.problems_path else None,
        "long_source": (
            "constructed_from_problem"
            if is_actadd
            else str(first_pair.get("long_source") or "unknown")
        ),
        "short_source": (
            "constructed_from_problem"
            if is_actadd
            else str(first_pair.get("short_source") or "unknown")
        ),
        "short_prompt_example": actadd_short_example if is_actadd else None,
        "long_prompt_example": actadd_long_example if is_actadd else None,
        "activation_batch_size": args.activation_batch_size,
        "output_vector_path": str(output_vector),
        "created_at_unix": time.time(),
    }
    utils.write_json(output_vector.with_suffix(output_vector.suffix + ".metadata.json"), metadata)

    print("\nSaved steering vectors")
    print(f"  path:             {output_vector}")
    print(f"  samples:          {vectors.shape[0]}")
    print(f"  hidden size:      {vectors.shape[1]}")
    print(f"  mean vector norm: {metadata['mean_vector_norm']:.6f}")


if __name__ == "__main__":
    main()
