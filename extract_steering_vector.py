"""
Extract ASC endpoint or ActAdd prompt-contrast vectors from a checked pair file.

This script only does the second half of the pipeline:
  1. Read a manually filtered pairs JSON file.
  2. Build either answer-endpoint texts or short/long contrast prompts.
  3. Extract matched target-layer activations and save their differences.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--output_vector_path", type=str, required=True)
    parser.add_argument("--layer_index", type=str, default="auto")
    parser.add_argument(
        "--vector_method",
        choices=["asc_endpoint", "actadd_prompt"],
        default="asc_endpoint",
        help=(
            "asc_endpoint compares the ends of complete short/long answers. "
            "actadd_prompt constructs matched concise/step-by-step prompts from "
            "each problem and compares them before any answer is generated."
        ),
    )
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--activation_batch_size", type=int, default=4)
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

    pairs_path = Path(args.pairs_path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = utils.read_json(pairs_path)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("--pairs_path must contain a non-empty JSON list.")
    if args.vector_method == "actadd_prompt":
        if args.direction != "short_minus_long":
            raise ValueError(
                "actadd_prompt tests the target-minus-source theory and therefore "
                "requires --direction short_minus_long."
            )
        if args.activation_site != "block_input":
            raise ValueError(
                "actadd_prompt requires --activation_site block_input to match "
                "the ActAdd residual-stream intervention site."
            )

    model, tokenizer, input_device = utils.load_model_and_tokenizer(args)
    layer_index = utils.resolve_layer_index(model, args.model_name, args.layer_index)

    print("Extracting steering vectors")
    print(f"  model:       {args.model_name}")
    print(f"  pairs:       {pairs_path}")
    print(f"  samples:     {len(pairs)}")
    print(f"  layer:       {layer_index}")
    print(f"  method:      {args.vector_method}")
    print(f"  direction:   {args.direction}")
    print(f"  site:        {args.activation_site}")
    text_mode = (
        "matched concise prompt vs paper_cot prompt (no generated answers)"
        if args.vector_method == "actadd_prompt"
        else "long_prompt + short/long cot"
    )
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
        utils.pair_prompts_for_activation(first_pair)
        if args.vector_method == "actadd_prompt"
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
            "prompt_only" if args.vector_method == "actadd_prompt" else "all_tokens"
        ),
        "formula": (
            "v = h(short) - h(long); h <- h + gamma*v"
            if args.direction == "short_minus_long"
            else "v = h(long) - h(short); h <- h - gamma*v"
        ),
        "extraction_formula": (
            "v_i = h_pre_L(P_short(q_i))[-1] - h_pre_L(P_long(q_i))[-1]"
            if args.vector_method == "actadd_prompt"
            else "v_i = h_L(long_prompt+short_cot)[-1] - h_L(long_prompt+long_cot)[-1]"
        ),
        "injection_formula": (
            "h_pre_L(P_long(q))[-1] <- h_pre_L(P_long(q))[-1] + gamma*mean(v_i)"
            if args.vector_method == "actadd_prompt"
            else "h_L[-1] <- h_L[-1] + sign*gamma*mean(v_i)"
        ),
        "text_mode": (
            "short_prompt_vs_long_prompt_last_token"
            if args.vector_method == "actadd_prompt"
            else "long_prompt+cot_last_token"
        ),
        "contains_generated_answers": args.vector_method == "asc_endpoint",
        "positive_text": (
            "matched_concise_prompt"
            if args.vector_method == "actadd_prompt"
            else "long_prompt+short_cot"
        ),
        "negative_text": (
            "paper_cot_prompt"
            if args.vector_method == "actadd_prompt"
            else "long_prompt+long_cot"
        ),
        "representation_token": (
            "last_prompt_token"
            if args.vector_method == "actadd_prompt"
            else "last_answer_token"
        ),
        "intervention_token": "last_prompt_token",
        "matching_prompt_mode": (
            "paper_cot" if args.vector_method == "actadd_prompt" else None
        ),
        "positive_gamma_only": args.vector_method == "actadd_prompt",
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
        "pairs_path": str(pairs_path),
        "long_source": (
            "constructed_from_problem"
            if args.vector_method == "actadd_prompt"
            else str(first_pair.get("long_source") or "unknown")
        ),
        "short_source": (
            "constructed_from_problem"
            if args.vector_method == "actadd_prompt"
            else str(first_pair.get("short_source") or "unknown")
        ),
        "short_prompt_example": (
            actadd_short_example if args.vector_method == "actadd_prompt" else None
        ),
        "long_prompt_example": (
            actadd_long_example if args.vector_method == "actadd_prompt" else None
        ),
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
