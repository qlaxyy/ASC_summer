"""Train a paper-faithful LoReFT adapter for concise GSM8K reasoning.

Only the low-rank residual-stream interventions are optimized.  Every base
language-model parameter remains frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from asc_steering_utils import ACTADD_LONG_PROMPT_TEMPLATE
from loreft_utils import LoReFTBundle
from loreft_utils import LoReFTHookController
from loreft_utils import install_loreft_hooks
from loreft_utils import prompt_position_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(piece.strip()) for piece in text.split(",") if piece.strip()})
    if not values or values[0] < 0:
        raise ValueError("--layer_indices must contain nonnegative integers")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def get_transformer_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError("Could not find transformer layers on this model")


def get_input_device(model: Any) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for key in ("model.embed_tokens", "transformer.wte"):
            device = model.hf_device_map.get(key)
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
    return next(model.parameters()).device


def read_selected_pairs(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("pairs", payload if isinstance(payload, list) else None)
    if not isinstance(rows, list):
        raise ValueError("Pair report must be a list or contain a 'pairs' list")
    selected = [row for row in rows if row.get("selected_for_extraction", True)]
    selected = [
        row
        for row in selected
        if row.get("concise_correct", True)
        and not row.get("concise_length_capped", False)
        and not row.get("concise_repetition_artifact", False)
        and not row.get("concise_corruption_artifact", False)
    ]
    if not selected:
        raise ValueError("No eligible concise trajectories found in pair report")
    return selected


class ConciseTrajectoryDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.skipped: list[int] = []
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        for row in rows:
            prompt = ACTADD_LONG_PROMPT_TEMPLATE.format(problem=row["question"])
            output = str(row["concise_output"])
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            output_ids = tokenizer(
                output,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            input_ids = [*prompt_ids, *output_ids]
            if not input_ids or input_ids[-1] != eos_id:
                input_ids.append(eos_id)
            if len(input_ids) > max_seq_length:
                self.skipped.append(int(row.get("source_row_index", -1)))
                continue
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "prompt_length": len(prompt_ids),
                    "source_row_index": int(row.get("source_row_index", -1)),
                }
            )
        if not self.examples:
            raise ValueError(
                "All training trajectories exceeded --max_seq_length; increase it "
                "or provide shorter clean trajectories"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def make_collator(pad_token_id: int):
    def collate(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(row["input_ids"].numel() for row in rows)
        input_ids = torch.full(
            (len(rows), max_length), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        labels = torch.full((len(rows), max_length), -100, dtype=torch.long)
        prompt_lengths = torch.empty(len(rows), dtype=torch.long)
        source_indices = torch.empty(len(rows), dtype=torch.long)
        for index, row in enumerate(rows):
            length = row["input_ids"].numel()
            input_ids[index, :length] = row["input_ids"]
            attention_mask[index, :length] = 1
            labels[index, :length] = row["labels"]
            prompt_lengths[index] = int(row["prompt_length"])
            source_indices[index] = int(row["source_row_index"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
            "source_row_indices": source_indices,
        }

    return collate


def split_rows(
    rows: list[dict[str, Any]], seed: int, validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be between 0 and 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    if validation_count >= len(shuffled):
        raise ValueError("Need at least one training and one validation example")
    return shuffled[validation_count:], shuffled[:validation_count]


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    prompt_lengths = batch.pop("prompt_lengths")
    source_indices = batch.pop("source_row_indices")
    model_batch = {key: value.to(device) for key, value in batch.items()}
    return model_batch, prompt_lengths, source_indices


def compute_validation_loss(
    model: Any,
    bundle: LoReFTBundle,
    controller: LoReFTHookController,
    loader: DataLoader,
    input_device: torch.device,
) -> float:
    model.eval()
    bundle.eval()
    losses = []
    with torch.no_grad():
        for raw_batch in loader:
            batch, prompt_lengths, _ = move_batch(raw_batch, input_device)
            mask = prompt_position_mask(
                batch["attention_mask"],
                prompt_lengths.to(input_device),
                bundle.position_spec,
            )
            controller.prepare_training_mask(mask)
            outputs = model(**batch, use_cache=False)
            losses.append(float(outputs.loss.detach().cpu().item()))
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-model LoReFT adapter for concise reasoning"
    )
    parser.add_argument(
        "--model_name",
        default="/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--pairs_path",
        default="pairs/gsm8k_behavior_trajectory_pairs_train40.json",
    )
    parser.add_argument(
        "--output_path",
        default="vectors/loreft_gsm8k_concise_train40.pt",
    )
    parser.add_argument("--layer_indices", default="12,16,20,24")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--positions", default="f5+l5")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=9e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn_impl", default="sdpa")
    parser.add_argument(
        "--device_map",
        choices=["single", "auto"],
        default="single",
        help="single is the reproducible default; auto is experimental for training.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more GPU memory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    rows = read_selected_pairs(args.pairs_path)
    train_rows, validation_rows = split_rows(
        rows, args.seed, args.validation_fraction
    )
    print("LoReFT concise-reasoning training")
    print(f"  model:       {args.model_name}")
    print(f"  pairs:       {args.pairs_path}")
    print(f"  split:       {len(train_rows)} train / {len(validation_rows)} validation")
    print(f"  layers:      {args.layer_indices}")
    print(f"  rank:        {args.rank}")
    print(f"  positions:   {args.positions}")
    print("  base model:  frozen")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map: Any = None
    if torch.cuda.is_available():
        device_map = {"": 0} if args.device_map == "single" else "auto"
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
    }
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    layer_indices = parse_int_list(args.layer_indices)
    layers = get_transformer_layers(model)
    if layer_indices[-1] >= len(layers):
        raise ValueError(
            f"Requested layer {layer_indices[-1]}, but model has {len(layers)} layers"
        )
    hidden_size = int(model.config.hidden_size)
    bundle = LoReFTBundle(
        hidden_size=hidden_size,
        layer_indices=layer_indices,
        rank=args.rank,
        position_spec=args.positions,
        dropout=args.dropout,
        metadata={
            "method": "LoReFT",
            "paper": "ReFT: Representation Finetuning for Language Models (NeurIPS 2024)",
            "model_name": args.model_name,
            "prompt_mode": "paper_cot",
            "training_pairs_path": args.pairs_path,
            "training_seed": args.seed,
            "objective": "teacher_forced_cross_entropy_on_concise_correct_trajectories",
            "base_model_frozen": True,
        },
    )
    controller = LoReFTHookController(args.positions, scale=1.0)
    handles = install_loreft_hooks(layers, bundle, controller)
    input_device = get_input_device(model)

    train_dataset = ConciseTrajectoryDataset(
        train_rows, tokenizer, args.max_seq_length
    )
    validation_dataset = ConciseTrajectoryDataset(
        validation_rows, tokenizer, args.max_seq_length
    )
    collator = make_collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    print(
        f"  usable:      {len(train_dataset)} train / "
        f"{len(validation_dataset)} validation"
    )
    skipped = [*train_dataset.skipped, *validation_dataset.skipped]
    if skipped:
        print(f"  skipped long rows: {skipped}")
    print(f"  trainable parameters: {bundle.trainable_parameter_count:,}")
    print(f"  input device: {input_device}")

    optimizer = torch.optim.AdamW(
        bundle.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = round(total_steps * args.warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            bundle.train()
            running_losses = []
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                batch, prompt_lengths, _ = move_batch(raw_batch, input_device)
                mask = prompt_position_mask(
                    batch["attention_mask"],
                    prompt_lengths.to(input_device),
                    bundle.position_spec,
                )
                controller.prepare_training_mask(mask)
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss: {loss.item()}")
                running_losses.append(float(loss.detach().cpu().item()))
                (loss / args.gradient_accumulation_steps).backward()

                should_step = (
                    batch_index % args.gradient_accumulation_steps == 0
                    or batch_index == len(train_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        bundle.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

            validation_loss = compute_validation_loss(
                model,
                bundle,
                controller,
                validation_loader,
                input_device,
            )
            train_loss = float(np.mean(running_losses))
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3g}"
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                bundle.metadata.update(
                    {
                        "best_epoch": epoch,
                        "best_validation_loss": validation_loss,
                        "train_examples": len(train_dataset),
                        "validation_examples": len(validation_dataset),
                        "train_source_row_indices": [
                            example["source_row_index"]
                            for example in train_dataset.examples
                        ],
                        "validation_source_row_indices": [
                            example["source_row_index"]
                            for example in validation_dataset.examples
                        ],
                        "training_config": vars(args),
                        "training_history": list(history),
                    }
                )
                bundle.save(args.output_path)
                print(f"  saved best: {args.output_path}")
    finally:
        controller.clear()
        for handle in handles:
            handle.remove()

    training_report = {
        "method": "LoReFT",
        "adapter_path": args.output_path,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "config": vars(args),
        "selected_pairs": len(rows),
        "usable_train_examples": len(train_dataset),
        "usable_validation_examples": len(validation_dataset),
        "skipped_source_row_indices": skipped,
    }
    report_path = Path(str(args.output_path) + ".training.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)

    print("RESULT")
    print(f"  best validation loss: {best_validation_loss:.4f}")
    print(f"  adapter: {args.output_path}")
    print(f"  training report: {report_path}")
    print("  next: evaluate scale 0 vs 1 on held-out data; do not tune on test.")


if __name__ == "__main__":
    main()
