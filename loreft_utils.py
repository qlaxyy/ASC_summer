"""Minimal LoReFT utilities with no dependency on pyreft/pyvene.

The intervention follows Wu et al. (NeurIPS 2024), equation (2):

    Phi(h) = h + R^T (W h + b - R h)

The base language model remains frozen.  This module intentionally implements
only the small subset needed by this project: residual-stream block-output
interventions at selected prompt positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


FORMAT_NAME = "asc_loreft_v1"


class LowRankRotateLayer(nn.Module):
    """Tall orthonormal projection matrix storing R^T with shape [d, r]."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if rank <= 0 or rank > hidden_size:
            raise ValueError("rank must satisfy 0 < rank <= hidden_size")
        self.weight = nn.Parameter(torch.empty(hidden_size, rank))
        nn.init.orthogonal_(self.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight


class LoReFTIntervention(nn.Module):
    """Low-rank linear subspace intervention from the LoReFT paper."""

    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        rotate = LowRankRotateLayer(hidden_size, rank)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate)
        self.learned_source = nn.Linear(hidden_size, rank, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)

    def forward(self, hidden: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if not math.isfinite(scale):
            raise ValueError("LoReFT scale must be finite")
        original_dtype = hidden.dtype
        # The selected prompt positions are few, so float32 adapter arithmetic is
        # inexpensive and avoids fragile bf16 orthogonal parametrization updates.
        base = hidden.to(dtype=self.learned_source.weight.dtype)
        projected_base = self.rotate_layer(base)
        projected_source = self.learned_source(base)
        delta = (projected_source - projected_base) @ self.rotate_layer.weight.T
        output = base + float(scale) * delta
        return self.dropout(output).to(dtype=original_dtype)


def parse_position_spec(spec: str) -> tuple[int, int]:
    """Parse pyreft-style prompt positions such as ``f5+l5`` or ``l1``."""
    first_n = 0
    last_n = 0
    for raw_part in spec.lower().replace(" ", "").split("+"):
        if not raw_part:
            continue
        prefix = raw_part[0]
        try:
            count = int(raw_part[1:])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid LoReFT position component: {raw_part!r}") from exc
        if count < 0 or prefix not in {"f", "l"}:
            raise ValueError(f"Invalid LoReFT position component: {raw_part!r}")
        if prefix == "f":
            first_n += count
        else:
            last_n += count
    if first_n + last_n <= 0:
        raise ValueError("LoReFT positions must select at least one prompt token")
    return first_n, last_n


def prompt_position_mask(
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    position_spec: str,
) -> torch.Tensor:
    """Build a mask for first/last non-padding prompt tokens in each row.

    ``attention_mask`` describes the full padded training sequence.  During
    inference it describes the left-padded prompt and ``prompt_lengths`` is its
    non-padding length.
    """
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != attention_mask.shape[0]:
        raise ValueError("prompt_lengths must have shape [batch]")
    first_n, last_n = parse_position_spec(position_spec)
    result = torch.zeros_like(attention_mask, dtype=torch.bool)
    for row in range(attention_mask.shape[0]):
        nonpad = torch.nonzero(attention_mask[row].bool(), as_tuple=False).flatten()
        prompt_len = int(prompt_lengths[row].item())
        if prompt_len <= 0:
            continue
        if prompt_len > nonpad.numel():
            raise ValueError("prompt length exceeds the number of non-padding tokens")
        prompt_positions = nonpad[:prompt_len]
        if first_n:
            result[row, prompt_positions[:first_n]] = True
        if last_n:
            result[row, prompt_positions[-last_n:]] = True
    return result


@dataclass
class LoReFTHookController:
    """Mutable per-forward mask shared by all layer hooks."""

    position_spec: str
    scale: float = 1.0
    position_mask: torch.Tensor | None = None

    def prepare_training_mask(self, mask: torch.Tensor) -> None:
        self.position_mask = mask.detach()

    def prepare_inference(self, attention_mask: torch.Tensor) -> None:
        prompt_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        self.position_mask = prompt_position_mask(
            attention_mask,
            prompt_lengths,
            self.position_spec,
        ).detach()

    def clear(self) -> None:
        self.position_mask = None


class LoReFTBundle(nn.Module):
    """One independent LoReFT intervention per selected transformer layer."""

    def __init__(
        self,
        hidden_size: int,
        layer_indices: Iterable[int],
        rank: int,
        position_spec: str,
        dropout: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        layers = sorted({int(layer) for layer in layer_indices})
        if not layers or layers[0] < 0:
            raise ValueError("layer_indices must contain nonnegative indices")
        parse_position_spec(position_spec)
        self.hidden_size = int(hidden_size)
        self.layer_indices = layers
        self.rank = int(rank)
        self.position_spec = str(position_spec)
        self.dropout_rate = float(dropout)
        self.metadata = dict(metadata or {})
        self.interventions = nn.ModuleDict(
            {
                str(layer): LoReFTIntervention(hidden_size, rank, dropout)
                for layer in layers
            }
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint(self) -> dict[str, Any]:
        metadata = {
            **self.metadata,
            "format": FORMAT_NAME,
            "formula": "h + R^T(Wh + b - Rh)",
            "hidden_size": self.hidden_size,
            "layer_indices": self.layer_indices,
            "rank": self.rank,
            "position_spec": self.position_spec,
            "dropout": self.dropout_rate,
            "injection_site": "block_output",
            "inference_scope": "prompt_positions_only",
            "trainable_parameters": self.trainable_parameter_count,
        }
        return {
            "metadata": metadata,
            "state_dict": {
                key: value.detach().cpu() for key, value in self.state_dict().items()
            },
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        torch.save(self.checkpoint(), temporary)
        temporary.replace(output)
        return output


def load_loreft_bundle(path: str | Path) -> LoReFTBundle:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "metadata" not in payload or "state_dict" not in payload:
        raise ValueError(f"Invalid LoReFT checkpoint: {path}")
    metadata = payload["metadata"]
    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(
            f"Unsupported LoReFT checkpoint format: {metadata.get('format')!r}"
        )
    bundle = LoReFTBundle(
        hidden_size=int(metadata["hidden_size"]),
        layer_indices=metadata["layer_indices"],
        rank=int(metadata["rank"]),
        position_spec=str(metadata["position_spec"]),
        dropout=float(metadata.get("dropout", 0.0)),
        metadata=metadata,
    )
    bundle.load_state_dict(payload["state_dict"], strict=True)
    return bundle


def _layer_device(layer: nn.Module) -> torch.device:
    try:
        return next(layer.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


def install_loreft_hooks(
    transformer_layers: Any,
    bundle: LoReFTBundle,
    controller: LoReFTHookController,
) -> list[Any]:
    """Install block-output hooks and return removable handles."""
    handles = []
    layer_count = len(transformer_layers)
    for layer_index in bundle.layer_indices:
        if layer_index >= layer_count:
            raise IndexError(
                f"LoReFT layer {layer_index} is outside model with {layer_count} layers"
            )
        layer = transformer_layers[layer_index]
        intervention = bundle.interventions[str(layer_index)]
        intervention.to(device=_layer_device(layer), dtype=torch.float32)

        def hook(_module, _inputs, output, *, adapter=intervention):
            hidden = output if torch.is_tensor(output) else output[0]
            mask = controller.position_mask
            if mask is None or hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
                return None
            mask = mask.to(device=hidden.device, dtype=torch.bool)
            flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
            if flat_indices.numel() == 0:
                return None
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            selected = flat_hidden.index_select(0, flat_indices)
            transformed = adapter(selected, scale=controller.scale)
            updated = flat_hidden.clone().index_copy(0, flat_indices, transformed)
            return _replace_hidden(output, updated.reshape_as(hidden))

        handles.append(layer.register_forward_hook(hook))
    return handles


def validate_loreft_for_model(
    bundle: LoReFTBundle,
    model: Any,
    model_name: str,
    prompt_mode: str,
    allow_mismatch: bool = False,
) -> None:
    errors = []
    model_hidden_size = int(getattr(model.config, "hidden_size", -1))
    if model_hidden_size != bundle.hidden_size:
        errors.append(
            f"hidden_size checkpoint={bundle.hidden_size}, model={model_hidden_size}"
        )
    expected_model = str(bundle.metadata.get("model_name", ""))
    if expected_model:
        expected_tail = Path(expected_model.rstrip("/\\")).name.lower()
        actual_tail = Path(model_name.rstrip("/\\")).name.lower()
        if expected_tail != actual_tail:
            errors.append(f"model checkpoint={expected_tail}, requested={actual_tail}")
    expected_prompt = bundle.metadata.get("prompt_mode")
    if expected_prompt and expected_prompt != prompt_mode:
        errors.append(f"prompt checkpoint={expected_prompt}, requested={prompt_mode}")
    if errors and not allow_mismatch:
        raise ValueError(
            "LoReFT checkpoint/evaluation mismatch: "
            + "; ".join(errors)
            + ". Pass --allow_loreft_metadata_mismatch only for a labeled diagnostic."
        )
