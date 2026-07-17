from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from loreft_utils import LoReFTBundle
from loreft_utils import LoReFTHookController
from loreft_utils import LoReFTIntervention
from loreft_utils import install_loreft_hooks
from loreft_utils import load_loreft_bundle
from loreft_utils import parse_position_spec
from loreft_utils import prompt_position_mask


def test_loreft_matches_paper_equation() -> None:
    torch.manual_seed(3)
    intervention = LoReFTIntervention(hidden_size=6, rank=2, dropout=0.0).eval()
    hidden = torch.randn(4, 6)
    rotate = intervention.rotate_layer.weight
    expected = hidden + (
        intervention.learned_source(hidden) - hidden @ rotate
    ) @ rotate.T
    actual = intervention(hidden)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_prompt_position_mask_handles_left_and_right_padding() -> None:
    attention = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    lengths = torch.tensor([4, 3])
    mask = prompt_position_mask(attention, lengths, "f1+l2")
    expected = torch.tensor(
        [
            [0, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)
    assert parse_position_spec("f5+l5") == (5, 5)


class _FakeBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    def forward(self, hidden: torch.Tensor):
        return (hidden + 1.0, "cache")


def test_hook_changes_only_selected_positions_and_preserves_tuple() -> None:
    layers = nn.ModuleList([_FakeBlock(4)])
    bundle = LoReFTBundle(4, [0], rank=2, position_spec="l1")
    controller = LoReFTHookController("l1", scale=1.0)
    controller.prepare_training_mask(torch.tensor([[False, True]]))
    handles = install_loreft_hooks(layers, bundle, controller)
    try:
        hidden = torch.zeros(1, 2, 4)
        output = layers[0](hidden)
    finally:
        for handle in handles:
            handle.remove()
    assert isinstance(output, tuple)
    assert output[1] == "cache"
    assert torch.equal(output[0][:, 0, :], torch.ones(1, 4))
    assert not torch.equal(output[0][:, 1, :], torch.ones(1, 4))


def test_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(7)
    bundle = LoReFTBundle(
        hidden_size=8,
        layer_indices=[1, 3],
        rank=2,
        position_spec="f1+l2",
        metadata={"model_name": "example/model", "prompt_mode": "paper_cot"},
    )
    path = tmp_path / "adapter.pt"
    bundle.save(path)
    restored = load_loreft_bundle(path)
    assert restored.layer_indices == [1, 3]
    assert restored.position_spec == "f1+l2"
    for key, value in bundle.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
