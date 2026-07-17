from __future__ import annotations

import unittest

try:
    import torch

    from reproduce_repe_pca_reading import (
        first_pca_direction,
        grouped_accuracy,
        grouped_accuracy_columns,
        select_rows,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the local CPU environment")
def test_pca_direction_recovers_and_orients_shared_axis() -> None:
    torch.manual_seed(0)
    axis = torch.tensor([1.0, 0.0, 0.0, 0.0])
    pairs = []
    positive_indices = []
    for index in range(12):
        center = torch.randn(4) * 0.05
        positive = center + axis * (1.0 + index * 0.02)
        negative = center - axis * (1.0 + index * 0.02)
        if index % 2:
            pairs.append(torch.stack([negative, positive]))
            positive_indices.append(1)
        else:
            pairs.append(torch.stack([positive, negative]))
            positive_indices.append(0)

    direction, diagnostics = first_pca_direction(
        torch.stack(pairs), positive_indices, "l2"
    )

    assert torch.dot(direction, axis) > 0.99
    assert diagnostics["train_pair_accuracy"] == 1.0
    assert diagnostics["train_mean_margin"] > 0


@unittest.skipIf(torch is None, "torch is not installed in the local CPU environment")
def test_grouped_accuracy_handles_variable_choice_counts() -> None:
    scores = torch.tensor([0.1, 0.8, 0.2, 0.9, 0.1])
    accuracy, predictions, correct = grouped_accuracy(scores, [2, 3], [1, 0])
    assert accuracy == 1.0
    assert predictions == [1, 0]
    assert correct == [True, True]


@unittest.skipIf(torch is None, "torch is not installed in the local CPU environment")
def test_grouped_accuracy_columns_scores_many_directions() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.1, 0.2],
        ]
    )
    accuracies = grouped_accuracy_columns(scores, [2, 3], [1, 0])
    torch.testing.assert_close(accuracies, torch.tensor([1.0, 0.0]))


@unittest.skipIf(torch is None, "torch is not installed in the local CPU environment")
def test_dataset_split_is_deterministic_and_disjoint() -> None:
    rows = [{"index": index} for index in range(20)]
    val_a, test_a, val_indices_a, test_indices_a = select_rows(rows, 5, 7, 9)
    val_b, test_b, val_indices_b, test_indices_b = select_rows(rows, 5, 7, 9)
    assert val_a == val_b
    assert test_a == test_b
    assert val_indices_a == val_indices_b
    assert test_indices_a == test_indices_b
    assert set(val_indices_a).isdisjoint(test_indices_a)
