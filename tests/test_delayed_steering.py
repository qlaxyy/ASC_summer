import pytest

try:
    import torch

    from eval_asc_paper import make_cached_input_steering_hook
    from eval_asc_paper import make_cached_output_steering_hook
except ModuleNotFoundError:
    torch = None


pytestmark = pytest.mark.skipif(
    torch is None,
    reason="torch/transformers are not installed in the local CPU environment",
)


def test_delayed_output_hook_starts_after_configured_generated_tokens():
    hook = make_cached_output_steering_hook(
        torch.ones(2),
        gamma=0.5,
        injection_sign="add",
        injection_scope="all_tokens",
        injection_token_count=1,
        injection_start_generated_token=2,
    )

    prefill = torch.zeros(1, 4, 2)
    assert hook(None, None, (prefill,)) is None
    assert torch.equal(prefill, torch.zeros_like(prefill))

    first_cached_step = torch.zeros(1, 1, 2)
    assert hook(None, None, (first_cached_step,)) is None
    assert torch.equal(first_cached_step, torch.zeros_like(first_cached_step))

    second_cached_step = torch.zeros(1, 1, 2)
    result = hook(None, None, (second_cached_step,))
    assert result is not None
    assert torch.equal(second_cached_step, torch.full_like(second_cached_step, 0.5))

    # The next prefill belongs to a new batch and must reset the counter.
    hook(None, None, (torch.zeros(1, 3, 2),))
    next_batch_first_step = torch.zeros(1, 1, 2)
    assert hook(None, None, (next_batch_first_step,)) is None


def test_delayed_input_hook_uses_the_same_generation_counter_semantics():
    hook = make_cached_input_steering_hook(
        torch.ones(3),
        gamma=1.0,
        injection_sign="add",
        injection_scope="all_tokens",
        injection_token_count=1,
        injection_start_generated_token=1,
    )

    prefill = torch.zeros(2, 5, 3)
    assert hook(None, (prefill,)) is None

    first_cached_step = torch.zeros(2, 1, 3)
    result = hook(None, (first_cached_step,))
    assert result is not None
    assert torch.equal(first_cached_step, torch.ones_like(first_cached_step))


def test_zero_start_preserves_immediate_all_tokens_behavior():
    hook = make_cached_output_steering_hook(
        torch.ones(2),
        gamma=1.0,
        injection_sign="add",
        injection_scope="all_tokens",
        injection_token_count=1,
        injection_start_generated_token=0,
    )
    prefill = torch.zeros(1, 3, 2)
    hook(None, None, (prefill,))
    assert torch.equal(prefill[:, :-1, :], torch.zeros(1, 2, 2))
    assert torch.equal(prefill[:, -1:, :], torch.ones(1, 1, 2))


def test_delayed_steering_rejects_incompatible_scope():
    with pytest.raises(ValueError, match="all_tokens"):
        make_cached_output_steering_hook(
            torch.ones(2),
            gamma=1.0,
            injection_sign="add",
            injection_scope="sequence_all",
            injection_token_count=1,
            injection_start_generated_token=128,
        )
