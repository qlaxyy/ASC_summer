import unittest

try:
    import torch

    from eval_asc_paper import make_cached_input_steering_hook
    from eval_asc_paper import make_cached_output_steering_hook
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the local CPU environment")
class ProjectionMatchingHookTests(unittest.TestCase):
    def test_output_hook_interpolates_every_selected_projection(self) -> None:
        hidden = torch.tensor([[[0.0, 2.0], [2.0, 3.0]]])
        hook = make_cached_output_steering_hook(
            torch.tensor([1.0, 0.0]),
            gamma=0.5,
            injection_sign="add",
            injection_scope="sequence_all",
            injection_token_count=1,
            intervention_mode="projection_match",
            projection_target=4.0,
        )

        output = hook(None, None, (hidden,))[0]

        torch.testing.assert_close(
            output,
            torch.tensor([[[2.0, 2.0], [3.0, 3.0]]]),
        )

    def test_input_hook_alpha_one_reaches_projection_target(self) -> None:
        hidden = torch.tensor([[[1.0, 5.0], [-2.0, 7.0]]])
        hook = make_cached_input_steering_hook(
            torch.tensor([1.0, 0.0]),
            gamma=1.0,
            injection_sign="add",
            injection_scope="sequence_all",
            injection_token_count=1,
            intervention_mode="projection_match",
            projection_target=3.0,
        )

        output = hook(None, (hidden,))[0]

        torch.testing.assert_close(
            output,
            torch.tensor([[[3.0, 5.0], [3.0, 7.0]]]),
        )

    def test_additive_mode_remains_unchanged(self) -> None:
        hidden = torch.tensor([[[1.0, 2.0]]])
        hook = make_cached_output_steering_hook(
            torch.tensor([0.5, -1.0]),
            gamma=2.0,
            injection_sign="add",
            injection_scope="sequence_all",
            injection_token_count=1,
        )

        output = hook(None, None, (hidden,))[0]

        torch.testing.assert_close(output, torch.tensor([[[2.0, 0.0]]]))

    def test_projection_mode_requires_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires projection_target"):
            make_cached_output_steering_hook(
                torch.tensor([1.0, 0.0]),
                gamma=0.5,
                injection_sign="add",
                injection_scope="sequence_all",
                injection_token_count=1,
                intervention_mode="projection_match",
            )


if __name__ == "__main__":
    unittest.main()
