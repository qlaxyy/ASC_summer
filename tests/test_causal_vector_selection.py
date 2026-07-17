import unittest

from select_causal_conciseness_vector import assess_candidate
from select_causal_conciseness_vector import choose_held_out_rows
from select_causal_conciseness_vector import interpolate_projection_threshold
from select_causal_conciseness_vector import parse_projection_alpha_list
from select_causal_conciseness_vector import select_best_candidate

try:
    from eval_asc_paper import has_obvious_corruption_artifact
except ModuleNotFoundError:
    has_obvious_corruption_artifact = None


class CausalVectorSelectionTests(unittest.TestCase):
    def baseline(self):
        return {
            "accuracy": 0.90,
            "avg_tokens": 1000.0,
            "repetition_artifact_rate": 0.0,
            "corruption_artifact_rate": 0.0,
            "length_capped_rate": 0.0,
        }

    def candidate(self, **overrides):
        row = {
            "accuracy": 0.88,
            "avg_tokens": 850.0,
            "repetition_artifact_rate": 0.0,
            "corruption_artifact_rate": 0.0,
            "length_capped_rate": 0.0,
            "gamma": 0.5,
        }
        row.update(overrides)
        return row

    def test_held_out_rows_exclude_extraction_indices_and_are_deterministic(self):
        first = choose_held_out_rows(20, {0, 1, 2}, 5, 123)
        second = choose_held_out_rows(20, {0, 1, 2}, 5, 123)
        self.assertEqual(first, second)
        self.assertTrue(set(first).isdisjoint({0, 1, 2}))

    def test_projection_alphas_are_bounded_and_deduplicated(self):
        self.assertEqual(
            parse_projection_alpha_list("0.25,0.5,0.25"),
            [0.25, 0.5],
        )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            parse_projection_alpha_list("1.1")

    def test_projection_threshold_interpolates_from_verbose_to_concise(self):
        self.assertAlmostEqual(
            interpolate_projection_threshold(-112.0, -48.0, 0.25),
            -96.0,
        )
        with self.assertRaisesRegex(ValueError, "must exceed"):
            interpolate_projection_threshold(-48.0, -112.0, 0.5)

    def test_candidate_passes_all_hard_constraints(self):
        result = assess_candidate(
            self.baseline(), self.candidate(), 0.05, 0.04, 0.04, 0.0, 0.04
        )
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["compression_fraction"], 0.15)

    def test_corruption_is_a_hard_rejection(self):
        result = assess_candidate(
            self.baseline(),
            self.candidate(corruption_artifact_rate=0.01),
            0.05,
            0.04,
            0.04,
            0.0,
            0.04,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("corruption_artifact", result["rejection_reasons"])

    def test_best_candidate_prefers_compression_after_filtering(self):
        candidates = [
            {**self.candidate(gamma=0.5), "eligible": True, "compression_fraction": 0.10},
            {**self.candidate(gamma=1.0), "eligible": True, "compression_fraction": 0.20},
        ]
        self.assertEqual(select_best_candidate(candidates)["gamma"], 1.0)

    def test_robust_checks_reject_outlier_driven_mean_compression(self):
        baseline = self.baseline()
        candidate = self.candidate(avg_tokens=800.0)
        baseline["detailed_results"] = [
            {"question": f"q{i}", "tokens": 1000} for i in range(10)
        ]
        candidate["detailed_results"] = [
            {"question": f"q{i}", "tokens": token_count}
            for i, token_count in enumerate([200, 200] + [1050] * 8)
        ]

        result = assess_candidate(
            baseline,
            candidate,
            0.05,
            0.04,
            0.04,
            0.0,
            0.04,
            min_trimmed_compression=0.02,
            min_pairwise_win_margin=0.0,
            trim_fraction=0.1,
        )

        self.assertFalse(result["eligible"])
        self.assertIn("negative_pairwise_win_margin", result["rejection_reasons"])
        self.assertEqual(result["paired_shorter_count"], 2)
        self.assertEqual(result["paired_longer_count"], 8)



@unittest.skipIf(
    has_obvious_corruption_artifact is None,
    "torch is not installed in the local CPU environment",
)
class CorruptionDetectorTests(unittest.TestCase):
    def test_obvious_corruption_detector_is_conservative(self):
        self.assertTrue(has_obvious_corruption_artifact("bad \ufffd output"))
        self.assertTrue(has_obvious_corruption_artifact("<0xE4><0xB8><0xAD>"))
        self.assertFalse(has_obvious_corruption_artifact("中文与 LaTeX: \\boxed{42}"))


if __name__ == "__main__":
    unittest.main()
