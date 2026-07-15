import unittest

from select_causal_conciseness_vector import assess_candidate
from select_causal_conciseness_vector import choose_held_out_rows
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
