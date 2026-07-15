import unittest

from extract_behavior_trajectory_vectors import relative_bin_bounds
from generate_behavior_trajectory_pairs import assess_pair_quality


class BehaviorPairQualityTests(unittest.TestCase):
    def test_clean_correct_compressed_pair_is_eligible(self) -> None:
        result = assess_pair_quality(
            concise_correct=True,
            verbose_correct=True,
            concise_tokens=300,
            verbose_tokens=1000,
            concise_capped=False,
            verbose_capped=False,
            concise_repetition=False,
            verbose_repetition=False,
            concise_corruption=False,
            verbose_corruption=False,
            min_pair_compression=0.30,
            min_concise_tokens=16,
        )
        self.assertTrue(result["quality_eligible"])
        self.assertAlmostEqual(result["pair_compression_fraction"], 0.70)

    def test_incorrect_or_not_shorter_pair_is_rejected(self) -> None:
        result = assess_pair_quality(
            concise_correct=False,
            verbose_correct=True,
            concise_tokens=800,
            verbose_tokens=1000,
            concise_capped=False,
            verbose_capped=False,
            concise_repetition=False,
            verbose_repetition=False,
            concise_corruption=False,
            verbose_corruption=False,
            min_pair_compression=0.30,
            min_concise_tokens=16,
        )
        self.assertFalse(result["quality_eligible"])
        self.assertIn("concise_answer_incorrect", result["rejection_reasons"])
        self.assertIn("insufficient_pair_compression", result["rejection_reasons"])


class RelativeTrajectoryBinningTests(unittest.TestCase):
    def test_bins_cover_every_token_once(self) -> None:
        bounds = relative_bin_bounds(101, 8)
        covered = [index for start, end in bounds for index in range(start, end)]
        self.assertEqual(covered, list(range(101)))
        self.assertTrue(all(end > start for start, end in bounds))

    def test_rejects_more_bins_than_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            relative_bin_bounds(7, 8)


if __name__ == "__main__":
    unittest.main()
