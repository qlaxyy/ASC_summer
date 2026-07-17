import unittest

try:
    from asc_steering_utils import pair_texts_for_activation
except (ImportError, OSError) as exc:  # pragma: no cover - local CPU env may lack torch
    pair_texts_for_activation = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(pair_texts_for_activation is None, f"optional ML deps unavailable: {IMPORT_ERROR}")
class RawEndpointTextTests(unittest.TestCase):
    def test_raw_problem_is_shared_prefix_for_both_cots(self) -> None:
        row = {
            "question": "What is 2 + 3?",
            "concise_output": "2+3=5. Final answer: 5",
            "verbose_output": "First add two and three. Final answer: 5",
        }

        short_text, long_text = pair_texts_for_activation(
            row, "asc_endpoint_raw"
        )

        self.assertEqual(short_text, "What is 2 + 3?\n2+3=5. Final answer: 5")
        self.assertEqual(
            long_text,
            "What is 2 + 3?\nFirst add two and three. Final answer: 5",
        )
        self.assertFalse(short_text.startswith("Question:"))
        self.assertNotIn("Let's think step by step", short_text)


if __name__ == "__main__":
    unittest.main()
