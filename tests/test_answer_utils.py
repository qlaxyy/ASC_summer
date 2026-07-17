import unittest

from answer_utils import compare_answers
from answer_utils import extract_answer
from answer_utils import extract_answer_with_trace
from answer_utils import evaluate_generation_answer


class FinalAnswerMarkdownTests(unittest.TestCase):
    def test_prefers_bold_rhs_of_final_total_equation(self) -> None:
        output = """
        The program had 60 downloads initially.
        Total downloads = 60 + 180 + 126 = **366**
        """
        self.assertEqual(extract_answer(output), "366")

    def test_accepts_latex_escaped_currency_on_rhs(self) -> None:
        output = r"""
        ### Conclusion
        Total Cost: \( \$40 + \$24 = \$64 \).
        Kylar pays **\$64** for all 16 glasses.
        """
        self.assertEqual(extract_answer(output), "64")

    def test_accepts_qualified_number_inside_boxed_text(self) -> None:
        output = r"""
        \[
        \boxed{\text{Approximately }95\text{ minutes}}
        \]
        """
        self.assertTrue(compare_answers(extract_answer(output), "95"))

    def test_real_archived_bold_total_is_not_misread_as_first_number(self) -> None:
        output = r"""
        The program had a total of **366 downloads** over the three months.
        - First Month: 60 downloads
        - Second Month: 3 x 60 = 180 downloads
        - Third Month: 70% of 180 = 126 downloads
        Total downloads = 60 + 180 + 126 = **366**
        """
        pred, correct, trace = evaluate_generation_answer(output, "366")
        self.assertEqual(pred, "366")
        self.assertTrue(correct)
        self.assertEqual(trace["confidence"], "high")

    def test_real_archived_conclusion_prefers_total_over_prior_subtotal(self) -> None:
        output = r"""
        Regular glasses cost \$40 and discounted glasses cost \$24.
        ### Conclusion
        Total Cost: \( \$40 + \$24 = \$64 \).
        Kylar pays **\$64** for all 16 glasses.
        """
        self.assertEqual(extract_answer(output), "64")

    def test_only_last_boxed_answer_controls_correctness(self) -> None:
        output = r"""
        Initial attempt: \boxed{42}.
        Correction: the final answer is \boxed{41}.
        """
        pred, correct, trace = evaluate_generation_answer(output, "42")
        self.assertEqual(pred, "41")
        self.assertFalse(correct)
        self.assertEqual(trace["candidates"], ["42", "41"])
        self.assertEqual(trace["selection_policy"], "last_explicit_candidate")
        self.assertFalse(trace["requires_review"])

    def test_unmarked_tail_guess_is_low_confidence(self) -> None:
        trace = extract_answer_with_trace(
            "I tried several possibilities. The remaining calculation gives 17"
        )
        self.assertEqual(trace["answer"], "17")
        self.assertEqual(trace["confidence"], "low")
        self.assertTrue(trace["requires_review"])
        self.assertIn("heuristic_numeric_fallback", trace["warnings"])

    def test_empty_generation_is_reviewable_and_incorrect(self) -> None:
        pred, correct, trace = evaluate_generation_answer("", "1")
        self.assertEqual(pred, "")
        self.assertFalse(correct)
        self.assertTrue(trace["requires_review"])


if __name__ == "__main__":
    unittest.main()
