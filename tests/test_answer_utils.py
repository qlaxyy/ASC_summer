import unittest

from answer_utils import extract_answer


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


if __name__ == "__main__":
    unittest.main()
