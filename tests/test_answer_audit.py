import json
import tempfile
import unittest
from pathlib import Path

from audit_saved_answers import audit_file


class SavedAnswerAuditTests(unittest.TestCase):
    def test_audit_detects_stale_judgment_without_mutating_source(self) -> None:
        payload = {
            "detailed_results": [
                {
                    "question": "How many?",
                    "model_output": "Total = 60 + 180 + 126 = **366**",
                    "pred_answer": "60",
                    "gt_answer": "366",
                    "correct": False,
                    "tokens": 10,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "saved.json"
            original = json.dumps(payload)
            source.write_text(original, encoding="utf-8")

            audit = audit_file(source)

            self.assertEqual(source.read_text(encoding="utf-8"), original)
            self.assertEqual(audit["summary"]["records"], 1)
            self.assertEqual(audit["summary"]["judgment_changes"], 1)
            self.assertEqual(audit["summary"]["old_correct"], 0)
            self.assertEqual(audit["summary"]["new_correct"], 1)
            self.assertEqual(len(audit["review_queue"]), 1)


if __name__ == "__main__":
    unittest.main()
