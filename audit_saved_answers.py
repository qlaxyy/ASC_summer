"""Re-score archived evaluation JSON without modifying the source files.

The output filename contains both the answer-parser version and the SHA-256 of
the source JSON. This makes each audit an immutable, reproducible sidecar and
keeps parser fixes separate from the original GPU generation archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from answer_utils import ANSWER_PARSER_VERSION
from answer_utils import evaluate_generation_answer


REQUIRED_DETAIL_KEYS = {"model_output", "pred_answer", "gt_answer", "correct"}
AUDIT_SCHEMA_VERSION = 2


def iter_detail_rows(value: Any, path: str = "root") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if REQUIRED_DETAIL_KEYS.issubset(value):
            yield path, value
        for key, child in value.items():
            yield from iter_detail_rows(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_detail_rows(child, f"{path}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_file(source_path: Path) -> dict[str, Any]:
    raw = source_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    records = list(iter_detail_rows(payload))

    old_correct = 0
    new_correct = 0
    prediction_changes = 0
    judgment_changes = 0
    confidence_counts: Counter[str] = Counter()
    review_queue: list[dict[str, Any]] = []

    for record_path, row in records:
        old_pred = str(row.get("pred_answer", ""))
        old_judgment = bool(row.get("correct", False))
        gt = str(row.get("gt_answer", ""))
        new_pred, new_judgment, trace = evaluate_generation_answer(
            str(row.get("model_output", "")), gt
        )

        old_correct += int(old_judgment)
        new_correct += int(new_judgment)
        pred_changed = new_pred != old_pred
        judgment_changed = new_judgment != old_judgment
        prediction_changes += int(pred_changed)
        judgment_changes += int(judgment_changed)
        confidence_counts[str(trace["confidence"])] += 1

        review_reasons = list(trace["warnings"])
        if pred_changed:
            review_reasons.append("prediction_changed_since_archive")
        if judgment_changed:
            review_reasons.append("judgment_changed_since_archive")
        if row.get("length_capped"):
            review_reasons.append("length_capped")
        if row.get("corruption_artifact"):
            review_reasons.append("corruption_artifact")

        if (
            trace["requires_review"]
            or pred_changed
            or judgment_changed
            or row.get("length_capped")
            or row.get("corruption_artifact")
        ):
            output = str(row.get("model_output", ""))
            review_queue.append(
                {
                    "record_path": record_path,
                    "question": str(row.get("question", "")),
                    "old_pred_answer": old_pred,
                    "new_pred_answer": new_pred,
                    "gt_answer": gt,
                    "old_correct": old_judgment,
                    "new_correct": new_judgment,
                    "answer_extraction": trace,
                    "review_reasons": list(dict.fromkeys(review_reasons)),
                    "model_output_tail": output[-1000:],
                }
            )

    source_hash = sha256_file(source_path)
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "answer_parser_version": ANSWER_PARSER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": source_path.as_posix(),
            "sha256": source_hash,
            "bytes": source_path.stat().st_size,
        },
        "summary": {
            "records": len(records),
            "old_correct": old_correct,
            "new_correct": new_correct,
            "prediction_changes": prediction_changes,
            "judgment_changes": judgment_changes,
            "review_queue_count": len(review_queue),
            "confidence_counts": dict(sorted(confidence_counts.items())),
        },
        "review_queue": review_queue,
    }


def write_audit_atomic(payload: dict[str, Any], output_path: Path) -> bool:
    if output_path.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable answer-parser audit sidecars for saved eval JSON."
    )
    parser.add_argument("paths", nargs="+", help="Saved evaluation JSON file(s).")
    parser.add_argument(
        "--output_dir",
        default="results/answer_audits",
        help="Directory for content-addressed audit sidecars.",
    )
    parser.add_argument(
        "--fail_on_judgment_change",
        action="store_true",
        help="Exit nonzero if re-scoring changes at least one correctness judgment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    changed_total = 0

    for raw_path in args.paths:
        source_path = Path(raw_path)
        audit = audit_file(source_path)
        source_hash = audit["source"]["sha256"]
        safe_version = ANSWER_PARSER_VERSION.replace(".", "_")
        output_path = output_dir / (
            f"{source_path.stem}.answer_audit.s{AUDIT_SCHEMA_VERSION}."
            f"v{safe_version}.{source_hash[:12]}.json"
        )
        created = write_audit_atomic(audit, output_path)
        summary = audit["summary"]
        changed_total += int(summary["judgment_changes"])
        action = "created" if created else "exists"
        print(
            f"{source_path}: records={summary['records']}, "
            f"correct={summary['old_correct']}->{summary['new_correct']}, "
            f"judgment_changes={summary['judgment_changes']}, "
            f"review={summary['review_queue_count']}"
        )
        print(f"  {action}: {output_path}")

    if args.fail_on_judgment_change and changed_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
