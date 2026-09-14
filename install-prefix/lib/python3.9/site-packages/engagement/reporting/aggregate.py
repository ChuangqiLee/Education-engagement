"""Per-student and classroom-level aggregation."""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List


def aggregate_records(records: Iterable[object], low_threshold: float = 50.0) -> Dict[str, object]:
    rows = list(records)
    grade_counts = Counter(row.engagement_level for row in rows)
    per_student = defaultdict(list)
    warnings = []
    for row in rows:
        per_student[str(row.track_id)].append(row)
        if row.engagement_level == "G3" or (
            row.engagement_score is not None and row.engagement_score < low_threshold
        ):
            warnings.append(
                {"timestamp": row.timestamp, "source_id": row.source_id, "track_id": row.track_id, "grade": row.engagement_level}
            )
    student_summary = {}
    for student, values in per_student.items():
        scores = [item.engagement_score for item in values if item.engagement_score is not None]
        student_summary[student] = {
            "observations": len(values),
            "grade_distribution": dict(Counter(item.engagement_level for item in values)),
            "average_score": None if not scores else sum(scores) / len(scores),
            "start": min(item.timestamp for item in values),
            "end": max(item.timestamp for item in values),
        }
    all_scores = [row.engagement_score for row in rows if row.engagement_score is not None]
    return {
        "observations": len(rows),
        "student_count": len(per_student),
        "grade_distribution": {grade: grade_counts.get(grade, 0) for grade in ("G1", "G2", "G3")},
        "average_engagement_score": None if not all_scores else sum(all_scores) / len(all_scores),
        "per_student": student_summary,
        "low_engagement_timeline": warnings,
    }

