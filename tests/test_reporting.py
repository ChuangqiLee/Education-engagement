from engagement.pipeline.engine import PipelineRecord
from engagement.reporting.aggregate import aggregate_records
from engagement.reporting.html_report import write_html_report


def _record(timestamp, grade, score):
    return PipelineRecord(timestamp, "front", 1, (0, 0, 10, 10), "neutral", 0.6, "turn_head", 0.7, 0, 0, 0, None, [0.1, 0.4, 0.5], score, grade, {"G1": "active", "G2": "partial", "G3": "passive"}[grade], "DEMO_UNTRAINED")


def test_aggregate_and_report(tmp_path):
    records = [_record(0, "G1", 80), _record(2, "G3", 35)]
    summary = aggregate_records(records)
    assert summary["student_count"] == 1
    assert len(summary["low_engagement_timeline"]) == 1
    output = write_html_report(records, tmp_path / "report.html")
    assert output.exists()
    assert "DEMO / UNTRAINED" in output.read_text(encoding="utf-8")

