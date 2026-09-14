"""Self-contained HTML report plus chart assets."""

import html
import json
from pathlib import Path
from typing import Iterable, Union

from .aggregate import aggregate_records


def write_html_report(records: Iterable[object], output: Union[str, Path], title: str = "Class engagement report") -> Path:
    records = list(records)
    summary = aggregate_records(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chart_name = output.stem + "_engagement_over_time.png"
    chart_path = output.parent / chart_name
    if records:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grouped = {}
        for row in records:
            if row.engagement_score is not None:
                grouped.setdefault(str(row.track_id), []).append((row.timestamp, row.engagement_score))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for student, points in grouped.items():
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label="Student " + student)
        ax.set(xlabel="Time (seconds)", ylabel="Demo/configured score", ylim=(0, 105))
        if grouped:
            ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
    warning_rows = "".join(
        "<tr><td>{timestamp:.2f}</td><td>{source_id}</td><td>{track_id}</td><td>{grade}</td></tr>".format(**item)
        for item in summary["low_engagement_timeline"]
    ) or "<tr><td colspan='4'>None</td></tr>"
    provenance = "DEMO / UNTRAINED" if any(getattr(row, "provenance", "") == "DEMO_UNTRAINED" for row in records) else "MODEL INFERENCE"
    document = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui;max-width:980px;margin:32px auto;color:#16202a}} .warning{{padding:12px;background:#fff2cc;border-left:5px solid #d99b00}} table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{white-space:pre-wrap}}</style></head>
<body><h1>{title}</h1><p class="warning"><strong>{provenance}</strong>. Scores depend on configured assumptions and are not validated educational measurements.</p>
<h2>Class summary</h2><pre>{summary}</pre>
{chart}
<h2>Low-engagement warning timeline</h2><table><thead><tr><th>Time</th><th>Source</th><th>Track</th><th>Grade</th></tr></thead><tbody>{warnings}</tbody></table>
</body></html>""".format(
        title=html.escape(title),
        provenance=provenance,
        summary=html.escape(json.dumps(summary, indent=2, ensure_ascii=False)),
        chart=("<h2>Engagement over time</h2><img src='%s' style='max-width:100%%'>" % html.escape(chart_name)) if records else "",
        warnings=warning_rows,
    )
    output.write_text(document, encoding="utf-8")
    return output

