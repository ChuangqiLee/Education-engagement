#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stats.stats_utils import equal_variance_t_from_summary, two_group_between_ss, welch_t_from_summary


PAPER = {
    "n1": 116,
    "mean1": 83.84,
    "sd1": 9.29336,
    "n2": 118,
    "mean2": 81.21,
    "sd2": 9.50577,
    "t_equal": 2.134,
    "df_equal": 232.0,
    "p_equal": 0.034,
    "mean_difference": 2.62324,
    "standard_error": 1.22918,
    "anova_f": 4.290,
    "anova_p": 0.039,
    "anova_between_ss": 380.435,
    "anova_within_ss": 20574.176,
}


def main() -> None:
    output = Path("outputs/statistics")
    output.mkdir(parents=True, exist_ok=True)
    equal = equal_variance_t_from_summary(PAPER["n1"], PAPER["mean1"], PAPER["sd1"], PAPER["n2"], PAPER["mean2"], PAPER["sd2"])
    welch = welch_t_from_summary(PAPER["n1"], PAPER["mean1"], PAPER["sd1"], PAPER["n2"], PAPER["mean2"], PAPER["sd2"])
    between = two_group_between_ss(PAPER["n1"], PAPER["mean1"], PAPER["n2"], PAPER["mean2"])
    rows = [
        ("mean_difference", PAPER["mean_difference"], equal["mean_difference"], "partial", "reported means are rounded"),
        ("equal_variance_t", PAPER["t_equal"], equal["t"], "partial", "derivable from rounded summary statistics"),
        ("equal_variance_df", PAPER["df_equal"], equal["df"], "yes", "n1+n2-2"),
        ("equal_variance_p", PAPER["p_equal"], equal["p_two_sided"], "partial", "rounding and SPSS display"),
        ("standard_error", PAPER["standard_error"], equal["standard_error"], "partial", "rounded summaries"),
        ("anova_F", PAPER["anova_f"], equal["equivalent_two_group_anova_f"], "no", "for two groups ordinary ANOVA F must equal t squared"),
        ("anova_between_SS", PAPER["anova_between_ss"], between, "no", "reported means/sample sizes imply a different between-group SS"),
    ]
    csv_path = output / "paper_vs_recomputed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "paper_value", "independently_recomputed_value", "reproducible", "note", "difference"))
        for name, paper, computed, status, note in rows:
            writer.writerow((name, paper, computed, status, note, computed - paper))
    report = """# Statistical reproduction from summaries

No raw student grades were available. K-S normality and Levene variance tests
cannot be recomputed from means/SDs. The values below use only published,
rounded summaries.

- Equal-variance t: {t:.6f}, df={df:.0f}, p={p:.6f}
- Welch t: {wt:.6f}, df={wdf:.6f}, p={wp:.6f}
- Two-group ANOVA identity F=t^2: {f:.6f}; paper reports 4.290
- Between-group SS implied by rounded means: {ss:.6f}; paper reports 380.435

The ANOVA inconsistency is not repaired or fitted away. Synthetic data, when
used by the raw-data demo, is labelled synthetic and is not evidence for the
paper's findings.
""".format(t=equal["t"], df=equal["df"], p=equal["p_two_sided"], wt=welch["t"], wdf=welch["df"], wp=welch["p_two_sided"], f=equal["equivalent_two_group_anova_f"], ss=between)
    (output / "statistical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"equal_variance": equal, "welch": welch, "between_ss": between, "output": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()

