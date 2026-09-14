"""Statistical checks possible from summary values."""

import math
from typing import Dict


def equal_variance_t_from_summary(n1: int, mean1: float, sd1: float, n2: int, mean2: float, sd2: float) -> Dict[str, float]:
    df = n1 + n2 - 2
    pooled_variance = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / df
    standard_error = math.sqrt(pooled_variance * (1.0 / n1 + 1.0 / n2))
    t_statistic = (mean1 - mean2) / standard_error
    try:
        from scipy.stats import t as student_t

        p_value = float(2 * student_t.sf(abs(t_statistic), df))
    except ImportError:
        p_value = float("nan")
    return {
        "mean_difference": mean1 - mean2,
        "pooled_variance": pooled_variance,
        "standard_error": standard_error,
        "t": t_statistic,
        "df": float(df),
        "p_two_sided": p_value,
        "equivalent_two_group_anova_f": t_statistic**2,
    }


def welch_t_from_summary(n1: int, mean1: float, sd1: float, n2: int, mean2: float, sd2: float) -> Dict[str, float]:
    variance = sd1**2 / n1 + sd2**2 / n2
    standard_error = math.sqrt(variance)
    statistic = (mean1 - mean2) / standard_error
    df = variance**2 / ((sd1**2 / n1) ** 2 / (n1 - 1) + (sd2**2 / n2) ** 2 / (n2 - 1))
    try:
        from scipy.stats import t as student_t

        p_value = float(2 * student_t.sf(abs(statistic), df))
    except ImportError:
        p_value = float("nan")
    return {"standard_error": standard_error, "t": statistic, "df": df, "p_two_sided": p_value}


def two_group_between_ss(n1: int, mean1: float, n2: int, mean2: float) -> float:
    grand = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    return n1 * (mean1 - grand) ** 2 + n2 * (mean2 - grand) ** 2

