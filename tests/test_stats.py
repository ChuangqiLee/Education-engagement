from stats.stats_utils import equal_variance_t_from_summary, two_group_between_ss


def test_summary_statistics_recompute_and_detect_anova_gap():
    result = equal_variance_t_from_summary(116, 83.84, 9.29336, 118, 81.21, 9.50577)
    assert result["df"] == 232
    assert 2.1 < result["t"] < 2.2
    assert abs(result["equivalent_two_group_anova_f"] - 4.290) > 0.1
    assert abs(two_group_between_ss(116, 83.84, 118, 81.21) - 380.435) > 1

