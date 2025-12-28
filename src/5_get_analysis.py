import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# -----------------------------
# データ読み込み
# -----------------------------
df = pd.read_csv("../data/processed/commit_summary_readability_3metrics.csv")

# -----------------------------
# 平均値・中央値の算出
# -----------------------------
metrics = [
    "mi_before_avg", "mi_after_avg", "mi_diff_avg",
    "cc_before_avg", "cc_after_avg", "cc_diff_avg",
    "loc_before_avg", "loc_after_avg", "loc_diff_avg"
]

summary_stats = pd.DataFrame({
    "Mean": df[metrics].mean(),
    "Median": df[metrics].median()
}).round(2)   # 小数第2位

print("=== Mean & Median ===")
print(summary_stats)
print()

# -----------------------------
# Effect size (r) の算出
# r = Z / sqrt(N)
# -----------------------------
def wilcoxon_effect_size(before, after):
    """
    Calculate effect size r for Wilcoxon signed-rank test.
    r = Z / sqrt(N)
    """
    diff = after - before
    n = np.sum(diff != 0)  # Number of non-zero differences

    # Edge case: no non-zero differences
    if n == 0:
        return np.nan, np.nan

    # Use method='approx' to get z-statistic directly (scipy >= 1.9.0)
    # wilcoxon(after, before) so positive z means after > before
    result = wilcoxon(after, before, method='approx')
    z = result.zstatistic
    p = result.pvalue

    r = z / np.sqrt(n)
    return r, p

effect_sizes = {}

effect_sizes["MI_r"], effect_sizes["MI_p"] = wilcoxon_effect_size(
    df["mi_before_avg"], df["mi_after_avg"]
)
effect_sizes["CC_r"], effect_sizes["CC_p"] = wilcoxon_effect_size(
    df["cc_before_avg"], df["cc_after_avg"]
)
effect_sizes["LOC_r"], effect_sizes["LOC_p"] = wilcoxon_effect_size(
    df["loc_before_avg"], df["loc_after_avg"]
)

effect_df = pd.DataFrame.from_dict(
    effect_sizes, orient="index", columns=["Value"]
)

# r は小数第2位、p-value はそのまま
effect_df.loc[effect_df.index.str.endswith("_r")] = \
    effect_df.loc[effect_df.index.str.endswith("_r")].astype(float).round(2)

effect_df.loc[effect_df.index.str.endswith("_p")] = \
    effect_df.loc[effect_df.index.str.endswith("_p")].astype(float)

print("=== Effect Size (r) & p-value ===")
print(effect_df)
print()

# -----------------------------
# 改善・悪化割合
# -----------------------------
n_commits = len(df)

# MI：増加が改善
mi_improve = (df["mi_diff_avg"] > 0).sum() / n_commits
mi_worsen  = (df["mi_diff_avg"] < 0).sum() / n_commits

# CC：減少が改善
cc_improve = (df["cc_diff_avg"] < 0).sum() / n_commits
cc_worsen  = (df["cc_diff_avg"] > 0).sum() / n_commits

# LOC：減少が改善
loc_improve = (df["loc_diff_avg"] < 0).sum() / n_commits
loc_worsen  = (df["loc_diff_avg"] > 0).sum() / n_commits

improvement_df = pd.DataFrame({
    "Improve (%)": [
        mi_improve * 100,
        cc_improve * 100,
        loc_improve * 100
    ],
    "Worsen (%)": [
        mi_worsen * 100,
        cc_worsen * 100,
        loc_worsen * 100
    ]
}, index=["MI", "CC", "LOC"]).round(1)   # 小数第1位（％）

print("=== Improvement / Worsening Ratio ===")
print(improvement_df)
