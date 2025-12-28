import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =================================================
# データ読み込み
# =================================================
df = pd.read_csv("../data/processed/commit_summary_readability_allmetrics.csv")

# =================================================
# diff（after - before）を生成
# =================================================
df["mi_diff_avg"]   = df["mi_after_avg"]  - df["mi_before_avg"]
df["cc_diff_avg"]   = df["cc_after_avg"]  - df["cc_before_avg"]

df["loc_diff_avg"]  = df["loc_after_avg"]  - df["loc_before_avg"]
df["lloc_diff_avg"] = df["lloc_after_avg"] - df["lloc_before_avg"]
df["sloc_diff_avg"] = df["sloc_after_avg"] - df["sloc_before_avg"]

df["h_volume_diff_avg"] = (
    df["h_volume_after_avg"] - df["h_volume_before_avg"]
)
df["h_difficulty_diff_avg"] = (
    df["h_difficulty_after_avg"] - df["h_difficulty_before_avg"]
)
df["h_effort_diff_avg"] = (
    df["h_effort_after_avg"] - df["h_effort_before_avg"]
)

# =================================================
# 平均値・中央値
# =================================================
metrics = [
    # MI
    "mi_before_avg", "mi_after_avg", "mi_diff_avg",

    # CC
    "cc_before_avg", "cc_after_avg", "cc_diff_avg",

    # Raw
    "loc_before_avg", "loc_after_avg", "loc_diff_avg",
    "lloc_before_avg", "lloc_after_avg", "lloc_diff_avg",
    "sloc_before_avg", "sloc_after_avg", "sloc_diff_avg",

    # Halstead
    "h_volume_before_avg", "h_volume_after_avg", "h_volume_diff_avg",
    "h_difficulty_before_avg", "h_difficulty_after_avg", "h_difficulty_diff_avg",
    "h_effort_before_avg", "h_effort_after_avg", "h_effort_diff_avg",
]

summary_stats = pd.DataFrame({
    "Mean": df[metrics].mean(),
    "Median": df[metrics].median()
}).round(2)

print("=== Mean & Median ===")
print(summary_stats)
print()

# =================================================
# Wilcoxon + Effect size (r)
# =================================================
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

# --- MI ---
effect_sizes["MI"] = wilcoxon_effect_size(
    df["mi_before_avg"], df["mi_after_avg"]
)

# --- CC ---
effect_sizes["CC"] = wilcoxon_effect_size(
    df["cc_before_avg"], df["cc_after_avg"]
)

# --- Raw ---
effect_sizes["LOC"] = wilcoxon_effect_size(
    df["loc_before_avg"], df["loc_after_avg"]
)
effect_sizes["LLOC"] = wilcoxon_effect_size(
    df["lloc_before_avg"], df["lloc_after_avg"]
)
effect_sizes["SLOC"] = wilcoxon_effect_size(
    df["sloc_before_avg"], df["sloc_after_avg"]
)

# --- Halstead ---
effect_sizes["Halstead Volume"] = wilcoxon_effect_size(
    df["h_volume_before_avg"], df["h_volume_after_avg"]
)
effect_sizes["Halstead Difficulty"] = wilcoxon_effect_size(
    df["h_difficulty_before_avg"], df["h_difficulty_after_avg"]
)
effect_sizes["Halstead Effort"] = wilcoxon_effect_size(
    df["h_effort_before_avg"], df["h_effort_after_avg"]
)

# =================================================
# 表記整形（r: 小数2位, p: <0.001）
# =================================================
def format_r(value):
    return f"{value:.2f}"

def format_p(value):
    if value < 0.001:
        return "<0.001"
    else:
        return f"{value:.3f}"

rows = []
for metric, (r, p) in effect_sizes.items():
    rows.append((
        metric,
        format_r(r),
        format_p(p)
    ))

effect_df = pd.DataFrame(
    rows, columns=["Metric", "Effect size (r)", "p-value"]
).set_index("Metric")

print("=== Effect Size (r) & p-value ===")
print(effect_df)
print()

# =================================================
# 改善・悪化割合
# =================================================
n_commits = len(df)

def improvement_ratio(diff, improve_if="increase"):
    if improve_if == "increase":
        improve = (diff > 0).sum() / n_commits
        worsen  = (diff < 0).sum() / n_commits
    else:
        improve = (diff < 0).sum() / n_commits
        worsen  = (diff > 0).sum() / n_commits
    return improve * 100, worsen * 100

rows = []

# MI（増加が改善）
rows.append(("MI", *improvement_ratio(df["mi_diff_avg"], "increase")))

# CC（減少が改善）
rows.append(("CC", *improvement_ratio(df["cc_diff_avg"], "decrease")))

# Raw
rows.append(("LOC",  *improvement_ratio(df["loc_diff_avg"], "decrease")))
rows.append(("LLOC", *improvement_ratio(df["lloc_diff_avg"], "decrease")))
rows.append(("SLOC", *improvement_ratio(df["sloc_diff_avg"], "decrease")))

# Halstead
rows.append(("Halstead Volume",
             *improvement_ratio(df["h_volume_diff_avg"], "decrease")))
rows.append(("Halstead Difficulty",
             *improvement_ratio(df["h_difficulty_diff_avg"], "decrease")))
rows.append(("Halstead Effort",
             *improvement_ratio(df["h_effort_diff_avg"], "decrease")))

improvement_df = pd.DataFrame(
    rows, columns=["Metric", "Improve (%)", "Worsen (%)"]
).set_index("Metric").round(1)

print("=== Improvement / Worsening Ratio ===")
print(improvement_df)
