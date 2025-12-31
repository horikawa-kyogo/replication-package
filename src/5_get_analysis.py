import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# =================================================
# Data loading
# =================================================
df = pd.read_csv("../data/processed/commit_summary_readability_allmetrics.csv")

# =================================================
# single_comments
# =================================================
df["single_comments_before_avg"] = (
    df["comments_before_avg"] - df["multi_before_avg"]
)
df["single_comments_after_avg"] = (
    df["comments_after_avg"] - df["multi_after_avg"]
)

df["single_comments_before_avg"] = df["single_comments_before_avg"].clip(lower=0)
df["single_comments_after_avg"] = df["single_comments_after_avg"].clip(lower=0)

# =================================================
# diff（after - before）
# =================================================
diff_metrics = [
    "mi", "cc",
    "loc", "lloc", "sloc",
    "comments", "single_comments", "multi", "blank",
    "h_volume", "h_difficulty", "h_effort"
]

for m in diff_metrics:
    df[f"{m}_diff_avg"] = df[f"{m}_after_avg"] - df[f"{m}_before_avg"]

# =================================================
# Mean and median
# =================================================
metrics = []
for m in diff_metrics:
    metrics.extend([
        f"{m}_before_avg",
        f"{m}_after_avg",
        f"{m}_diff_avg"
    ])

summary_stats = pd.DataFrame({
    "Mean": df[metrics].mean(),
    "Median": df[metrics].median()
}).round(2)

print("=== Mean & Median ===")
print(summary_stats)
print()

# =================================================
# Cliff's delta（paired）
# =================================================
def cliffs_delta_paired(before, after):
    """
    Paired Cliff's delta.
    delta = ( #after>before - #after<before ) / N_nonzero
    """
    diff = after - before
    diff = diff[diff != 0]
    n = len(diff)

    if n == 0:
        return np.nan

    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)

    return (gt - lt) / n

# =================================================
# Wilcoxon p-value + Cliff's delta
# =================================================
rows = []
for m in diff_metrics:
    before = df[f"{m}_before_avg"]
    after  = df[f"{m}_after_avg"]

    # p-value
    try:
        p = wilcoxon(after, before, method="approx").pvalue
    except ValueError:
        p = np.nan

    # effect size
    delta = cliffs_delta_paired(before, after)

    rows.append((m, delta, p))

effect_df = pd.DataFrame(
    rows, columns=["Metric", "Cliff's delta", "p-value"]
).set_index("Metric")

def format_delta(x):
    return f"{x:.2f}" if pd.notna(x) else "–"

def format_p(x):
    if pd.isna(x):
        return "–"
    return "<0.001" if x < 0.001 else f"{x:.3f}"

effect_df["Effect size"] = effect_df["Cliff's delta"].apply(format_delta)
effect_df["p-value"] = effect_df["p-value"].apply(format_p)
effect_df = effect_df[["Effect size", "p-value"]]

print("=== Cliff's delta & p-value ===")
print(effect_df)
print()

# =================================================
# Improvement/worsening rate
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

improve_policy = {
    "mi": "increase",
    "cc": "decrease",
    "loc": "decrease",
    "lloc": "decrease",
    "sloc": "decrease",
    "comments": "increase",
    "single_comments": "increase",
    "multi": "decrease",
    "blank": "neutral",
    "h_volume": "decrease",
    "h_difficulty": "decrease",
    "h_effort": "decrease",
}

rows = []
for m, policy in improve_policy.items():
    if policy == "neutral":
        continue
    imp, wors = improvement_ratio(df[f"{m}_diff_avg"], policy)
    rows.append((m, imp, wors))

improvement_df = pd.DataFrame(
    rows, columns=["Metric", "Improve (%)", "Worsen (%)"]
).set_index("Metric").round(1)

print("=== Improvement / Worsening Ratio ===")
print(improvement_df)
