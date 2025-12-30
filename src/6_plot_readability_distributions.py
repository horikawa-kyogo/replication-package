import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("../data/processed/commit_summary_readability_3metrics.csv")
# -----------------------------
# Specify the destination directory
# -----------------------------
out_dir = Path("../figure")
out_dir.mkdir(parents=True, exist_ok=True)


metrics = [
    ("MI",  "mi_before_avg",  "mi_after_avg"),
    ("CC",  "cc_before_avg",  "cc_after_avg"),
    ("LOC", "loc_before_avg", "loc_after_avg")
]

palette = {
    "Before": "#f2a7a7",
    "After":  "#a9c6ea"
}

fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False)

for ax, (name, before_col, after_col) in zip(axes, metrics):

    before = df[before_col].dropna()
    after  = df[after_col].dropna()

    # -------------------------
    # split violin
    # -------------------------
    plot_df = pd.DataFrame({
        "Value": pd.concat([before, after], ignore_index=True),
        "Stage": ["Before"] * len(before) + ["After"] * len(after)
    })

    sns.violinplot(
        data=plot_df,
        x=[""] * len(plot_df),
        y="Value",
        hue="Stage",
        split=True,
        inner=None,
        palette=palette,
        cut=0,
        linewidth=1.0,
        ax=ax
    )

    # -------------------------
    # Boxplot
    # -------------------------
    ax.boxplot(
        before,
        positions=[-0.16],      # ← Shift left
        widths=0.1,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.4),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2)
    )

    ax.boxplot(
        after,
        positions=[0.16],       # ← Shift right
        widths=0.1,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.4),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2)
    )

    # -------------------------
    # Average
    # -------------------------
    ax.scatter(
        [-0.28, 0.28],
        [before.mean(), after.mean()],
        color="white",
        edgecolor="black",
        s=90,
        zorder=5
    )

    # -------------------------
    # Configure subplot appearance (title, axes, grid)
    # -------------------------
    ax.set_title(name, fontsize=18)
    ax.set_ylabel(name, fontsize=13)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=12)

    if ax != axes[0]:
        ax.get_legend().remove()

# legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:2], labels[:2], frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(
    out_dir / "mi_cc_loc_before_after_distribution.png",
    dpi=300,                 # paper quality
    bbox_inches="tight",     # adjust margins
    facecolor="white"        # background
)
plt.show()
