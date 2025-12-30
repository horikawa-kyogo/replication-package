import pandas as pd
import os
from pathlib import Path

# --- output file path ---
output_path_parquet = "../data/processed/filtered_commits.parquet"
output_path_csv = "../data/processed/filtered_commits.csv"  # CSV出力先

# --- Loading data ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

commits_df = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
prs_df = pd.read_parquet(DATA_DIR / "all_pull_request.parquet")
details_df = pd.read_parquet(DATA_DIR / "pr_commit_details.parquet")

# --- Only commits that change Python files ---
py_details = details_df[details_df["filename"].str.endswith(".py", na=False)]
py_commits = set(py_details["sha"].unique())

# --- Readability Keywords ---
keywords = [
    "readability", "readable", "understandability", "understandable",
    "clarity", "legibility", "easier to read", "comprehensible"
]

# --- Keyword filter + Python file changes ---
filtered_commits = commits_df[
    commits_df["message"].fillna("").str.lower().apply(
        lambda m: any(kw in m for kw in keywords)
    )
    & commits_df["sha"].isin(py_commits)
]

# --- Add PR information ---
filtered_commits = filtered_commits.merge(
    prs_df[["id", "repo_url"]],
    left_on="pr_id",
    right_on="id",
    how="left"
)

# --- Save Parquet ---
output_dir = os.path.dirname(output_path_parquet)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

filtered_commits.to_parquet(output_path_parquet)
print(f"✅ Parquet Save completed: {output_path_parquet}（{len(filtered_commits)}件のコミット）")

# --- CSV output ---
# 必要なカラムだけ抽出（例: sha, message, pr_id, repo_url）
columns_to_export = ["sha", "message", "pr_id", "repo_url"]
filtered_commits[columns_to_export].to_csv(
    output_path_csv, index=False, encoding="utf-8-sig"
)
print(f"✅ CSV Save completed: {output_path_csv}（{len(filtered_commits)}件のコミット）")
