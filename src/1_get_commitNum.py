import pandas as pd
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
# --- Loading data ---
commits_df = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
prs_df = pd.read_parquet(DATA_DIR / "all_pull_request.parquet")
details_df = pd.read_parquet(DATA_DIR / "pr_commit_details.parquet")

# --- Split repo_url into owner/repo ---
prs_df["repo_full"] = prs_df["repo_url"].str.replace("https://github.com/", "", regex=False)
prs_df["repo_name"] = prs_df["repo_full"].apply(lambda x: x.split("/")[-1])
prs_df["repo_owner"] = prs_df["repo_full"].apply(lambda x: x.split("/")[0])

main_repo_owner = prs_df.groupby("repo_name")["repo_owner"].agg(lambda x: Counter(x).most_common(1)[0][0])
prs_df["is_fork"] = prs_df.apply(lambda row: row["repo_owner"] != main_repo_owner[row["repo_name"]], axis=1)

# --- Readability Keywords ---
keywords = [
    "readability", "readable", "understandability", "understandable",
    "clarity", "legibility", "easier to read", "comprehensible"
]

# --- ① commits containing keywords ---
keyword_commits = commits_df[
    commits_df["message"].fillna("").str.lower().apply(
        lambda m: any(kw in m for kw in keywords)
    )
]

# --- ② Combine PR information and exclude forks ---
keyword_commits = keyword_commits.merge(
    prs_df[["id", "is_fork"]],
    left_on="pr_id",
    right_on="id",
    how="left"
)
not_fork_commits = keyword_commits[keyword_commits["is_fork"] == False]

# --- ③ Python file change commit extract ---
py_details = details_df[details_df["filename"].str.endswith(".py", na=False)]
py_commits = set(py_details["sha"].unique())
py_filtered_commits = not_fork_commits[
    not_fork_commits["sha"].isin(py_commits)
]

# --- Results ---
print(f"0️⃣ Total number of commits: {len(commits_df)}")
print(f"1️⃣ Number of commits containing the keyword: {len(keyword_commits)}")
print(f"2️⃣ Number of commits after fork removal: {len(not_fork_commits)}")
print(f"3️⃣ Number of commits in which Python files were changed: {len(py_filtered_commits)}")
