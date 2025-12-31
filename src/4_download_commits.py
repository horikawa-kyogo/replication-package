import os
import subprocess
import pandas as pd
import numpy as np
from radon.metrics import mi_visit, h_visit
from radon.complexity import cc_visit
from radon.raw import analyze as raw_analyze
import tempfile
import shutil
from pathlib import Path
import ast

# =================================================
# GitHub PAT
# =================================================
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")

# =================================================
# Data loading
# =================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

commits_df = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
prs_df = pd.read_parquet(DATA_DIR / "all_pull_request.parquet")
details_df = pd.read_parquet(DATA_DIR / "pr_commit_details.parquet")

# =================================================
# Commit containing Python files
# =================================================
py_commits = set(
    details_df[details_df["filename"].str.endswith(".py", na=False)]["sha"].unique()
)

# =================================================
# readability keywords
# =================================================
keywords = [
    "readability", "readable", "understandability", "understandable",
    "clarity", "legibility", "easier to read", "comprehensible"
]

filtered = commits_df[
    commits_df["sha"].isin(py_commits) &
    commits_df["message"].fillna("").str.lower().apply(
        lambda m: any(k in m for k in keywords)
    )
]

if filtered.empty:
    print("No target commit")
    exit(0)

# =================================================
# temporary directory
# =================================================
tmp_root = tempfile.mkdtemp(prefix="repo_radon_")
print(f"🧰 temporary directory: {tmp_root}")

# =================================================
# AST analysis
# =================================================
def count_definitions(code: str):
    try:
        tree = ast.parse(code)
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        n_functions = 0
        n_classes = 0
        n_methods = 0
        n_docstrings = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                n_functions += 1
                if isinstance(parents.get(node), ast.ClassDef):
                    n_methods += 1
                if ast.get_docstring(node):
                    n_docstrings += 1
            elif isinstance(node, ast.ClassDef):
                n_classes += 1
                if ast.get_docstring(node):
                    n_docstrings += 1
            elif isinstance(node, ast.Module):
                if ast.get_docstring(node):
                    n_docstrings += 1

        return n_functions, n_classes, n_methods, n_docstrings
    except Exception:
        return 0, 0, 0, 0

# =================================================
# Radon
# =================================================
def analyze_code(code: str):
    try:
        # --- MI ---
        mi = mi_visit(code, True)

        # --- CC ---
        cc_list = cc_visit(code)
        cc_total = sum(c.complexity for c in cc_list)

        # --- Raw ---
        raw = raw_analyze(code)
        n_functions, n_classes, n_methods, n_docstrings = count_definitions(code)

        raw_metrics = {
            "loc": raw.loc,
            "lloc": raw.lloc,
            "sloc": raw.sloc,
            "comments": raw.comments,
            "multi": raw.multi,
            "blank": raw.blank,
            "n_functions": n_functions,
            "n_classes": n_classes,
            "n_methods": n_methods,
            "n_docstrings": n_docstrings,
        }

        # --- Halstead ---
        try:
            h = h_visit(code)
            if h and h.total and h.total.length > 0:
                halstead_metrics = {
                    "h1": h.total.h1,
                    "h2": h.total.h2,
                    "N1": h.total.N1,
                    "N2": h.total.N2,
                    "h_volume": h.total.volume,
                    "h_difficulty": h.total.difficulty,
                    "h_effort": h.total.effort,
                    "t": h.total.time,
                    "b": h.total.bugs,
                }
            else:
                raise ValueError
        except Exception:
            halstead_metrics = {
                k: np.nan for k in
                ["h1","h2","N1","N2","h_volume","h_difficulty","h_effort","t","b"]
            }

        return mi, cc_total, raw_metrics, halstead_metrics

    except Exception as e:
        print(f"⚠️ Radon analysis failure: {e}")
        return None, None, None, None

# =================================================
# Git utilities
# =================================================
def get_parent_commit(repo_dir, sha):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{sha}^"],
            cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        return None

# =================================================
# Result storage
# =================================================
file_results = []
summary_results = []

skipped_no_parent = 0
skipped_clone_fail = 0

# =================================================
# main loop
# =================================================
for _, row in filtered.iterrows():
    sha = row["sha"]
    pr_id = row["pr_id"]

    repo_row = prs_df[prs_df["id"] == pr_id]
    if repo_row.empty:
        continue

    repo_url = repo_row.iloc[0]["repo_url"]
    if "api.github.com/repos" in repo_url:
        repo_url = repo_url.replace(
            "https://api.github.com/repos/", "https://github.com/"
        ) + ".git"

    if GITHUB_PAT and repo_url.startswith("https://github.com/"):
        repo_url = repo_url.replace(
            "https://github.com/",
            f"https://{GITHUB_PAT}@github.com/"
        )

    repo_name = os.path.basename(repo_url).replace(".git", "")
    repo_dir = os.path.join(tmp_root, repo_name)

    if not os.path.exists(os.path.join(repo_dir, ".git")):
        res = subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            skipped_clone_fail += 1
            continue

    parent_sha = get_parent_commit(repo_dir, sha)
    if parent_sha is None:
        skipped_no_parent += 1
        continue

    diff = subprocess.run(
        ["git", "diff", "--name-only", parent_sha, sha],
        cwd=repo_dir, capture_output=True, text=True
    ).stdout.splitlines()

    py_files = [f for f in diff if f.endswith(".py")]
    if not py_files:
        continue

    mi_b, mi_a = [], []
    cc_b, cc_a = [], []
    raw_b, raw_a = [], []
    hal_b, hal_a = [], []

    for f in py_files:
        before = subprocess.run(
            ["git", "show", f"{parent_sha}:{f}"],
            cwd=repo_dir, capture_output=True, text=True
        ).stdout

        after = subprocess.run(
            ["git", "show", f"{sha}:{f}"],
            cwd=repo_dir, capture_output=True, text=True
        ).stdout

        r1 = analyze_code(before)
        r2 = analyze_code(after)
        if r1[0] is None or r2[0] is None:
            continue

        mi_before, cc_before, raw_before, hal_before = r1
        mi_after, cc_after, raw_after, hal_after = r2

        file_data = {
            "repo_url": repo_url,
            "sha": sha,
            "file": f,
            "mi_before": mi_before,
            "mi_after": mi_after,
            "mi_diff": mi_after - mi_before,
            "cc_before": cc_before,
            "cc_after": cc_after,
            "cc_diff": cc_after - cc_before,
        }

        for k in raw_before:
            file_data[f"{k}_before"] = raw_before[k]
            file_data[f"{k}_after"] = raw_after[k]

        for k in hal_before:
            file_data[f"{k}_before"] = hal_before[k]
            file_data[f"{k}_after"] = hal_after[k]

        file_results.append(file_data)

        mi_b.append(mi_before); mi_a.append(mi_after)
        cc_b.append(cc_before); cc_a.append(cc_after)
        raw_b.append(raw_before); raw_a.append(raw_after)
        hal_b.append(hal_before); hal_a.append(hal_after)

    if mi_b:
        summary = {
            "repo_url": repo_url,
            "sha": sha,
            "mi_before_avg": np.mean(mi_b),
            "mi_after_avg": np.mean(mi_a),
            "cc_before_avg": np.mean(cc_b),
            "cc_after_avg": np.mean(cc_a),
        }

        for k in raw_b[0]:
            summary[f"{k}_before_avg"] = np.mean([r[k] for r in raw_b])
            summary[f"{k}_after_avg"] = np.mean([r[k] for r in raw_a])

        for k in hal_b[0]:
            summary[f"{k}_before_avg"] = np.nanmean([h[k] for h in hal_b])
            summary[f"{k}_after_avg"] = np.nanmean([h[k] for h in hal_a])

        summary_results.append(summary)

# =================================================
# Save
# =================================================
OUT = BASE_DIR / "data" / "processed" / "commit_summary_readability_allmetrics.csv"
pd.DataFrame(summary_results).to_csv(OUT, index=False)

print(f"✅ Save completed: {OUT}")
print(f"⚠️ Excluded commits without a parent: {skipped_no_parent}")
print(f"⚠️ clone failure exclusion: {skipped_clone_fail}")

shutil.rmtree(tmp_root, ignore_errors=True)
print("🧹 Temporary directory deleted")
