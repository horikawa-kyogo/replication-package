import os
import subprocess
import pandas as pd
import tempfile
import shutil
import time
from pathlib import Path
import ast

from radon.metrics import mi_visit, h_visit
from radon.complexity import cc_visit
from radon.raw import analyze as raw_analyze

# =================================================
# setting
# =================================================
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")
SLEEP_PER_COMMIT = 2.0
BATCH_SIZE = 30            # None means no limit (process all)
REPO_CACHE_DIR = Path("../repo_cache").resolve()

REPO_CACHE_DIR.mkdir(exist_ok=True)

# =================================================
# Data loading
# =================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_CSV = Path("../data/processed/commit_summary_readability_allmetrics.csv")

commits_df = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
prs_df = pd.read_parquet(DATA_DIR / "all_pull_request.parquet")
details_df = pd.read_parquet(DATA_DIR / "pr_commit_details.parquet")

# =================================================
# Processed SHA (for resumption)
# =================================================
done_shas = set()
if OUT_CSV.exists():
    done_shas = set(pd.read_csv(OUT_CSV)["sha"].astype(str))
    print(f"🔁 Resume mode: {len(done_shas)} commits already done")

# =================================================
# Extract target commit
# =================================================
py_commits = set(
    details_df[details_df["filename"].str.endswith(".py", na=False)]["sha"].astype(str)
)

keywords = [
    "readability", "readable", "understandability", "understandable",
    "clarity", "legibility", "easier to read", "comprehensible"
]

filtered = commits_df[
    commits_df["sha"].astype(str).isin(py_commits) &
    commits_df["message"].fillna("").str.lower().apply(
        lambda m: any(k in m for k in keywords)
    )
]

filtered = filtered[~filtered["sha"].astype(str).isin(done_shas)]

if BATCH_SIZE:
    filtered = filtered.head(BATCH_SIZE)

if filtered.empty:
    print("✅ All commits processed")
    exit(0)

# =================================================
# AST auxiliary
# =================================================
def count_definitions(code: str):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        n_functions = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
        n_classes   = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        n_methods   = sum(
            isinstance(n, ast.FunctionDef) and isinstance(getattr(n, "parent", None), ast.ClassDef)
            for n in ast.walk(tree)
        )
        n_docstrings = sum(
            ast.get_docstring(n) is not None
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
        )
        return n_functions, n_classes, n_methods, n_docstrings
    except Exception:
        return 0, 0, 0, 0

# =================================================
# Radon
# =================================================
def analyze_code(code: str):
    try:
        mi = mi_visit(code, True)
        cc = sum(c.complexity for c in cc_visit(code))

        raw = raw_analyze(code)
        nf, nc, nm, nd = count_definitions(code)

        raw_m = {
            "loc": raw.loc, "lloc": raw.lloc, "sloc": raw.sloc,
            "comments": raw.comments, "multi": raw.multi, "blank": raw.blank,
            "n_functions": nf, "n_classes": nc, "n_methods": nm, "n_docstrings": nd
        }

        h = h_visit(code).total
        hal_m = {
            "h1": h.h1, "h2": h.h2, "N1": h.N1, "N2": h.N2,
            "h_volume": h.volume, "h_difficulty": h.difficulty,
            "h_effort": h.effort, "t": h.time, "b": h.bugs
        }

        return mi, cc, raw_m, hal_m
    except Exception:
        return None, None, None, None

# =================================================
# main loop
# =================================================
results = []

for _, row in filtered.iterrows():
    sha = str(row["sha"])
    pr_id = row["pr_id"]

    repo_row = prs_df[prs_df["id"] == pr_id]
    if repo_row.empty:
        continue

    repo_url = repo_row.iloc[0]["repo_url"]
    if "api.github.com/repos" in repo_url:
        repo_url = repo_url.replace(
            "https://api.github.com/repos/", "https://github.com/"
        ) + ".git"

    auth_url = repo_url.replace(
        "https://github.com/", f"https://{GITHUB_PAT}@github.com/"
    )

    repo_dir = REPO_CACHE_DIR / Path(repo_url).stem

    if not repo_dir.exists():
        print(f"⬇ clone {repo_dir.name}")
        if subprocess.run(
            ["git", "clone", "--quiet", auth_url, str(repo_dir)]
        ).returncode != 0:
            continue

    parent = subprocess.run(
        ["git", "rev-parse", f"{sha}^"],
        cwd=repo_dir,
        capture_output=True, text=True
    )
    if parent.returncode != 0:
        continue

    parent_sha = parent.stdout.strip()

    diff = subprocess.run(
        ["git", "diff", "--name-only", parent_sha, sha],
        cwd=repo_dir,
        capture_output=True, text=True
    )

    py_files = [f for f in diff.stdout.splitlines() if f.endswith(".py")]
    if not py_files:
        continue

    mi_b = []; mi_a = []
    cc_b = []; cc_a = []
    raw_b = []; raw_a = []
    hal_b = []; hal_a = []

    for f in py_files:
        before = subprocess.run(
            ["git", "show", f"{parent_sha}:{f}"],
            cwd=repo_dir,
            capture_output=True, text=True
        )
        after = subprocess.run(
            ["git", "show", f"{sha}:{f}"],
            cwd=repo_dir,
            capture_output=True, text=True
        )

        if before.returncode != 0 or after.returncode != 0:
            continue

        mib, ccb, rb, hb = analyze_code(before.stdout)
        mia, cca, ra, ha = analyze_code(after.stdout)

        if mib is None:
            continue

        mi_b.append(mib); mi_a.append(mia)
        cc_b.append(ccb); cc_a.append(cca)
        raw_b.append(rb); raw_a.append(ra)
        hal_b.append(hb); hal_a.append(ha)

    if not mi_b:
        continue

    row_out = {
        "repo_url": repo_url,
        "sha": sha,
        "mi_before_avg": sum(mi_b)/len(mi_b),
        "mi_after_avg": sum(mi_a)/len(mi_a),
        "cc_before_avg": sum(cc_b)/len(cc_b),
        "cc_after_avg": sum(cc_a)/len(cc_a),
    }

    for k in raw_b[0]:
        row_out[f"{k}_before_avg"] = sum(r[k] for r in raw_b)/len(raw_b)
        row_out[f"{k}_after_avg"]  = sum(r[k] for r in raw_a)/len(raw_a)

    for k in hal_b[0]:
        row_out[f"{k}_before_avg"] = sum(h[k] for h in hal_b)/len(hal_b)
        row_out[f"{k}_after_avg"]  = sum(h[k] for h in hal_a)/len(hal_a)

    results.append(row_out)

    print(f"✅ done {sha}")
    time.sleep(SLEEP_PER_COMMIT)

# =================================================
# Save
# =================================================
df_out = pd.DataFrame(results)
if OUT_CSV.exists():
    df_out = pd.concat([pd.read_csv(OUT_CSV), df_out], ignore_index=True)

df_out.to_csv(OUT_CSV, index=False)
print(f"📄 saved: {OUT_CSV}")
