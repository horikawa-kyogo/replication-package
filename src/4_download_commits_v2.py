import os
import subprocess
import pandas as pd
from radon.metrics import mi_visit, h_visit
from radon.complexity import cc_visit
from radon.raw import analyze as raw_analyze
import tempfile
import shutil
from pathlib import Path

# --- GitHub PAT（Private対応） ---
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")

# --- データ読み込み ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

commits_df = pd.read_parquet(DATA_DIR / "pr_commits.parquet")
prs_df = pd.read_parquet(DATA_DIR / "all_pull_request.parquet")
details_df = pd.read_parquet(DATA_DIR / "pr_commit_details.parquet")

# --- Pythonファイル変更コミット ---
py_commits = set(details_df[details_df["filename"].str.endswith(".py", na=False)]["sha"].unique())

# --- キーワード ---
keywords = [
    "readability", "readable", "understandability", "understandable",
    "clarity", "legibility", "easier to read", "comprehensible"
]

# --- 対象コミット抽出 ---
filtered = commits_df[
    commits_df["sha"].isin(py_commits) &
    commits_df["message"].fillna("").str.lower().apply(lambda m: any(kw in m for kw in keywords))
]

if filtered.empty:
    print("対象コミットなし")
    exit(0)

# --- 一時ディレクトリ ---
tmp_root = tempfile.mkdtemp(prefix="repo_radon_")
print(f"🧰 一時ディレクトリ: {tmp_root}")

# --- Radon解析関数 ---
def analyze_code(code):
    try:
        # --- MI ---
        mi = mi_visit(code, True)

        # --- CC ---
        cc_list = cc_visit(code)
        cc_total = sum(c.complexity for c in cc_list)

        # --- Raw Metrics ---
        raw = raw_analyze(code)
        raw_metrics = {
            "loc": raw.loc,
            "lloc": raw.lloc,
            "sloc": raw.sloc,
            "comments": raw.comments,
            "multi": raw.multi,
            "blank": raw.blank,
        }

        # --- Halstead Metrics ---
        h = h_visit(code)
        if h.total.length > 0:
            halstead_metrics = {
                "h_volume": h.total.volume,
                "h_difficulty": h.total.difficulty,
                "h_effort": h.total.effort,
            }
        else:
            halstead_metrics = {
                "h_volume": 0,
                "h_difficulty": 0,
                "h_effort": 0,
            }

        return mi, cc_total, raw_metrics, halstead_metrics

    except Exception as e:
        print(f"⚠️ Radon解析失敗: {e}")
        return None, None, None, None

# --- 結果リスト ---
file_results = []
summary_results = []

for idx, commit_row in filtered.iterrows():
    sha = commit_row["sha"]
    pr_id = commit_row["pr_id"]

    repo_row = prs_df[prs_df["id"] == pr_id]
    if repo_row.empty:
        continue

    repo_url = repo_row.iloc[0]["repo_url"]

    # --- API URL → HTTPS URL ---
    if "api.github.com/repos" in repo_url:
        repo_url = repo_url.replace("https://api.github.com/repos/", "https://github.com/") + ".git"

    # --- PAT埋め込み ---
    if GITHUB_PAT and repo_url.startswith("https://github.com/"):
        parts = repo_url.split("https://github.com/")
        repo_url = f"https://{GITHUB_PAT}@github.com/{parts[1]}"

    repo_name = os.path.basename(repo_url).replace(".git", "")
    repo_dir = os.path.join(tmp_root, repo_name)

    # --- Git clone ---
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        res = subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        if res.returncode != 0:
            print(f"❌ Clone失敗: {repo_url}")
            continue

    # --- 親コミット取得 ---
    parent_res = subprocess.run(
        ["git", "rev-parse", f"{sha}^"],
        cwd=repo_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    parent_sha = parent_res.stdout.strip() if parent_res.returncode == 0 else None
    if not parent_sha:
        continue

    # --- Pythonファイル差分 ---
    diff_res = subprocess.run(
        ["git", "diff", "--name-only", parent_sha, sha],
        cwd=repo_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    py_files = [f for f in diff_res.stdout.splitlines() if f.endswith(".py")]
    if not py_files:
        continue

    # --- 集計用 ---
    mi_b, mi_a = [], []
    cc_b, cc_a = [], []
    loc_b, loc_a = [], []
    lloc_b, lloc_a = [], []
    sloc_b, sloc_a = [], []
    comm_b, comm_a = [], []
    multi_b, multi_a = [], []
    blank_b, blank_a = [], []
    hv_b, hv_a = [], []
    hd_b, hd_a = [], []
    he_b, he_a = [], []

    for f in py_files:
        # --- before ---
        before_code = subprocess.run(
            ["git", "show", f"{parent_sha}:{f}"],
            cwd=repo_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        ).stdout

        mi_before, cc_before, raw_before, hal_before = analyze_code(before_code)

        # --- after ---
        after_code = subprocess.run(
            ["git", "show", f"{sha}:{f}"],
            cwd=repo_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        ).stdout

        mi_after, cc_after, raw_after, hal_after = analyze_code(after_code)

        if None in (mi_before, mi_after):
            continue

        # --- file単位結果 ---
        file_results.append({
            "repo_url": repo_url,
            "sha": sha,
            "file": f,

            "mi_before": mi_before,
            "mi_after": mi_after,
            "mi_diff": mi_after - mi_before,

            "cc_before": cc_before,
            "cc_after": cc_after,
            "cc_diff": cc_after - cc_before,

            "loc_before": raw_before["loc"],
            "loc_after": raw_after["loc"],

            "lloc_before": raw_before["lloc"],
            "lloc_after": raw_after["lloc"],

            "sloc_before": raw_before["sloc"],
            "sloc_after": raw_after["sloc"],

            "comments_before": raw_before["comments"],
            "comments_after": raw_after["comments"],

            "multi_before": raw_before["multi"],
            "multi_after": raw_after["multi"],

            "blank_before": raw_before["blank"],
            "blank_after": raw_after["blank"],

            "h_volume_before": hal_before["h_volume"],
            "h_volume_after": hal_after["h_volume"],

            "h_difficulty_before": hal_before["h_difficulty"],
            "h_difficulty_after": hal_after["h_difficulty"],

            "h_effort_before": hal_before["h_effort"],
            "h_effort_after": hal_after["h_effort"],
        })

        # --- summary用 ---
        mi_b.append(mi_before); mi_a.append(mi_after)
        cc_b.append(cc_before); cc_a.append(cc_after)
        loc_b.append(raw_before["loc"]); loc_a.append(raw_after["loc"])
        lloc_b.append(raw_before["lloc"]); lloc_a.append(raw_after["lloc"])
        sloc_b.append(raw_before["sloc"]); sloc_a.append(raw_after["sloc"])
        comm_b.append(raw_before["comments"]); comm_a.append(raw_after["comments"])
        multi_b.append(raw_before["multi"]); multi_a.append(raw_after["multi"])
        blank_b.append(raw_before["blank"]); blank_a.append(raw_after["blank"])
        hv_b.append(hal_before["h_volume"]); hv_a.append(hal_after["h_volume"])
        hd_b.append(hal_before["h_difficulty"]); hd_a.append(hal_after["h_difficulty"])
        he_b.append(hal_before["h_effort"]); he_a.append(hal_after["h_effort"])

    if mi_b:
        summary_results.append({
            "repo_url": repo_url,
            "sha": sha,

            "mi_before_avg": sum(mi_b)/len(mi_b),
            "mi_after_avg": sum(mi_a)/len(mi_a),

            "cc_before_avg": sum(cc_b)/len(cc_b),
            "cc_after_avg": sum(cc_a)/len(cc_a),

            "loc_before_avg": sum(loc_b)/len(loc_b),
            "loc_after_avg": sum(loc_a)/len(loc_a),

            "lloc_before_avg": sum(lloc_b)/len(lloc_b),
            "lloc_after_avg": sum(lloc_a)/len(lloc_a),

            "sloc_before_avg": sum(sloc_b)/len(sloc_b),
            "sloc_after_avg": sum(sloc_a)/len(sloc_a),

            "comments_before_avg": sum(comm_b)/len(comm_b),
            "comments_after_avg": sum(comm_a)/len(comm_a),

            "multi_before_avg": sum(multi_b)/len(multi_b),
            "multi_after_avg": sum(multi_a)/len(multi_a),

            "blank_before_avg": sum(blank_b)/len(blank_b),
            "blank_after_avg": sum(blank_a)/len(blank_a),

            "h_volume_before_avg": sum(hv_b)/len(hv_b),
            "h_volume_after_avg": sum(hv_a)/len(hv_a),

            "h_difficulty_before_avg": sum(hd_b)/len(hd_b),
            "h_difficulty_after_avg": sum(hd_a)/len(hd_a),

            "h_effort_before_avg": sum(he_b)/len(he_b),
            "h_effort_after_avg": sum(he_a)/len(he_a),
        })

# --- CSV 保存 ---
summary_csv = "../data/processed/commit_summary_readability_allmetrics.csv"
pd.DataFrame(summary_results).to_csv(summary_csv, index=False)
print(f"✅ 集計結果保存: {summary_csv}（{len(summary_results)}件）")

# --- 作業ディレクトリ削除 ---
shutil.rmtree(tmp_root, ignore_errors=True)
print("🧹 一時ディレクトリ削除完了")
