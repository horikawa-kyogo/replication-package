import pandas as pd

# --- Input file ---
input_csv = "../data/processed/filtered_commits.csv"

# --- Specify the number of items  ---
N = 231

# --- CSV loading ---
df = pd.read_csv(input_csv)

# --- Randomly select N items ---
subset_df = df.sample(n=N)  # Maintaining reproducibility with random_state

# --- Embed the number in the output file name ---
output_csv = f"../data/results/filtered_commits_{N}.csv"

# --- Save ---
subset_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ CSV Save completed: {output_csv} ({len(subset_df)}commits)")
