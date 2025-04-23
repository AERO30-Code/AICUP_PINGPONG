import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set file paths
info_path = "original/train/train_info.csv"
fig_dir = "EDA/figures"
os.makedirs(fig_dir, exist_ok=True)

# Read train_info.csv
df = pd.read_csv(info_path)

# Robust parse cut_point string to list of int
def parse_cutpoint(cp_str):
    cp_str = str(cp_str).replace('[', '').replace(']', '').replace('\n', ' ')
    tokens = [s for s in cp_str.strip().split(' ') if s != '']
    return list(map(int, tokens))

# Prepare all segment lengths and related labels
segment_info = []
for idx, row in df.iterrows():
    try:
        cut_points = parse_cutpoint(row['cut_point'])
        lengths = np.diff(cut_points)
        for seg_len in lengths:
            segment_info.append({
                "mode": row["mode"],
                "gender": row["gender"],
                "hold racket handed": row["hold racket handed"],
                "play years": row["play years"],
                "level": row["level"],
                "segment_length": seg_len
            })
    except Exception as e:
        print(f"Error parsing cut_point for unique_id {row['unique_id']}: {e}")

segment_df = pd.DataFrame(segment_info)

# Define all target labels
target_labels = [
    ("mode", "Mode"),
    ("gender", "Gender"),
    ("hold racket handed", "Hold Racket Handed"),
    ("play years", "Play Years"),
    ("level", "Level"),
]

# For each label, plot all categories' segment length distributions
for col, col_title in target_labels:
    categories = sorted(segment_df[col].unique())
    n_cat = len(categories)
    ncols = 3
    nrows = int(np.ceil(n_cat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4))
    axes = axes.flatten()
    for i, cat in enumerate(categories):
        ax = axes[i]
        segs = segment_df[segment_df[col] == cat]["segment_length"]
        ax.hist(segs, bins=20, color="#6699cc", edgecolor="black")
        ax.set_title(f"{col_title}: {cat}")
        ax.set_xlabel("Segment Length")
        ax.set_ylabel("Count")
    # Hide unused subplots
    for j in range(i+1, nrows*ncols):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"cutpoint_by_{col.replace(' ', '_')}.png"))
    plt.close()
    print(f"Saved cutpoint distribution by {col}.")

print("All cutpoint distribution plots by label generated.")