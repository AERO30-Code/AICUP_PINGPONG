import os
import pandas as pd
import matplotlib.pyplot as plt

# Set file paths
info_path = "original/train/train_info.csv"
output_dir = "EDA/report"
fig_dir = "EDA/figures"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# Read train_info.csv
df = pd.read_csv(info_path)

# Check missing values
missing = df.isnull().sum()
with open(os.path.join(output_dir, "info_missing.txt"), "w") as f:
    f.write("Missing values per column:\n")
    f.write(str(missing))
print("Missing values check done.")

# Define valid ranges for each label
valid_ranges = {
    "gender": [1, 2],
    "hold racket handed": [1, 2],
    "play years": [0, 1, 2],
    "level": [2, 3, 4, 5]
}

# Check for invalid values in each label column
for col, valid in valid_ranges.items():
    if col in df.columns:
        invalid = df[~df[col].isin(valid)]
        if not invalid.empty:
            print(f"Warning: Invalid values found in column '{col}'.")
print("Invalid value check done.")

# Value counts and percentages for each label column
display_labels = [
    ("gender", [1, 2]),
    ("hold racket handed", [1, 2]),
    ("play years", [0, 1, 2]),
    ("level", [2, 3, 4, 5]),
    ("mode", sorted(df["mode"].unique()))
]
stats = []

# Prepare subplots: 2 rows x 3 cols, last subplot will be hidden
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (col, label_order) in enumerate(display_labels):
    if col in df.columns:
        counts = df[col].value_counts().reindex(label_order, fill_value=0)
        percents = df[col].value_counts(normalize=True).reindex(label_order, fill_value=0) * 100
        temp_df = pd.DataFrame({
            "count": counts,
            "percent": percents.round(2)
        })
        temp_df["label"] = col
        temp_df["value"] = temp_df.index
        stats.append(temp_df.reset_index(drop=True))
        # Draw bar chart in the corresponding subplot
        ax = axes[idx]
        bars = ax.bar([str(v) for v in label_order], counts.values, color="#6699cc")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        # Add percent labels on top of bars
        for bar, percent in zip(bars, percents.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{percent:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        ax.set_ylim(0, max(counts.values)*1.15)
        # Set x-axis order
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([str(v) for v in label_order])

# Hide the last unused subplot
if len(display_labels) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "info_label_distributions.png"))
plt.close()

# Concatenate and save stats
all_stats = pd.concat(stats, ignore_index=True)
all_stats = all_stats[["label", "value", "count", "percent"]]
all_stats.to_csv(os.path.join(output_dir, "info_stats.csv"), index=False)
print("Label distribution stats saved and figure generated.")