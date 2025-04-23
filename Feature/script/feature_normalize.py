import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

NORMALIZE_METHOD = "minmax"  # "zscore" or "minmax"
MINMAX_RANGE = (0, 1)        # Only when NORMALIZE_METHOD is "minmax"
INPUT_PATH = "Feature/output/features_train_Az_median.csv"

df = pd.read_csv(INPUT_PATH)

exclude_cols = ["unique_id", "mode", "gender", "hold racket handed", "play years", "level"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

if NORMALIZE_METHOD == "zscore":
    scaler = StandardScaler()
    suffix = "zscore"
elif NORMALIZE_METHOD == "minmax":
    scaler = MinMaxScaler(feature_range=MINMAX_RANGE)
    suffix = f"minmax_{MINMAX_RANGE[0]}_{MINMAX_RANGE[1]}"

df_norm = df.copy()
df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
base, ext = os.path.splitext(INPUT_PATH)
OUTPUT_PATH = f"{base}_{suffix}{ext}"

print(f"Saving normalized features to {OUTPUT_PATH}")
df_norm.to_csv(OUTPUT_PATH, index=False)
