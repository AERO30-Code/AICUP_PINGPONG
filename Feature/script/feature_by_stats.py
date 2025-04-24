import pandas as pd
import os

SUMMARY_COLS = [
    "unique_id",
    "mode",
    "gender",
    "hold racket handed",
    "play years",
    "level",
]

SUFFIX_LIST = ["median"]
SENSORS = ["Az"]

INPUT_PATH = "Feature/output/features_test.csv"
OUTPUT_PATH = "Feature/output/features_test_Az_median.csv"

def main():
    df = pd.read_csv(INPUT_PATH)
    
    keep_cols = [col for col in SUMMARY_COLS if col in df.columns]

    for sensor in SENSORS:
        for suffix in SUFFIX_LIST:
            matched_cols = [col for col in df.columns if (sensor in col and col.endswith(suffix))]
            keep_cols.extend(matched_cols)

    keep_cols = list(dict.fromkeys(keep_cols))
    df_out = df[keep_cols]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df_out.columns)} columns to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
