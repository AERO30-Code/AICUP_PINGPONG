import pandas as pd

N_TOP_FEATURES = 100
FEATURE_IMPORTANCE_CSV = 'Model/analysis/analysis_output/train_noplayerid_swing5to27_meanrank_4models_top800.csv'
FEATURES_INPUT_CSV = 'Feature/katsu/train_noplayerid_swing5to27.csv'
OUTPUT_CSV = f'Feature/output/features_gender_top_Test/train_noplayerid_swing5to27_4models_top{N_TOP_FEATURES}.csv'

META_COLS = [
    'unique_id', 'mode', 'gender', 'hold racket handed', 'play years', 'level'
]
# META_COLS = [
#     'unique_id', 'mode'
# ]

importance_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
selected_features = importance_df.sort_values('mean_rank')['feature'].head(N_TOP_FEATURES).tolist()

features_df = pd.read_csv(FEATURES_INPUT_CSV)
cols_to_keep = META_COLS + [col for col in selected_features if col in features_df.columns]
filtered_df = features_df[cols_to_keep]

filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"save to {OUTPUT_CSV}")
