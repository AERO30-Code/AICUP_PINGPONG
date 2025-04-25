import pandas as pd
import os
import matplotlib.pyplot as plt

# ===== User-customizable parameters =====
model_csv_paths = [
    'Model/output/20250425_153905/xgboost_gender_feature_importance.csv',
    'Model/output/20250425_153922/RandomForest_gender_feature_importance.csv',
    'Model/output/20250425_153939/lightgbm_gender_feature_importance.csv',
    'Model/output/20250425_154001/catboost_gender_feature_importance.csv',
]
output_dir = 'Model/analysis/analysis_output'
TOP_N = 800

# Create output_dir if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read feature importance from all models
model_dfs = []
for path in model_csv_paths:
    df = pd.read_csv(path, header=None, names=['feature', 'importance'])
    df = df.iloc[1:]  # Remove the first row (index column)
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    model_dfs.append(df)

# Collect all features from all models
all_features = set()
for df in model_dfs:
    all_features.update(df['feature'])
all_features = list(all_features)

# Calculate ranking for each model
rank_dfs = []
for i, df in enumerate(model_dfs):
    rank_df = pd.DataFrame({'feature': all_features})
    feature2rank = {f: rank+1 for rank, f in enumerate(df['feature'])}
    max_rank = len(df) + 1
    rank_df[f'rank_{i+1}'] = rank_df['feature'].map(lambda x: feature2rank.get(x, max_rank))
    rank_dfs.append(rank_df)

# Merge all rankings
merged_rank = rank_dfs[0]
for i in range(1, len(rank_dfs)):
    merged_rank = pd.merge(merged_rank, rank_dfs[i], on='feature')

# Calculate mean rank
rank_cols = [col for col in merged_rank.columns if col.startswith('rank_')]
merged_rank['mean_rank'] = merged_rank[rank_cols].astype(float).mean(axis=1)
merged_rank = merged_rank.sort_values('mean_rank')

# Output top TOP_N features
mean_rank_top = merged_rank.head(TOP_N)
mean_rank_top.to_csv(os.path.join(output_dir, f'train_noplayerid_swing5to27_meanrank_4models_top{TOP_N}.csv'), index=False)

# ===== Plotting: Line plot for all 800 features =====
plot_csv_path = os.path.join(output_dir, f'train_noplayerid_swing5to27_meanrank_4models_top{TOP_N}.csv')
plot_df = pd.read_csv(plot_csv_path)

plt.figure(figsize=(32, 8))
plt.plot(plot_df['feature'], plot_df['rank_1'], label='rank_1', marker='', linewidth=1)
plt.plot(plot_df['feature'], plot_df['rank_2'], label='rank_2', marker='', linewidth=1)
plt.plot(plot_df['feature'], plot_df['rank_3'], label='rank_3', marker='', linewidth=1)
plt.plot(plot_df['feature'], plot_df['rank_4'], label='rank_4', marker='', linewidth=1)
plt.plot(plot_df['feature'], plot_df['mean_rank'], label='mean_rank', marker='', linewidth=2, color='black')

plt.title('Feature Ranks Across 4 Models (Top 800 Features)')
plt.xlabel('Feature')
plt.ylabel('Rank')
plt.legend()

# Only show every 40th x-tick label to avoid clutter
step = 40
plt.xticks(ticks=range(0, len(plot_df['feature']), step), labels=plot_df['feature'][::step], rotation=60, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'train_noplayerid_swing5to27_meanrank_4models_top{TOP_N}_lineplot.png'))
plt.close()
