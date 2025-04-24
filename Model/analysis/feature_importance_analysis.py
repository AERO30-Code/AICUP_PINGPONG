import pandas as pd
import matplotlib.pyplot as plt
import os

rf_path = 'Model/output/20250421_224234/RandomForest_gender_feature_importance.csv'
xgb_path = 'Model/output/20250421_221845/xgboost_gender_feature_importance.csv'
output_dir = 'Model/analysis/analysis_output'
TOP_N = 800

# Create output_dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 讀取CSV
rf = pd.read_csv(rf_path, header=None, names=['feature', 'importance'])
xgb = pd.read_csv(xgb_path, header=None, names=['feature', 'importance'])

# 去除第一行（index欄）
rf = rf.iloc[1:]
xgb = xgb.iloc[1:]

# 依重要性排序
rf_sorted = rf.sort_values('importance', ascending=False).reset_index(drop=True)
xgb_sorted = xgb.sort_values('importance', ascending=False).reset_index(drop=True)

rf_top = rf_sorted.head(TOP_N)
xgb_top = xgb_sorted.head(TOP_N)

# 交集法
intersection = pd.merge(rf_top, xgb_top, on='feature')
intersection = intersection[['feature']]
intersection.to_csv(os.path.join(output_dir, f'feature_intersection_top{TOP_N}.csv'), index=False)

# 平均排名法
rf_sorted['rf_rank'] = rf_sorted.index + 1
xgb_sorted['xgb_rank'] = xgb_sorted.index + 1
rank_df = pd.merge(rf_sorted[['feature', 'rf_rank']], xgb_sorted[['feature', 'xgb_rank']], on='feature', how='outer')
print(f'[Info] Some features only appear in a single model. Assigning rank {TOP_N * 2}.')
rank_df = rank_df.fillna(TOP_N * 2)
rank_df['mean_rank'] = (rank_df['rf_rank'] + rank_df['xgb_rank']) / 2
rank_df = rank_df.sort_values('mean_rank')
mean_rank_top = rank_df.head(TOP_N)
mean_rank_top.to_csv(os.path.join(output_dir, f'feature_meanrank_top{TOP_N}.csv'), index=False)

# 可視化 (只保留平均排名前30)
mean_rank_plot = pd.merge(mean_rank_top, rf[['feature', 'importance']], on='feature', how='left')
mean_rank_plot = pd.merge(mean_rank_plot, xgb[['feature', 'importance']], on='feature', how='left', suffixes=('_rf', '_xgb'))
mean_rank_plot = mean_rank_plot.fillna(0)
fig_height = max(7, min(TOP_N * 0.35, 30))
plt.figure(figsize=(10, fig_height))
plt.barh(mean_rank_plot['feature'][::-1], mean_rank_plot['mean_rank'][::-1], color='green')
plt.title(f'Top {TOP_N} Features by Mean Rank (Lower = More Important)')
plt.xlabel('Mean Rank')
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f'feature_meanrank_top{TOP_N}.png'))
plt.close()

print('Analysis complete. Results saved to analysis directory.')
