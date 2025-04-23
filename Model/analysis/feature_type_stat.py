import pandas as pd
import os
import re

rf_path = 'Model/output/20250421_224234/RandomForest_gender_feature_importance.csv'
xgb_path = 'Model/output/20250421_221845/xgboost_gender_feature_importance.csv'
TOP_N = 50

# 統計類型設定
stat_types = ['median', 'mean', 'std', 'max', 'min']
axis_types = ['Az', 'Ax', 'Ay', 'Gz', 'Gx', 'Gy']
segment_pattern = re.compile(r'segment(\d+)_')

def extract_stats(feature_names):
    stat_count = {s: 0 for s in stat_types}
    axis_count = {a: 0 for a in axis_types}
    segment_count = {}
    for feat in feature_names:
        # 統計量
        for s in stat_types:
            if f'_{s}' in feat:
                stat_count[s] += 1
        # 軸
        for a in axis_types:
            if f'_{a}_' in feat or feat.endswith(f'_{a}'):
                axis_count[a] += 1
        # segment
        m = segment_pattern.match(feat)
        if m:
            seg = m.group(1)
            segment_count[seg] = segment_count.get(seg, 0) + 1
    # 移除數量為0的項目
    stat_count = {k: v for k, v in stat_count.items() if v > 0}
    axis_count = {k: v for k, v in axis_count.items() if v > 0}
    segment_count = {k: v for k, v in segment_count.items() if v > 0}
    return stat_count, axis_count, segment_count

def print_stats(title, stat_count, axis_count, segment_count, file=None):
    lines = []
    lines.append(f'==== {title} ====')
    lines.append('[Stat Type Count]')
    for k, v in stat_count.items():
        lines.append(f'{k}: {v}')
    lines.append('[Axis Type Count]')
    for k, v in axis_count.items():
        lines.append(f'{k}: {v}')
    lines.append('[Segment Count]')
    for k, v in sorted(segment_count.items(), key=lambda x: int(x[0])):
        lines.append(f'segment{k}: {v}')
    lines.append('')
    # 螢幕輸出
    for line in lines:
        print(line)
    # txt輸出
    if file:
        for line in lines:
            file.write(line + '\n')

# 讀取CSV
rf = pd.read_csv(rf_path, header=None, names=['feature', 'importance'])
xgb = pd.read_csv(xgb_path, header=None, names=['feature', 'importance'])
rf = rf.iloc[1:]
xgb = xgb.iloc[1:]

rf_top = rf.sort_values('importance', ascending=False).head(TOP_N)['feature']
xgb_top = xgb.sort_values('importance', ascending=False).head(TOP_N)['feature']

rf_stat, rf_axis, rf_seg = extract_stats(rf_top)
xgb_stat, xgb_axis, xgb_seg = extract_stats(xgb_top)

output_path = 'Model/analysis/analysis_output/feature_type_stat.txt'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    print_stats('RandomForest Top 50', rf_stat, rf_axis, rf_seg, file=f)
    print_stats('XGBoost Top 50', xgb_stat, xgb_axis, xgb_seg, file=f)
