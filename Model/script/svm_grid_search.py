import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, make_scorer

CONFIG_PATH = "Model/config/model_config.yaml"
FEATURE_PATH = "Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top100.csv"

# label 對應 features_train.csv 欄位名
target = "gender"
target_col = "gender"

# 讀取 config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# 讀取 features
print("讀取特徵...")
df = pd.read_csv(FEATURE_PATH)

# 只排除 unique_id 與所有標籤（除了自己的 y_col）
excluded_cols = ["unique_id", "hold racket handed", "play years", "level"]
feature_cols = [col for col in df.columns if col not in excluded_cols]
X = df[feature_cols]
y = df[target_col]

# gender: 1=male, 2=female → 0/1
if y.min() == 1 and y.max() == 2:
    y = y - 1

# 讀取 train_info.csv 並 merge 取得 player_id 與 stratify 欄位
info_df = pd.read_csv('original/train/train_info.csv')
stratify_col = target_col
player_df = info_df[['player_id', stratify_col]].drop_duplicates()

# 以 player_id 為單位做 stratified split
from sklearn.model_selection import train_test_split as sk_train_test_split
player_train, player_val = sk_train_test_split(
    player_df, test_size=0.2, random_state=42, stratify=player_df[stratify_col]
)
train_player_ids = set(player_train['player_id'])
val_player_ids = set(player_val['player_id'])
df_merged = df.merge(info_df[['unique_id', 'player_id']], on='unique_id', how='left')
train_idx = df_merged[df_merged['player_id'].isin(train_player_ids)].index
val_idx = df_merged[df_merged['player_id'].isin(val_player_ids)].index
X_train = X.loc[train_idx].reset_index(drop=True)
X_val = X.loc[val_idx].reset_index(drop=True)
y_train = y.loc[train_idx].reset_index(drop=True)
y_val = y.loc[val_idx].reset_index(drop=True)

# 特徵 scaling（與 config 一致）
scaling_mode = config.get('feature_scaling', 'none').lower()
if scaling_mode == "zscore":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
elif scaling_mode == "minmax":
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
# 若 scaling_mode == "none"，則不做 scaling

# 使用 RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# 第一階段 coarse grid
param_grid1 = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto'],
    'kernel': ['rbf'],
    'probability': [True],
    'random_state': [42]
}
print("\n===== 第一階段 coarse GridSearchCV (cross-validation 平均分數) =====")
svc = SVC()
grid1 = GridSearchCV(
    svc,
    param_grid1,
    scoring=make_scorer(roc_auc_score, needs_proba=True),
    cv=cv,
    verbose=2,
    n_jobs=-1
)
grid1.fit(X_train, y_train)
print(f"第一階段最佳參數: {grid1.best_params_}")
print(f"第一階段 cross-validation 平均 AUC: {grid1.best_score_:.4f}")

# 根據 coarse 結果自動細化範圍
best_C = grid1.best_params_['C']
best_gamma = grid1.best_params_['gamma']
# 細緻範圍設計：C 在 best_C 附近，gamma 在 best_gamma 附近
import numbers
def refine_range(val, scale=2, steps=5, log=True):
    if isinstance(val, str):
        return [val]
    if log:
        base = np.log10(val) if val > 0 else 0
        rng = np.logspace(base-scale, base+scale, steps)
        return sorted(set([round(float(x), 6) for x in rng]))
    else:
        rng = np.linspace(max(0.0001, val-scale), val+scale, steps)
        return sorted(set([round(float(x), 6) for x in rng]))
C_range2 = refine_range(best_C, scale=0.5, steps=5, log=True)
gamma_range2 = refine_range(best_gamma, scale=0.5, steps=5, log=True)
param_grid2 = {
    'C': C_range2,
    'gamma': gamma_range2,
    'kernel': ['rbf'],
    'probability': [True],
    'random_state': [42]
}
print("\n===== 第二階段 fine GridSearchCV (cross-validation 平均分數) =====")
grid2 = GridSearchCV(
    svc,
    param_grid2,
    scoring=make_scorer(roc_auc_score, needs_proba=True),
    cv=cv,
    verbose=2,
    n_jobs=-1
)
grid2.fit(X_train, y_train)
print(f"第二階段最佳參數: {grid2.best_params_}")
print(f"第二階段 cross-validation 平均 AUC: {grid2.best_score_:.4f}")

# 在驗證集上評估
best_model = grid2.best_estimator_
y_val_proba = best_model.predict_proba(X_val)
auc = roc_auc_score(y_val, y_val_proba[:, 1])
print(f"驗證集 AUC: {auc:.4f}")

print("\n請將下列參數複製到 model_config.yaml 的 gender 區塊：")
print("params:")
for k, v in grid2.best_params_.items():
    print(f"  {k}: {v}") 