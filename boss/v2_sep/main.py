import os
import sys
import math
import glob
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# 為 XGBoost 模型，如有需要請先安裝 xgboost
model_name = "XGBoost"  # 或 "RandomForest"
group_size = 27  # 每組資料筆數


#####################
# 模型選擇與聚合函式
#####################

def model_selector(model_name):
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb = XGBClassifier(random_state=42)
        grid = GridSearchCV(xgb, param_grid, scoring='roc_auc_ovr', cv=3)
        return grid
    else:
        print("no model name")
        assert 0


def _FOR_NOT_AGGR_USED_aggregate_proba_eval_binary(clf, X_eval, y_eval, group_size):
    predicted = clf.predict_proba(X_eval)
    num_groups = len(predicted) // group_size
    group_probs = []
    y_eval_agg = []
    for i in range(num_groups):
        group_slice = predicted[i * group_size:(i + 1) * group_size]
        avg_proba = group_slice.mean(axis=0)
        group_probs.append(avg_proba)
        y_eval_agg.append(y_eval[i * group_size])
    group_probs = np.array(group_probs)
    y_eval_agg = np.array(y_eval_agg)
    auc_score = roc_auc_score(y_eval_agg, group_probs[:, 1])
    return group_probs, y_eval_agg, auc_score

def aggregate_proba_eval_binary(clf, X_eval, y_eval, group_size=1):
    # 直接預測 aggregated sample
    predicted = clf.predict_proba(X_eval)
    # predicted 形狀為 (n_samples, 2)
    try:
        auc_score = roc_auc_score(y_eval, predicted[:, 1])
    except Exception as e:
        print("ROC AUC 計算失敗:", e)
        auc_score = 0
    return predicted, y_eval, auc_score


def _FOR_NOT_AGGR_USED_aggregate_proba_eval_multi(clf, X_eval, y_eval, group_size):
    predicted = clf.predict_proba(X_eval)
    num_groups = len(predicted) // group_size
    group_probs = []
    y_eval_agg = []
    for i in range(num_groups):
        group_slice = predicted[i * group_size:(i + 1) * group_size]
        avg_proba = group_slice.mean(axis=0)
        group_probs.append(avg_proba)
        y_eval_agg.append(y_eval[i * group_size])
    group_probs = np.array(group_probs)
    y_eval_agg = np.array(y_eval_agg)
    # 這裡僅印出 AUC，如不支援多類 AUC計算可自行處理
    try:
        auc_score = roc_auc_score(y_eval_agg, group_probs, multi_class='ovr')
    except:
        auc_score = 0
    return group_probs, y_eval_agg, auc_score

def aggregate_proba_eval_multi(clf, X_eval, y_eval, group_size=1):
    # 直接預測，因為每個 sample 就是一個受測者的 aggregated feature
    predicted = clf.predict_proba(X_eval)
    # 在這裡 y_eval 也應該是一維 array，每一筆對應一個受測者
    try:
        auc_score = roc_auc_score(y_eval, predicted, average="micro", multi_class='ovr')
    except Exception as e:
        print("ROC AUC 計算失敗:", e)
        auc_score = 0
    return predicted, y_eval, auc_score


def aggregate_proba_test_binary(clf, X_test, group_size):
    predicted = clf.predict_proba(X_test)
    num_groups = len(predicted) // group_size
    group_probs = []
    for i in range(num_groups):
        group_slice = predicted[i * group_size:(i + 1) * group_size]
        avg_proba = group_slice.mean(axis=0)
        group_probs.append(avg_proba)
    return np.array(group_probs)


def aggregate_proba_test_multi(clf, X_test, group_size):
    predicted = clf.predict_proba(X_test)
    num_groups = len(predicted) // group_size
    group_probs = []
    for i in range(num_groups):
        group_slice = predicted[i * group_size:(i + 1) * group_size]
        avg_proba = group_slice.mean(axis=0)
        group_probs.append(avg_proba)
    return np.array(group_probs)


def model_binary(X_train, y_train, X_eval, y_eval, X_test, model_name, type_label):
    clf = model_selector(model_name)
    clf.fit(X_train, y_train)
    group_probs_eval, y_eval_agg, auc_score_eval = aggregate_proba_eval_binary(clf, X_eval, y_eval, group_size)
    print(f"{type_label} AUC (eval) = {auc_score_eval:.4f}")
    group_probs_test = aggregate_proba_test_binary(clf, X_test, group_size) if X_test is not None else None
    return clf, group_probs_eval, y_eval_agg, auc_score_eval, group_probs_test


def model_multiary(X_train, y_train, X_eval, y_eval, X_test, model_name, type_label,sample_weight=None):
    clf = model_selector(model_name)

    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
    group_probs_eval, y_eval_agg, auc_score_eval = aggregate_proba_eval_multi(clf, X_eval, y_eval, group_size)
    print(f"{type_label} Multi-class AUC (eval) = {auc_score_eval:.4f}")

    group_probs_test = aggregate_proba_test_multi(clf, X_test, group_size) if X_test is not None else None
    return clf, group_probs_eval, y_eval_agg, auc_score_eval, group_probs_test



###############################
# 資料讀取與分組函式（基於 CSV檔案，unique_id 為檔名）
###############################

# def load_and_group_train_data(info, feature_data_dir, train_players, eval_players, target_mask):
#     # 建立空的 DataFrame（分組：Group A: mode in [9,10]；Group B: mode in [0-8]）
#     x_train_A, y_train_A = pd.DataFrame(), pd.DataFrame(columns=target_mask)
#     x_eval_A, y_eval_A = pd.DataFrame(), pd.DataFrame(columns=target_mask)
#     x_train_B, y_train_B = pd.DataFrame(), pd.DataFrame(columns=target_mask)
#     x_eval_B, y_eval_B = pd.DataFrame(), pd.DataFrame(columns=target_mask)
#
#     datalist = glob.glob(os.path.join(feature_data_dir, '*.csv'))
#     for file in datalist:
#         unique_id = int(Path(file).stem)
#         row = info[info['unique_id'] == unique_id]
#         if row.empty:
#             continue
#         mode_value = int(row['mode'].iloc[0])
#         player_id = row['player_id'].iloc[0]
#         data = pd.read_csv(file)  # shape=(group_size, num_features)
#
#         target = row[target_mask]
#         target_repeated = pd.concat([target] * len(data), ignore_index=True)
#         if player_id in train_players:
#             if mode_value in [9, 10]:
#                 x_train_A = pd.concat([x_train_A, data], ignore_index=True)
#                 y_train_A = pd.concat([y_train_A, target_repeated], ignore_index=True)
#             else:
#                 x_train_B = pd.concat([x_train_B, data], ignore_index=True)
#                 y_train_B = pd.concat([y_train_B, target_repeated], ignore_index=True)
#         elif player_id in eval_players:
#             if mode_value in [9, 10]:
#                 x_eval_A = pd.concat([x_eval_A, data], ignore_index=True)
#                 y_eval_A = pd.concat([y_eval_A, target_repeated], ignore_index=True)
#             else:
#                 x_eval_B = pd.concat([x_eval_B, data], ignore_index=True)
#                 y_eval_B = pd.concat([y_eval_B, target_repeated], ignore_index=True)
#     return x_train_A, y_train_A, x_eval_A, y_eval_A, x_train_B, y_train_B, x_eval_B, y_eval_B
#
#
# def load_test_data(test_info_path, test_feature_data_dir):
#     test_info = pd.read_csv(test_info_path)
#     test_datalist = glob.glob(os.path.join(test_feature_data_dir, '*.csv'))
#     base_names = [os.path.splitext(os.path.basename(file))[0] for file in test_datalist]
#     mode_dict = {}
#     for idx, row in test_info.iterrows():
#         mode_dict[row['unique_id']] = int(row['mode'])
#     x_test = pd.DataFrame()
#     test_ids = []
#     for file in test_datalist:
#         unique_id = int(Path(file).stem)
#         data = pd.read_csv(file)
#         x_test = pd.concat([x_test, data], ignore_index=True)
#         test_ids.extend([unique_id] * len(data))
#     return x_test, test_ids, mode_dict, base_names

def load_and_group_train_data(info, feature_data_dir, train_players, eval_players, target_mask):
    # 建立空的列表，分別儲存 Group A 與 Group B 的聚合資料
    # Group A：mode in [9,10]
    x_train_A_list, y_train_A_list = [], []
    x_eval_A_list, y_eval_A_list = [], []

    # Group B：mode 不在 [9,10]
    x_train_B_list, y_train_B_list = [], []
    x_eval_B_list, y_eval_B_list = [], []

    datalist = glob.glob(os.path.join(feature_data_dir, '*.csv'))
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        mode_value = int(row['mode'].iloc[0])
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)  # 原始 shape = (27, num_features)

        # 將27次揮拍聚合成一筆 aggregated feature
        aggregated_data = aggregate_swings(data)
        # 將 aggregated_data 轉成 DataFrame (一筆資料，每個欄位為 aggregated 特徵)
        aggregated_df = pd.DataFrame([aggregated_data])

        # 取得 target (假設 target_mask 是多欄位，可以直接保存)
        target = row[target_mask]

        # 依據 player_id 在 train 或 eval 中、以及 mode_value，分別放到 Group A 或 Group B
        if player_id in train_players:
            if mode_value in [9, 10]:
                x_train_A_list.append(aggregated_df)
                y_train_A_list.append(target)
            else:
                x_train_B_list.append(aggregated_df)
                y_train_B_list.append(target)
        elif player_id in eval_players:
            if mode_value in [9, 10]:
                x_eval_A_list.append(aggregated_df)
                y_eval_A_list.append(target)
            else:
                x_eval_B_list.append(aggregated_df)
                y_eval_B_list.append(target)

    # 將各列表合併成 DataFrame，每個受測者只有一筆資料
    x_train_A = pd.concat(x_train_A_list, ignore_index=True)
    y_train_A = pd.concat(y_train_A_list, ignore_index=True)
    x_eval_A = pd.concat(x_eval_A_list, ignore_index=True)
    y_eval_A = pd.concat(y_eval_A_list, ignore_index=True)

    x_train_B = pd.concat(x_train_B_list, ignore_index=True)
    y_train_B = pd.concat(y_train_B_list, ignore_index=True)
    x_eval_B = pd.concat(x_eval_B_list, ignore_index=True)
    y_eval_B = pd.concat(y_eval_B_list, ignore_index=True)

    return x_train_A, y_train_A, x_eval_A, y_eval_A, x_train_B, y_train_B, x_eval_B, y_eval_B


def load_test_data(test_info_path, test_feature_data_dir):
    test_info = pd.read_csv(test_info_path)
    test_datalist = glob.glob(os.path.join(test_feature_data_dir, '*.csv'))
    # 取檔名（不含副檔名）作為 unique_id 字串，或轉成 int 視你的資料而定
    base_names = [os.path.splitext(os.path.basename(file))[0] for file in test_datalist]

    mode_dict = {}
    # 將 test_info 中的 unique_id 與 mode 對應起來
    for idx, row in test_info.iterrows():
        mode_dict[row['unique_id']] = int(row['mode'])

    aggregated_list = []
    test_ids = []
    # 對每個 CSV 檔，做聚合，每個檔案只產生一筆 aggregated sample
    for file in test_datalist:
        unique_id = int(Path(file).stem)
        data = pd.read_csv(file)
        # 呼叫你定義的 aggregate_swings 函式聚合該檔案的27次揮拍
        aggregated_features = aggregate_swings(data)
        # 將 aggregated_features 轉成一個 DataFrame 的一行
        aggregated_list.append(pd.DataFrame([aggregated_features]))
        test_ids.append(unique_id)

    # 合併所有 aggregated 的資料，這裡每筆資料對應一個受測者
    x_test = pd.concat(aggregated_list, ignore_index=True)
    return x_test, test_ids, mode_dict, base_names

def aggregate_swings(swing_data):
    """
    swing_data: pandas DataFrame 或 numpy array，形狀 (27, n_features)
    回傳 aggregated feature vector，其包含每個特徵的 mean, std, max, min。
    """
    # 如果 swing_data 是 DataFrame，轉成 numpy 陣列
    if isinstance(swing_data, pd.DataFrame):
        swing_data = swing_data.values

    # 計算每個特徵的統計量 (沿 axis=0 即 27 次的資料)
    agg_mean = np.mean(swing_data, axis=0)
    agg_std = np.std(swing_data, axis=0)
    agg_max = np.max(swing_data, axis=0)
    agg_min = np.min(swing_data, axis=0)

    # 合併所有統計量
    aggregated_features = np.concatenate([agg_mean, agg_std, agg_max, agg_min])
    return aggregated_features


#############################################
# 測試預測函式 (根據 group_size 與 mode 決定模型)
#############################################

def _FOR_NOT_AGGR_USED_predict_test_binary(X_test, group_size, test_ids, mode_dict, clf_A, clf_B):
    num_groups = len(X_test) // group_size
    final_probs = []
    for i in range(num_groups):
        group_data = X_test[i * group_size:(i + 1) * group_size]
        unique_id = test_ids[i * group_size]
        mode = mode_dict.get(unique_id, None)
        if mode is None:
            pred = clf_B.predict_proba(group_data).mean(axis=0)
        elif mode in [9, 10]:
            pred = clf_A.predict_proba(group_data).mean(axis=0)
        else:
            pred = clf_B.predict_proba(group_data).mean(axis=0)
        final_probs.append(pred)
    return np.array(final_probs)


def _FOR_NOT_AGGR_USED_predict_test_multi(X_test, group_size, test_ids, mode_dict, clf_A, clf_B):
    num_groups = len(X_test) // group_size
    final_probs = []
    for i in range(num_groups):
        group_data = X_test[i * group_size:(i + 1) * group_size]
        unique_id = test_ids[i * group_size]
        mode = mode_dict.get(unique_id, None)
        if mode is None:
            pred = clf_B.predict_proba(group_data).mean(axis=0)
        elif mode in [9, 10]:
            pred = clf_A.predict_proba(group_data).mean(axis=0)
        else:
            pred = clf_B.predict_proba(group_data).mean(axis=0)
        final_probs.append(pred)
    return np.array(final_probs)

def predict_test_binary(X_test, group_size, test_ids, mode_dict, clf_A, clf_B):
    """
    針對每筆 aggregated 測試資料進行預測 (二分類)：
      - X_test: 每一行代表一個受測者的 aggregated feature
      - test_ids: list of unique_id (與 X_test 行順序一致)
      - mode_dict: 字典，將 unique_id 對應到 mode
      - clf_A / clf_B: 分別為模型 (例如 Group A 與 Group B)
    """
    final_probs = []
    # 每筆資料直接預測
    for i, unique_id in enumerate(test_ids):
        mode = mode_dict.get(unique_id, None)
        sample = X_test[i:i+1]  # 取出單筆資料，保持 2D 輸入
        if mode is None:
            pred = clf_B.predict_proba(sample)[0]
        elif mode in [9, 10]:
            pred = clf_A.predict_proba(sample)[0]
        else:
            pred = clf_B.predict_proba(sample)[0]
        final_probs.append(pred)
    return np.array(final_probs)


def predict_test_multi(X_test, group_size, test_ids, mode_dict, clf_A, clf_B):
    """
    同上，適用於多分類情境，直接預測 aggregated sample
    """
    final_probs = []
    for i, unique_id in enumerate(test_ids):
        mode = mode_dict.get(unique_id, None)
        sample = X_test[i:i+1]
        if mode is None:
            pred = clf_B.predict_proba(sample)[0]
        elif mode in [9, 10]:
            pred = clf_A.predict_proba(sample)[0]
        else:
            pred = clf_B.predict_proba(sample)[0]
        final_probs.append(pred)
    return np.array(final_probs)

def compute_sample_weights(y_encoded):
    """依照每個類別的數量來計算 sample_weight"""
    counter = Counter(y_encoded)  # e.g. {0: 100, 1: 30, 2: 10}
    total = len(y_encoded)
    n_classes = len(counter)
    # 建立 每個class: weight
    class_weights = {}
    for cls, cnt in counter.items():
        class_weights[cls] = total / (n_classes * cnt)

    # 建立與 y_encoded 等長的 sample_weight
    sample_weight = np.array([class_weights[cls] for cls in y_encoded])
    return sample_weight

#############################################
# 主程式
#############################################
def main():
    # 讀取訓練資訊
    train_info_path = "data/39_Training_Dataset/train_info.csv"
    info = pd.read_csv(train_info_path)
    # 分割玩家以建立 train 與 eval
    unique_players = info['player_id'].unique()
    train_players, eval_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    test_info_path = "data/39_Test_Dataset/test_info.csv"

    feature_data_dir = "data/39_Training_Dataset/tabular_data_train"
    test_feature_data_dir = "data/39_Test_Dataset/tabular_data_train"

    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # 讀取並依照 mode 分組: Group A (mode in [9,10]) 與 Group B (mode in [0-8])
    (x_train_A, y_train_A, x_eval_A, y_eval_A,
     x_train_B, y_train_B, x_eval_B, y_eval_B) = load_and_group_train_data(info, feature_data_dir, train_players,
                                                                           eval_players, target_mask)

    # 特徵縮放
    scaler_A = MinMaxScaler()
    X_train_scaled_A = scaler_A.fit_transform(x_train_A)
    X_eval_scaled_A = scaler_A.transform(x_eval_A)

    scaler_B = MinMaxScaler()
    X_train_scaled_B = scaler_B.fit_transform(x_train_B)
    X_eval_scaled_B = scaler_B.transform(x_eval_B)

    # LabelEncoder - 分別處理各目標欄位
    # gender (binary)
    le_gender = LabelEncoder()
    y_train_le_gender_A = le_gender.fit_transform(y_train_A['gender'])
    y_eval_le_gender_A = le_gender.transform(y_eval_A['gender'])
    y_train_le_gender_B = le_gender.fit_transform(y_train_B['gender'])
    y_eval_le_gender_B = le_gender.transform(y_eval_B['gender'])

    # hold racket handed (binary)
    le_hold = LabelEncoder()
    y_train_le_hold_A = le_hold.fit_transform(y_train_A['hold racket handed'])
    y_eval_le_hold_A = le_hold.transform(y_eval_A['hold racket handed'])
    y_train_le_hold_B = le_hold.fit_transform(y_train_B['hold racket handed'])
    y_eval_le_hold_B = le_hold.transform(y_eval_B['hold racket handed'])

    # play years (multi, 假設3類)
    le_years = LabelEncoder()
    y_train_le_years_A = le_years.fit_transform(y_train_A['play years'])
    y_eval_le_years_A = le_years.transform(y_eval_A['play years'])
    y_train_le_years_B = le_years.fit_transform(y_train_B['play years'])
    y_eval_le_years_B = le_years.transform(y_eval_B['play years'])

    # level (multi, 假設4類)
    le_level = LabelEncoder()
    y_train_le_level_A = le_level.fit_transform(y_train_A['level'])
    y_eval_le_level_A = le_level.transform(y_eval_A['level'])
    y_train_le_level_B = le_level.fit_transform(y_train_B['level'])
    y_eval_le_level_B = le_level.transform(y_eval_B['level'])

    # 訓練模型：
    # Group A:
    clf_gender_A, _, _, auc_gender_A, _ = model_binary(X_train_scaled_A, y_train_le_gender_A,
                                                       X_eval_scaled_A, y_eval_le_gender_A, None, model_name,
                                                       "gender (Group A)")
    clf_hold_A, _, _, auc_hold_A, _ = model_binary(X_train_scaled_A, y_train_le_hold_A,
                                                   X_eval_scaled_A, y_eval_le_hold_A, None, model_name,
                                                   "hold (Group A)")
    clf_years_A, _, _, auc_years_A, _ = model_multiary(X_train_scaled_A, y_train_le_years_A,
                                                       X_eval_scaled_A, y_eval_le_years_A, None, model_name,
                                                       "play years (Group A)")
    clf_level_A, _, _, auc_level_A, _ = model_multiary(X_train_scaled_A, y_train_le_level_A,
                                                       X_eval_scaled_A, y_eval_le_level_A, None, model_name,
                                                       "level (Group A)")
    # Group B:
    clf_gender_B, _, _, auc_gender_B, _ = model_binary(X_train_scaled_B, y_train_le_gender_B,
                                                       X_eval_scaled_B, y_eval_le_gender_B, None, model_name,
                                                       "gender (Group B)")
    clf_hold_B, _, _, auc_hold_B, _ = model_binary(X_train_scaled_B, y_train_le_hold_B,
                                                   X_eval_scaled_B, y_eval_le_hold_B, None, model_name,
                                                   "hold (Group B)")
    clf_years_B, _, _, auc_years_B, _ = model_multiary(X_train_scaled_B, y_train_le_years_B,
                                                       X_eval_scaled_B, y_eval_le_years_B, None, model_name,
                                                       "play years (Group B)")
    clf_level_B, _, _, auc_level_B, _ = model_multiary(X_train_scaled_B, y_train_le_level_B,
                                                       X_eval_scaled_B, y_eval_le_level_B, None, model_name,
                                                       "level (Group B)")
    print("Group A AUCs: gender:", "play years:", auc_years_A)
    print("Group B AUCs: gender:", "play years:", auc_years_B)

    # print("Group A AUCs: gender:", auc_gender_A, "hold:", auc_hold_A, "play years:", auc_years_A, "level:", auc_level_A)
    # print("Group B AUCs: gender:", auc_gender_B, "hold:", auc_hold_B, "play years:", auc_years_B, "level:", auc_level_B)

    # 載入測試資料
    if 1:
        x_test, test_ids, mode_dict, base_names = load_test_data(test_info_path, test_feature_data_dir)
        scaler_test = MinMaxScaler()
        X_test_scaled = scaler_test.fit_transform(x_test)

        # 測試預測（依照 unique_id 的 mode 使用 Group A 或 B 模型）
        final_gender_probs = predict_test_binary(X_test_scaled, group_size, test_ids, mode_dict, clf_gender_A, clf_gender_B)
        final_hold_probs = predict_test_binary(X_test_scaled, group_size, test_ids, mode_dict, clf_hold_A, clf_hold_B)
        final_years_probs = predict_test_multi(X_test_scaled, group_size, test_ids, mode_dict, clf_years_A, clf_years_B)
        final_level_probs = predict_test_multi(X_test_scaled, group_size, test_ids, mode_dict, clf_level_A, clf_level_B)

        # 組合輸出結果
        # 輸出欄位：['unique_id', 'gender', 'hold racket handed', 'play years_0', 'play years_1', 'play years_2',
        #            'level_2', 'level_3', 'level_4', 'level_5']
        columns = ['unique_id', 'gender', 'hold racket handed',
                   'play years_0', 'play years_1', 'play years_2',
                   'level_2', 'level_3', 'level_4', 'level_5']
        results = []
        num_groups = final_gender_probs.shape[0]
        for i in range(num_groups):
            unique_id = base_names[i]  # 假設 base_names 中的順序與 group 順序一致
            # 對於 binary 任務，取正類機率（假設 index 1）
            gender_prob = f"{1-final_gender_probs[i, 1]:.4f}"
            hold_prob = f"{1-final_hold_probs[i, 1]:.4f}"
            # 對於多類任務，展開每個類別的機率
            play_years_list = [f"{p:.4f}" for p in final_years_probs[i]]  # 3個數值
            level_list = [f"{p:.4f}" for p in final_level_probs[i]]  # 4個數值
            row = [unique_id, gender_prob, hold_prob] + play_years_list + level_list
            results.append(row)

        df_out = pd.DataFrame(results, columns=columns)
        df_out.to_csv('predictions.csv', index=False)
        print("Saved predictions to predictions.csv")


if __name__ == '__main__':
    main()
    # not use aggr
    # A : 0.97, 1, 0.669, 0.87
    # B : 1, 1, 0.62, 0.837
    #

    # aggr
    # A : 0.9941, 0.999, 0.6919, 0.8561
    # B : 0.9979, 1, 0.6840, 0.6828
    # Total 0.82