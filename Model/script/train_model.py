import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import datetime
CONFIG_PATH = "Model/config/model_config.yaml"
FEATURE_PATH = "Feature/output/features_train_gender_top/features_train_top50.csv"
MODEL_DIR = "Model/output/"

# 目標與型態設定
TARGETS = {
    "gender": "binary",
    "hold_racket_handed": "binary",
    "play_years": "multiclass",
    "level": "multiclass"
}
# label 對應 features_train.csv 欄位名
TARGET_COLS = {
    "gender": "gender",
    "hold_racket_handed": "hold racket handed",
    "play_years": "play years",
    "level": "level"
}

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_model(name, params):
    if name.lower() == "randomforest":
        return RandomForestClassifier(**params)
    elif name.lower() == "lightgbm":
        return LGBMClassifier(**params)
    elif name.lower() == "xgboost":
        return XGBClassifier(**params)
    elif name.lower() == "svm":
        return SVC(**params)
    elif name.lower() == "catboost":
        return CatBoostClassifier(**params)
    elif name.lower() == "logisticregression":
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    config = load_config()
    df = pd.read_csv(FEATURE_PATH)

    # 取得本次要訓練的 targets
    train_targets = config.get('train_targets', None)
    if train_targets is None:
        selected_targets = list(TARGETS.keys())
    else:
        selected_targets = [t for t in train_targets if t in TARGETS]

    # 建立以日期時間命名的 output 子資料夾
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(MODEL_DIR, now)
    os.makedirs(run_output_dir, exist_ok=True)

    log_lines = []
    log_lines.append(f"Training time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Feature path: {FEATURE_PATH}")
    # 讀取 scaling 方式
    scaling_mode = config.get('feature_scaling', 'none').lower()
    log_lines.append(f"Feature scaling: {scaling_mode}")
    log_lines.append(f"Trained targets: {', '.join(selected_targets)}")

    for target in selected_targets:
        mode = TARGETS[target]
        print(f"\n===== Training for target: {target} ({mode}) =====")
        y_col = TARGET_COLS[target]
        # 僅排除 unique_id 與其他標籤（除了自己的 y_col）
        excluded_cols = ["unique_id"] + list(TARGET_COLS.values())
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        X = df[feature_cols]
        y = df[y_col]

        # 標籤轉換 Readme(train).txt：
        # gender: 1=male, 2=female
        # hold racket handed: 1=right, 2=left
        # play years: 0=low, 1=mid, 2=high
        # level: 2=大專甲組, 3=大專乙組, 4=青少年國手, 5=青少年選手
        #
        # gender/hold racket handed: 1/2 → 0/1（減1）
        # play years: 已經是 0/1/2，不需轉換
        # level: 2/3/4/5 → 0/1/2/3（減去最小值2）
        if mode == "binary" and y.min() == 1 and y.max() == 2:
            y = y - 1
        if mode == "multiclass":
            y = y - y.min()
            
        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model_name = config[target]["model"]

        # 特徵 scaling（僅針對 feature_cols）
        if scaling_mode == "zscore":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            scaler_path = os.path.join(run_output_dir, f"{model_name}_{target}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
        elif scaling_mode == "minmax":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            scaler_path = os.path.join(run_output_dir, f"{model_name}_{target}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
        # 若 scaling_mode == "none"，則不做 scaling

        # 取得模型與參數
        params = config[target]["params"]
        model = get_model(model_name, params)

        log_lines.append("")
        log_lines.append(f"Model: {model_name}_{target}")
        log_lines.append(f"Parameters: {params}")

        # fit
        model.fit(X_train, y_train)

        # 預測機率
        y_val_proba = model.predict_proba(X_val)
        if mode == "binary":
            auc = roc_auc_score(y_val, y_val_proba[:, 1])
        else:
            auc = roc_auc_score(y_val, y_val_proba, average="micro", multi_class="ovr")
        print(f"Validation ROC AUC: {auc:.4f}")

        # Save model
        model_path = os.path.join(run_output_dir, f"{model_name}_{target}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Save feature importance
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            imp_path = os.path.join(run_output_dir, f"{model_name}_{target}_feature_importance.csv")
            imp.to_csv(imp_path)
            print(f"Feature importance saved to {imp_path}.")

        # Log AUC
        log_lines.append(f"Validation ROC AUC for {model_name}_{target}: {auc:.4f}")
        
    # Write log.txt
    log_path = os.path.join(run_output_dir, "log.txt")
    with open(log_path, "w") as f:
        for line in log_lines:
            f.write(str(line) + "\n")
    print(f"Training log saved to {log_path}")

if __name__ == "__main__":
    main()