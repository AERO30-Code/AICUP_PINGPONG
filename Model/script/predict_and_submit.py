import os
import pandas as pd
import joblib
import numpy as np

FEATURE_PATH = "Feature/katsu/test_v3.csv"
MODEL_DIR = "Model/output/20250501_180625"
MODEL_FOLDER_NAME = os.path.basename(MODEL_DIR.rstrip('/'))
SUBMIT_PATH = f"Model/submission/submission_{MODEL_FOLDER_NAME}.csv"

TARGETS = {
    "gender": "binary",
    "hold_racket_handed": "binary",
    "play_years": "multiclass",
    "level": "multiclass"
}
TARGET_COLS = {
    "gender": "gender",
    "hold_racket_handed": "hold racket handed",
    "play_years": "play years",
    "level": "level"
}
PLAY_YEARS_CLASSES = [0, 1, 2]
LEVEL_CLASSES = [2, 3, 4, 5]

def main():
    # 讀取 sample_submission 欄位順序
    sample_sub_path = "Model/submission/sample_submission.csv"
    sample_sub = pd.read_csv(sample_sub_path, nrows=1)
    submission_columns = list(sample_sub.columns)

    df = pd.read_csv(FEATURE_PATH)
    submit_df = pd.DataFrame()
    submit_df["unique_id"] = df["unique_id"]

    # 自動偵測有哪些模型
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    available_targets = set()
    for f in model_files:
        for t in TARGETS:
            if f.endswith(f"_{t}.pkl"):
                available_targets.add(t)
    
    # gender
    if "gender" in available_targets:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(f"_gender.pkl")]
        model_path = os.path.join(MODEL_DIR, model_files[0])
        model = joblib.load(model_path)
        # 動態取得模型名稱
        model_name = model_files[0].replace("_gender.pkl", "")
        excluded_cols = ["unique_id"] + list(TARGET_COLS.values())
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_gender_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(df[feature_cols])
        else:
            X_test = df[feature_cols]
        proba = model.predict_proba(X_test)[:, 0]
        submit_df["gender"] = proba
    else:
        submit_df["gender"] = 0.5

    # hold_racket_handed
    if "hold_racket_handed" in available_targets:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(f"_hold_racket_handed.pkl")]
        model_path = os.path.join(MODEL_DIR, model_files[0])
        model = joblib.load(model_path)
        model_name = model_files[0].replace("_hold_racket_handed.pkl", "")
        excluded_cols = ["unique_id"] + list(TARGET_COLS.values())
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_hold_racket_handed_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(df[feature_cols])
        else:
            X_test = df[feature_cols]
        proba = model.predict_proba(X_test)[:, 0]
        submit_df["hold racket handed"] = proba
    else:
        submit_df["hold racket handed"] = 0.5

    # play_years
    if "play_years" in available_targets:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(f"_play_years.pkl")]
        model_path = os.path.join(MODEL_DIR, model_files[0])
        model = joblib.load(model_path)
        model_name = model_files[0].replace("_play_years.pkl", "")
        excluded_cols = ["unique_id"] + list(TARGET_COLS.values())
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_play_years_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(df[feature_cols])
        else:
            X_test = df[feature_cols]
        proba = model.predict_proba(X_test)
        for i, c in enumerate(["play years_0", "play years_1", "play years_2"]):
            submit_df[c] = proba[:, i]
    else:
        for c in ["play years_0", "play years_1", "play years_2"]:
            submit_df[c] = 0.333333

    # level
    if "level" in available_targets:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(f"_level.pkl")]
        model_path = os.path.join(MODEL_DIR, model_files[0])
        model = joblib.load(model_path)
        model_name = model_files[0].replace("_level.pkl", "")
        excluded_cols = ["unique_id"] + list(TARGET_COLS.values())
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_level_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(df[feature_cols])
        else:
            X_test = df[feature_cols]
        proba = model.predict_proba(X_test)
        for i, c in enumerate(["level_2", "level_3", "level_4", "level_5"]):
            submit_df[c] = proba[:, i]
    else:
        for c in ["level_2", "level_3", "level_4", "level_5"]:
            submit_df[c] = 0.25

    # 依 sample_submission 欄位順序輸出
    submit_df = submit_df[submission_columns]
    submit_df.to_csv(SUBMIT_PATH, index=False, float_format="%.6f")
    print(f"Submission saved to {SUBMIT_PATH}")

if __name__ == "__main__":
    main()