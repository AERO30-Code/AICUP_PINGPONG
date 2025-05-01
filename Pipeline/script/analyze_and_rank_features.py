import os
import yaml
import datetime
import pandas as pd
import importlib.util
import sys
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Constants (可以考慮移到共用設定檔或模組) ---
# 假設 TARGETS_INFO 與 run_pipeline.py 中的一致
TARGETS_INFO = {
    "gender": {"type": "binary", "col": "gender"},
    "hold_racket_handed": {"type": "binary", "col": "hold racket handed"},
    "play_years": {"type": "multiclass", "col": "play years"},
    "level": {"type": "multiclass", "col": "level"}
}

# --- Helper Functions (Adapted from run_pipeline.py) ---

def load_pipeline_config(config_path="config/pipeline_config.yaml"):
    """載入 Pipeline 設定檔"""
    abs_config_path = os.path.abspath(config_path)
    print(f"Loading pipeline config from: {abs_config_path}")
    try:
        with open(abs_config_path, "r") as f:
            config = yaml.safe_load(f)
            return config, os.path.dirname(abs_config_path)
    except FileNotFoundError:
        print(f"Error: Pipeline config file not found at {abs_config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing pipeline config: {e}")
        sys.exit(1)

def resolve_paths(config, config_dir):
    """將 YAML 中的相對路徑轉換為絕對路徑"""
    def resolve(path):
        if not path:
            return path
        if not os.path.isabs(path):
            resolved_path = os.path.abspath(os.path.join(config_dir, path))
            return resolved_path
        return path

    for key in config.get('paths', {}):
        config['paths'][key] = resolve(config['paths'].get(key))
    for key in config.get('output_paths', {}):
        config['output_paths'][key] = resolve(config['output_paths'].get(key))
    if 'generate_features_script' in config.get('paths', {}):
         config['paths']['generate_features_script'] = resolve(config['paths']['generate_features_script'])
    # 解析 analysis 和 selection 的路徑
    if 'output_rank_file' in config.get('feature_analysis', {}):
         # 輸出排名檔案路徑是相對於 output_paths.model_runs 的 *父* 目錄
         base_output_dir = os.path.dirname(config['output_paths'].get('model_runs', '.'))
         relative_rank_path = config['feature_analysis']['output_rank_file']
         config['feature_analysis']['_resolved_output_rank_file'] = os.path.abspath(os.path.join(base_output_dir, relative_rank_path))

    if 'rank_file_path' in config.get('feature_selection', {}):
         # 特徵選擇讀取的排名檔案路徑，也相對於 config 目錄解析
         config['feature_selection']['_resolved_rank_file_path'] = resolve(config['feature_selection']['rank_file_path'])


    return config

def import_generate_features(script_path):
    """動態導入 generate_features 模組"""
    if not script_path or not os.path.exists(script_path):
         print(f"Error: generate_features script path '{script_path}' not specified or file not found.")
         sys.exit(1)
    try:
        module_name = "generate_features"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
             raise ImportError(f"Could not load spec for module at {script_path}")
        generate_features = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_features)
        print(f"Successfully imported generate_features from: {script_path}")
        return generate_features
    except Exception as e:
        print(f"Error dynamically importing generate_features from {script_path}: {e}")
        sys.exit(1)

def get_model_instance(model_config):
    """根據設定檔創建模型實例 (與 run_pipeline.py 相同)"""
    model_type = model_config['model_type'].lower()
    params = model_config.get('params', {})

    if model_type == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        # 自動處理 XGBoost 的參數
        params.setdefault('use_label_encoder', False)
        is_multiclass = 'objective' in params and 'multi' in params['objective']
        params.setdefault('eval_metric', 'mlogloss' if is_multiclass else 'logloss')
        return XGBClassifier(**params)
    elif model_type == "svm":
        from sklearn.svm import SVC
        params.setdefault('probability', True) # 通常分析不需要 probability，但保留以防萬一
        return SVC(**params)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    # 添加其他模型...
    else:
        raise ValueError(f"Unknown model type for analysis: {model_type}")

def get_feature_importance(model, feature_names):
    """提取模型特徵重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'): # 例如 Logistic Regression
        # 對於多分類的 coef_ 可能需要特殊處理，這裡簡化為取絕對值的平均或第一個類別
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        print(f"Warning: Cannot extract feature importance from model type {type(model)}. Returning zeros.")
        importances = np.zeros(len(feature_names))

    if len(importances) != len(feature_names):
         print(f"Warning: Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}).")
         # 嘗試處理，或返回 None/空 DataFrame
         return pd.DataFrame({'feature': feature_names, 'importance': 0.0}) # 返回 0

    return pd.DataFrame({'feature': feature_names, 'importance': importances})


# --- Main Analysis Logic ---

def main():
    # 1. 載入設定檔
    script_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(script_dir, "../config/pipeline_config.yaml"))
    config, config_dir = load_pipeline_config(config_path)
    config = resolve_paths(config, config_dir)

    analysis_config = config.get('feature_analysis', {})
    if not analysis_config.get('enable', False):
        print("Feature analysis is disabled in the configuration. Exiting.")
        return

    print("\n--- Running Feature Analysis and Ranking ---")

    # --- 檢查必要設定 ---
    target_key = analysis_config.get('target_for_analysis')
    analysis_models_config = analysis_config.get('analysis_models')
    output_rank_file_path = analysis_config.get('_resolved_output_rank_file')
    gen_config = config.get('feature_generation', {})
    paths_config = config.get('paths', {})
    output_paths_config = config.get('output_paths', {})

    if not target_key or target_key not in TARGETS_INFO:
        print(f"Error: 'target_for_analysis' ({target_key}) is missing or invalid in config.")
        sys.exit(1)
    if not analysis_models_config:
        print("Error: 'analysis_models' section is missing or empty in config.")
        sys.exit(1)
    if not output_rank_file_path:
        print("Error: Could not resolve 'output_rank_file' path.")
        sys.exit(1)
    if not paths_config.get('train_info') or not paths_config.get('train_data'):
        print("Error: Missing 'train_info' or 'train_data' path in config.")
        sys.exit(1)

    target_info = TARGETS_INFO[target_key]
    y_col = target_info['col']
    print(f"Analysis target: {target_key} (column: '{y_col}')")

    # 2. 導入 generate_features 模組
    generate_features_script_path = config.get('paths', {}).get('generate_features_script')
    generate_features_module = import_generate_features(generate_features_script_path)

    # 3. 執行特徵生成 (僅訓練資料)
    print("\nGenerating full features for training data...")
    sensors = generate_features_module.ALL_SENSORS if gen_config.get('sensors_to_process', 'ALL') == 'ALL' else gen_config['sensors_to_process']
    stats = generate_features_module.ALL_STATS if gen_config.get('stats_to_calculate', 'ALL') == 'ALL' else gen_config['stats_to_calculate']
    # 分析時，檔名可以固定或使用分析目標命名
    train_output_filename = f"features_train_s{gen_config['start_swing']}-e{gen_config['end_swing']}_FOR_ANALYSIS.csv"
    generated_features_dir = output_paths_config.get('generated_features', os.path.join(config_dir, '../..', 'Feature/generated')) # 提供預設值
    os.makedirs(generated_features_dir, exist_ok=True)
    train_feature_path = os.path.join(generated_features_dir, train_output_filename)

    generate_features_module.process_data(
        info_path=paths_config['train_info'],
        data_dir=paths_config['train_data'],
        output_path=train_feature_path,
        start_swing=gen_config['start_swing'],
        end_swing=gen_config['end_swing'],
        sensors_to_process=sensors,
        stats_to_calculate=stats,
        extra_features_to_include=gen_config.get('extra_features_to_include', []),
        is_train_data=True,
        # 分析時通常不需要 player_id，除非你想做 player-based split 分析
        include_player_id_in_train=False
    )
    print(f"Full training features generated at: {train_feature_path}")

    # 4. 載入完整訓練特徵
    try:
        df_train = pd.read_csv(train_feature_path)
    except FileNotFoundError:
        print(f"Error: Generated training feature file not found at {train_feature_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading generated features: {e}")
        sys.exit(1)

    # 5. 準備 X (完整特徵) 和 y (分析目標)
    excluded_cols = ["unique_id"] + [t_info['col'] for t_info in TARGETS_INFO.values() if t_info['col'] in df_train.columns]
    feature_cols = [col for col in df_train.columns if col not in excluded_cols]

    if not feature_cols:
        print("Error: No feature columns found after exclusions.")
        sys.exit(1)

    X = df_train[feature_cols]
    if y_col not in df_train.columns:
        print(f"Error: Target column '{y_col}' for analysis not found in features.")
        sys.exit(1)
    y = df_train[y_col]

    # 標籤編碼 (與 run_pipeline.py 一致)
    le = LabelEncoder()
    original_min_label = None
    if target_info['type'] == "binary":
         if y.min() == 1 and y.max() == 2: # 假設 1, 2 -> 0, 1
             y = y - 1
         elif set(y.unique()) != {0, 1}:
             print("Warning: Binary target not {1, 2} or {0, 1}. Using LabelEncoder.")
             y = le.fit_transform(y) # 確保是 0, 1
    elif target_info['type'] == "multiclass":
        # 檢查是否為數字，如果不是則用 LabelEncoder
        if not pd.api.types.is_numeric_dtype(y):
             print(f"Warning: Multiclass target '{y_col}' is not numeric. Using LabelEncoder.")
             y = le.fit_transform(y)
        else:
             original_min_label = y.min()
             y = y - original_min_label # 平移到從 0 開始

    print(f"Shape of X for analysis: {X.shape}")
    print(f"Value counts for y ('{y_col}') for analysis:\n{y.value_counts()}")


    # 6. 訓練分析模型並提取重要性
    importance_dfs = []
    model_names = []
    for model_name, model_config in analysis_models_config.items():
        print(f"\nTraining analysis model: {model_name} ({model_config['model_type']})")
        try:
            model = get_model_instance(model_config)
            # 某些模型可能需要 verbose=0 或類似設定來減少輸出
            if 'verbose' in model.get_params():
                model.set_params(verbose=0)
            if 'silent' in model.get_params(): # For older XGBoost/LGBM
                model.set_params(silent=True)

            model.fit(X, y)
            imp_df = get_feature_importance(model, feature_cols)
            if not imp_df.empty:
                 importance_dfs.append(imp_df)
                 model_names.append(model_name)
                 print(f"Extracted importance for {len(imp_df)} features.")
            else:
                 print(f"Could not extract importance for model {model_name}")

        except Exception as e:
            print(f"Error training or getting importance for model {model_name}: {e}")
            # 可以選擇跳過或中止
            # continue

    if not importance_dfs:
        print("Error: No feature importances could be extracted from any analysis model. Aborting.")
        sys.exit(1)

    # 7. 計算平均排名 (類似 feature_importance_analysis_4models.py)
    print("\nCalculating mean feature ranks...")
    all_features = set()
    for df in importance_dfs:
        all_features.update(df['feature'])
    all_features = list(all_features)
    print(f"Total unique features across analysis models: {len(all_features)}")

    # 創建一個包含所有特徵的基礎 DataFrame
    merged_rank = pd.DataFrame({'feature': all_features})

    # 為每個模型的 feature importance 計算排名
    for i, imp_df in enumerate(importance_dfs):
        # 按重要性降序排序，排名從 1 開始
        imp_df_sorted = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
        feature_to_rank = {f: rank + 1 for rank, f in enumerate(imp_df_sorted['feature'])}
        # 對於在某模型中重要性為 0 或未出現的特徵，給予最大排名 + 1
        max_rank = len(imp_df_sorted) + 1
        rank_col_name = f'rank_{model_names[i]}'
        merged_rank[rank_col_name] = merged_rank['feature'].map(lambda x: feature_to_rank.get(x, max_rank))

    # 計算平均排名
    rank_cols = [col for col in merged_rank.columns if col.startswith('rank_')]
    merged_rank['mean_rank'] = merged_rank[rank_cols].mean(axis=1)
    merged_rank = merged_rank.sort_values('mean_rank').reset_index(drop=True)

    print("Top 5 features based on mean rank:")
    print(merged_rank.head())

    # 8. 儲存排名檔案
    try:
        output_dir = os.path.dirname(output_rank_file_path)
        os.makedirs(output_dir, exist_ok=True)
        merged_rank.to_csv(output_rank_file_path, index=False)
        print(f"\nMean rank file saved to: {output_rank_file_path}")
    except Exception as e:
        print(f"Error saving mean rank file: {e}")

    print("\n--- Feature Analysis and Ranking Completed ---")


if __name__ == "__main__":
    main()
