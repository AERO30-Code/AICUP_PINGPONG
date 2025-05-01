import os
import yaml
import datetime
import pandas as pd
import importlib.util
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score 

TARGETS_INFO = {
    "gender": {"type": "binary", "col": "gender"},
    "hold_racket_handed": {"type": "binary", "col": "hold racket handed"},
    "play_years": {"type": "multiclass", "col": "play years"},
    "level": {"type": "multiclass", "col": "level"}
}

def load_pipeline_config(config_path="config/pipeline_config.yaml"):
    """
    載入 Pipeline 設定檔。
    現在假定 config_path 是相對於執行 run_pipeline.py 的當前工作目錄，或是絕對路徑。
    """
    abs_config_path = os.path.abspath(config_path)
    print(f"Loading pipeline config from: {abs_config_path}")
    try:
        with open(abs_config_path, "r") as f:
            config = yaml.safe_load(f)
            return config, os.path.dirname(abs_config_path) # 返回設定檔和其所在目錄
    except FileNotFoundError:
        print(f"Error: Pipeline config file not found at {abs_config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing pipeline config: {e}")
        sys.exit(1)

def resolve_paths(config, config_dir):
    """
    將 YAML 中的相對路徑轉換為絕對路徑 (相對於設定檔所在目錄)。
    """
    # print("Resolving paths specified in config...") # 移除此行減少輸出
    def resolve(path):
        if not path: # 檢查 path 是否為 None 或空字串
            return path
        if not os.path.isabs(path): # 將相對於設定檔目錄的路徑轉換為絕對路徑
            resolved_path = os.path.abspath(os.path.join(config_dir, path))
            return resolved_path
        return path

    for key in config.get('paths', {}):
        config['paths'][key] = resolve(config['paths'].get(key))

    for key in config.get('output_paths', {}):
        config['output_paths'][key] = resolve(config['output_paths'].get(key))

    if 'generate_features_script' in config.get('paths', {}):
         config['paths']['generate_features_script'] = resolve(config['paths']['generate_features_script'])

    # print("Path resolution completed.") # 移除此行減少輸出
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


def run_feature_generation_stage(config, generate_features_module):
    """執行特徵生成階段"""
    print("\n--- Running Feature Generation Stage ---")
    gen_config = config['feature_generation']
    paths_config = config['paths']
    output_paths_config = config['output_paths']

    # 檢查必要路徑是否存在
    required_paths = [paths_config.get('train_info'), paths_config.get('train_data'),
                      paths_config.get('test_info'), paths_config.get('test_data'),
                      output_paths_config.get('generated_features')]
    if None in required_paths:
         print("Error: Missing required paths for feature generation in config. Aborting stage.")
         return None, None

    sensors = generate_features_module.ALL_SENSORS if gen_config.get('sensors_to_process', 'ALL') == 'ALL' else gen_config['sensors_to_process']
    stats = generate_features_module.ALL_STATS if gen_config.get('stats_to_calculate', 'ALL') == 'ALL' else gen_config['stats_to_calculate']

    train_output_filename = f"features_train_s{gen_config['start_swing']}-e{gen_config['end_swing']}.csv"
    test_output_filename = f"features_test_s{gen_config['start_swing']}-e{gen_config['end_swing']}.csv"
    train_output_path = os.path.join(output_paths_config['generated_features'], train_output_filename)
    test_output_path = os.path.join(output_paths_config['generated_features'], test_output_filename)

    generate_features_module.process_data(
        info_path=paths_config['train_info'],
        data_dir=paths_config['train_data'],
        output_path=train_output_path,
        start_swing=gen_config['start_swing'],
        end_swing=gen_config['end_swing'],
        sensors_to_process=sensors,
        stats_to_calculate=stats,
        extra_features_to_include=gen_config.get('extra_features_to_include', []),
        is_train_data=True,
        include_player_id_in_train=gen_config.get('include_player_id_in_train', True)
    )

    generate_features_module.process_data(
        info_path=paths_config['test_info'],
        data_dir=paths_config['test_data'],
        output_path=test_output_path,
        start_swing=gen_config['start_swing'],
        end_swing=gen_config['end_swing'],
        sensors_to_process=sensors,
        stats_to_calculate=stats,
        extra_features_to_include=gen_config.get('extra_features_to_include', []),
        is_train_data=False,
        include_player_id_in_train=False
    )

    print("--- Feature Generation Stage Completed ---")
    return train_output_path, test_output_path

def get_model_instance(model_config):
    """根據設定檔創建模型實例"""
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
        params.setdefault('use_label_encoder', False)
        if 'objective' not in params or 'multi' not in params['objective']:
             params.setdefault('eval_metric', 'logloss')
        else:
             params.setdefault('eval_metric', 'mlogloss')
        return XGBClassifier(**params)
    elif model_type == "svm":
        from sklearn.svm import SVC
        params.setdefault('probability', True)
        return SVC(**params)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        # CatBoost 可能需要特別處理 params (e.g., iterations)
        # 可以在這裡加入檢查或調整
        return CatBoostClassifier(**params)
    elif model_type == "logisticregression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_training_stage(config, train_feature_path):
    """執行模型訓練階段"""
    print("\n--- Running Training Stage ---")
    train_config = config['training']
    model_configs = train_config.get('models', {})
    selected_targets = [t for t in train_config.get('targets_to_train', []) if t in TARGETS_INFO]
    output_paths_config = config['output_paths']
    paths_config = config['paths']

    if not selected_targets:
        print("No targets specified for training in config. Skipping training stage.")
        return None

    if not train_feature_path or not os.path.exists(train_feature_path):
        print(f"Error: Training feature file not found at {train_feature_path}. Aborting training.")
        return None

    # 建立本次運行的輸出目錄
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = os.path.join(output_paths_config['model_runs'], now)
    try:
        os.makedirs(current_run_output_dir, exist_ok=True)
        print(f"Model outputs will be saved to: {current_run_output_dir}")
    except OSError as e:
        print(f"Error creating output directory {current_run_output_dir}: {e}")
        return None

    # 載入訓練特徵數據
    # print(f"Loading training features from: {train_feature_path}") # 移除
    try:
        df = pd.read_csv(train_feature_path)
    except Exception as e:
        print(f"Error loading training feature file {train_feature_path}: {e}")
        return None

    # 準備日誌記錄
    log_lines = []
    log_lines.append(f"Pipeline Run Time: {now}")
    log_lines.append(f"Training Feature Path: {os.path.basename(train_feature_path)}")
    log_lines.append(f"Output Directory: {current_run_output_dir}")

    # --- 在日誌中加入 Feature Generation 設定 ---
    log_lines.append("\n--- Feature Generation Configuration Used ---")
    feature_gen_config = config.get('feature_generation', {})
    for key, value in feature_gen_config.items():
        log_lines.append(f"  {key}: {value}")
    log_lines.append("-----------------------------------------")

    log_lines.append(f"\nFeature Scaling: {train_config.get('feature_scaling', 'none')}")
    log_lines.append(f"Trained Targets: {', '.join(selected_targets)}")


    # 載入原始 info 以便分割 player_id
    try:
        info_df = pd.read_csv(paths_config['train_info'])
    except FileNotFoundError:
        print(f"Error: Original train_info file not found at {paths_config['train_info']}. Cannot perform player-based split. Aborting training.")
        return None
    except Exception as e:
        print(f"Error reading train_info file {paths_config['train_info']}: {e}")
        return None

    # 針對每個目標進行訓練
    validation_results = {}
    for target in selected_targets:
        if target not in model_configs:
            print(f"Warning: Configuration for target '{target}' not found in training.models. Skipping target.")
            continue

        target_info = TARGETS_INFO[target]
        mode = target_info['type']
        y_col = target_info['col']
        current_model_config = model_configs[target]

        print(f"Training for target: {target} ({current_model_config['model_type']})")

        # 準備特徵 X 和標籤 y
        excluded_cols = ["unique_id"] + [t_info['col'] for t_info in TARGETS_INFO.values()]
        if config['feature_generation'].get('include_player_id_in_train', False) and 'player_id' in df.columns:
            excluded_cols.append('player_id')
        excluded_cols = [col for col in excluded_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in excluded_cols]

        if not feature_cols:
             print(f"Error: No feature columns found for target {target} after exclusions. Skipping.")
             continue

        X = df[feature_cols]
        if y_col not in df.columns:
             print(f"Error: Target column '{y_col}' not found in features file {train_feature_path}. Skipping target {target}.") # 保留錯誤
             continue
        y = df[y_col]

        # 標籤轉換
        original_min_label = None
        if mode == "binary" and y.min() == 1 and y.max() == 2:
            y = y - 1
        if mode == "multiclass":
            original_min_label = y.min()
            y = y - original_min_label

        # --- Player-based Stratified Split ---
        stratify_col = y_col
        if stratify_col not in info_df.columns:
             print(f"Error: Stratify column '{stratify_col}' not found in {paths_config['train_info']}. Aborting training for {target}.") # 保留錯誤
             continue

        player_df = info_df[['player_id', stratify_col]].drop_duplicates('player_id')
        try:
            # 檢查每個類別的玩家數量是否至少為 2
            class_counts = player_df[stratify_col].value_counts()
            if (class_counts < 2).any():
                 print(f"Warning: Target '{target}' has classes with less than 2 unique players for stratification ({class_counts[class_counts < 2].to_dict()}). Stratified split might fail or be unreliable. Trying without stratification for player split.")
                 player_train, player_val = train_test_split(
                     player_df, test_size=0.2, random_state=42 # 不進行分層
                 )
            else:
                player_train, player_val = train_test_split(
                    player_df, test_size=0.2, random_state=42, stratify=player_df[stratify_col]
                )
        except ValueError as e:
             print(f"Error during stratified split for target {target}: {e}") # 保留錯誤
             print("Skipping training for this target.") # 保留
             continue

        train_player_ids = set(player_train['player_id'])
        val_player_ids = set(player_val['player_id'])

        # 合併 player_id (如果需要)
        if 'player_id' not in df.columns:
            df_merged = df.merge(info_df[['unique_id', 'player_id']], on='unique_id', how='left')
            if df_merged['player_id'].isnull().any():
                 print(f"Warning: Some unique_ids could not be matched to player_ids. Excluding these rows.") # 保留警告
                 df_merged.dropna(subset=['player_id'], inplace=True)
        else:
            df_merged = df

        train_idx = df_merged[df_merged['player_id'].isin(train_player_ids)].index
        val_idx = df_merged[df_merged['player_id'].isin(val_player_ids)].index

        # 檢查是否有足夠的驗證數據
        if len(val_idx) == 0:
            print(f"Warning: No validation data generated for target {target} after player split. Skipping validation.") # 保留警告
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype) # 創建空的 DataFrame/Series
        else:
            X_train = X.loc[train_idx].reset_index(drop=True)
            X_val = X.loc[val_idx].reset_index(drop=True)
            y_train = y.loc[train_idx].reset_index(drop=True)
            y_val = y.loc[val_idx].reset_index(drop=True)

        # --- 特徵縮放 ---
        scaling_mode = train_config.get('feature_scaling', 'none').lower()
        scaler_path = None
        scaler_saved = False
        if scaling_mode in ["zscore", "minmax"]:
            scaler = StandardScaler() if scaling_mode == "zscore" else MinMaxScaler()
            try:
                X_train = scaler.fit_transform(X_train)
                if not X_val.empty:
                    X_val = scaler.transform(X_val)
                scaler_filename = f"{current_model_config['model_type']}_{target}_scaler.pkl"
                scaler_path = os.path.join(current_run_output_dir, scaler_filename)
                joblib.dump(scaler, scaler_path)
                scaler_saved = True
            except ValueError as ve: # Handle cases like only one sample
                 print(f"Warning: Feature scaling failed for target {target} (possibly due to single sample in train/val): {ve}. Skipping scaling.") # 保留警告
            except Exception as e:
                 print(f"Error during feature scaling for target {target}: {e}") # 保留錯誤
                 continue

        # --- 模型訓練 ---
        try:
            model = get_model_instance(current_model_config)
            log_lines.append("")
            log_lines.append(f"Target: {target}")
            log_lines.append(f"  Model Type: {current_model_config['model_type']}")
            log_lines.append(f"  Parameters: {current_model_config.get('params', {})}")
            if scaler_saved:
                log_lines.append(f"  Scaler Used: {scaling_mode} (Saved: {os.path.basename(scaler_path)})")
            else:
                log_lines.append(f"  Scaler Used: {scaling_mode}")

            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during model fitting for target {target}: {e}") # 保留錯誤
            log_lines.append(f"  Error during fitting: {e}")
            continue

        # --- 驗證與評估 ---
        auc = None
        if not y_val.empty:
            try:
                y_val_proba = model.predict_proba(X_val)
                if mode == "binary":
                    if y_val_proba.shape[1] >= 2:
                        auc = roc_auc_score(y_val, y_val_proba[:, 1])
                    else:
                        print(f"Warning: predict_proba for binary target {target} returned only one column. AUC might be inaccurate.") # 保留警告
                        auc = roc_auc_score(y_val, y_val_proba[:, 0])
                else:
                    auc = roc_auc_score(y_val, y_val_proba, average="micro", multi_class="ovr")
                print(f"Validation ROC AUC: {auc:.4f}")
                log_lines.append(f"  Validation ROC AUC: {auc:.4f}")
                validation_results[target] = auc
            except Exception as e:
                print(f"Error during validation/scoring for target {target}: {e}")
                log_lines.append(f"  Error during validation: {e}")
        else:
             log_lines.append("  Validation Skipped (empty validation set)")

        # --- 儲存模型 ---
        model_filename = f"{current_model_config['model_type']}_{target}.pkl"
        model_path = os.path.join(current_run_output_dir, model_filename)
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            print(f"Error saving model for target {target}: {e}") # 保留錯誤

    # --- 儲存運行日誌 ---
    log_path = os.path.join(current_run_output_dir, "log.txt")
    try:
        # --- 移除 AUC 摘要 ---
        # log_lines.append("\n--- Validation AUC Summary ---")
        # if validation_results:
        #     for target, auc in validation_results.items():
        #         log_lines.append(f"{target}: {auc:.4f}" if auc is not None else f"{target}: Error/Skipped")
        # else:
        #     log_lines.append("No validation results.")

        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))
        print(f"\nTraining log saved to: {log_path}")
    except Exception as e:
        print(f"Error saving log file: {e}")

    return current_run_output_dir

def run_prediction_stage(config, test_feature_path, model_run_dir_to_use):
    """執行預測與提交階段"""
    print("\n--- Running Prediction Stage ---")
    paths_config = config['paths']
    output_paths_config = config['output_paths']
    # 獲取設定檔中定義要訓練的目標 (用於檢查是否該搜尋模型)
    trained_targets_in_config = set(config.get('training', {}).get('targets_to_train', []))

    if not model_run_dir_to_use or not os.path.isdir(model_run_dir_to_use):
        print(f"Error: Model run directory '{model_run_dir_to_use}' not specified or not found. Aborting prediction.")
        return

    if not test_feature_path or not os.path.exists(test_feature_path):
        print(f"Error: Test feature file not found at {test_feature_path}. Aborting prediction.")
        return

    print(f"Using models from run: {os.path.basename(model_run_dir_to_use)}")

    # 載入測試特徵數據
    try:
        df_test = pd.read_csv(test_feature_path)
    except Exception as e:
        print(f"Error loading test feature file {test_feature_path}: {e}")
        return

    # 讀取 sample submission
    try:
        sample_sub = pd.read_csv(paths_config['sample_submission'], nrows=1)
        submission_columns = list(sample_sub.columns)
    except FileNotFoundError:
        print(f"Error: Sample submission file not found at {paths_config['sample_submission']}. Aborting prediction.")
        return
    except Exception as e:
        print(f"Error reading sample submission file {paths_config['sample_submission']}: {e}")
        return

    submit_df = pd.DataFrame()
    submit_df["unique_id"] = df_test["unique_id"]

    # --- 遍歷所有可能的目標進行預測 ---
    targets_predicted = 0
    for target, target_info in TARGETS_INFO.items():
        mode = target_info['type']
        y_col_original = target_info['col']

        if target not in trained_targets_in_config:
            # 直接填充預設值並繼續
            if mode == 'binary':
                 submit_df[y_col_original] = float("{:.4f}".format(0.5))
            elif target == 'play_years':
                 default_prob = float("{:.4f}".format(1/3))
                 for i in range(3): submit_df[f"play years_{i}"] = default_prob
            elif target == 'level':
                 default_prob = float("{:.4f}".format(0.25))
                 for i in range(2, 6): submit_df[f"level_{i}"] = default_prob
            continue # 繼續下一個 target

        model_path = None
        scaler_path = None
        model_name_prefix = None
        try:
            found_model = False
            for filename in os.listdir(model_run_dir_to_use):
                if filename.endswith(f"_{target}.pkl"):
                    model_path = os.path.join(model_run_dir_to_use, filename)
                    model_name_prefix = filename.replace(f"_{target}.pkl", "")
                    scaler_filename = f"{model_name_prefix}_{target}_scaler.pkl"
                    potential_scaler_path = os.path.join(model_run_dir_to_use, scaler_filename)
                    if os.path.exists(potential_scaler_path):
                        scaler_path = potential_scaler_path
                    found_model = True
                    break
            if not found_model:
                 print(f"Warning: Model file for target '{target}' (configured for training) not found in {model_run_dir_to_use}. Filling default probabilities.")
                 model_path = None
        except FileNotFoundError:
             print(f"Error accessing model run directory {model_run_dir_to_use}. Filling defaults for {target}.")
             model_path = None
        except Exception as e:
             print(f"Error listing files in {model_run_dir_to_use}: {e}. Filling defaults for {target}.")
             model_path = None

        # --- 如果找不到模型 (即使設定要訓練)，填充預設值並繼續 ---
        if not model_path:
            if mode == 'binary':
                 submit_df[y_col_original] = float("{:.4f}".format(0.5))
            elif target == 'play_years':
                 default_prob = float("{:.4f}".format(1/3))
                 for i in range(3): submit_df[f"play years_{i}"] = default_prob
            elif target == 'level':
                 default_prob = float("{:.4f}".format(0.25))
                 for i in range(2, 6): submit_df[f"level_{i}"] = default_prob
            continue

        # --- 載入模型和 Scaler ---
        try:
            model = joblib.load(model_path)
            scaler = None
            if scaler_path:
                scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading model or scaler for target {target}: {e}")
            print(f"Filling default probabilities for {target}")
            if mode == 'binary':
                 submit_df[y_col_original] = float("{:.4f}".format(0.5))
            elif target == 'play_years':
                 default_prob = float("{:.4f}".format(1/3))
                 for i in range(3): submit_df[f"play years_{i}"] = default_prob
            elif target == 'level':
                 default_prob = float("{:.4f}".format(0.25))
                 for i in range(2, 6): submit_df[f"level_{i}"] = default_prob
            continue

        # --- 準備預測數據 X_test ---
        excluded_cols = ["unique_id"] + [t_info['col'] for t_info in TARGETS_INFO.values() if t_info['col'] in df_test.columns]
        if 'player_id' in df_test.columns:
             excluded_cols.append('player_id')
        feature_cols = [col for col in df_test.columns if col not in excluded_cols]

        if not feature_cols:
            print(f"Error: No feature columns found for test data. Cannot predict for {target}.") # 保留錯誤
            if mode == 'binary': submit_df[y_col_original] = float("{:.4f}".format(0.5))
            elif target == 'play_years':
                default_prob = float("{:.4f}".format(1/3)); [submit_df.update({f"play years_{i}": default_prob}) for i in range(3)]
            elif target == 'level':
                default_prob = float("{:.4f}".format(0.25)); [submit_df.update({f"level_{i}": default_prob}) for i in range(2, 6)]
            continue

        X_test = df_test[feature_cols]

        # --- 特徵縮放 ---
        if scaler:
            try:
                X_test = scaler.transform(X_test)
            except ValueError as ve:
                 print(f"Error applying scaler to test data for target {target}: {ve}") # 保留錯誤
                 print("This might be due to a mismatch between training and test features.") # 保留提示
                 print(f"Filling default probabilities for {target}") # 保留提示
                 if mode == 'binary': submit_df[y_col_original] = float("{:.4f}".format(0.5))
                 elif target == 'play_years':
                     default_prob = float("{:.4f}".format(1/3)); [submit_df.update({f"play years_{i}": default_prob}) for i in range(3)]
                 elif target == 'level':
                     default_prob = float("{:.4f}".format(0.25)); [submit_df.update({f"level_{i}": default_prob}) for i in range(2, 6)]
                 continue
            except Exception as e:
                print(f"Unexpected error applying scaler to test data for target {target}: {e}") # 保留錯誤
                print(f"Filling default probabilities for {target}") # 保留提示
                if mode == 'binary': submit_df[y_col_original] = float("{:.4f}".format(0.5))
                elif target == 'play_years':
                    default_prob = float("{:.4f}".format(1/3)); [submit_df.update({f"play years_{i}": default_prob}) for i in range(3)]
                elif target == 'level':
                    default_prob = float("{:.4f}".format(0.25)); [submit_df.update({f"level_{i}": default_prob}) for i in range(2, 6)]
                continue

        # --- 進行預測 ---
        try:
            proba = model.predict_proba(X_test)
            targets_predicted += 1

            # --- 填充提交 DataFrame ---
            if mode == 'binary':
                if proba.shape[1] >= 1:
                    submit_df[y_col_original] = proba[:, 0] # 取索引 0 (通常對應原始的 1)
                else:
                    print(f"Warning: predict_proba for binary target {target} returned unexpected shape {proba.shape}. Filling with 0.5000.") # 保留警告
                    submit_df[y_col_original] = 0.5000
            elif target == 'play_years':
                if hasattr(model, 'classes_') and proba.shape[1] == len(model.classes_):
                     for i, class_label in enumerate(model.classes_):
                         submit_col_name = f"play years_{class_label}"
                         if submit_col_name in submission_columns:
                             submit_df[submit_col_name] = proba[:, i]
                else:
                    print(f"Warning: Cannot determine class order or probability shape mismatch for {target}. Filling defaults.") # 保留警告
                    default_prob = float("{:.4f}".format(1/3)); [submit_df.update({f"play years_{i}": default_prob}) for i in range(3)]
            elif target == 'level':
                if hasattr(model, 'classes_') and proba.shape[1] == len(model.classes_):
                    original_min_level = 2
                    for i, class_label_encoded in enumerate(model.classes_):
                         original_class_label = class_label_encoded + original_min_level
                         submit_col_name = f"level_{original_class_label}"
                         if submit_col_name in submission_columns:
                             submit_df[submit_col_name] = proba[:, i]
                else:
                    print(f"Warning: Cannot determine class order or probability shape mismatch for {target}. Filling defaults.") # 保留警告
                    default_prob = float("{:.4f}".format(0.25)); [submit_df.update({f"level_{i}": default_prob}) for i in range(2, 6)]

        except AttributeError as ae:
             print(f"Error: Model for {target} might not support predict_proba ({ae}). Filling defaults.") # 保留錯誤
             if mode == 'binary': submit_df[y_col_original] = float("{:.4f}".format(0.5))
             elif target == 'play_years':
                 default_prob = float("{:.4f}".format(1/3)); [submit_df.update({f"play years_{i}": default_prob}) for i in range(3)]
             elif target == 'level':
                 default_prob = float("{:.4f}".format(0.25)); [submit_df.update({f"level_{i}": default_prob}) for i in range(2, 6)]
             continue
        except Exception as e:
            print(f"Error during prediction for target {target}: {e}") # 保留錯誤
            print(f"Filling default probabilities for {target}") # 保留提示
            if mode == 'binary':
                 submit_df[y_col_original] = float("{:.4f}".format(0.5))
            elif target == 'play_years':
                 default_prob = float("{:.4f}".format(1/3)); [submit_df.update({f"play years_{i}": default_prob}) for i in range(3)]
            elif target == 'level':
                 default_prob = float("{:.4f}".format(0.25)); [submit_df.update({f"level_{i}": default_prob}) for i in range(2, 6)]
            continue


    # --- 檢查並填充缺失的提交欄位 ---
    missing_cols_filled = False
    for col in submission_columns:
        if col not in submit_df.columns:
            if not missing_cols_filled:
                print(f"Warning: Filling missing columns in submission DataFrame with defaults:") # 保留警告
                missing_cols_filled = True
            print(f"  - Filling '{col}'") # 保留填充的欄位名
            # 填充預設值 (使用格式化)
            if "gender" in col or "hold racket handed" in col: submit_df[col] = float("{:.4f}".format(0.5))
            elif "play_years" in col: submit_df[col] = float("{:.4f}".format(1/3))
            elif "level" in col: submit_df[col] = float("{:.4f}".format(0.25))
            elif col != "unique_id": submit_df[col] = 0.0000

    if targets_predicted == 0 and not missing_cols_filled:
        print("\nWarning: No targets were successfully predicted. Submission file might be incomplete.") # 保留警告

    # --- 儲存提交檔案 ---
    try:
        output_submission_dir = output_paths_config['submission']
        os.makedirs(output_submission_dir, exist_ok=True)
        model_run_name = os.path.basename(model_run_dir_to_use.rstrip('/'))
        submission_filename = f"submission_{model_run_name}.csv"
        submission_path = os.path.join(output_submission_dir, submission_filename)

        # 依照 sample submission 的欄位順序排列並儲存
        submit_df = submit_df[submission_columns]
        # 使用 float_format 控制輸出精度
        submit_df.to_csv(submission_path, index=False, float_format="%.4f")
        print(f"Submission file saved to: {submission_path}")
    except KeyError as ke:
         print(f"Error: Column '{ke}' not found when reordering submission columns. Check sample submission and predictions.") # 保留錯誤
    except Exception as e:
        print(f"Error saving submission file: {e}")


def main():
    """主流程函數"""
    # 1. 載入設定檔
    # 使用相對於此腳本的固定相對路徑載入 config
    script_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(script_dir, "../config/pipeline_config.yaml"))
    config, config_dir = load_pipeline_config(config_path)
    config = resolve_paths(config, config_dir) # 將相對路徑轉換為絕對路徑

    # --- 導入 generate_features ---
    generate_features_script_path = config.get('paths', {}).get('generate_features_script')
    generate_features_module = import_generate_features(generate_features_script_path)

    # --- 初始化階段性結果 ---
    train_feature_path = None
    test_feature_path = None
    model_run_dir = None

    # 2. (可選) 執行特徵生成
    if config.get('run_stages', {}).get('feature_generation', False):
        train_feature_path, test_feature_path = run_feature_generation_stage(config, generate_features_module)
        if train_feature_path is None or test_feature_path is None:
             print("Feature generation failed. Aborting pipeline.") # 保留錯誤
             sys.exit(1)
    else:
        print("\nSkipping Feature Generation Stage (as configured).") # 保留跳過訊息
        gen_config = config['feature_generation']
        output_feature_dir = config['output_paths']['generated_features']
        train_output_filename = f"features_train_s{gen_config['start_swing']}-e{gen_config['end_swing']}.csv"
        test_output_filename = f"features_test_s{gen_config['start_swing']}-e{gen_config['end_swing']}.csv"
        train_feature_path = os.path.join(output_feature_dir, train_output_filename)
        test_feature_path = os.path.join(output_feature_dir, test_output_filename)
        if not os.path.exists(train_feature_path) or not os.path.exists(test_feature_path):
            print(f"Error: Assumed feature files not found. Expected:") # 保留錯誤
            print(f"  - {train_feature_path}")
            print(f"  - {test_feature_path}")
            print("Please run the feature generation stage first or check paths.") # 保留提示
            sys.exit(1)


    # 3. (可選) 執行模型訓練
    if config.get('run_stages', {}).get('training', False):
        model_run_dir = run_training_stage(config, train_feature_path)
        if model_run_dir is None:
             print("Training stage failed or was skipped for all targets. Aborting pipeline if prediction is required.") # 保留錯誤/警告
             if config.get('run_stages', {}).get('prediction', False):
                 sys.exit(1)
    else:
        print("\nSkipping Training Stage (as configured).") # 保留跳過訊息
        model_run_name = config.get('prediction', {}).get('model_run_to_use', None)
        if model_run_name and model_run_name != "YYYYMMDD_HHMMSS":
            model_run_dir = os.path.join(config['output_paths']['model_runs'], model_run_name)
            print(f"Using pre-trained models from: {model_run_dir}") # 保留使用的模型目錄訊息
            if not os.path.isdir(model_run_dir):
                 print(f"Error: Specified model run directory '{model_run_dir}' does not exist. Aborting.") # 保留錯誤
                 sys.exit(1)
        else:
             if config.get('run_stages', {}).get('prediction', False):
                 print("Error: Training skipped and no valid 'model_run_to_use' specified in prediction config. Cannot run prediction.") # 保留錯誤
                 sys.exit(1)
             model_run_dir = None

    # 4. (可選) 執行預測與提交
    if config.get('run_stages', {}).get('prediction', False):
        if not model_run_dir:
             print("Error: Cannot run prediction stage because no valid model directory is available (either training failed or was skipped without specifying 'model_run_to_use').") # 保留錯誤
        elif not test_feature_path or not os.path.exists(test_feature_path):
             print(f"Error: Test features path '{test_feature_path}' not available or file does not exist. Cannot run prediction stage.") # 保留錯誤
        else:
            run_prediction_stage(config, test_feature_path, model_run_dir)
    else:
        print("\nSkipping Prediction Stage (as configured).") # 保留跳過訊息

    print("\nPipeline finished.") # 保留結束訊息

if __name__ == "__main__":
    main()
