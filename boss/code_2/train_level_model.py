import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
from scipy import stats
from scipy.signal import find_peaks
import argparse
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler, PowerTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy.signal import welch


def load_raw_data(file_path):
    """Load raw sensor data from text file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 6:  # Ensure we have 6 values (3 for accel, 3 for gyro)
                data.append(values)

    # Convert to numpy array
    data = np.array(data)
    return data

def _filter_features_by_model(features, model_name):
    if model_name == "svm_standard":
        FILTER_KEY = []
    elif model_name == 'svm_robust':
        FILTER_KEY = []
    elif model_name == 'svm_minmax':
        FILTER_KEY = []
    else:
        FILTER_KEY = []

    return features.drop(columns=FILTER_KEY, errors='ignore') # {i:features[i] for i in features if i not in FILTER_KEY}

def calculate_peak_features(acc_mag):
    """Calculate features related to peaks in the acceleration magnitude."""
    peaks, properties = find_peaks(acc_mag, height=0)
    if len(peaks) > 0:
        filter_peak_start = int(len(peaks) * 0.1)
        peak_heights = properties['peak_heights'][filter_peak_start:-filter_peak_start, ]
        return {
            # 'num_peaks': len(peaks),
            'mean_peak_height': np.mean(peak_heights),
            'max_peak_height': np.max(peak_heights),
            'min_peak_height': np.min(peak_heights),
            'std_peak_height': np.std(peak_heights),
            'range_peak_height': np.max(peak_heights) - np.min(peak_heights),
            'q25_peak_height': np.percentile(peak_heights, 25),
            'q75_peak_height': np.percentile(peak_heights, 75),
            'iqr_peak_height': np.percentile(peak_heights, 75) - np.percentile(peak_heights, 25)
        }
    return {
        # 'num_peaks': 0,
        'mean_peak_height': 0,
        'max_peak_height': 0,
        'min_peak_height': 0,
        'std_peak_height': 0,
        'range_peak_height': 0,
        'q25_peak_height': 0,
        'q75_peak_height': 0,
        'iqr_peak_height': 0
    }

def spectral_entropy(x, fs=85, nperseg=None):
    # x: 一維時序信號；fs: 取樣頻率
    f, Pxx = welch(x, fs=fs, nperseg=nperseg or len(x)//2)
    P = Pxx / np.sum(Pxx)           # 正規化
    P = P[P>0]                      # 去掉 0 項，避免 log(0)
    return -np.sum(P * np.log(P))   # 熵

def extract_swing_features(data, n_swings=27, fs=85):
    """把一筆含有 27 swings 的 data 分成 27 段，各自跑 extract_features，最後把 key 改名拼在一起。"""
    N = data.shape[0]
    window = N // n_swings
    all_feats = {}
    for i in range(n_swings):
        seg = data[i*window:(i+1)*window, :]
        seg_ax, seg_ay, seg_az = seg[:, 0], seg[:, 1], seg[:, 2]
        seg_acc_mag = np.sqrt(seg_ax ** 2 + seg_ay ** 2 + seg_az ** 2)
        feats = {
            # f'max_acc_{i}': np.max(seg_acc_mag),
            # f'mean_acc_{i}': np.mean(seg_acc_mag),
            # f'std_acc_{i}': np.std(seg_acc_mag),

            # Axis-specific features
            f'x_std_{i}': np.std(seg_ax),
            f'y_std_{i}': np.std(seg_ay),
            f'z_std_{i}': np.std(seg_az),
            f'x_mean_{i}': np.mean(np.abs(seg_ax)),
            f'y_mean_{i}': np.mean(np.abs(seg_ay)),
            f'z_mean_{i}': np.mean(np.abs(seg_az)),

            # Percentile features
            # f'acc_75th_{i}': np.percentile(seg_acc_mag, 75),
            # f'acc_25th_{i}': np.percentile(seg_acc_mag, 25),

            # Statistical features
            f'acc_skew_{i}': stats.skew(seg_acc_mag),
            f'acc_kurtosis_{i}': stats.kurtosis(seg_acc_mag),
            f'acc_rms_{i}': np.sqrt(np.mean(seg_acc_mag ** 2)),
            f'acc_iqr_{i}': np.percentile(seg_acc_mag, 75) - np.percentile(seg_acc_mag, 25),

            # Axis-specific statistical features
            f'x_skew_{i}': stats.skew(seg_ax),
            f'y_skew_{i}': stats.skew(seg_ay),
            f'z_skew_{i}': stats.skew(seg_az),
            f'x_kurtosis_{i}': stats.kurtosis(seg_ax),
            f'y_kurtosis_{i}': stats.kurtosis(seg_ay),
            f'z_kurtosis_{i}': stats.kurtosis(seg_az),

            # Correlation features
            # f'xy_corr_{i}': np.corrcoef(seg_ax, seg_ay)[0, 1],
            # f'xz_corr_{i}': np.corrcoef(seg_ax, seg_az)[0, 1],
            # f'yz_corr_{i}': np.corrcoef(seg_ay, seg_az)[0, 1],
        }
        all_feats = {**all_feats, **feats}

    return all_feats

def extract_features(data):
    """Extract significant features from the raw data."""
    if 1:
        swing_features = extract_swing_features(data)
    else:
        swing_features = {}
    # Calculate acceleration for each axis
    filtered_index = int(data.shape[0] * 0.1)
    ax, ay, az = data[filtered_index:-filtered_index, 0], data[filtered_index:-filtered_index, 1], data[filtered_index:-filtered_index, 2]
    gx, gy, gz = data[:, 3], data[:, 4], data[:, 5]

    # Calculate acceleration magnitude
    acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyr_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # Calculate features that showed significance in analysis
    features = {
        # Basic acceleration features
        'max_acc': np.max(acc_mag),
        'mean_acc': np.mean(acc_mag),
        'std_acc': np.std(acc_mag),

        # 'max_gyr_acc': np.max(gyr_mag),
        # 'mean_gyr_acc': np.mean(gyr_mag),
        # 'std_gyr_acc': np.std(gyr_mag),

        # Axis-specific features
        'x_std': np.std(ax),
        'y_std': np.std(ay),
        'z_std': np.std(az),
        'x_mean': np.mean(np.abs(ax)),
        'y_mean': np.mean(np.abs(ay)),
        'z_mean': np.mean(np.abs(az)),
        # 'ax_rms': np.sqrt(np.mean(ax ** 2)),
        # 'ay_rms': np.sqrt(np.mean(ay ** 2)),
        # 'az_rms': np.sqrt(np.mean(az ** 2)),

        # 'x_gyr_std': np.std(gx),
        # 'y_gyr_std': np.std(gy),
        # 'z_gyr_std': np.std(gz),
        # 'x_gyr_mean': np.mean(np.abs(gx)),
        # 'y_gyr_mean': np.mean(np.abs(gy)),
        # 'z_gyr_mean': np.mean(np.abs(gz)),

        # Range features
        # 'x_range': np.max(ax) - np.min(ax),
        # 'y_range': np.max(ay) - np.min(ay),
        # 'z_range': np.max(az) - np.min(az),

        # 'x_gry_range': np.max(gx) - np.min(gx),
        # 'y_gry_range': np.max(gy) - np.min(gy),
        # 'z_gry_range': np.max(gz) - np.min(gz),

        # Percentile features
        'acc_75th': np.percentile(acc_mag, 75),
        'acc_25th': np.percentile(acc_mag, 25),
        # 'acc_75th_ax': np.percentile(ax, 75),
        # 'acc_25th_ax': np.percentile(ax, 25),
        # 'acc_75th_ay': np.percentile(ay, 75),
        # 'acc_25th_ay': np.percentile(ay, 25),
        # 'acc_75th_az': np.percentile(az, 75),
        # 'acc_25th_az': np.percentile(az, 25),

        # 'acc_gry_75th': np.percentile(gyr_mag, 75),
        # 'acc_gry_25th': np.percentile(gyr_mag, 25),

        # Statistical features
        'acc_skew': stats.skew(acc_mag),
        'acc_kurtosis': stats.kurtosis(acc_mag),
        'acc_rms': np.sqrt(np.mean(acc_mag ** 2)),
        'acc_iqr': np.percentile(acc_mag, 75) - np.percentile(acc_mag, 25),

        # 'acc_gry_skew': stats.skew(gyr_mag),
        # 'acc_gry_kurtosis': stats.kurtosis(gyr_mag),
        # 'acc_gry_rms': np.sqrt(np.mean(gyr_mag ** 2)),
        # 'acc_gry_iqr': np.percentile(gyr_mag, 75) - np.percentile(gyr_mag, 25),

        # Axis-specific statistical features
        'x_skew': stats.skew(ax),
        'y_skew': stats.skew(ay),
        'z_skew': stats.skew(az),
        'x_kurtosis': stats.kurtosis(ax),
        'y_kurtosis': stats.kurtosis(ay),
        'z_kurtosis': stats.kurtosis(az),

        # 'x_gry_skew': stats.skew(gx),
        # 'y_gry_skew': stats.skew(gy),
        # 'z_gry_skew': stats.skew(gz),
        # 'x_gry_kurtosis': stats.kurtosis(gx),
        # 'y_gry_kurtosis': stats.kurtosis(gy),
        # 'z_gry_kurtosis': stats.kurtosis(gz),

        # Correlation features
        'xy_corr': np.corrcoef(ax, ay)[0, 1],
        'xz_corr': np.corrcoef(ax, az)[0, 1],
        'yz_corr': np.corrcoef(ay, az)[0, 1],

        # 'xy_gry_corr': np.corrcoef(gx, gy)[0, 1],
        # 'xz_gry_corr': np.corrcoef(gx, gz)[0, 1],
        # 'yz_gry_corr': np.corrcoef(gy, gz)[0, 1]
    }
    dt = 1/85.0
    # jerk = np.diff(acc_mag)/dt
    # features.update({
    #                 'mean_jerk': np.mean(jerk),
    #                 'std_jerk': np.std(jerk),
    #                 'max_jerk': np.max(jerk)
    # })

    # spectral features
    fft_vals = np.abs(np.fft.rfft(acc_mag)) ** 2
    freqs = np.fft.rfftfreq(len(acc_mag), d=dt)
    total = np.sum(fft_vals)
    for low, high in [(0, 5), (5, 15), (15, 30)]:
        mask = (freqs >= low) & (freqs < high)
        features[f'fft_acc_{low}_{high}'] = np.sum(fft_vals[mask]) / total
    # features['acc_dom_freq'] = freqs[np.argmax(fft_vals)]
    # features['acc_spec_entropy'] = spectral_entropy(acc_mag, fs=85)

    # Add peak acceleration features
    # peak_features = calculate_peak_features(acc_mag)
    # features.update(peak_features)

    # Add peak angle acceleration features
    # gyr_peak = calculate_peak_features(gyr_mag)
    # for k, v in gyr_peak.items():
    #     features[f'gyr_{k}'] = v
    features = {**features, **swing_features}
    return features


def process_training_data(train_data_dir, train_info_path, mode_filter):
    """Process training data and prepare for model training."""
    # Load training info
    train_info = pd.read_csv(train_info_path)

    # Filter for specified modes
    if isinstance(mode_filter, list):
        mode_info = train_info[train_info['mode'].isin(mode_filter)].copy()
    else:
        mode_info = train_info[train_info['mode'] == mode_filter].copy()

    # Process each training file
    features_list = []
    level_list = []

    for _, row in mode_info.iterrows():
        file_path = os.path.join(train_data_dir, f"{row['unique_id']}.txt")
        if os.path.exists(file_path):
            # Load and process data
            data = load_raw_data(file_path)
            features = extract_features(data)
            features_list.append(features)
            level_list.append(row['level'])

    # Convert to DataFrame
    X_train = pd.DataFrame(features_list)
    y_train = np.array(level_list)

    return X_train, y_train


def process_test_data(test_data_dir, test_info_path, mode_filter):
    """Process test data and prepare for prediction."""
    # Load test info
    test_info = pd.read_csv(test_info_path)

    # Filter for specified modes
    if isinstance(mode_filter, list):
        mode_info = test_info[test_info['mode'].isin(mode_filter)]
    else:
        mode_info = test_info[test_info['mode'] == mode_filter]

    features_list = []
    file_ids = []

    # Process each test file
    for _, row in mode_info.iterrows():
        filename = f"{row['unique_id']}.txt"
        file_path = os.path.join(test_data_dir, filename)

        if os.path.exists(file_path):
            # Load and process data
            data = load_raw_data(file_path)
            features = extract_features(data)
            features_list.append(features)
            file_ids.append(row['unique_id'])

    # Convert to DataFrame
    X_test = pd.DataFrame(features_list)

    return X_test, file_ids


def get_model(model_name, random_state=42):
    """Get the specified model with default parameters.

    Args:
        model_name: Name of the model ('random_forest', 'xgboost', 'lightgbm', 'catboost', or 'stacking')
        random_state: Random state for reproducibility

    Returns:
        Model instance
    """
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == 'xgboost':
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    elif model_name == 'lightgbm':
        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            min_child_samples=5,  # Prevent overfitting
            min_child_weight=1,  # Prevent overfitting
            num_leaves=31,  # Control tree complexity
            verbose=-1  # Suppress warnings
        )
    elif model_name == 'catboost':
        return CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_seed=random_state,
            verbose=False
        )
    elif model_name == 'svm': # 'svm_robust', 'svm_minmax', 'svm_standard'
        # 以 Pipeline 包裝標準化與 SVM
        return Pipeline([
            ('scaler', RobustScaler()),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=1,
                random_state=random_state
            ))
        ])
    elif model_name == 'svm_robust': # 'svm_robust', 'svm_minmax', 'svm_standard'
        # 以 Pipeline 包裝標準化與 SVM
        return Pipeline([
            ('scaler', RobustScaler()),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=1,
                random_state=random_state
            ))
        ])
    elif model_name == 'svm_minmax': # 'svm_robust', 'svm_minmax', 'svm_standard'
        # 以 Pipeline 包裝標準化與 SVM
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=1,
                random_state=random_state
            ))
        ])
    elif model_name == 'svm_standard': # 'svm_robust', 'svm_minmax', 'svm_standard'
        # 以 Pipeline 包裝標準化與 SVM
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=1,
                random_state=random_state
            ))
        ])
    elif model_name == 'stacking':
        # Create base models with better parameters
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                random_state=random_state
            )),
            ('xgb', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                random_state=random_state
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_samples=5,
                num_leaves=31,
                random_state=random_state,
                verbose=-1
            ))
        ]
        # Create stacking classifier with CatBoost as final estimator
        return StackingClassifier(
            estimators=estimators,
            final_estimator=CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=5,
                random_seed=random_state,
                verbose=False
            ),
            cv=5,
            n_jobs=-1  # Use all available cores
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_feature_importance(model, feature_names):
    """Get feature importance from different model types."""
    if isinstance(model, StackingClassifier):
        # For stacking, only use base models' feature importances
        importances = np.zeros(len(feature_names))
        n_models = 0

        # Get importances from base models
        for name, base_model in model.named_estimators_.items():
            if hasattr(base_model, 'feature_importances_'):
                # Ensure the importances array has the correct shape
                if len(base_model.feature_importances_) == len(feature_names):
                    importances += base_model.feature_importances_
                    n_models += 1

        # Average the importances
        if n_models > 0:
            importances = importances / n_models
        return importances
    elif hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    else:
        return np.zeros(len(feature_names))


def train_and_collect(model_names, X_train, y_train):
    """
    對每個 model_names 依序 train，再把 predict_proba 堆疊回傳
    回傳形狀 (n_models, n_samples, n_classes)
    """
    trained_model = {}
    for name in model_names:
        training_features = _filter_features_by_model(X_train, name)
        model = get_model(name)
        print(f"training_features shape for ensemble mode: {training_features.shape}")
        model.fit(training_features, y_train)
        trained_model[name] = model
    return trained_model


def train_and_predict_for_mode(mode_filter, model_name, eval_mode=True, model_names=['random_forest']):
    """Train model and make predictions for specific mode(s)

    Args:
        mode_filter: List of modes or single mode to filter
        model_name: Name of the model for output files
        eval_mode: If True, use 80/20 train/val split. If False, use all data for training
        model_type: Type of model to use ('random_forest' or 'xgboost')
    """
    # Define paths
    train_data_dir = 'Original/train/train_data'
    train_info_path = 'Original/train/train_info.csv'
    test_data_dir = 'Original/test/test_data'
    test_info_path = 'Original/test/test_info.csv'

    # Process training data
    print(f"Processing training data for {model_name}...")
    X_train_full, y_train_full = process_training_data(train_data_dir, train_info_path, mode_filter)

    if eval_mode:
        # Split into train and validation sets (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full  # Ensure balanced split across classes
        )
    else:
        # Use all data for training
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = None, None

    if len(model_names) == 1:
        # Get and train model
        print(f"Training model for {model_name}...")
        training_features = _filter_features_by_model(X_train, model_names[0])
        print(f"Training features shape for one model mode: {training_features.shape}")
        model = get_model(model_names[0])
        model.fit(training_features, y_train)

        # Calculate validation score if in eval mode
        val_auc = None
        if eval_mode:
            eval_features = _filter_features_by_model(X_val, model_names[0])
            val_probs = model.predict_proba(eval_features)
            val_auc = roc_auc_score(y_val, val_probs, multi_class='ovr', average='micro')
            print(f"\nValidation Micro-averaged One-vs-Rest ROC AUC for {model_name}: {val_auc:.4f}")

        # Process test data
        print(f"Processing test data for {model_name}...")
        X_test, file_ids = process_test_data(test_data_dir, test_info_path, mode_filter)
        testing_features = _filter_features_by_model(X_test, model_names[0])

        # Make probability predictions
        print(f"Making predictions for {model_name}...")
        probabilities = model.predict_proba(testing_features)
    else:
        # Process test data
        print(f"Processing test data for {model_name}...")
        X_test, file_ids = process_test_data(test_data_dir, test_info_path, mode_filter)

        trained_models = train_and_collect(model_names, X_train, y_train)
        val_auc = None
        val_probs_list = []
        test_probs_list = []

        for model_name in trained_models:
            model = trained_models[model_name]
            if eval_mode:
                eval_features = _filter_features_by_model(X_val, model_name)
                val_probs = model.predict_proba(eval_features)
                val_probs_list.append(val_probs)
            testing_features = _filter_features_by_model(X_test, model_name)
            print(f"Testing features shape for {model_name}: {testing_features.shape}")
            probabilities = model.predict_proba(testing_features)
            test_probs_list.append(probabilities)
        if eval_mode:
            val_stack = np.stack(val_probs_list, axis=0)
            avg_val = np.mean(val_stack, axis=0)
            val_auc = roc_auc_score(y_val, avg_val, multi_class='ovr', average='micro')
            print(f"Ensemble Validation AUC ({','.join(model_names)}): {val_auc:.4f}")
        test_stack = np.stack(test_probs_list, axis=0)
        probabilities = np.mean(test_stack, axis=0)

    # Create predictions DataFrame
    results_df = pd.DataFrame({
        'unique_id': file_ids,
        'level_2': probabilities[:, 0],
        'level_3': probabilities[:, 1],
        'level_4': probabilities[:, 2],
        'level_5': probabilities[:, 3]
    })

    # Sort by unique_id
    results_df = results_df.sort_values('unique_id')

    # Print feature importances
    feature_names = X_train.columns
    importances = get_feature_importance(model, feature_names)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Only print if we have valid importances
    if np.any(importances > 0):
        print(f"\nTop 10 Most Important Features for {model_name}:")
        print(feature_importance.head(10))
    else:
        print(f"\nFeature importance not available for {model_name}")
    imp_vals = importances
    imp_names = feature_names
    idx = np.argsort(imp_vals)[::-1]
    plt.figure(figsize=(10, 8))
    plt.bar(np.array(imp_names)[idx][-20:], imp_vals[idx][-20:])
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.savefig(f'{str(mode_filter)}_{model_name}_feature_importance.png')

    return results_df, val_auc


def combine_predictions(eval_mode=True, model_list=['random_forest']):
    """Combine predictions from all models into a single file

    Args:
        eval_mode: If True, use 80/20 train/val split. If False, use all data for training
        model_type: Type of model to use ('random_forest' or 'xgboost')
    """
    # Train and predict for each mode group
    print("Processing mode 0-8...")
    pred_0_8, val_auc_0_8 = train_and_predict_for_mode(list(range(9)), "mode_0_8", eval_mode, model_list)

    print("\nProcessing mode 9...")
    pred_9, val_auc_9 = train_and_predict_for_mode(list(range(9, 11)), "mode_9", eval_mode, model_list)

    # print("\nProcessing mode 10...")
    # pred_10, val_auc_10 = train_and_predict_for_mode(10, "mode_10", eval_mode, model_type)

    # Print overall validation scores if in eval mode
    # if eval_mode:
    #     print("\nOverall Validation Scores:")
    #     print(f"Mode 0-8: {val_auc_0_8:.4f}")
    #     print(f"Mode 9: {val_auc_9:.4f}")
        # print(f"Mode 10: {val_auc_10:.4f}")
        # print(f"Average: {(val_auc_0_8 + val_auc_9 + val_auc_10) / 3:.4f}")

    # Combine all predictions
    all_predictions = pd.concat([pred_0_8, pred_9]) # , pred_10
    all_predictions = all_predictions.sort_values('unique_id')

    # Save combined predictions
    output_path = 'boss/submissions_final/levels_predictions.csv'
    all_predictions.to_csv(output_path, index=False)
    print(f"\nCombined predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict play years')
    parser.add_argument('--mode', type=str, choices=['all', 'eval'], default='eval',
                        help='Mode to run in: "all" for no evaluation split, "eval" for 80/20 split')
    parser.add_argument('--model', type=str, nargs='+',
                        choices=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'stacking', 'svm',
                                 'svm_robust', 'svm_minmax', 'svm_standard'],
                        default=['random_forest'],
                        help='Model to use: random_forest, xgboost, lightgbm, catboost, or stacking')
    args = parser.parse_args()

    eval_mode = args.mode == 'eval'
    model_list = args.model
    combine_predictions(eval_mode=eval_mode, model_list=model_list)
    # python train_level_model.py --model svm --mode all

    # tbrain -> single -> baseline
    # 0.58412221 -> 0.83648884

"""
/opt/anaconda3/envs/aicup/bin/python /Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/boss/code_2/train_level_model.py  --model svm --mode all
/opt/anaconda3/envs/aicup/bin/python /Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/boss/code_2/train_level_model.py  --model svm_robust svm_standard  --mode all
/opt/anaconda3/envs/aicup/bin/python /Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/boss/code_2/train_level_model.py  --model svm_robust svm_standard  --mode eval
"""