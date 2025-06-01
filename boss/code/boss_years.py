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


def extract_features(data):
    """Extract significant features from the raw data."""
    # Calculate acceleration for each axis
    ax, ay, az = data[:, 0], data[:, 1], data[:, 2]
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

    # Add peak acceleration features
    peak_features = calculate_peak_features(acc_mag)
    features.update(peak_features)

    # Add peak angle acceleration features
    # gyr_peak = calculate_peak_features(gyr_mag)
    # for k, v in gyr_peak.items():
    #     features[f'gyr_{k}'] = v

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
    play_years_list = []

    for _, row in mode_info.iterrows():
        file_path = os.path.join(train_data_dir, f"{row['unique_id']}.txt")
        if os.path.exists(file_path):
            # Load and process data
            data = load_raw_data(file_path)
            features = extract_features(data)
            features_list.append(features)
            play_years_list.append(row['play years'])

    # Convert to DataFrame
    X_train = pd.DataFrame(features_list)
    y_train = np.array(play_years_list)

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
    elif model_name == 'svm':
        # 以 Pipeline 包裝標準化與 SVM
        return Pipeline([
            ('scaler', PowerTransformer()),
            ('svm', SVC(
                probability=True,
                kernel='rbf',
                C=0.1,
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


def train_and_predict_for_mode(mode_filter, model_name, eval_mode=True, model_type='random_forest'):
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

    # Get and train model
    print(f"Training {model_type} model for {model_name}...")
    model = get_model(model_type)
    model.fit(X_train, y_train)

    # Calculate validation score if in eval mode
    val_auc = None
    if eval_mode:
        val_probs = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_probs, multi_class='ovr', average='micro')
        print(f"\nValidation Micro-averaged One-vs-Rest ROC AUC for {model_name}: {val_auc:.4f}")

    # Process test data
    print(f"Processing test data for {model_name}...")
    X_test, file_ids = process_test_data(test_data_dir, test_info_path, mode_filter)

    # Make probability predictions
    print(f"Making predictions for {model_name}...")
    probabilities = model.predict_proba(X_test)

    # Create predictions DataFrame
    results_df = pd.DataFrame({
        'unique_id': file_ids,
        'play years_0': probabilities[:, 0],
        'play years_1': probabilities[:, 1],
        'play years_2': probabilities[:, 2]
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


def combine_predictions(eval_mode=True, model_type='random_forest'):
    """Combine predictions from all models into a single file

    Args:
        eval_mode: If True, use 80/20 train/val split. If False, use all data for training
        model_type: Type of model to use ('random_forest' or 'xgboost')
    """
    # Train and predict for each mode group
    print("Processing mode 0-8...")
    pred_0_8, val_auc_0_8 = train_and_predict_for_mode(list(range(9)), "mode_0_8", eval_mode, model_type)

    print("\nProcessing mode 9...")
    pred_9, val_auc_9 = train_and_predict_for_mode(9, "mode_9", eval_mode, model_type)

    print("\nProcessing mode 10...")
    pred_10, val_auc_10 = train_and_predict_for_mode(10, "mode_10", eval_mode, model_type)

    # Print overall validation scores if in eval mode
    if eval_mode:
        print("\nOverall Validation Scores:")
        print(f"Mode 0-8: {val_auc_0_8:.4f}")
        print(f"Mode 9: {val_auc_9:.4f}")
        print(f"Mode 10: {val_auc_10:.4f}")
        print(f"Average: {(val_auc_0_8 + val_auc_9 + val_auc_10) / 3:.4f}")

    # Combine all predictions
    all_predictions = pd.concat([pred_0_8, pred_9, pred_10])
    all_predictions = all_predictions.sort_values('unique_id')

    # Save combined predictions
    output_path = 'boss/submissions_final/play_years_predictions.csv'
    all_predictions.to_csv(output_path, index=False)
    print(f"\nCombined predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict play years')
    parser.add_argument('--mode', type=str, choices=['all', 'eval'], default='eval',
                        help='Mode to run in: "all" for no evaluation split, "eval" for 80/20 split')
    parser.add_argument('--model', type=str,
                        choices=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'stacking', 'svm'],
                        default='random_forest',
                        help='Model to use: random_forest, xgboost, lightgbm, catboost, or stacking')
    args = parser.parse_args()

    eval_mode = args.mode == 'eval'
    combine_predictions(eval_mode=eval_mode, model_type=args.model)

    # Using xgboost
    # 80/20
    # Overall Validation Scores:
    # Mode 0-8: 0.9377
    # Mode 9: 0.9982
    # Mode 10: 0.9854
    # Average: 0.9738

    # all
    # 0.55271058 (random + pred play years)

    # Using random_forest
    # 80/20
    # Overall Validation Scores:
    # Mode 0-8: 0.9451
    # Mode 9: 0.9951
    # Mode 10: 0.9883
    # Average: 0.9762
    #

    # all
    # 0.55386613 (random + pred play years)
    # 0.55529738 (from last 80 % data)

    # Using svm
    # 0.55999280

    # Using lightgbm
    # 80/20
    # Overall Validation Scores:
    # Mode 0-8: 0.9426
    # Mode 9: 0.9986
    # Mode 10: 0.9886
    # Average: 0.9766

    # all   / 80/20
    # 0.5500/ 0.5517

    # Using catboost
    # 80/20
    # Overall Validation Scores:
    # Mode 0-8: 0.9632
    # Mode 9: 0.9934
    # Mode 10: 0.9771
    # Average: 0.9779

    # all
    # 0.5507

    # Using stacking
    # 80/20
    # Overall Validation Scores:
    # Mode 0-8: 0.9339
    # Mode 9: 0.9992
    # Mode 10: 0.9845
    # Average: 0.9726

    # all
    # 0.53832828

    # score + 0.2853137 = best
    # (score - 0.375) * 4 = this spec score


"""
/opt/anaconda3/envs/aicup/bin/python /Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/boss/code/boss_years.py --model svm --mode all
"""