import os
import yaml
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file and return as dictionary."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_instance(model_type: str, params: Dict[str, Any]):
    """Instantiate ML model by type and params from config."""
    model_type = model_type.lower()
    if model_type == 'svm':
        return SVC(**params)
    elif model_type == 'logisticregression':
        return LogisticRegression(**params)
    elif model_type == 'randomforest':
        return RandomForestClassifier(**params)
    elif model_type == 'xgboost':
        return XGBClassifier(**params)
    elif model_type == 'lightgbm':
        return LGBMClassifier(**params)
    elif model_type == 'catboost':
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_scaler(method: str):
    """Return scaler instance based on scaling method string."""
    if method == 'zscore':
        return StandardScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    elif method == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

def train_models(config_path: str):
    """Train model(s) based on configuration file.

    Supports multiple targets and models. Saves trained models and scalers.
    Automatically creates timestamped output subfolder to avoid overwrite.
    """
    config = load_config(config_path)
    df = pd.read_csv(config['input']['feature_csv'])

    if config['input'].get('use_top_n', False):
        rank_df = pd.read_csv(config['input']['rank_csv'])
        selected_features = rank_df['feature'].values[:config['input']['top_n']].tolist()
    else:
        selected_features = [col for col in df.columns if col not in ['unique_id', 'mode', 'player_id',
                                                                       'gender', 'hold racket handed',
                                                                       'play years', 'level']]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = config['output']['model_dir']
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    for target in config['training']['targets_to_train']:
        print(f"Training target: {target}")
        if target not in df.columns:
            print(f"  Skipped. Target column '{target}' not found in feature file.")
            continue

        X = df[selected_features].copy()
        y = df[target]

        if y.dtype == object or y.dtype.name == 'category':
            y = LabelEncoder().fit_transform(y)
        elif y.min() == 1 and y.max() == 2:
            y = y - 1

        scaler = get_scaler(config['training'].get('feature_scaling', 'none'))
        if scaler is not None:
            X = scaler.fit_transform(X)

        model_cfg = config['training']['models'].get(target, None)
        if model_cfg is None:
            print(f"  Skipped. No model defined for target '{target}' in config.")
            continue

        model = get_model_instance(model_cfg['model_type'], model_cfg['params'])

        try:
            model.fit(X, y)
            model_path = os.path.join(output_dir, f"{model_cfg['model_type']}_{target}.pkl")
            joblib.dump(model, model_path)
            print(f"  Model saved to {model_path}")

            if scaler is not None:
                scaler_path = os.path.join(output_dir, f"{model_cfg['model_type']}_{target}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
                print(f"  Scaler saved to {scaler_path}")
        except Exception as e:
            print(f"  Error training {target}: {e}")

if __name__ == "__main__":
    CONFIG_PATH = "Pipeline_reconstruction/configs/training_config.yaml"
    train_models(CONFIG_PATH)
    