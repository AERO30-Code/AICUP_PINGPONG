import os
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_scaler(path: str):
    return joblib.load(path) if os.path.exists(path) else None

def load_model(path: str):
    return joblib.load(path) if os.path.exists(path) else None

def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(X)
    except:
        return None

def run_prediction(config_path: str):
    """Main entry to perform prediction and generate submission."""
    config = load_config(config_path)

    test_df = pd.read_csv(config['input']['test_feature_csv'])
    sample_sub = pd.read_csv(config['input']['sample_submission_csv'])
    output_cols = sample_sub.columns.tolist()
    submission = pd.DataFrame({'unique_id': test_df['unique_id']})

    if config['prediction'].get('use_top_n', False):
        rank_df = pd.read_csv(config['input']['rank_csv'])
        selected_features = rank_df['feature'].values[:config['prediction']['top_n']].tolist()
    else:
        exclude = ["unique_id", "mode", "player_id", "gender", "hold racket handed", "play years", "level"]
        selected_features = [col for col in test_df.columns if col not in exclude]

    test_X = test_df[selected_features].copy()
    predicted_columns = set()

    all_targets = set(config.get('defaults', {}).keys()).union(config.get('targets', {}).keys())

    for target in all_targets:
        meta = config['targets'].get(target, {})
        model_type = meta.get('model_type')
        output_format = meta.get('output_columns')

        # Handle defaults
        default_vals = config['defaults'].get(target, {})
        if not isinstance(default_vals, dict):
            default_vals = {}

        if not output_format:
            output_format = list(default_vals.keys())

        for col in output_format:
            submission[col] = default_vals.get(col, 0.0)

        if not meta:
            print(f"Info: No prediction configured for '{target}'. Using defaults only.")
            continue

        model_path = os.path.join(config['input']['model_dir'], f"{model_type}_{target}.pkl")
        scaler_path = os.path.join(config['input']['model_dir'], f"{model_type}_{target}_scaler.pkl")

        if not os.path.exists(model_path):
            print(f"Warning: Model for {target} not found. Using default values.")
            continue

        model = load_model(model_path)
        scaler = load_scaler(scaler_path)
        X_input = scaler.transform(test_X) if scaler else test_X.values

        probs = predict_model(model, X_input)
        if probs is None:
            print(f"Warning: Prediction failed for {target}. Using default values.")
            continue

        if probs.ndim == 1 or probs.shape[1] == 1:
            submission[output_format[0]] = np.round(probs if probs.ndim == 1 else probs[:, 0], 4)
            predicted_columns.add(output_format[0])
        elif len(output_format) == probs.shape[1]:
            for i, col in enumerate(output_format):
                submission[col] = np.round(probs[:, i], 4)
                predicted_columns.add(col)
        elif probs.shape[1] == 2 and len(output_format) == 1:
            submission[output_format[0]] = np.round(probs[:, 0], 4)
            predicted_columns.add(output_format[0])
        else:
            print(f"Warning: Output shape mismatch for {target}. Using default values.")

    # Ensure all required output columns exist
    for col in output_cols:
        if col not in submission.columns:
            print(f"Warning: Column '{col}' missing in submission. Filling with 0.0.")
            submission[col] = 0.0

    submission = submission[output_cols]

    # Construct dynamic filename
    model_dir = config['input']['model_dir']
    run_id = os.path.basename(model_dir.rstrip('/'))
    output_name = f"submission_{run_id}.csv"
    output_path = os.path.join(config['output']['submission_dir'], output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")

if __name__ == "__main__":
    CONFIG_PATH = "Pipeline_reconstruction/configs/predict_config.yaml"
    run_prediction(CONFIG_PATH)