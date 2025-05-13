import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_instance(model_type: str, params: Dict[str, Any]):
    model_type = model_type.lower()
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        params.setdefault('use_label_encoder', False)
        params.setdefault('eval_metric', 'logloss')
        return XGBClassifier(**params)
    elif model_type == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
    else:
        return pd.DataFrame({
            'feature': feature_names,
            'importance': np.zeros(len(feature_names))
        })

def select_features(config_path: str):
    config = load_config(config_path)

    feature_df = pd.read_csv(config['input']['feature_csv'])
    target = config['target']

    if target not in feature_df.columns:
        print(f"Error: target '{target}' not found in feature file. Please ensure it's included.")
        return

    X = feature_df.drop(columns=["unique_id", target], errors='ignore')
    excluded_targets = ["gender", "play years", "level", "hold racket handed"]
    X = X.drop(columns=[col for col in excluded_targets if col in X.columns], errors='ignore')
    y = feature_df[target]

    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)
    elif y.min() == 1 and y.max() == 2:
        y = y - 1

    importance_dfs = []
    for name, model_cfg in config['models'].items():
        print(f"Training model: {name} ({model_cfg['model_type']})")
        model = get_model_instance(model_cfg['model_type'], model_cfg['params'])
        try:
            model.fit(X, y)
            imp_df = get_feature_importance(model, X.columns.tolist())
            imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
            imp_df.rename(columns={'importance': f"importance_{name}"}, inplace=True)
            importance_dfs.append(imp_df)
        except Exception as e:
            print(f"Error training model {name}: {e}")

    if not importance_dfs:
        print("Error: No feature importance could be computed.")
        return

    all_features = set(X.columns)
    merged_df = pd.DataFrame({'feature': list(all_features)})

    for imp_df in importance_dfs:
        merged_df = pd.merge(merged_df, imp_df, on='feature', how='left')

    rank_cols = []
    for col in merged_df.columns:
        if col.startswith('importance_'):
            rank_col = col.replace('importance', 'rank')
            merged_df[rank_col] = merged_df[col].rank(ascending=False)
            rank_cols.append(rank_col)

    merged_df['mean_rank'] = merged_df[rank_cols].mean(axis=1)

    merged_df = merged_df[[col for col in merged_df.columns if not col.startswith('importance_')]]
    merged_df = merged_df.sort_values('mean_rank').reset_index(drop=True)

    output_path = config['output']['rank_csv']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Feature ranking saved to: {output_path}")

if __name__ == "__main__":
    CONFIG_PATH = "Pipeline_reconstruction/configs/features_selection_config.yaml"
    select_features(CONFIG_PATH)
