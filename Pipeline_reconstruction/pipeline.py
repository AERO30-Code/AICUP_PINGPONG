import os
import shutil
import yaml
from datetime import datetime
import subprocess


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config_path, config):
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_configs(source_paths, dest_dir):
    ensure_dir(dest_dir)
    for path in source_paths:
        if os.path.exists(path):
            shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))

def run_python_module(module_path):
    print(f"Running: {module_path}")
    subprocess.run(["python", module_path], check=True)

def run_pipeline(pipeline_config_path):
    config = load_config(pipeline_config_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join("Pipeline_reconstruction/outputs/runs", timestamp)
    ensure_dir(run_dir)

    # snapshot configs
    config_snapshot_dir = os.path.join(run_dir, "configs")
    copy_configs([
        config['paths']['features_config'],
        config['paths']['selection_config'],
        config['paths']['training_config'],
        config['paths']['predict_config']
    ], config_snapshot_dir)

    # === Stage: Feature Generation ===
    if config['run_stages'].get('feature_generation', False):
        print("[Stage] Generating features...")
        run_python_module("Pipeline_reconstruction/features_generate.py")

    # === Stage: Feature Selection ===
    if config['run_stages'].get('feature_selection', False):
        print("[Stage] Selecting features...")
        run_python_module("Pipeline_reconstruction/features_selection.py")

    # === Stage: Model Training ===
    model_timestamp = None
    if config['run_stages'].get('training', False):
        print("[Stage] Training models...")
        run_python_module("Pipeline_reconstruction/model_train.py")

        # 尋找最新的模型輸出資料夾
        model_base_dir = config['paths']['model_base_dir']
        all_subdirs = [d for d in os.listdir(model_base_dir) if os.path.isdir(os.path.join(model_base_dir, d))]
        latest = sorted(all_subdirs)[-1]  # 時間排序取最新
        model_timestamp = latest
        print(f"Detected latest model output folder: {model_timestamp}")

        # 寫入 predict_config.yaml 中的 input.model_dir
        predict_config_path = config['paths']['predict_config']
        predict_config = load_config(predict_config_path)
        predict_config['input']['model_dir'] = os.path.join(model_base_dir, model_timestamp)
        save_config(predict_config_path, predict_config)
        print(f"Updated predict_config.yaml with model_dir: {predict_config['input']['model_dir']}")

    # === Stage: Prediction ===
    if config['run_stages'].get('prediction', False):
        print("[Stage] Generating submission...")
        run_python_module("Pipeline_reconstruction/model_predict.py")

    print(f"\nPipeline run complete. All outputs saved under: {run_dir}")

if __name__ == "__main__":
    PIPELINE_CONFIG_PATH = "Pipeline_reconstruction/configs/pipeline_config.yaml"
    run_pipeline(PIPELINE_CONFIG_PATH)