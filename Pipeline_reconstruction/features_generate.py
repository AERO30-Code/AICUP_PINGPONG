import os
import yaml
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

ALL_SENSORS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
ALL_STATS = ["mean", "std", "min", "max", "median", "rms"]
ALL_EXTRA_FEATURES = ["test_time"]

def read_sensor_txt(file_path: str) -> np.ndarray:
    """Read a sensor txt file and return a numpy array of shape (N, 6)."""
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 1 and data.shape[0] == 6:
            data = data.reshape(1, 6)
        elif data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(f"File {file_path} does not contain 6 columns.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Returning empty array.")
        return np.empty((0, 6))
    return data

def calculate_swing_stats(segment_data: np.ndarray, stats_to_calculate: List[str]) -> Dict[str, float]:
    results = {}
    if segment_data.size == 0:
        for stat in stats_to_calculate:
            results[stat] = 0.0
        return results

    if "mean" in stats_to_calculate:
        results["mean"] = np.mean(segment_data)
    if "std" in stats_to_calculate:
        results["std"] = np.std(segment_data)
    if "min" in stats_to_calculate:
        results["min"] = np.min(segment_data)
    if "max" in stats_to_calculate:
        results["max"] = np.max(segment_data)
    if "median" in stats_to_calculate:
        results["median"] = np.median(segment_data)
    if "rms" in stats_to_calculate:
        results["rms"] = np.sqrt(np.mean(np.square(segment_data)))

    for stat in stats_to_calculate:
        if stat not in results:
            results[stat] = 0.0
    return results

def calculate_extra_features(full_data: np.ndarray, extra_features_to_include: List[str]) -> Dict[str, Any]:
    extra_features = {}
    num_lines = full_data.shape[0]
    if "test_time" in extra_features_to_include:
        sampling_rate = 85.0
        test_time = round(num_lines / sampling_rate, 4) if sampling_rate else 0
        extra_features["test_time"] = test_time
    return extra_features

def process_data(config: dict, is_train: bool) -> pd.DataFrame:
    """Process sensor data and return a DataFrame of extracted features."""
    info_path = config["input"]["train_info"] if is_train else config["input"]["test_info"]
    data_dir = config["input"]["train_data"] if is_train else config["input"]["test_data"]
    start_swing = config["start_swing"]
    end_swing = config["end_swing"]
    sensors_to_process = config["sensors"]
    stats_to_calculate = config["stats"]
    extra_features_to_include = config.get("extra_features", [])
    include_player_id = config.get("include_player_id", False)

    try:
        info_df = pd.read_csv(info_path)
    except FileNotFoundError:
        print(f"Error: Info file not found at {info_path}. Aborting.")
        return pd.DataFrame()

    feature_rows = []
    required_labels = ["mode"] + (['gender', 'hold racket handed', 'play years', 'level'] if is_train else [])

    for _, row in info_df.iterrows():
        row_dict = {"unique_id": row["unique_id"]}
        for label in required_labels:
            row_dict[label] = row.get(label, None)
        if is_train and include_player_id:
            row_dict["player_id"] = row.get("player_id", None)

        data_path = os.path.join(data_dir, f"{row['unique_id']}.txt")
        if not os.path.exists(data_path):
            continue

        full_sensor_data = read_sensor_txt(data_path)
        if full_sensor_data.size == 0:
            continue

        try:
            cut_point_str = re.sub(r"[\[\]\n]", "", str(row["cut_point"]))
            cut_point_str = re.sub(r"\s+", " ", cut_point_str).strip()
            cut_points = np.fromstring(cut_point_str, sep=' ', dtype=int)
            if len(cut_points) != 28:
                print(f"Warning: Expected 28 cut points, got {len(cut_points)}. Skipping {row['unique_id']}.")
                continue
        except Exception:
            print(f"Error parsing cut points for {row['unique_id']}. Skipping.")
            continue

        num_defined_swings = len(cut_points) - 1
        if start_swing < 0 or end_swing >= num_defined_swings:
            print(f"Error: swing index out of bounds for {row['unique_id']}. Allowed range is 0 to {num_defined_swings - 1}. Skipping.")
            continue

        max_data_index = full_sensor_data.shape[0]

        for swing_idx in range(start_swing, end_swing + 1):
            segment_data = np.empty((0, 6))
            seg_start = cut_points[swing_idx]
            seg_end = cut_points[swing_idx + 1]
            if seg_start < 0 or seg_start >= max_data_index:
                continue
            seg_end = min(seg_end, max_data_index)
            if seg_start < seg_end:
                segment_data = full_sensor_data[seg_start:seg_end]

            for sensor_idx, sensor_name in enumerate(ALL_SENSORS):
                if sensor_name in sensors_to_process:
                    sensor_segment_data = segment_data[:, sensor_idx] if segment_data.size > 0 else np.array([])
                    stats_results = calculate_swing_stats(sensor_segment_data, stats_to_calculate)
                    for stat_name, stat_value in stats_results.items():
                        col_name = f"swing{swing_idx}_{stat_name}_{sensor_name}"
                        row_dict[col_name] = stat_value

        row_dict.update(calculate_extra_features(full_sensor_data, extra_features_to_include))
        feature_rows.append(row_dict)

    if not feature_rows:
        print(f"No features generated for {'train' if is_train else 'test'} data.")
        return pd.DataFrame()

    features_df = pd.DataFrame(feature_rows)
    features_df.fillna(0, inplace=True)
    return features_df

def generate_features(config_path: str):
    """Main entry to generate training and test features from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_output_path = config["output"]["train_features"]
    test_output_path = config["output"]["test_features"]
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

    print("Generating training features...")
    train_df = process_data(config, is_train=True)
    if not train_df.empty:
        train_df.to_csv(train_output_path, index=False)
        print(f"Training features saved to: {train_output_path}")

    print("Generating test features...")
    test_df = process_data(config, is_train=False)
    if not test_df.empty:
        test_df.to_csv(test_output_path, index=False)
        print(f"Test features saved to: {test_output_path}")

if __name__ == "__main__":
    CONFIG_PATH = "Pipeline_reconstruction/configs/features_config.yaml"
    generate_features(CONFIG_PATH)