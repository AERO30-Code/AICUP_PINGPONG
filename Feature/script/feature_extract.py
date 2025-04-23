import os
import pandas as pd
import numpy as np
import re

# Path settings
# INFO_PATH = "original/train/train_info.csv"
# DATA_DIR = "original/train/train_data"
# OUTPUT_PATH = "Feature/output/features_train.csv"

INFO_PATH = "original/test/test_info.csv"
DATA_DIR = "original/test/test_data"
OUTPUT_PATH = "Feature/output/features_test.csv"

# Sensors and statistics to extract
SENSORS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
STATS = ["mean", "std", "min", "max", "median"]

def read_sensor_txt(file_path):
    """
    Read sensor txt file and return numpy array.
    """
    data = np.loadtxt(file_path)
    return data

def extract_segment_features(segment_data):
    """
    Calculate statistical features for a segment.
    Returns a dict: {sensor_stat: value}
    """
    features = {}
    for idx, sensor in enumerate(SENSORS):
        values = segment_data[:, idx]
        features[f"{sensor}_mean"] = np.mean(values)
        features[f"{sensor}_std"] = np.std(values)
        features[f"{sensor}_min"] = np.min(values)
        features[f"{sensor}_max"] = np.max(values)
        features[f"{sensor}_median"] = np.median(values)
    return features

def main():
    print("Loading info csv...")
    info_df = pd.read_csv(INFO_PATH)
    print(f"Total unique_id: {len(info_df)}")

    feature_rows = []
    for idx, row in info_df.iterrows():
        unique_id = row["unique_id"]
        cut_point_str = re.sub(r"[\[\]\n]", "", str(row["cut_point"]))
        cut_points = np.fromstring(cut_point_str, sep=' ', dtype=int)
        data_path = os.path.join(DATA_DIR, f"{unique_id}.txt")
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skip.")
            continue
        data = read_sensor_txt(data_path)
        segment_features = {}
        for seg_idx in range(len(cut_points) - 1):
            seg_start = cut_points[seg_idx]
            seg_end = cut_points[seg_idx + 1]
            segment_data = data[seg_start:seg_end]
            feat = extract_segment_features(segment_data)
            for k, v in feat.items():
                col_name = f"segment{seg_idx}_{k}"
                segment_features[col_name] = v
        # Add unique_id and labels
        row_dict = {"unique_id": unique_id}
        row_dict.update(segment_features)
        # Add labels (for train)
        for label in ["mode", "gender", "hold racket handed", "play years", "level"]:
            if label in row:
                row_dict[label] = row[label]
        feature_rows.append(row_dict)
        if (idx + 1) % 20 == 0 or idx == len(info_df) - 1:
            print(f"Processed {idx + 1}/{len(info_df)} unique_ids")

    print("Building DataFrame...")
    features_df = pd.DataFrame(feature_rows)
    print(f"Shape of features DataFrame: {features_df.shape}")
    print(f"Saving to {OUTPUT_PATH} ...")
    features_df.to_csv(OUTPUT_PATH, index=False)
    print("Feature extraction completed.")

if __name__ == "__main__":
    main()