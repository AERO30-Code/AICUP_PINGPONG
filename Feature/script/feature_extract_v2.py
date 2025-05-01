import os
import pandas as pd
import numpy as np
import re

INFO_PATH = "original/train/train_info.csv"
DATA_DIR = "original/train/train_data"
OUTPUT_PATH = "Feature/output/features_train_v2.csv"

SENSORS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]  
STATS = ["mean", "std", "min", "max", "median", "rms"]
SEGMENT_RANGE = (5, 27)  # Inclusive, e.g., (4, 27) means segment4 to segment27. Set to (1, None) for all.

def read_sensor_txt(file_path):
    try:
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return None

def extract_segment_features(segment_data):
    """
    Calculate statistical features for a segment.
    Returns a dict: {sensor_stat: value}
    If segment_data is invalid, return all zeros.
    """
    features = {}
    if segment_data is None or segment_data.shape[0] == 0 or segment_data.shape[1] != len(SENSORS):
        # Fill zeros for all features
        for sensor in SENSORS:
            for stat in STATS:
                features[f"{sensor}_{stat}"] = 0
        features["length"] = 0
        return features
    for idx, sensor in enumerate(SENSORS):
        values = segment_data[:, idx]
        features[f"{sensor}_mean"] = np.mean(values)
        features[f"{sensor}_std"] = np.std(values)
        features[f"{sensor}_min"] = np.min(values)
        features[f"{sensor}_max"] = np.max(values)
        features[f"{sensor}_median"] = np.median(values)
        features[f"{sensor}_rms"] = np.sqrt(np.mean(np.square(values)))
    features["length"] = segment_data.shape[0]
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
        # Calculate test_time feature
        test_time = 0
        if data is not None and data.shape[0] > 0:
            test_time = round(data.shape[0] / 85, 4)
        segment_features["test_time"] = test_time
        # Determine segment range
        seg_start = SEGMENT_RANGE[0]
        seg_end = SEGMENT_RANGE[1] if SEGMENT_RANGE[1] is not None else len(cut_points) - 1
        # For allsegments stats
        allsegments_data = []
        segment_lengths = []  # For segment length stats
        # For mid/late segment stats
        mid_segments_data = []  # 9-18
        late_segments_data = [] # 19-27
        # Extract features for each segment in range
        for seg_idx in range(seg_start, seg_end + 1):
            if seg_idx >= len(cut_points):
                print(f"Warning: unique_id {unique_id} has only {len(cut_points)-1} segments, expected at least {seg_end}.")
                # Fill zeros for missing segments
                feat = extract_segment_features(None)
            else:
                seg_start_idx = cut_points[seg_idx - 1]
                seg_end_idx = cut_points[seg_idx] if seg_idx < len(cut_points) else data.shape[0]
                segment_data = data[seg_start_idx:seg_end_idx] if data is not None else None
                if segment_data is None or segment_data.shape[0] == 0 or segment_data.shape[1] != len(SENSORS):
                    print(f"Warning: unique_id {unique_id} segment{seg_idx} data invalid, filling with zeros.")
                feat = extract_segment_features(segment_data)
                # Collect for allsegments
                if segment_data is not None and segment_data.shape[0] > 0 and segment_data.shape[1] == len(SENSORS):
                    allsegments_data.append(segment_data)
                    segment_lengths.append(segment_data.shape[0])
                    # Collect for mid/late segments
                    if 9 <= seg_idx <= 18:
                        mid_segments_data.append(segment_data)
                    if 19 <= seg_idx <= 27:
                        late_segments_data.append(segment_data)
                else:
                    segment_lengths.append(0)
            for k, v in feat.items():
                if k == "length":
                    col_name = f"segment{seg_idx}_length"
                else:
                    col_name = f"segment{seg_idx}_{k}"
                segment_features[col_name] = v
        # Segment length stats
        if len(segment_lengths) > 0:
            segment_features["segment_length_mean"] = np.mean(segment_lengths)
            segment_features["segment_length_std"] = np.std(segment_lengths)
            segment_features["segment_length_min"] = np.min(segment_lengths)
            segment_features["segment_length_max"] = np.max(segment_lengths)
        else:
            segment_features["segment_length_mean"] = 0
            segment_features["segment_length_std"] = 0
            segment_features["segment_length_min"] = 0
            segment_features["segment_length_max"] = 0
        # Allsegments stats
        if len(allsegments_data) > 0:
            allsegments_data = np.vstack(allsegments_data)
            for idx, sensor in enumerate(SENSORS):
                values = allsegments_data[:, idx]
                segment_features[f"allsegments_mean_{sensor}"] = np.mean(values)
                segment_features[f"allsegments_std_{sensor}"] = np.std(values)
                segment_features[f"allsegments_min_{sensor}"] = np.min(values)
                segment_features[f"allsegments_max_{sensor}"] = np.max(values)
                segment_features[f"allsegments_median_{sensor}"] = np.median(values)
                segment_features[f"allsegments_rms_{sensor}"] = np.sqrt(np.mean(np.square(values)))
        else:
            for sensor in SENSORS:
                for stat in STATS:
                    segment_features[f"allsegments_{stat}_{sensor}"] = 0
        # Mid (9-18) and Late (19-27) segment stats and their difference
        for stat in ["mean", "std"]:
            for sensor_idx, sensor in enumerate(SENSORS):
                # Mid
                if len(mid_segments_data) > 0:
                    mid_data = np.vstack(mid_segments_data)
                    if stat == "mean":
                        mid_val = np.mean(mid_data[:, sensor_idx])
                    else:
                        mid_val = np.std(mid_data[:, sensor_idx])
                else:
                    mid_val = 0
                segment_features[f"mid_{stat}_{sensor}"] = mid_val
                # Late
                if len(late_segments_data) > 0:
                    late_data = np.vstack(late_segments_data)
                    if stat == "mean":
                        late_val = np.mean(late_data[:, sensor_idx])
                    else:
                        late_val = np.std(late_data[:, sensor_idx])
                else:
                    late_val = 0
                segment_features[f"late_{stat}_{sensor}"] = late_val
                # Difference
                segment_features[f"late_minus_mid_{stat}_{sensor}"] = late_val - mid_val
        # Add unique_id and labels if present
        row_dict = {"unique_id": unique_id}
        row_dict.update(segment_features)
        for label in ["mode", "gender", "hold racket handed", "play years", "level"]:
            if label in row:
                row_dict[label] = row[label]
        feature_rows.append(row_dict)
        if (idx + 1) % 50 == 0 or idx == len(info_df) - 1:
            print(f"Processed {idx + 1}/{len(info_df)} unique_ids")

    print("Building DataFrame...")
    features_df = pd.DataFrame(feature_rows)
    print(f"Shape of features DataFrame: {features_df.shape}")
    print(f"Saving to {OUTPUT_PATH} ...")
    features_df.to_csv(OUTPUT_PATH, index=False)
    print("Feature extraction completed.")

if __name__ == "__main__":
    main() 