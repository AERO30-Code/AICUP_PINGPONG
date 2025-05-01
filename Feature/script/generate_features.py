import os
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

ALL_SENSORS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
ALL_STATS = ["mean", "std", "min", "max", "median", "rms"]
ALL_EXTRA_FEATURES = ["test_time"]

def read_sensor_txt(file_path: str) -> np.ndarray:
    """
    讀取感測器 txt 檔案並返回 numpy 陣列。
    此版本不跳過第一行，以確保 cut_point 索引能正確對應。
    """
    try:
        data = np.loadtxt(file_path)
        # 檢查欄位數是否正確 (應為 6)
        if data.ndim == 1 and data.shape[0] == 6:
             data = data.reshape(1, 6)
        elif data.ndim != 2 or data.shape[1] != 6:
             raise ValueError(f"File {file_path} does not contain 6 columns.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Returning empty array.")
        return np.empty((0, 6))
    return data

def calculate_swing_stats(segment_data: np.ndarray, stats_to_calculate: List[str]) -> Dict[str, float]:
    """
    為單一感測器在一個揮拍片段 (segment) 內的數據計算指定的統計特徵。
    """
    results = {}
    # 處理空片段：所有統計量設為 0
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

    # 確保所有請求的統計量都有值 (即使計算失敗也補 0)
    for stat in stats_to_calculate:
        if stat not in results:
             print(f"Warning: Stat '{stat}' calculation might have failed for a segment.")
             results[stat] = 0.0

    return results

def calculate_extra_features(full_data: np.ndarray, extra_features_to_include: List[str]) -> Dict[str, Any]:
    """
    計算指定的額外/全局特徵。
    """
    extra_features = {}
    if not extra_features_to_include:
        return extra_features

    num_lines = full_data.shape[0]

    if "test_time" in extra_features_to_include:
        sampling_rate = 85.0
        test_time = round(num_lines / sampling_rate, 4) if sampling_rate else 0
        extra_features["test_time"] = test_time
    # 在此處添加其他額外特徵的計算邏輯
    # 例如：if 'total_energy' in extra_features_to_include:
    #          extra_features['total_energy'] = calculate_energy(full_data)

    return extra_features

def process_data(info_path: str, data_dir: str, output_path: str,
                 start_swing: int, end_swing: int,
                 sensors_to_process: List[str], stats_to_calculate: List[str],
                 extra_features_to_include: List[str],
                 is_train_data: bool, include_player_id_in_train: bool):
    """
    根據 info 檔案和設定處理感測器數據，並將特徵儲存為 CSV。
    """
    # 使用英文打印輸出訊息
    print(f"\nProcessing data from: {info_path}")
    print(f"Sensor data directory: {data_dir}")
    print(f"Outputting features to: {output_path}")
    print(f"Configuration:")
    print(f"  - Swings: {start_swing} to {end_swing}")
    print(f"  - Sensors: {sensors_to_process}")
    print(f"  - Stats: {stats_to_calculate}")
    print(f"  - Extra Features: {extra_features_to_include}")
    if is_train_data:
        print(f"  - Include Player ID: {include_player_id_in_train}")

    try:
        info_df = pd.read_csv(info_path)
    except FileNotFoundError:
        print(f"Error: Info file not found at {info_path}. Aborting.")
        return
    print(f"Total unique_ids in info file: {len(info_df)}")

    feature_rows = []
    required_labels = ["mode", "gender", "hold racket handed", "play years", "level"] if is_train_data else ["mode"]

    # --- 驗證參數有效性 ---
    valid_sensors = [s for s in sensors_to_process if s in ALL_SENSORS]
    if len(valid_sensors) != len(sensors_to_process):
        print(f"Warning: Invalid sensor names provided. Using only valid sensors: {valid_sensors}")
        sensors_to_process = valid_sensors

    valid_stats = [s for s in stats_to_calculate if s in ALL_STATS]
    if len(valid_stats) != len(stats_to_calculate):
        print(f"Warning: Invalid stat names provided. Using only valid stats: {valid_stats}")
        stats_to_calculate = valid_stats

    valid_extra_features = [f for f in extra_features_to_include if f in ALL_EXTRA_FEATURES]
    if len(valid_extra_features) != len(extra_features_to_include):
        print(f"Warning: Invalid extra feature names provided. Using only valid ones: {valid_extra_features}")
        extra_features_to_include = valid_extra_features

    # --- 逐一處理 unique_id ---
    for idx, row in info_df.iterrows():
        row_dict = {}
        unique_id = row["unique_id"]
        row_dict["unique_id"] = unique_id

        # --- 添加標籤 ---
        # 加入 info 檔案中存在的必要標籤
        for label in required_labels:
             if label in row:
                 row_dict[label] = row[label]
             else:
                 print(f"Warning: Label '{label}' not found for unique_id {unique_id} in {info_path}")
                 row_dict[label] = None 

        # 如果是訓練資料且需要，則加入 player_id
        if is_train_data and include_player_id_in_train:
            if "player_id" in row:
                row_dict["player_id"] = row["player_id"]
            else:
                # 英文警告
                print(f"Warning: player_id requested but not found for unique_id {unique_id} in {info_path}")


        # --- 讀取感測器數據 ---
        data_path = os.path.join(data_dir, f"{unique_id}.txt")
        if not os.path.exists(data_path):
            print(f"Warning: Data file {data_path} not found for unique_id {unique_id}. Skipping this unique_id.")
            continue # 跳過此 unique_id 的後續處理
        full_sensor_data = read_sensor_txt(data_path)
        if full_sensor_data.size == 0:
             print(f"Warning: Failed to read or empty data in {data_path} for unique_id {unique_id}. Skipping this unique_id.")
             continue # 跳過此 unique_id


        # --- 解析切割點 (Cut Points) ---
        cut_points = []
        try:
            # 清理字串：移除方括號、換行符、多餘空格
            cut_point_str = re.sub(r"[\[\]\n]", "", str(row["cut_point"]))
            cut_point_str = re.sub(r"\s+", " ", cut_point_str).strip() # 將多個空格替換為單個
            cut_points = np.fromstring(cut_point_str, sep=' ', dtype=int)
            # 檢查切割點數量是否為預期的 28 個 (定義 27 個區間)
            if len(cut_points) != 28:
                print(f"Warning: Incorrect number of cut points ({len(cut_points)}) for unique_id {unique_id}. Expected 28. Proceeding cautiously.")
                # 可根據需要決定如何處理，例如跳過、填充或繼續 (目前選擇繼續)
        except Exception as e:
            print(f"Error parsing cut_points for unique_id {unique_id}: {e}. Skipping this unique_id.")
            continue # 跳過此 unique_id


        # --- 提取揮拍 (Swing) 特徵 ---
        # 揮拍編號是 1-based (從 1 到 27)
        num_defined_swings = len(cut_points) - 1 # 實際定義的揮拍數量
        max_data_index = full_sensor_data.shape[0] # 資料陣列的實際長度 (索引上限為 length - 1)

        for swing_idx_1based in range(start_swing, end_swing + 1):
            segment_data = np.empty((0, 6)) # 預設為空片段

            # 檢查請求的揮拍編號是否在定義的範圍內
            if swing_idx_1based <= num_defined_swings:
                seg_start = cut_points[swing_idx_1based - 1]
                seg_end = cut_points[swing_idx_1based]

                # --- 邊界條件檢查與修正 ---
                valid_indices = True
                # 檢查起始點
                if seg_start < 0:
                     # 英文警告
                    print(f"Warning: Invalid start cut point {seg_start} (< 0) for swing {swing_idx_1based}, unique_id {unique_id}. Features for this swing will be 0.")
                    valid_indices = False
                elif seg_start >= max_data_index:
                    # 如果起始點已經超出或等於最大長度，則此片段不可能有數據
                     # 英文警告
                    print(f"Warning: Start cut point {seg_start} is out of data bounds (max index {max_data_index-1}) for swing {swing_idx_1based}, unique_id {unique_id}. Features for this swing will be 0.")
                    valid_indices = False

                # 只有在起始點有效時才檢查結束點
                if valid_indices:
                    original_seg_end = seg_end
                    # 如果結束點超出實際數據長度 (允許等於長度，因為 Python 切片不包含結束索引)
                    if seg_end > max_data_index:
                        # 英文警告，並修正結束點
                        print(f"Warning: End cut point {original_seg_end} exceeds data length ({max_data_index}) for swing {swing_idx_1based}, unique_id {unique_id}. Using {max_data_index} as end point.")
                        seg_end = max_data_index # 修正為數組長度

                    # 確保起始點仍然小於修正後的結束點，以進行切片
                    # 注意：即使 seg_end == max_data_index，切片 data[start:end] 也是有效的
                    if seg_start < seg_end:
                        segment_data = full_sensor_data[seg_start:seg_end]
                    else:
                        # 如果 start >= end (可能因為 start 就很大或 end 被修正後變小)，片段為空
                         print(f"Note: Cut points [{seg_start}, {original_seg_end}] (adjusted end: {seg_end}) result in an empty segment for swing {swing_idx_1based}, unique_id {unique_id}. Features for this swing will be 0.")
                         segment_data = np.empty((0, 6)) # 確保是空陣列
            else:
                 # 如果請求的揮拍編號超出了 cut_points 定義的數量 (通常不應發生，除非 cut_points 數量不足 28)
                 print(f"Warning: Requested swing {swing_idx_1based} exceeds number of defined swings ({num_defined_swings}) for unique_id {unique_id}. Features for this swing will be 0.")


            # --- 計算指定感測器的統計量 ---
            for sensor_idx, sensor_name in enumerate(ALL_SENSORS):
                if sensor_name in sensors_to_process:
                    # 提取當前感測器的數據 (如果 segment_data 非空)
                    sensor_segment_data = segment_data[:, sensor_idx] if segment_data.size > 0 else np.array([])

                    # 計算請求的統計量
                    stats_results = calculate_swing_stats(sensor_segment_data, stats_to_calculate)

                    # 將結果添加到主字典中
                    for stat_name, stat_value in stats_results.items():
                        # 使用 f-string 格式化欄位名稱
                        col_name = f"swing{swing_idx_1based}_{stat_name}_{sensor_name}"
                        row_dict[col_name] = stat_value

        # --- 計算額外特徵 ---
        extra_features_results = calculate_extra_features(full_sensor_data, extra_features_to_include)
        # 將額外特徵結果更新到主字典
        row_dict.update(extra_features_results)

        # 將此 unique_id 的所有特徵添加到結果列表中
        feature_rows.append(row_dict)

        if (idx + 1) % 500 == 0 or idx == len(info_df) - 1:
            print(f"Processed {idx + 1}/{len(info_df)} unique_ids...")

    if not feature_rows:
        print("No features were generated. Please check input data and configurations.")
        return

    features_df = pd.DataFrame(feature_rows)
    print(f"Shape of features DataFrame before handling NaNs: {features_df.shape}")
    features_df.fillna(0, inplace=True)
    print(f"Shape after filling NaNs: {features_df.shape}")
    print(f"Saving features to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving file to {output_path}: {e}")

    print(f"Feature extraction for {'train' if is_train_data else 'test'} data completed.")


# --- 主執行區塊 ---
if __name__ == "__main__":
    TRAIN_INFO_PATH = "/Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/Original/train/train_info.csv"
    TRAIN_DATA_DIR = "/Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/Original/train/train_data"
    TEST_INFO_PATH = "/Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/Original/test/test_info.csv"
    TEST_DATA_DIR = "/Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/Original/test/test_data"
    OUTPUT_DIR = "/Users/charlie/MBP16/Master_Data/AICUP/PINGPONG/Feature/generated"

    # 揮拍範圍 (1-based)。例如：1 到 27 代表所有揮拍。
    START_SWING = 1
    END_SWING = 27

    SENSORS_TO_PROCESS = ALL_SENSORS # 使用所有感測器: ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

    STATS_TO_CALCULATE = ALL_STATS # 計算所有可用統計量: ["mean", "std", "min", "max", "median", "rms"]

    # 要包含的額外/全局特徵列表 (每個 unique_id 計算一次)
    EXTRA_FEATURES_TO_INCLUDE = ["test_time"] # 計算 test_time

    INCLUDE_PLAYER_ID_IN_TRAIN = False

    train_output_filename = f"features_train_s{START_SWING}-e{END_SWING}.csv"
    test_output_filename = f"features_test_s{START_SWING}-e{END_SWING}.csv"

    TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, train_output_filename)
    TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, test_output_filename)

    process_data(
        info_path=TRAIN_INFO_PATH,
        data_dir=TRAIN_DATA_DIR,
        output_path=TRAIN_OUTPUT_PATH,
        start_swing=START_SWING,
        end_swing=END_SWING,
        sensors_to_process=SENSORS_TO_PROCESS,
        stats_to_calculate=STATS_TO_CALCULATE,
        extra_features_to_include=EXTRA_FEATURES_TO_INCLUDE,
        is_train_data=True,
        include_player_id_in_train=INCLUDE_PLAYER_ID_IN_TRAIN
    )

    process_data(
        info_path=TEST_INFO_PATH,
        data_dir=TEST_DATA_DIR,
        output_path=TEST_OUTPUT_PATH,
        start_swing=START_SWING,
        end_swing=END_SWING,
        sensors_to_process=SENSORS_TO_PROCESS,
        stats_to_calculate=STATS_TO_CALCULATE,
        extra_features_to_include=EXTRA_FEATURES_TO_INCLUDE,
        is_train_data=False,
        include_player_id_in_train=False
    )
    
    print("\nAll processing finished.")
