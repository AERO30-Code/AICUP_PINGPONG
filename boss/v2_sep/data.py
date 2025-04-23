import os
import csv
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
import glob

# ========== 參數設定 ========== #
TRAIN_RAW_DATA_DIR = "data/39_Training_Dataset/train_data"
TEST_RAW_DATA_DIR = "data/39_Test_Dataset/test_data"
train_info_path = "data/39_Training_Dataset/train_info.csv"
# 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
TRAIN_FEATURE_DIR = 'data/39_Training_Dataset/tabular_data_train'
TEST_FEATURE_DIR = 'data/39_Test_Dataset/tabular_data_train'

# TRAIN_RAW_DATA_DIR = 'path_to_train_txt'
# TEST_RAW_DATA_DIR = 'path_to_test_txt'
# TRAIN_FEATURE_DIR = 'train_feature'
# TEST_FEATURE_DIR = 'test_feature'
NUM_SWINGS = 28  # 根據你的資料需求, 若是27次揮拍就設27+1分點

# (可選) 若要濾波，使用 scipy.signal
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(signal, cutoff=30, fs=85, order=4):
    """對1D向量作低通濾波"""
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


def read_txt_file(file_path):
    """
    讀取txt檔, 回傳陣列:
    All_data: shape = (N, 6), 依序為 [ax, ay, az, gx, gy, gz]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    All_data = []
    for line in lines:
        line = line.strip()
        if not line:
            # 空行跳過
            continue
        # 嘗試以空白切割
        tokens = line.split()
        if len(tokens) < 6:
            # 可能是標頭或其他雜訊行
            continue

        row = list(map(int, tokens[:6]))  # 只取前6個, ax,ay,az,gx,gy,gz
        All_data.append(row)

    return np.array(All_data)  # shape=(N,6)


def segment_data(data_array, num_segments=28):
    """
    以等分方式將 data_array 切割成 (num_segments-1) 段
    回傳: list of segments, 每段 shape=(segment_len,6)
    """
    length = len(data_array)
    # 建立切分點 (等分)
    swing_index = np.linspace(0, length, num_segments, dtype=int)
    segments = []
    for i in range(1, len(swing_index)):
        start_idx = swing_index[i - 1]
        end_idx = swing_index[i]
        seg = data_array[start_idx:end_idx]
        segments.append(seg)
    return segments


def compute_features(segment):
    """
    針對 segment (shape=(k,6))，計算各種統計特徵 (時域+頻域)
    回傳: list or dict of feature values
    """
    # shape = (k,6)
    # 分成ax, ay, az, gx, gy, gz
    ax = segment[:, 0]
    ay = segment[:, 1]
    az = segment[:, 2]
    gx = segment[:, 3]
    gy = segment[:, 4]
    gz = segment[:, 5]

    # (可選) 濾波
    # ax = butter_lowpass_filter(ax)
    # ay = butter_lowpass_filter(ay)
    # az = butter_lowpass_filter(az)
    # gx = butter_lowpass_filter(gx)
    # gy = butter_lowpass_filter(gy)
    # gz = butter_lowpass_filter(gz)

    # 計算幅度 (magnitude)
    a_magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    g_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # --------- 時域特徵 ----------
    def get_stats(x):
        mean_ = np.mean(x)
        std_ = np.std(x)
        rms_ = np.sqrt(np.mean(x ** 2))
        max_ = np.max(x)
        min_ = np.min(x)
        # 偏度 (skew) & 峰度 (kurtosis), 用 scipy.stats 也可以
        n = len(x)
        if n > 1:
            skew_ = np.sum((x - mean_) ** 3) / n / (np.sum((x - mean_) ** 2) / n) ** 1.5
            kurt_ = np.sum((x - mean_) ** 4) / n / (np.sum((x - mean_) ** 2) / n) ** 2
        else:
            skew_ = 0
            kurt_ = 0
        return mean_, std_, rms_, max_, min_, skew_, kurt_

    ax_mean, ax_std, ax_rms, ax_max, ax_min, ax_skew, ax_kurt = get_stats(ax)
    ay_mean, ay_std, ay_rms, ay_max, ay_min, ay_skew, ay_kurt = get_stats(ay)
    az_mean, az_std, az_rms, az_max, az_min, az_skew, az_kurt = get_stats(az)
    gx_mean, gx_std, gx_rms, gx_max, gx_min, gx_skew, gx_kurt = get_stats(gx)
    gy_mean, gy_std, gy_rms, gy_max, gy_min, gy_skew, gy_kurt = get_stats(gy)
    gz_mean, gz_std, gz_rms, gz_max, gz_min, gz_skew, gz_kurt = get_stats(gz)

    a_mag_mean, a_mag_std, a_mag_rms, a_mag_max, a_mag_min, a_mag_skew, a_mag_kurt = get_stats(a_magnitude)
    g_mag_mean, g_mag_std, g_mag_rms, g_mag_max, g_mag_min, g_mag_skew, g_mag_kurt = get_stats(g_magnitude)

    # --------- 頻域特徵 (FFT) ----------
    # 如果 segment 長度不足，或不是2的次方，也可以直接 fft，不必特地 padding
    # 這裡只計算幅度向量的 FFT 做示範, 也可對每一軸做 FFT
    a_fft = np.fft.fft(a_magnitude)
    g_fft = np.fft.fft(g_magnitude)
    # 取絕對值 (幅度譜)
    a_fft_abs = np.abs(a_fft)
    g_fft_abs = np.abs(g_fft)
    # 例如取平均頻譜值
    a_fft_mean = np.mean(a_fft_abs)
    g_fft_mean = np.mean(g_fft_abs)
    # 也可以計算能量 (psd)
    a_psd = np.sum(a_fft_abs ** 2)
    g_psd = np.sum(g_fft_abs ** 2)

    # 簡易的頻譜熵 (spectrum entropy)
    def spectrum_entropy(magnitude):
        mag_sum = np.sum(magnitude)
        if mag_sum == 0:
            return 0
        p = magnitude / mag_sum
        # 避免 log(0) 出現nan
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        return ent

    a_entropy = spectrum_entropy(a_fft_abs)
    g_entropy = spectrum_entropy(g_fft_abs)

    # 將特徵整合
    # (依需求定順序, 這裡示範將各種特徵放一起)
    features = [
        ax_mean, ax_std, ax_rms, ax_max, ax_min, ax_skew, ax_kurt,
        ay_mean, ay_std, ay_rms, ay_max, ay_min, ay_skew, ay_kurt,
        az_mean, az_std, az_rms, az_max, az_min, az_skew, az_kurt,
        gx_mean, gx_std, gx_rms, gx_max, gx_min, gx_skew, gx_kurt,
        gy_mean, gy_std, gy_rms, gy_max, gy_min, gy_skew, gy_kurt,
        gz_mean, gz_std, gz_rms, gz_max, gz_min, gz_skew, gz_kurt,
        a_mag_mean, a_mag_std, a_mag_rms, a_mag_max, a_mag_min, a_mag_skew, a_mag_kurt,
        g_mag_mean, g_mag_std, g_mag_rms, g_mag_max, g_mag_min, g_mag_skew, g_mag_kurt,
        a_fft_mean, g_fft_mean, a_psd, g_psd, a_entropy, g_entropy
    ]
    return features


def data_generate():
    # 指定要輸出的CSV欄位名稱 (對應 compute_features 的順序)
    header = [
        'ax_mean', 'ax_std', 'ax_rms', 'ax_max', 'ax_min', 'ax_skew', 'ax_kurt',
        'ay_mean', 'ay_std', 'ay_rms', 'ay_max', 'ay_min', 'ay_skew', 'ay_kurt',
        'az_mean', 'az_std', 'az_rms', 'az_max', 'az_min', 'az_skew', 'az_kurt',
        'gx_mean', 'gx_std', 'gx_rms', 'gx_max', 'gx_min', 'gx_skew', 'gx_kurt',
        'gy_mean', 'gy_std', 'gy_rms', 'gy_max', 'gy_min', 'gy_skew', 'gy_kurt',
        'gz_mean', 'gz_std', 'gz_rms', 'gz_max', 'gz_min', 'gz_skew', 'gz_kurt',
        'a_mag_mean', 'a_mag_std', 'a_mag_rms', 'a_mag_max', 'a_mag_min', 'a_mag_skew', 'a_mag_kurt',
        'g_mag_mean', 'g_mag_std', 'g_mag_rms', 'g_mag_max', 'g_mag_min', 'g_mag_skew', 'g_mag_kurt',
        'a_fft_mean', 'g_fft_mean', 'a_psd', 'g_psd', 'a_entropy', 'g_entropy'
    ]

    # === 1) 處理 train 資料 ===
    pathlist_txt = glob.glob(os.path.join(TRAIN_RAW_DATA_DIR, "*.txt"))
    print("===== extracting train data feature =====")
    print("Total txt files:", len(pathlist_txt))

    os.makedirs(TRAIN_FEATURE_DIR, exist_ok=True)

    for file in tqdm(pathlist_txt):
        All_data = read_txt_file(file)  # shape=(N,6)
        # 等分切割
        segments = segment_data(All_data, NUM_SWINGS)

        # 輸出CSV (同檔名)
        out_csv = os.path.join(TRAIN_FEATURE_DIR, f"{Path(file).stem}.csv")
        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            # 對每段揮拍計算特徵
            for seg in segments:
                if len(seg) == 0:
                    continue
                feats = compute_features(seg)
                writer.writerow(feats)

    print("training data gen finished.")

    # === 2) 處理 test 資料 (同理) ===
    pathlist_txt = glob.glob(os.path.join(TEST_RAW_DATA_DIR, "*.txt"))
    print("===== extracting test data feature =====")
    print("Total test txt files:", len(pathlist_txt))

    os.makedirs(TEST_FEATURE_DIR, exist_ok=True)

    for file in tqdm(pathlist_txt):
        All_data = read_txt_file(file)
        segments = segment_data(All_data, NUM_SWINGS)

        out_csv = os.path.join(TEST_FEATURE_DIR, f"{Path(file).stem}.csv")
        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for seg in segments:
                if len(seg) == 0:
                    continue
                feats = compute_features(seg)
                writer.writerow(feats)

    print("test data gen finished.")


if __name__ == '__main__':
    data_generate()
