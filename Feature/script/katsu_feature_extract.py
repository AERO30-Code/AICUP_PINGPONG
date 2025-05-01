import pandas as pd
import numpy as np
import os

def data_generate_train():
    datapath = 'Original/train/train_data'
    info_csv = 'Original/train/train_info.csv'

    df = pd.read_csv(info_csv)
    output_df = df[['unique_id', 'player_id', 'mode', 'gender', 'hold racket handed', 'play years', 'level']].copy()

    axis_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    stat_names = ['mean', 'std', 'max', 'min', 'rms', 'median']

    features = {f'swing{i}_{stat}_{axis}': [] for i in range(5, 24) for stat in stat_names for axis in axis_names}
    features['unique_id'] = []
    features['test_time'] = []

    for filename in sorted(os.listdir(datapath)):
        if filename.endswith(".txt"):
            unique_id = int(os.path.splitext(filename)[0])
            row = df[df['unique_id'] == unique_id]
            if row.empty:
                continue

            try:
                cutpoints = list(map(int, row['cut_point'].values[0].strip().strip('[]').split()))
            except Exception as e:
                print(f"{unique_id} cut_point failed: {e}")
                continue

            filepath = os.path.join(datapath, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()[1:]

            features['unique_id'].append(unique_id)
            features['test_time'].append(round(len(lines) / 85, 4))

            for i in range(1, len(cutpoints)):
                if i < 5 or i > 23:
                    continue
                    
                segment_lines = lines[cutpoints[i - 1]:cutpoints[i]]
                segment_data = [np.fromstring(line.strip(), sep=' ', dtype=int) for line in segment_lines]
                segment_array = np.array(segment_data)
                
                if segment_array.shape[0] == 0 or segment_array.shape[1] != 6:
                    for stat in stat_names:
                        for axis in axis_names:
                            features[f'swing{i}_{stat}_{axis}'].append(0)
                    continue

                for stat in stat_names:
                    if stat == 'mean':
                        vals = segment_array.mean(axis=0)
                    elif stat == 'std':
                        vals = segment_array.std(axis=0)
                    elif stat == 'max':
                        vals = segment_array.max(axis=0)
                    elif stat == 'min':
                        vals = segment_array.min(axis=0)
                    elif stat == 'rms':
                        vals = np.sqrt(np.mean(np.square(segment_array), axis=0))
                    elif stat == 'median':
                        vals = np.median(segment_array, axis=0)
                    for axis, val in zip(axis_names, vals):
                        features[f'swing{i}_{stat}_{axis}'].append(val)

            if len(cutpoints) != 28:
                print(f"{unique_id} 的切割點數不正確，應為28段但實際為 {len(cutpoints)} 段")
                for i in range(len(cutpoints), 28):
                    for stat in stat_names:
                        for axis in axis_names:
                            features[f'swing{i}_{stat}_{axis}'].append(0)

    feature_df = pd.DataFrame(features)
    output_df = output_df.merge(feature_df, on='unique_id', how='left')
    output_df.to_csv('Feature/katsu/train_v3.csv', index=False)
    # print("✅ 特徵儲存完成：data/train.csv")


def data_generate_test():
    datapath = 'Original/test/test_data'
    info_csv = 'Original/test/test_info.csv'

    df = pd.read_csv(info_csv)
    output_df = df[['unique_id', 'mode']].copy()

    axis_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    stat_names = ['mean', 'std', 'max', 'min', 'rms', 'median']

    features = {f'swing{i}_{stat}_{axis}': [] for i in range(5, 24) for stat in stat_names for axis in axis_names}
    features['unique_id'] = []
    features['test_time'] = []

    for filename in sorted(os.listdir(datapath)):
        if filename.endswith(".txt"):
            unique_id = int(os.path.splitext(filename)[0])
            row = df[df['unique_id'] == unique_id]
            if row.empty:
                continue

            try:
                cutpoints = list(map(int, row['cut_point'].values[0].strip().strip('[]').split()))
            except Exception as e:
                print(f"{unique_id} cut_point failed: {e}")
                continue

            filepath = os.path.join(datapath, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()[1:]

            features['unique_id'].append(unique_id)
            features['test_time'].append(round(len(lines) / 85, 4))

            for i in range(1, len(cutpoints)):
                if i < 5 or i > 23:
                    continue

                segment_lines = lines[cutpoints[i - 1]:cutpoints[i]]
                segment_data = [np.fromstring(line.strip(), sep=' ', dtype=int) for line in segment_lines]
                segment_array = np.array(segment_data)

                if segment_array.shape[0] == 0 or segment_array.shape[1] != 6:
                    for stat in stat_names:
                        for axis in axis_names:
                            features[f'swing{i}_{stat}_{axis}'].append(0)
                    continue

                for stat in stat_names:
                    if stat == 'mean':
                        vals = segment_array.mean(axis=0)
                    elif stat == 'std':
                        vals = segment_array.std(axis=0)
                    elif stat == 'max':
                        vals = segment_array.max(axis=0)
                    elif stat == 'min':
                        vals = segment_array.min(axis=0)
                    elif stat == 'rms':
                        vals = np.sqrt(np.mean(np.square(segment_array), axis=0))
                    elif stat == 'median':
                        vals = np.median(segment_array, axis=0)
                    for axis, val in zip(axis_names, vals):
                        features[f'swing{i}_{stat}_{axis}'].append(val)

            if len(cutpoints) != 28:
                print(f"{unique_id} 的切割點數不正確，應為28段但實際為 {len(cutpoints)} 段")
                for i in range(len(cutpoints), 28):
                    for stat in stat_names:
                        for axis in axis_names:
                            features[f'swing{i}_{stat}_{axis}'].append(0)

    feature_df = pd.DataFrame(features)
    output_df = output_df.merge(feature_df, on='unique_id', how='left')
    output_df.to_csv('Feature/katsu/test_v3.csv', index=False)
    # print("✅ 特徵儲存完成：data/test.csv")

if __name__ == '__main__':
    data_generate_train()
    data_generate_test()