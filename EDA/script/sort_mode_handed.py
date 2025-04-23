import os
import shutil
import pandas as pd

handed_base = 'EDA/katsu/handed'
left_dir = os.path.join(handed_base, 'left')
right_dir = os.path.join(handed_base, 'right')
csv_path = 'original/train/train_info.csv'
out_base = 'EDA/katsu/mode_handed'

print('Reading train_info.csv...')
df = pd.read_csv(csv_path)
info_dict = {}
for _, row in df.iterrows():
    info_dict[str(row['unique_id'])] = {
        'mode': row['mode'],
        'handed': row['hold racket handed']
    }

def process_folder(folder_path, handed_label):
    total = 0
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.png'):
            continue
        unique_id = os.path.splitext(fname)[0]
        if unique_id not in info_dict:
            print(f"[Warning] {fname} not found in CSV, skipping.")
            continue
        mode = info_dict[unique_id]['mode']
        handed = info_dict[unique_id]['handed']
        # 1: right, 2: left
        if handed_label == 'right' and handed != 1:
            print(f"[Warning] {fname} in 'right' but handed is not 1, skipping.")
            continue
        if handed_label == 'left' and handed != 2:
            print(f"[Warning] {fname} in 'left' but handed is not 2, skipping.")
            continue
        out_dir = os.path.join(out_base, f"mode{mode}", handed_label)
        os.makedirs(out_dir, exist_ok=True)
        src = os.path.join(folder_path, fname)
        dst = os.path.join(out_dir, fname)
        shutil.copy2(src, dst)
        total += 1
    return total

if __name__ == '__main__':
    print('Processing right-handed images...')
    right_count = process_folder(right_dir, 'right')
    print('Processing left-handed images...')
    left_count = process_folder(left_dir, 'left')
    print(f'Total processed images: {right_count + left_count}')
    print('Done!')
