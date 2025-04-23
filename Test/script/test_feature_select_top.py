import os
import shutil
import subprocess
import re
import csv
import matplotlib.pyplot as plt

# 設定路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_SCRIPT = os.path.join(BASE_DIR, 'Feature', 'script', 'feature_select_top.py')
MODEL_SCRIPT = os.path.join(BASE_DIR, 'Model', 'script', 'train_model.py')
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, 'Test', 'output')

# 備份檔名
FEATURE_SCRIPT_BAK = FEATURE_SCRIPT + '.bak'
MODEL_SCRIPT_BAK = MODEL_SCRIPT + '.bak'

# 參數範圍
N_TOP_FEATURES_LIST = list(range(60, 651, 10))

# 結果儲存
RESULT_CSV = os.path.join(TEST_OUTPUT_DIR, 'feature_auc_result.csv')
RESULT_PNG = os.path.join(TEST_OUTPUT_DIR, 'feature_auc_result.png')

# 主要流程
if __name__ == '__main__':
    # 確保 output 目錄存在
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # 備份原始檔案
    shutil.copyfile(FEATURE_SCRIPT, FEATURE_SCRIPT_BAK)
    shutil.copyfile(MODEL_SCRIPT, MODEL_SCRIPT_BAK)

    results = []
    try:
        for n in N_TOP_FEATURES_LIST:
            print(f"\n===== 測試 N_TOP_FEATURES = {n} =====")
            # 1. 修改 feature_select_top.py
            with open(FEATURE_SCRIPT_BAK, 'r', encoding='utf-8') as f:
                feature_lines = f.readlines()
            with open(FEATURE_SCRIPT, 'w', encoding='utf-8') as f:
                for line in feature_lines:
                    if line.strip().startswith('N_TOP_FEATURES'):
                        f.write(f'N_TOP_FEATURES = {n}\n')
                    else:
                        f.write(line)
            # 2. 執行 feature_select_top.py
            result = subprocess.run(['python', FEATURE_SCRIPT], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"feature_select_top.py 執行失敗\n{result.stderr}")
                raise RuntimeError('feature_select_top.py failed')
            # 3. 修改 train_model.py 的 FEATURE_PATH
            feature_csv_path = f"Feature/output/features_train_gender_top/features_train_top{n}.csv"
            with open(MODEL_SCRIPT_BAK, 'r', encoding='utf-8') as f:
                model_lines = f.readlines()
            with open(MODEL_SCRIPT, 'w', encoding='utf-8') as f:
                for line in model_lines:
                    if line.strip().startswith('FEATURE_PATH'):
                        f.write(f'FEATURE_PATH = "{feature_csv_path}"\n')
                    else:
                        f.write(line)
            # 4. 執行 train_model.py 並擷取 Validation ROC AUC
            result = subprocess.run(['python', MODEL_SCRIPT], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"train_model.py 執行失敗\n{result.stderr}")
                raise RuntimeError('train_model.py failed')
            auc = None
            for line in result.stdout.splitlines():
                m = re.search(r'Validation ROC AUC: ([0-9.]+)', line)
                if m:
                    auc = float(m.group(1))
                    break
            print(f"N_TOP_FEATURES={n}, Validation ROC AUC={auc}")
            results.append((n, auc))
        # 5. 寫入 CSV
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['N_TOP_FEATURES', 'Validation_ROC_AUC'])
            for row in results:
                writer.writerow(row)
        # 6. 畫圖
        x = [r[0] for r in results]
        y = [r[1] for r in results]
        plt.figure(figsize=(10,6))
        plt.plot(x, y, marker='o')
        plt.xlabel('N_TOP_FEATURES')
        plt.ylabel('Validation ROC AUC')
        plt.title('Validation ROC AUC vs N_TOP_FEATURES')
        plt.grid(True)
        plt.savefig(RESULT_PNG)
        print(f"\n所有測試完成，結果已儲存於 {RESULT_CSV} 與 {RESULT_PNG}")
    finally:
        # 還原原始檔案
        shutil.move(FEATURE_SCRIPT_BAK, FEATURE_SCRIPT)
        shutil.move(MODEL_SCRIPT_BAK, MODEL_SCRIPT)
        print("\n已還原 feature_select_top.py 與 train_model.py 原始內容。")
