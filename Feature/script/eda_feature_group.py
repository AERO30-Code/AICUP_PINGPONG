import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_PATH = "Feature/output/features_train_zscore.csv"
FIGURE_DIR = "Feature/figures"

LABEL_COLS = ["gender"]
SENSORS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
STATS = ["mean", "median"] # "mean", "std", "min", "max", "median"
SEGMENTS = list(range(27))  # 0-26

# ====== 使用者可自訂：只畫哪些感測軸和統計量 ======
# 例如只畫 Ax, Ay 的 mean：SELECTED_SENSORS = ["Ax", "Ay"]，SELECTED_STATS = ["mean"]
SELECTED_SENSORS = SENSORS
SELECTED_STATS = STATS


def main():
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_PATH)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 只針對 SELECTED_SENSORS 和 SELECTED_STATS 畫箱型圖
    for sensor in SELECTED_SENSORS:
        for stat in SELECTED_STATS:
            fig, axes = plt.subplots(3, 9, figsize=(27, 9), sharey=False)
            axes = axes.flatten()
            for seg in SEGMENTS:
                col = f"segment{seg}_{sensor}_{stat}"
                if col not in df.columns:
                    continue
                ax = axes[seg]
                sns.boxplot(x="gender", y=col, data=df, ax=ax)
                ax.set_title(f"segment{seg}")
                ax.set_xlabel("")
                ax.set_ylabel(col)
            plt.suptitle(f"{sensor}_{stat} by gender (27 segments)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            label_str = "_".join(LABEL_COLS)
            fig_path = os.path.join(FIGURE_DIR, f"boxplot_{sensor}_{stat}_by_{label_str}.png")
            # fig_path = os.path.join(FIGURE_DIR, "zscore", f"boxplot_{sensor}_{stat}_by_{label_str}_zscore.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"Saved {fig_path}")

if __name__ == "__main__":
    main()