# Model Training Leaderboard Record (Gender)

leaderboard score * 4 - 1.5 = real score

---
### 測試不同基於樹的模型
**Experiment 1**

*   **Experiment Path:** `Model/output/20250424_150853`
*   **Model:** `xgboost_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
    ```
*   **Leaderboard Score:** 0.60316654

**Experiment 2**

*   **Experiment Path:** `Model/output/20250424_150937`
*   **Model:** `RandomForest_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 200, 'max_depth': 8, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.58577194

**Experiment 3**

*   **Experiment Path:** `Model/output/20250424_151014`
*   **Model:** `lightgbm_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
    ```
*   **Leaderboard Score:** 0.60775533

**Experiment 4**

*   **Experiment Path:** `Model/output/20250424_151057`
*   **Model:** `catboost_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'iterations': 200, 'depth': 8, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': False, 'thread_count': -1}
    ```
*   **Leaderboard Score:** 0.59949770

**Experiment 5**

*   **Experiment Path:** `Model/output/20250424_152656`
*   **Model:** `lightgbm_gender`
*   **Feature Path:** `Feature/output/features_train_mean_median_std.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
    ```
*   **Leaderboard Score:** 0.60506890
---
### 測試 svm logisticregression 不同正規化方法
**Experiment 6**

*   **Experiment Path:** `Model/output/20250424_153225`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61169126

**Experiment 7**

*   **Experiment Path:** `Model/output/20250424_153448`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** minmax
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60036610

**Experiment 8**

*   **Experiment Path:** `Model/output/20250424_160007`
*   **Model:** `logisticregression_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.57938543

**Experiment 9**

*   **Experiment Path:** `Model/output/20250424_160027`
*   **Model:** `logisticregression_gender`
*   **Feature Path:** `Feature/output/features_train.csv`
*   **Scaling:** minmax
*   **Parameters:**
    ```python
    {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.58044594
---
### 測試 svm 訓練資料只保留 Az_median
**Experiment 10**

*   **Experiment Path:** `Model/output/20250424_161522`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_train_Az_median.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.55659923

**Experiment 11**

*   **Experiment Path:** `Model/output/20250424_161537`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_train_Az_median.csv`
*   **Scaling:** minmax
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.55059226
---
### 測試 svm 訓練資料只保留「基於樹模型的重要特徵 不同top數」4models
**Experiment 12**

*   **Experiment Path:** `Model/output/20250424_170212`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/features_train_4models_top500.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61219887

**Experiment 13**

*   **Experiment Path:** `Model/output/20250424_170219`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/features_train_4models_top200.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60970611

**Experiment 14**

*   **Experiment Path:** `Model/output/20250424_170224`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/features_train_4models_top100.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60404119
---
### 測試 svm 訓練資料只保留「基於樹模型的重要特徵 不同top數」2models
**Experiment 15**

*   **Experiment Path:** `Model/output/20250424_170508`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/features_train_2models_top500.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61174436

**Experiment 16**

*   **Experiment Path:** `Model/output/20250424_170513`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/features_train_2models_top200.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60759758

**Experiment 17**

*   **Experiment Path:** `Model/output/20250424_170517`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/features_train_2models_top100.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.59598660
---
### 測試 svm 訓練資料用 katsu 的資料 
**Experiment 18**

*   **Experiment Path:** `Model/output/20250424_172147`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/katsu/train_noplayerid.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61248313
---
### 測試 svm 訓練資料用 katsu 的資料 砍前面揮拍
**Experiment 19**

*   **Experiment Path:** `Model/output/20250424_183636`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/katsu/train_noplayerid_swing5to27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.6136723
---
### 測試 svm 訓練資料用 katsu 的資料 砍前面揮拍 不同基於樹的模型
**Experiment 20**

*   **Experiment Path:** `Model/output/20250425_153905`
*   **Model:** `xgboost_gender`
*   **Feature Path:** `Feature/katsu/train_noplayerid_swing5to27.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
    ```
*   **Note:** Not uploaded to leaderboard, mainly for obtaining feature importance from tree-based models.

**Experiment 21**

*   **Experiment Path:** `Model/output/20250425_153922`
*   **Model:** `RandomForest_gender`
*   **Feature Path:** `Feature/katsu/train_noplayerid_swing5to27.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 200, 'max_depth': 8, 'random_state': 42}
    ```
*   **Note:** Not uploaded to leaderboard, mainly for obtaining feature importance from tree-based models.

**Experiment 22**

*   **Experiment Path:** `Model/output/20250425_153939`
*   **Model:** `lightgbm_gender`
*   **Feature Path:** `Feature/katsu/train_noplayerid_swing5to27.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
    ```
*   **Note:** Not uploaded to leaderboard, mainly for obtaining feature importance from tree-based models.

**Experiment 23**

*   **Experiment Path:** `Model/output/20250425_154001`
*   **Model:** `catboost_gender`
*   **Feature Path:** `Feature/katsu/train_noplayerid_swing5to27.csv`
*   **Scaling:** none
*   **Parameters:**
    ```python
    {'iterations': 200, 'depth': 8, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': False, 'thread_count': -1}
    ```
*   **Note:** Not uploaded to leaderboard, mainly for obtaining feature importance from tree-based models.
---
### 測試 svm 訓練資料用 katsu 的資料 砍前面揮拍 只保留「基於樹模型的重要特徵 不同top數」4models
**Experiment 24** BEST

*   **Experiment Path:** `Model/output/20250425_161611`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top500.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61533511

**Experiment 25**

*   **Experiment Path:** `Model/output/20250425_161615`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top200.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61513207

**Experiment 26**

*   **Experiment Path:** `Model/output/20250425_161618`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top100.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60552966
---
### 測試 svm 訓練資料用 katsu 的資料 砍前面揮拍 只保留「基於樹模型的重要特徵 不同top數」4models 調整參數
**Experiment 27**

*   **Experiment Path:** `Model/output/20250425_170409`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top500.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61107119

**Experiment 28**

*   **Experiment Path:** `Model/output/20250425_171527`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top500.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'C': 0.003162, 'gamma': 0.000316, 'kernel': 'rbf', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.59220686

**Experiment 29**

*   **Experiment Path:** `Model/output/20250425_172552`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/output/features_gender_top/train_noplayerid_swing5to27_4models_top500.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'C': 0.003162, 'gamma': 0.000316, 'kernel': 'rbf', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.59220686
---
### 測試 svm 訓練資料用 katsu 的資料 砍前後揮拍 這邊亂測試的不準
**Experiment 30**

*   **Experiment Path:** `Model/output/20250501_180031`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/katsu/train_v2.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.59722361

**Experiment 31**

*   **Experiment Path:** `Model/output/20250501_180625`
*   **Model:** `svm_gender`
*   **Feature Path:** `Feature/katsu/train_v3.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60783499
---
### 測試 svm 重構的程式碼和之前的分數比對 不同拍數
**Experiment 32** (Exp18 0.61248313) 

*   **Experiment Path:** `Pipeline/output/20250502_043008`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s1-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61256747

**Experiment 33** (Exp19 0.6136723) 

*   **Experiment Path:** `Pipeline/output/20250502_043734`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61349835

**Experiment 34**

*   **Experiment Path:** `Pipeline/output/20250502_043758`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e24.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61214576
---
### 測試 svm 重構的程式碼 去掉GxGyGz 不同拍數
**Experiment 35**

*   **Experiment Path:** `Pipeline/output/20250502_061834`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s1-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60164684

**Experiment 36**

*   **Experiment Path:** `Pipeline/output/20250502_061856`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60214351

**Experiment 37**

*   **Experiment Path:** `Pipeline/output/20250502_061914`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e24.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60090026
---
### 測試 svm 重構的程式碼 s4-e27 只保留基於樹模型的重要特徵 4models 不同top數
**Experiment 38** (Exp24 0.61533511) top500 BEST

*   **Experiment Path:** `Pipeline/output/20250502_070240`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61499931

**Experiment 39** top300

*   **Experiment Path:** `Pipeline/output/20250502_070527`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61472129

**Experiment 40** top100

*   **Experiment Path:** `Pipeline/output/20250502_070538`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.60397872

**Experiment 40** top600

*   **Experiment Path:** `Pipeline/output/20250502_071333`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61452762

**Experiment 40** top400

*   **Experiment Path:** `Pipeline/output/20250502_071339`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s4-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61491653
---
### 測試 svm 重構的程式碼 s5-e27 只保留基於樹模型的重要特徵 4models 不同top數
**Experiment 41** top500

*   **Experiment Path:** `Pipeline/output/20250502_180217`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s5-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61466975

**Experiment 42** top400 

*   **Experiment Path:** `Pipeline/output/20250502_180239`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s5-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61518205 
---
#### 之後都 調整精度 .6f
---
**Experiment 43** top400 BEST

*   **Experiment Path:** `Pipeline/output/20250502_181245`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s5-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 0.61517892

**Experiment 44** top450

*   **Experiment Path:** `Pipeline/output/20250502_181633`
*   **Model:** `svm_gender`
*   **Feature Path:** `features_train_s5-e27.csv`
*   **Scaling:** zscore
*   **Parameters:**
    ```python
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    ```
*   **Leaderboard Score:** 