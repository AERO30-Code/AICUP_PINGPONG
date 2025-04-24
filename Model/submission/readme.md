紀錄上傳成績 leaderboard*4-1.5=

path: Model/output/20250424_150853
target: gender
model: xgboost_gender
Feature path: Feature/output/features_train.csv
scaling: none
Parameters: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
leaderboard: 0.60316654

path: Model/output/20250424_150937
target: gender
model: RandomForest_gender
Feature path: Feature/output/features_train.csv
scaling: none
Parameters: {'n_estimators': 200, 'max_depth': 8, 'random_state': 42}
leaderboard: 0.58577194

path: Model/output/20250424_151014
target: gender
model: lightgbm_gender
Feature path: Feature/output/features_train.csv
scaling: none
Parameters: {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
leaderboard: 0.60775533

path: Model/output/20250424_151057
target: gender
model: catboost_gender
Feature path: Feature/output/features_train.csv
scaling: none
Parameters: {'iterations': 200, 'depth': 8, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': False, 'thread_count': -1}
leaderboard: 0.59949770
====================
path: Model/output/20250424_152656
target: gender
model: lightgbm_gender
Feature path: Feature/output/features_train_mean_median_std.csv
scaling: none
Parameters: {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
leaderboard: 0.60506890
====================
path: Model/output/20250424_153225
target: gender
model: svm_gender
Feature path: Feature/output/features_train.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 0.61169126

path: Model/output/20250424_153448
target: gender
model: svm_gender
Feature path: Feature/output/features_train.csv
scaling: minmax
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 0.60036610

path: Model/output/20250424_160007
target: gender
model: logisticregression_gender
Feature path: Feature/output/features_train.csv
scaling: zscore
Parameters: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
leaderboard: 0.57938543

path: Model/output/20250424_160027
target: gender
model: logisticregression_gender
Feature path: Feature/output/features_train.csv
scaling: minmax
Parameters: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
leaderboard: 0.58044594
====================
path: Model/output/20250424_161522
target: gender
model: svm_gender
Feature path: Feature/output/features_train_Az_median.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 0.55659923

path: Model/output/20250424_161537
target: gender
model: svm_gender
Feature path: Feature/output/features_train_Az_median.csv
scaling: minmax
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 0.55059226
====================
path: Model/output/20250424_170212
target: gender
model: svm_gender
Feature path: Feature/output/features_gender_top/features_train_4models_top500.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 0.61219887

path: Model/output/20250424_170219
target: gender
model: svm_gender
Feature path: Feature/output/features_gender_top/features_train_4models_top200.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 

path: Model/output/20250424_170224
target: gender
model: svm_gender
Feature path: Feature/output/features_gender_top/features_train_4models_top100.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 

path: Model/output/20250424_170508
target: gender
model: svm_gender
Feature path: Feature/output/features_gender_top/features_train_2models_top500.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 

path: Model/output/20250424_170513
target: gender
model: svm_gender
Feature path: Feature/output/features_gender_top/features_train_2models_top200.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 

path: Model/output/20250424_170517
target: gender
model: svm_gender
Feature path: Feature/output/features_gender_top/features_train_2models_top100.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 
====================
path: Model/output/20250424_172147
target: gender
model: svm_gender
Feature path: Feature/katsu/train_noplayerid.csv
scaling: zscore
Parameters: {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42}
leaderboard: 0.61248313

path: 
target: gender
model: 
Feature path: 
scaling: 
Parameters: 
leaderboard: 

path: 
target: gender
model: 
Feature path: 
scaling: 
Parameters: 
leaderboard: 

path: 
target: gender
model: 
Feature path: 
scaling: 
Parameters: 
leaderboard: 