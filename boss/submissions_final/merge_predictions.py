\
import pandas as pd
import os
from datetime import datetime

def main():
    """
    主要功能：合併多個預測CSV檔案至一個參考CSV檔案的格式中。

    此腳本會讀取一個基準參考CSV檔案，然後將多個包含預測數據的CSV檔案
    （例如級別、握拍方式、打球年資和性別的預測）的內容更新至基準檔案中。
    更新是基於 'unique_id' 欄位進行的。
    如果預測檔案中的欄位名稱與參考檔案不完全匹配，將會輸出提示。
    最終合併後的檔案將以 'MMDD_HHMMSS.csv' 的格式儲存，並且浮點數會被格式化到指定精度。

    腳本設計為從 PINGPONG 專案根目錄執行。
    """
    # --- Configuration ---
    # Path to the reference CSV file (template for columns and initial values)
    # Relative from PINGPONG directory (where this script is executed)
    reference_csv_path = 'boss/code/fill_missing_predictions.csv'

    # Paths to the prediction CSV files
    # Paths are relative from PINGPONG directory
    predictions_files_info = [
        {
            # Level
            # "path": 'boss/submissions_final/only_target/only_levels_0522_140101.csv', # L1
            # "path": 'boss/submissions_final/only_target/only_levels_0522_162954.csv', # L7
            # "path": 'boss/submissions_final/only_target/only_levels_0523_113629.csv', # L8
            "path": 'boss/submissions_final/only_target/only_levels_0601_233801.csv', # L14
            "columns_to_use": None,  # None means use all columns except unique_id
            "id_col": "unique_id"
        },
        {
            # Hold racket handed
            "path": 'boss/submissions_final/0521_153128/holds_predictions.csv',
            "columns_to_use": None,
            "id_col": "unique_id"
        },
        {
            # Play years
            # "path": 'boss/submissions_final/only_target/only_play_years_0521_175053.csv', # P1
            # "path": 'boss/submissions_final/only_target/only_play_years_0522_122626.csv', # P2
            # "path": 'boss/submissions_final/only_target/only_play_years_0522_131944.csv', # P5
            "path": 'boss/submissions_final/only_target/only_play_years_0601_225025.csv', # P6

            "columns_to_use": ['play years_0', 'play years_1', 'play years_2'],
            "id_col": "unique_id"
        },
        {
            # Gender
            "path": 'Pipeline_reconstruction/outputs/submissions/submission_20250521_144623.csv', # G1
            # "path": 'Pipeline_reconstruction/outputs/submissions/submission_20250521_160424.csv', # G2
            # "path": 'Pipeline_reconstruction/outputs/submissions/submission_20250521_162003.csv', # G3

            "columns_to_use": ['gender'],  # Only use 'gender' column from this file
            "id_col": "unique_id"
        }
    ]

    # Output directory for the merged CSV file
    # Relative from PINGPONG directory
    output_dir = 'boss/submissions_final/'
    
    # Desired float precision for the output CSV file (e.g., '%.4f' for 4 decimal places)
    output_float_format = '%.4f'

    # --- End Configuration ---

    print(f"Processing reference file: {reference_csv_path}")
    try:
        base_df = pd.read_csv(reference_csv_path)
    except FileNotFoundError:
        print(f"Error: Reference file '{reference_csv_path}' not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading reference file '{reference_csv_path}': {e}")
        return

    original_columns = base_df.columns.tolist()
    print(f"Reference original columns: {original_columns}")

    if 'unique_id' not in base_df.columns:
        print("Error: Reference file missing 'unique_id' column. Exiting.")
        return

    try:
        base_df = base_df.set_index('unique_id', drop=True)
    except KeyError:
        print("Error: Failed to set 'unique_id' as index for reference file.")
        return

    for file_info in predictions_files_info:
        pred_file_path = file_info["path"]
        specified_cols_to_use = file_info["columns_to_use"]
        id_col = file_info["id_col"]
        
        print(f"\nProcessing prediction file: {pred_file_path}")

        try:
            pred_df = pd.read_csv(pred_file_path)
        except FileNotFoundError:
            print(f"  Info: Prediction file '{pred_file_path}' not found. Skipped.")
            continue
        except Exception as e:
            print(f"  Error reading prediction file '{pred_file_path}': {e}. Skipped.")
            continue

        if id_col not in pred_df.columns:
            print(f"  Info: Prediction file '{pred_file_path}' missing ID column '{id_col}'. Skipped.")
            continue

        potential_data_cols = []
        if specified_cols_to_use:
            potential_data_cols = [col for col in specified_cols_to_use if col in pred_df.columns and col != id_col]
            missing_spec_cols = [col for col in specified_cols_to_use if col not in pred_df.columns]
            if missing_spec_cols:
                print(f"  Info: Prediction file '{pred_file_path}' missing specified data columns: {missing_spec_cols}.")
        else:
            potential_data_cols = [col for col in pred_df.columns if col != id_col]

        if not potential_data_cols:
            print(f"  Info: No usable data columns found in '{pred_file_path}' (excluding ID). Skipped.")
            continue

        valid_update_cols_for_this_file = []
        # cols_to_select_from_pred_df = [id_col] # Re-initialize for clarity, though not strictly needed here

        for col in potential_data_cols:
            if col not in base_df.columns: # Check against base_df's columns (index is 'unique_id' at this point)
                print(f"  Info: Column '{col}' from '{pred_file_path}' not in reference file. Will be ignored.")
            else:
                valid_update_cols_for_this_file.append(col)
                # cols_to_select_from_pred_df.append(col) # This logic is handled below more cleanly
        
        if not valid_update_cols_for_this_file:
            print(f"  Info: No columns from '{pred_file_path}' could be mapped to reference file. Update skipped.")
            continue
        
        try:
            # Select only id_col and valid_update_cols_for_this_file for the update
            cols_for_pred_df_subset = [id_col] + valid_update_cols_for_this_file
            # Ensure unique columns in case id_col was somehow in valid_update_cols_for_this_file
            pred_df_for_update = pred_df[list(dict.fromkeys(cols_for_pred_df_subset))].copy()
            pred_df_for_update = pred_df_for_update.set_index(id_col, drop=True)
        except KeyError:
            print(f"  Info: Error selecting columns or setting index '{id_col}' for '{pred_file_path}'. Skipped.")
            continue
        except Exception as e:
            print(f"  Error preparing prediction file '{pred_file_path}' for update: {e}. Skipped.")
            continue
            
        print(f"  Updating data using columns {valid_update_cols_for_this_file} from '{pred_file_path}'...")
        base_df.update(pred_df_for_update)

    base_df.reset_index(inplace=True)

    try:
        base_df = base_df[original_columns]
    except KeyError as e:
        print(f"Warning: Error reordering columns to original. Missing columns: {e}")
        print(f"Current columns: {base_df.columns.tolist()}")
        print("Saving with current column order or subset of original columns.")
        present_original_columns = [col for col in original_columns if col in base_df.columns]
        if present_original_columns:
            base_df = base_df[present_original_columns]


    current_time_str = datetime.now().strftime('%m%d_%H%M%S')
    output_filename = f"{current_time_str}.csv"
    
    # Ensure output directory exists (relative to CWD)
    os.makedirs(output_dir, exist_ok=True) 
    final_output_path = os.path.join(output_dir, output_filename)

    try:
        base_df.to_csv(final_output_path, index=False, float_format=output_float_format)
        print(f"\nSuccessfully merged files. Output: {final_output_path} (float format: {output_float_format})")
    except Exception as e:
        print(f"Error saving merged file to '{final_output_path}': {e}")

if __name__ == '__main__':
    main()
