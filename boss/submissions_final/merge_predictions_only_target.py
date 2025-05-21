\
import pandas as pd
import os
from datetime import datetime
import re

def get_target_theme_from_filename(filepath):
    """
    從預測檔案的路徑中提取目標主題。
    預期檔案名稱格式為 XXX_predictions.csv 或包含 XXX_predictions 的模式。
    例如：'play_years_predictions.csv' -> 'play_years'
           'some/path/levels_predictions.csv' -> 'levels'
    """
    filename = os.path.basename(filepath)
    match = re.match(r"^(.*?)_predictions\.csv", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback if the pattern is slightly different, e.g. part of a longer name
    if "_predictions" in filename:
        return filename.split("_predictions")[0]
    return "unknown_target"

def main():
    """
    主要功能：將特定預測檔案的數據更新至一個基準提交檔案中。

    此腳本會讀取一個完整的參考提交檔案（例如 sample_submission.csv）作為基礎。
    然後，它會讀取一個只包含部分欄位預測的特定預測檔案（例如 play_years_predictions.csv）。
    腳本會基於 'unique_id' 將特定預測檔案中的數據更新到參考提交檔案的對應欄位。
    特定預測檔案更新的欄位中的浮點數會被格式化到指定精度，
    而參考檔案中未被更新的欄位將保持其原始格式。
    腳本會檢查 unique_id 和欄位名稱的匹配情況，並輸出提示。
    最終合併後的檔案將儲存到 'boss/submissions_final/only_target/' 目錄下，
    檔名格式為 'only_目標主題_MMDD_HHMMSS.csv'。

    腳本設計為從 PINGPONG 專案根目錄執行。
    """
    # --- Configuration ---
    # Path to the main reference submission CSV file
    # Relative from PINGPONG directory (where this script is executed)
    reference_submission_path = 'Original/test/sample_submission.csv'

    # Path to the specific prediction CSV file to patch into the reference
    # Relative from PINGPONG directory
    # !! 請用戶根據實際情況修改此路徑 !!
    prediction_patch_file_path = 'boss/submissions_final/0521_153128/play_years_predictions.csv' # 範例路徑

    # Output directory for the patched submission CSV file
    # Relative from PINGPONG directory
    output_base_dir = 'boss/submissions_final/only_target/'
    
    # Desired float precision for the output CSV file (e.g., '%.4f' for 4 decimal places)
    output_float_format = '%.4f'
    
    id_column_name = 'unique_id'
    # --- End Configuration ---

    print(f"Starting patch process...")
    print(f"Reference submission file: {reference_submission_path}")
    print(f"Prediction patch file: {prediction_patch_file_path}")

    # Load reference submission file
    try:
        base_df = pd.read_csv(reference_submission_path)
    except FileNotFoundError:
        print(f"Error: Reference submission file '{reference_submission_path}' not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading reference submission file '{reference_submission_path}': {e}")
        return

    if id_column_name not in base_df.columns:
        print(f"Error: ID column '{id_column_name}' not found in reference submission file '{reference_submission_path}'. Exiting.")
        return
    original_columns_order = base_df.columns.tolist()

    # Load prediction patch file
    try:
        patch_df = pd.read_csv(prediction_patch_file_path)
    except FileNotFoundError:
        print(f"Error: Prediction patch file '{prediction_patch_file_path}' not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading prediction patch file '{prediction_patch_file_path}': {e}")
        return

    if id_column_name not in patch_df.columns:
        print(f"Error: ID column '{id_column_name}' not found in prediction patch file '{prediction_patch_file_path}'. Exiting.")
        return

    # Extract target theme for output filename
    target_theme = get_target_theme_from_filename(prediction_patch_file_path)
    print(f"Determined target theme: {target_theme}")

    # --- Unique ID Mismatch Checks ---
    base_ids = set(base_df[id_column_name])
    patch_ids = set(patch_df[id_column_name])

    ids_in_patch_not_in_base = patch_ids - base_ids
    if ids_in_patch_not_in_base:
        print(f"  Info: {len(ids_in_patch_not_in_base)} IDs found in patch file but not in reference file. These rows from patch file will be ignored for update if not in base. Example IDs: {list(ids_in_patch_not_in_base)[:5]}")

    ids_in_base_not_in_patch = base_ids - patch_ids
    if ids_in_base_not_in_patch:
        print(f"  Info: {len(ids_in_base_not_in_patch)} IDs found in reference file but not in patch file. These rows in reference will retain original values for patched columns. Example IDs: {list(ids_in_base_not_in_patch)[:5]}")

    # Set index for update
    try:
        base_df = base_df.set_index(id_column_name)
        patch_df = patch_df.set_index(id_column_name)
    except KeyError:
        print(f"Error: Failed to set '{id_column_name}' as index. This should not happen if previous checks passed.")
        return

    # --- Column Handling & Update ---
    patch_columns = [col for col in patch_df.columns if col in base_df.columns] # Columns in patch that are also in base
    ignored_patch_cols = [col for col in patch_df.columns if col not in base_df.columns]

    if ignored_patch_cols:
        print(f"  Info: The following columns from patch file '{prediction_patch_file_path}' are not in reference file and will be ignored: {ignored_patch_cols}")
    
    if not patch_columns:
        print(f"  Info: No common columns (other than ID) found between patch file and reference file. No update will be performed.")
    else:
        print(f"  Updating reference with columns from patch file: {patch_columns}")
        base_df.update(patch_df[patch_columns])

    # Reset index and reorder columns
    base_df.reset_index(inplace=True)
    try:
        # Ensure original column order, dropping any new columns that might have been added if update logic changes
        base_df = base_df[original_columns_order]
    except KeyError as e:
        print(f"Warning: Could not strictly enforce original column order. Missing original columns: {e}. Current columns: {base_df.columns.tolist()}")
        # Fallback: ensure all original columns that are still present are first, in order
        present_original_columns = [col for col in original_columns_order if col in base_df.columns]
        other_columns = [col for col in base_df.columns if col not in present_original_columns]
        base_df = base_df[present_original_columns + other_columns]

    # Apply specific float formatting only to the updated columns
    if patch_columns: # Ensure there were columns to update
        for col in patch_columns:
            if col in base_df.columns and pd.api.types.is_numeric_dtype(base_df[col]):
                # Apply formatting, ensuring NaNs are handled correctly
                base_df[col] = base_df[col].apply(
                    lambda x: (output_float_format % x) if pd.notnull(x) else x
                )
                print(f"  Applied format '{output_float_format}' to updated column: {col}")


    # --- Output ---
    current_time_str = datetime.now().strftime('%m%d_%H%M%S')
    output_filename = f"only_{target_theme}_{current_time_str}.csv"
    
    # Ensure output directory exists (relative to CWD which is PINGPONG root)
    os.makedirs(output_base_dir, exist_ok=True) 
    final_output_path = os.path.join(output_base_dir, output_filename)

    try:
        base_df.to_csv(final_output_path, index=False) # Removed global float_format
        print(f"\\nSuccessfully patched submission. Output: {final_output_path} (updated columns formatted to: {output_float_format})")
    except Exception as e:
        print(f"Error saving patched submission to '{final_output_path}': {e}")

if __name__ == '__main__':
    main()
