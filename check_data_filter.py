from pathlib import Path
import numpy as np
import pandas as pd
import csv
import os

def check_filtering_result(data_filter_res_path, model_config, dataset_name, df_res_path):
    """
    根據資料過濾結果的 excel 檔，檢查五折交叉驗證的資料過濾正確性
    Input:
        X_reserved: 包含五折交叉驗證的過濾結果 (保留樣本的 index)
        m: 基本模型數量
        b: 集成挑選選擇模型數量
    """

    # 從 model_config 中解析模型參數
    m = model_config['num_base_models']
    b = model_config['selection_model_nums']

    # 讀取 Excel 並載入所有工作表
    excel_file = data_filter_res_path
    sheets = pd.read_excel(excel_file, sheet_name=None, index_col=0)  # 讀取所有工作表

    # 儲存那些被過濾錯誤或保留錯誤的樣本
    reserved_err_samples = {f"fold_{fold + 1}": [] for fold in range(5)}
    filtered_err_samples = {f"fold_{fold + 1}": [] for fold in range(5)}

    is_reserved_err = False
    is_filtered_err = False

    n_fold_filter = []  # 記錄每折的資料過濾筆數
    print("------------------------------------------------------\nStart checking data filtering result...")
    # 針對每個 fold 進行檢查
    for sheet_name, df in sheets.items():
        n_filter = 0
        print(f"Checking sheet: {sheet_name}")
        # 依序檢查每一筆樣本的資料過濾正確性
        for idx, row in df.iterrows():
            is_reserved, Vyj, max_Vcf, class_nums = row['reserved'], row['Vyj'], row['Max Vcf'], row['class_nums']
            # 被過濾樣本，檢查是否三個條件有任一成立
            if is_reserved == 0:
                if (not (Vyj - (m - b) > max_Vcf) and not (Vyj < b / class_nums) and not (Vyj + (m - b) < max_Vcf)):
                    filtered_err_samples[sheet_name].append(idx)
                    is_filtered_err = True
                n_filter += 1   # 資料過濾樣本加 1
            # 被保留的樣本，檢查三個條件是否都不成立
            else:
                if ((Vyj - (m - b) > max_Vcf) or (Vyj < b / class_nums) or (Vyj + (m - b) < max_Vcf)):
                    reserved_err_samples[sheet_name].append(idx)
                    is_reserved_err = True
        n_fold_filter.append(n_filter)
    """
    計算平均資料過濾比率
    """
    avg_filter_nums = sum(n_fold_filter) / 5
    avg_filter_rate = (avg_filter_nums / df.shape[0]) * 100

    # 將兩個準確率 寫入 csv（使用 append 模式，避免被覆蓋）
    with open(df_res_path, mode='a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name ,f"{avg_filter_nums}/{df.shape[0]}", f"{avg_filter_rate}%"] )
    
    for fold in range(5):
        if is_reserved_err:
            print(f"Fold {fold + 1} - Reserved Error Samples: {len(reserved_err_samples[f'fold_{fold + 1}'])}")
        elif is_filtered_err:
            print(f"Fold {fold + 1} - Filtered Error Samples: {len(filtered_err_samples[f'fold_{fold + 1}'])}")
    
    if not is_reserved_err and not is_filtered_err:
        print(f"{dataset_name} filtering is correct")

    

