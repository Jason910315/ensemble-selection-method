import numpy as np
import pandas as pd
import os

def process_filtering(y_train, data_filter_table, m, b, fold):
    """
    執行多類別資料過濾，並將過濾結果寫入 Excel
    Input:
        y_train: 訓練集 y
        data_filter_table: 資料過濾表，shape = (樣本數, 基本模型數, 類別數) (N, B, C)
        m: 集成挑選選擇模型數量
        b: 集成挑選選擇模型數量
    Return:
        指定資料集保留下來的訓練樣本
    """

    class_nums = len(np.unique(y_train))

    # 計算每個樣本所得到的每個類別票數
    all_samples_votes = np.sum(data_filter_table, axis = 1)  # shape(N, C)

    X_reserved = []

    # 初始化用來存 Excel 資料的 List
    excel_data_rows = [] 

    # 檔案路徑
    excel_file = 'data_filter_var.xlsx'

    for j in range(len(y_train)):
        true_class = y_train[j]
        temp_votes = all_samples_votes[j].copy()
        
        # 計算樣本 j 的真實類別票數
        Vyj = temp_votes[true_class]
        temp_votes[true_class] = -1  # Mask
        max_Vcf = np.max(temp_votes) # 最大錯誤類別票數

        # 過濾條件
        is_reserved = True
        if ((Vyj - (m - b) > max_Vcf) or (Vyj < b / class_nums) or (Vyj + (m - b) < max_Vcf)):
            is_reserved = False
        else:
            X_reserved.append(j)
            is_reserved = True

        # 收集每一列的數據
        row_data = [j, Vyj, m, b, max_Vcf, class_nums, true_class, 1 if is_reserved else 0]
        excel_data_rows.append(row_data)

    """
    將資料過濾結果寫入 Excel
    """
    columns = ['Sample', 'Vyj', 'm', 'b', 'Max Vcf', 'class_nums', 'true_class', 'reserved']
    
    # 建立 DataFrame
    result_df = pd.DataFrame(excel_data_rows, columns=columns)
    result_df.set_index('Sample', inplace=True)

    # 如果檔案存在就用 'a'，不存在就用 'w'
    mode = 'a' if os.path.exists(excel_file) else 'w'
    if_sheet_exists = 'replace' if mode == 'a' else None
        
    try:   
        # 寫入檔案
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            result_df.to_excel(writer, sheet_name=f"fold_{fold + 1}")
    except Exception as e:
        print(f"Error writing to Excel: {e}")

    return X_reserved