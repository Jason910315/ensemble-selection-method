import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

def calculate_equal_width_splits(df, att, nbin):
    """
    計算 equal width 切點，並回傳切點列表
    """
    split_points = []

    # 針對每個特徵，計算 equal width 切點
    for i in att:
        # 判斷特徵是否為數值型 (int、float)
        is_numeric = pd.api.types.is_numeric_dtype(df[i])

        # 若特徵為非數值型，不進行離散化，但須進行轉碼
        if not is_numeric:
            df[i] = pd.factorize(df[i])[0]

        # 若特徵的數值少於 10 個，不進行離散化，但仍須處理 encode
        # 不用 factorize 防止數值大小順序被打亂
        if df[i].nunique() <= 10:
            encoder = LabelEncoder()
            df[i] = encoder.fit_transform(df[i])
            split_points.append(None)
            continue

        max_value = df[i].max(skipna=True)         # 計算寬度
        min_value = df[i].min(skipna=True)
        width = (max_value - min_value) / nbin

        s = []         # 計算分割點
        for j in range(1, nbin):
            point = min_value + j * width
            s.append(point)
        split_points.append(s)
    return split_points

# 將特徵值轉為 bin 編號
def convert_value_to_bin(value, bins):
    for id, point in enumerate(bins):
        if value <= point:
            return id 
    return id + 1

def ew_discretize(df, att, split_points):
    convert_data = df.copy()
    for i, col in enumerate(att):
        # 若特徵不進行離散化，則跳過
        if split_points[i] is None:
            continue
        convert_data[col] = convert_data[col].apply(
            lambda x: convert_value_to_bin(x, split_points[i]))
    return convert_data

def transfer_class(df):
    """
    將 class 欄位轉為數值型，並且從 0 開始
    """
    is_numeric = pd.api.types.is_numeric_dtype(df['class'])
    if not is_numeric:
        df['class'] = pd.factorize(df['class'])[0]
    # 特徵 class 為數值型，還需使用 LabelEncoder 轉為使其 0 開始
    else:
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
    return df

# 原始資料夾和輸出資料夾
input_folder = 'datasets/原始資料集/多類別'  # 原始資料所在的資料夾
output_folder = "datasets/離散化資料集/多類別"  # 離散化後的資料夾
os.makedirs(output_folder, exist_ok=True)  # 如果文件夾不存在，則創建

# 離散化的參數
nbin = 10 # 分箱數量

# 處理資料夾中的所有 CSV 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # 只處理 CSV 文件
        try:
            file_path = os.path.join(input_folder, file_name)

            # 讀取資料
            df = pd.read_csv(file_path)
            n_columns = df.shape[1]
            df.columns = list(range(1, n_columns)) + ['class'] # 修改欄位名稱

            # 計算等寬分箱
            att = list(range(1, n_columns))
            split = calculate_equal_width_splits(df, att, nbin)

            # 進行等寬離散化
            df_discretized = ew_discretize(df, att, split)
            # 進行類別處理
            df_discretized = transfer_class(df_discretized)

            # 儲存處理後的文件
            output_file_path = os.path.join(output_folder, file_name)
            df_discretized.to_csv(output_file_path, index=False)
            print(f"離散化文件已儲存至: {output_file_path}")
        except Exception as e:
            print(f"處理檔案 {file_name} 時出現錯誤，跳過。錯誤信息: {e}")