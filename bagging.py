import pickle
import random, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import os, time, json, csv
from pathlib import Path
from tqdm import tqdm
import config
import data_filtering
import check_data_filter

def naive_bayes_classifier(df_train, attributes, class_nums):
    """
    使用訓練資料集建構簡易貝氏分類器，attributes: 特徵集合
    Return:
        prior: 先驗機率
        p_Xi_Cj_dict: 條件機率
    """
    all_classes = np.arange(class_nums)  # 將類別數轉為列表
    nj_counts = df_train['class'].value_counts()
    # 缺失類別用 0 補齊
    nj = nj_counts.reindex(all_classes, fill_value=0)

    prior = nj / len(df_train)    # p(cj)
    att_value_counts = {x: df_train[x].nunique() for x in attributes}  # 計算每個屬性 x 的可能值個數
    
    p_Xi_Cj_dict = {}
    # 針對每個特徵 x
    for att in attributes:
        # 依照 (att, class) 分組，並計算每個組合的出現次數，最後轉置成 (att, class) 的 DataFrame
        nij = df_train.groupby([att, 'class']).size().unstack(
            fill_value=0)  # p(xi | cj)
        p_Xi_Cj_dict[att] = (nij + 1) / (nj + att_value_counts[att])  # Laplace estimator
    # p_Xi_Cj_dict 的格式為 {特徵名稱: 儲存特徵類別組合機率的 DataFrame}
    return prior, p_Xi_Cj_dict, att_value_counts

def predict(df_train, df_predict, prior, p_Xi_Cj_dict, att_value_counts):

    """
    根據 df_train 得到的機率，預測 df_predict 的樣本類別 (多類別)
    Return:
        1. 樣本的預測向量，shape = (N, C)，裡面每個元素為 1d array，裡面有 C 個元素，代表 C 個類別的預測與否，1/0
        2. 所有樣本實際預測類別，shape = (N, )
    """
    nj = df_train['class'].value_counts()  
    pred_class = []
    pred_vector = []
    # 預測每個樣本
    for i in range(len(df_predict)):
        attributes = df_predict.iloc[i, :-1]   # 每個樣本的特徵值不同，要個別取出
        max_prob = -np.inf
        pred_class_i = None   # 樣本 i 的預測類別
        vector_i = np.zeros_like(prior, dtype=int)   # 樣本 i 的預測向量

        # 計算第 i 個樣本的特徵值組合在 Cj 類別下發生機率
        for Cj in nj.index:
            class_condition = 1  # class condition 就是類別下特徵組合機率
            # 計算每個特徵值在樣本 i 的出現機率
            for att in attributes.index:
                if pd.notna(df_predict[att][i]):  # 忽略NaN機率值
                    try:
                        # 取出特徵 att 的 likelihood DataFrame，再取出 Cj 欄位下，att 值的對應機率
                        class_condition *= p_Xi_Cj_dict[att][Cj][attributes[att]]
                    except KeyError:
                        class_condition *= 1 / (nj[Cj] + att_value_counts[att])  # 如果值不在訓練集中，使用拉普拉斯平滑
            # 後驗機率
            posterior_prob = prior[Cj] * class_condition
           
            if posterior_prob > max_prob:
                pred_class_i = Cj
                max_prob = posterior_prob
        
        vector_i[pred_class_i] = 1
        pred_vector.append(vector_i)
        pred_class.append(pred_class_i)
    
    return np.array(pred_vector), np.array(pred_class)

def cross_validation_with_ensemble(file_path, target_column, model_config, dataset_name):
    """
    進行五折交叉驗證訓練，以下每個步驟都是在每一折內各自進行
    1. 每一折訓練產生 25 個基本模型 (Bagging)
    2. 每個基本模型預測每個訓練樣本建構資料過濾表
    3. 進行資料過濾
    4. 使用過濾後資料集進行最佳化集成挑選
    5. 進行集成，預測測試集樣本
    """
    # 從 model_config 中解析模型參數
    k = model_config['k_folds']
    num_base_models = model_config['num_base_models']
    selection_model_nums = model_config['selection_model_nums']

    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]  # 特徵名稱集合
    X = data.drop(columns = [target_column]).values
    y = data[target_column].values
    class_nums = len(np.unique(y))  # 計算類別數量

   # 對 X 別做 LabelEncode，防止空箱問題產生
    for i in range(X.shape[1]): # 針對每個特徵
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i]) 

    start_time = time.time()  # 計時開始

    # 分割訓練與測試
    kf = KFold(n_splits=k, shuffle=True, random_state=42) 

    training_accuracies = []    # 儲存每個 fold 基本模型對 training set 的預測準確率
    test_accuracies = []    # 儲存每個 fold 集成模型對 test set 的預測準確率
    data_filter_table = {f"fold_{fold + 1}": [] for fold in range(k)}  # 儲存每個基本模型對每個訓練樣本的預測向量

    fold_models = {}  # 儲存每個折數的模型 

    X_reserved = {f"fold_{fold + 1}": [] for fold in range(k)}  # 儲存資料過濾結果

    # 進行五折交叉驗證訓練
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train_data = pd.DataFrame(X_train, columns=attributes)
        train_data['class'] = y_train
        test_data = pd.DataFrame(X_test, columns=attributes)
        test_data['class'] = y_test

        # fold_data 儲存模型資訊
        fold_data = {
            "test_index": test_index.tolist(),
            "train_class": y_train.tolist(),
            "class_nums": class_nums,
        }
        bagging_models = []  # 記錄訓練出的模型

        # 計算訓練集樣本數
        N = len(X_train)
        # 每個基本模型對測試集樣本的預測，單一折數中基本模型的預測結果，(樣本數, 模型數)
        model_test_predictions = np.zeros((len(test_data), num_base_models), dtype=int)   

        # 集成 num_base_models 個 base models 的預測結果  
        nums = 0
        with tqdm(total = num_base_models, desc = f"Fold {fold + 1} - Building Models") as pbar:
            while nums < num_base_models:
                # 生成 0 ~ N-1 範圍內的隨機亂數，總共生成 N 個
                # 這步驟代表 bagging 的取後放回抽樣，陣列裡的每個元素即為抽到的訓練集樣本索引
                sampled_indices = random.choices(range(N), k = N)
                bag_train_data = train_data.iloc[sampled_indices]

                fold_data["bag_train_data"] = bag_train_data

                # 計算訓練資料集的先驗機率、似然機率
                prior, p_Xi_Cj_dict, att_value_counts = naive_bayes_classifier(bag_train_data, attributes, class_nums)
                bagging_models.append((prior, p_Xi_Cj_dict, att_value_counts))

                # 對原始訓練集的預測
                pred_vector, pred_class = predict(bag_train_data, train_data, prior, p_Xi_Cj_dict, att_value_counts)
                training_accuracies.append(np.mean(y_train == pred_class))  # 記錄訓練集準確率
                data_filter_table[f"fold_{fold + 1}"].append(pred_vector)  # 儲存該基本模型對所有樣本的預測向量
                
                # 測試集預測
                pred_vector, pred_class = predict(bag_train_data, test_data, prior, p_Xi_Cj_dict, att_value_counts)
                model_test_predictions[:,nums] = pred_class  # 基本模型 i 對所有測試樣本的預測結果

                nums += 1
                pbar.update(1)

        fold_data["Bagging"] = bagging_models  # 記錄所有 Bagging 基本模型資訊

        # 保存該折模型與樣本資訊
        fold_models[f"fold_{fold + 1}"] = fold_data
                
        # 集成所有基本模型的預測成為最終預測 (投票)
        ensemble_test_predictions = np.apply_along_axis(
            lambda x : np.bincount(x.astype(int), minlength = class_nums).argmax(),
            axis = 1, arr = model_test_predictions
        )

        test_accuracy = np.mean(y_test == ensemble_test_predictions)
        test_accuracies.append(test_accuracy)  # 儲存該 fold 的集成模型測試集準確率

        # 將 fold_{fold + 1} 的資料過濾表轉置，shape = (N, B, C)
        data_filter_table[f"fold_{fold + 1}"] = np.array(data_filter_table[f"fold_{fold + 1}"]).transpose(1, 0, 2).tolist()

        # 計算資料過濾所需變數
        X_reserved[f"fold_{fold + 1}"] = data_filtering.process_filtering(y_train, data_filter_table[f"fold_{fold + 1}"], num_base_models, selection_model_nums, fold)
    
    end_time = time.time()
    exec_time = end_time - start_time  # 訓練結束，計算總時間

    path = config.PATH.get("Bagging")

    # 將所有預測資訊寫入
    write_json_data(path["training_accuracy_path"], dataset_name, training_accuracies)  # 將五折交叉驗證中的訓練集樣本預測正確率寫入
    write_json_data(path["training_pred_vector_path"], dataset_name, data_filter_table)  # 將五折交叉驗證中的訓練集樣本預測向量寫入
    write_json_data(path["data_filter_reserved_path"] , dataset_name, X_reserved) 

    # 保存模型至文件
    output_path = os.path.join(path["model_path"], f"{dataset_name}_models.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(fold_models, f)

    print(f"模型已保存至 {output_path}")

    return np.mean(training_accuracies), np.mean(test_accuracies), exec_time

# 寫入 json 檔案，並壓縮內層
def write_json_data(path, dataset_name, content):
    with open(path, "r", encoding="utf-8") as r:
        json_data = json.load(r)
       
    json_data[f"{dataset_name}"] = content
    json_str = json.dumps(json_data, ensure_ascii=False, indent=3)
    json_str = re.sub(
        r'\[\s*([0-9\.\,\s\-]+?)\s*\]', 
        lambda m: '[' + ', '.join([x.strip() for x in m.group(1).split(',')]) + ']',
        json_str
    )

    json_str = re.sub(
        r'(\[\s*(?:\[[0-9\.\,\s\-]+\]\s*,?\s*)+\])',
        lambda m: re.sub(r'\s+', ' ', m.group(1)).replace(' [', '[').replace('] ]', ']]'),
        json_str
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)

if __name__ == "__main__":
    random.seed(42)
    model_config = config.MODEL_CONFIG
    pso_config = config.PSO_CONFIG
    path = config.PATH.get("Bagging")
    
    # 處理多類別資料
    data_folder = "datasets/離散化資料集/多類別" # 使用離散化後的資料
    dataset_list = config.DATASET_LIST
    df_res_path = path["data_filter_result_path"]

    # 建儲存資料過濾筆數與比例
    with open(df_res_path, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "avg filtered count", "avg filtered rate"])

    # 建立 csv 檔案，用以儲存 PSO_TRENB 的 training 和 test 準確率
    with open(path["log_file"], mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Training_Accuracy", "Test_Accuracy", "Time"])

    for filename in dataset_list: # 處理每個資料集
        print(f"處理資料集: {filename}")
        file_path = os.path.join(data_folder,filename + '.csv')
        target_column = "class"    # 類別欄位設為'class'

        # 進行 TRENB 訓練，得到五折交叉驗證後的訓練集、測試集準確率
        training_accuracy, test_accuracy, exec_time = cross_validation_with_ensemble(file_path, target_column, model_config, filename)

        # 將兩個準確率 寫入 csv（使用 append 模式，避免被覆蓋）
        with open(path["log_file"], mode='a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, training_accuracy, test_accuracy, exec_time] )
            
            parent_path = Path.cwd()
            data_filter_var_path = os.path.join(parent_path, "data_filter_var.xlsx")

            check_data_filter.check_filtering_result(data_filter_var_path, model_config, filename, df_res_path)