import pickle
import random
import copy, re, traceback
import numpy as np
import pandas as pd
import os, time, json, csv
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import check_data_filter
import data_filtering
import config

def pso_predict(X, prior, likelihood):
    N = len(X)
    F = len(likelihood) 
    # 預測每個樣本，先初始 posterior
    posterior = np.tile(prior, (N, 1))   # 重複貼上 N 次 prior，形成一個 (N, C) 的矩陣

    for f in range(F):
        posterior *= likelihood[f][:, X[:,f]].T 

    # 多類別投票一樣取最大值，且為了解決平手問題，每個票數都加上隨機雜訊
    max_prob_index = np.argmax(posterior, axis = 1)

    # 生成預測的向量
    pred_vector = np.zeros_like(posterior, dtype=int)
    pred_vector[np.arange(len(pred_vector)), max_prob_index] = 1   # 將每個樣本機率最大的位置設為 1，其他位置設為 0

    # 實際預測的類別結果
    pred_class = max_prob_index
    return np.array(pred_vector), np.array(pred_class)

def bagging_predict(nj, df_predict, prior, p_Xi_Cj_dict, att_value_counts):
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

# 載入模型
def load_models(method, dataset_name):
    path = config.PATH
    model_path = os.path.join(path[method]["model_path"], f"{dataset_name}_models.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models

# 將 PSO 與 Bagging 模型混合並預測資料集
def test_save_models(file_path, target_column, model_config):
    k = model_config['k_folds']
    num_base_models = model_config['num_base_models']
    selection_model_nums = model_config['selection_model_nums']

    # 載入資料集
    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]  # 特徵名稱集合
    X = data.drop(columns = [target_column]).values
    y = data[target_column].values
    class_nums = len(np.unique(y))  # 計算類別數量
    sample_nums = len(X)

    test_accuracies = []  # 儲存每個 fold 集成模型對 test set 的預測準確率
    training_accuracies = []    # 儲存每個 fold 基本模型對 training set 的預測準確率
    data_filter_table = {f"fold_{fold + 1}": [] for fold in range(k)}  # 儲存每個基本模型對每個訓練樣本的預測向量
    X_reserved = {f"fold_{fold + 1}": [] for fold in range(k)}

    # 對 X、y 分別做 LabelEncode
    for i in range(X.shape[1]): # 針對每個特徵
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i]) 
    le = LabelEncoder()
    y = le.fit_transform(y)

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    # 載入模型
    pso_trenb_models = load_models("PSO_TRENB", dataset_name)
    bagging_models = load_models("Bagging", dataset_name)

    for fold in range(k):
        pso_fold_data = pso_trenb_models[f"fold_{fold + 1}"]
        bagging_fold_data = bagging_models[f"fold_{fold + 1}"]

        # 還原測試集索引 (用其中一個還原就好，兩者的 test_data 一樣)
        test_index = bagging_fold_data["test_index"]
        X_test = X[test_index]
        y_test = y[test_index]

        # 還原訓練集索引
        train_index = [i for i in range(sample_nums) if i not in test_index]
        X_train = X[train_index]
        y_train = y[train_index]

        train_data = pd.DataFrame(X_train, columns=attributes)
        train_data['class'] = y_train
        test_data = pd.DataFrame(X_test, columns=attributes)
        test_data['class'] = y_test

        # 取得兩種模型
        pso_temp_models = pso_fold_data["PSO_TRENB"]
        bagging_trenb_models = bagging_fold_data["Bagging"]

        # 每個基本模型對測試集樣本的預測
        model_test_predictions = []

        # 隨機抽取 25 個模型
        selected_bagging_models = random.sample(bagging_trenb_models, k=25)
        selected_pso_models = random.sample(pso_temp_models, k=25)

        """
        以下開始進行訓練集與測試集預測 (還未集成挑選)
        """

        # Bagging 方法預測
        for model in selected_bagging_models:
            prior, p_Xi_Cj_dict, att_value_counts, nj = model   
            # 訓練集預測
            pred_vector, pred_class = bagging_predict(nj, train_data, prior, p_Xi_Cj_dict, att_value_counts)
            training_accuracies.append(np.mean(y_train == pred_class))  # 記錄訓練集準確率
            data_filter_table[f"fold_{fold + 1}"].append(pred_vector)  # 儲存該基本模型對所有樣本的預測向量

            # 測試集預測
            pred_vector, pred_class = bagging_predict(nj, test_data, prior, p_Xi_Cj_dict, att_value_counts)
            model_test_predictions.append(pred_class)  # 基本模型 i 對所有測試樣本的預測結果

        # PSO 方法預測測
        for model in selected_pso_models:
            prior, likelihood = model 
            pred_vector, pred_class = pso_predict(X_train, prior, likelihood)
            training_accuracies.append(np.mean(y_train == pred_class))  
            data_filter_table[f"fold_{fold + 1}"].append(pred_vector)  

            # 測試集預測
            pred_vector, pred_class = pso_predict(X_test, prior, likelihood)
            model_test_predictions.append(pred_class)  # 基本模型 i 對所有測試樣本的預測結果


        # 將 fold_{fold + 1} 的資料過濾表轉置，shape = (N, B, C)
        data_filter_table[f"fold_{fold + 1}"] = np.array(data_filter_table[f"fold_{fold + 1}"]).transpose(1, 0, 2).tolist()
        # 進行資料過濾
        X_reserved[f"fold_{fold + 1}"]  = data_filtering.process_filtering(y_train, data_filter_table[f"fold_{fold + 1}"], num_base_models, selection_model_nums, fold)

        # --- 進行測試集集成投票 ---
        model_test_predictions = np.array(model_test_predictions).T  # 轉為 numpy array 且轉置才能做投票

        # 集成所有基本模型的預測成為最終預測 (投票)
        ensemble_test_predictions = np.apply_along_axis(
            lambda x : np.bincount(x.astype(int), minlength = class_nums).argmax(),
            axis = 1, arr = model_test_predictions
        )
        test_accuracy = np.mean(y_test == ensemble_test_predictions)
        test_accuracies.append(test_accuracy)  # 儲存該 fold 的集成模型測試集準確率
        
    path = config.PATH.get("PSO_Bagging")

    # 將所有預測資訊寫入
    write_json_data(path["training_accuracy_path"], dataset_name, training_accuracies)  # 將五折交叉驗證中的訓練集樣本預測正確率寫入
    write_json_data(path["training_pred_vector_path"], dataset_name, data_filter_table)  # 將五折交叉驗證中的訓練集樣本預測向量寫入
    write_json_data(path["data_filter_reserved_path"] , dataset_name, X_reserved) 
    
    return np.mean(training_accuracies), np.mean(test_accuracies)

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
    path = config.PATH.get("PSO_Bagging")
    
    # 處理多類別資料
    data_folder = "datasets/離散化資料集/多類別" # 使用離散化後的資料
    dataset_list = config.DATASET_LIST
    df_res_path = path["data_filter_result_path"]

    # 建立 csv 檔案，用以儲存 PSO_Bagging 的 training 和 test 準確率
    with open(path["log_file"], mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Training_Accuracy", "Test_Accuracy", "Time"])
    
    # 建儲存資料過濾筆數與比例
    with open(df_res_path, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "avg filtered count", "avg filtered rate"])

    # 建立儲存集成挑選後的模型索引，以及集成挑選後的測試集正確率
    with open(path["es_result_path"], mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset"] + ["fold"] + [str(i+1) for i in range(25)] + ["Obj"] + ["Test Accuracy"])

    k = model_config['k_folds']
    num_base_models = model_config['num_base_models']

    for filename in dataset_list: # 處理每個資料集
        print(f"處理資料集: {filename}")
        file_path = os.path.join(data_folder,filename + '.csv')
        target_column = "class"    # 類別欄位設為'class'

        # 進行 PSO_TRENB 訓練，得到五折交叉驗證後的訓練集
        training_accuracy, test_accuracy  = test_save_models(file_path, target_column, model_config)
            
        # 將兩個準確率 寫入 csv（使用 append 模式，避免被覆蓋）
        with open(path["log_file"], mode='a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, training_accuracy, test_accuracy] )
            
            parent_path = Path.cwd()
            data_filter_var_path = os.path.join(parent_path, "data_filter_var.xlsx")

            # 檢查資料過濾正確性，寫過濾結果
            check_data_filter.check_filtering_result(data_filter_var_path, model_config, filename, df_res_path)

