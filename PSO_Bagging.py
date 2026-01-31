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
def test_save_models(file_path, target_column, k, num_base_models):
    # 載入資料集
    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]  # 特徵名稱集合
    X = data.drop(columns = [target_column]).values
    y = data[target_column].values
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

    path = config.PATH

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

        # 取得兩種模型
        pso_models = pso_fold_data["pso_models"]
        bagging_models = bagging_fold_data["bagging_models"]

        # 每個基本模型對測試集樣本的預測
        model_test_predictions = []

        # 隨機抽取 25 個 模型
        selected_bagging_models = random.choice(bagging_models, size=25)
        selected_pso_models = random.choice(pso_models, size=25)

        # 挑出基本模型使用 Bagging 方法預測
        for model in selected_bagging_models:
            prior, p_Xi_Cj_dict, nj, att_value_counts = model
            # 訓練集預測
            pred_vector, pred_class = bagging_predict(nj, train_data, prior, p_Xi_Cj_dict, att_value_counts)
            training_accuracies["Bagging"].append(np.mean(y_train == pred_class))  # 記錄訓練集準確率
            data_filter_table[f"fold_{fold + 1}"].append(pred_vector)  # 儲存該基本模型對所有樣本的預測向量
            # 測試集預測
            pred_vector, pred_class = bagging_predict(nj, test_data, prior, p_Xi_Cj_dict, att_value_counts)
            model_test_predictions.append(pred_class)  # 基本模型 i 對所有測試樣本的預測結果

        # 挑出基本模型使用 PSO 方法預測測
        for model in selected_pso_models:
            prior, likelihood = model 
            pred_vector, pred_class = pso_predict(X_train, prior, likelihood)
            training_accuracies["PSO"].append(np.mean(y_train == pred_class))  
            data_filter_table[f"fold_{fold + 1}"].append(pred_vector)  
            # 測試集預測
            pred_vector, pred_class = pso_predict(X_test, prior, likelihood)
            model_test_predictions.append(pred_class)  # 基本模型 i 對所有測試樣本的預測結果


        # 記錄訓練集預測的資訊
        training_accuracies_path = os.path.join(parent_path, "training_accuracies","PSO_Bagging.json")
        write_json_data(training_accuracies_path, dataset_name, training_accuracies)  # 將五折交叉驗證中的訓練集樣本預測正確率寫入

        # 將 fold_{fold + 1} 的資料過濾表轉置，shape = (N, B, C)
        data_filter_table[f"fold_{fold + 1}"] = np.array(data_filter_table[f"fold_{fold + 1}"]).transpose(1, 0, 2).tolist()
        # 進行資料過濾
        X_reserved[f"fold_{fold + 1}"]  = data_filtering(X_train, y_train, data_filter_table[f"fold_{fold + 1}"], num_base_models, selection_model_nums, fold)

        # --- 進行測試集集成投票 ---
        model_test_predictions = np.array(model_test_predictions).T  # 轉為 numpy array 且轉置才能做投票

        # 集成所有基本模型的預測成為最終預測 (投票)
        ensemble_test_predictions = np.apply_along_axis(
            lambda x : np.bincount(x.astype(int), minlength = class_nums).argmax(),
            axis = 1, arr = model_test_predictions
        )
        test_accuracy = np.mean(y_test == ensemble_test_predictions)
        test_accuracies.append(test_accuracy)  # 儲存該 fold 的集成模型測試集準確率
        
    all_accuracies = training_accuracies["PSO"] + training_accuracies["Bagging"]
    
    return np.mean(all_accuracies), np.mean(test_accuracies)

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
    k = 5                 # k 折交叉驗證次數
    num_base_models = 50   # 基本模型數量
    selection_model_nums = 25   # 集成挑選選擇模型數量
    np.random.seed(42)   # 設定全域的隨機種子

    # 初始化 PSO 參數 
    bounds = (1e-10, 1)
    num_particles = 50  # 粒子數
    max_iter = 100

    bagging_ratio = 0.6   # Bagging 模型比例
    
    # 處理多類別資料
    data_folder = "datasets/離散化資料集/多類別" # 使用離散化後的資料
    dataset_list = ["Abalone_3class","Acoustic","Balance Scale","Car","Choice","DryBean","Ecoli","Glass","Iris","Landsat","Leaf","Modeling","Performance","Seeds","Student","Vertebral","Winequality-red","Winequality-white","Yeast","Zoo"]
    dataset_list = ["Abalone_3class","Acoustic","Balance Scale","Choice","Iris","Student","Performance","Winequality-white","Yeast","Zoo"]

    # 建立 csv 檔案，用以儲存 PSO_Bagging 的 training 和 test 準確率
    csv_filename = f"PSO_Bagging.csv"
    with open(csv_filename, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "Training_Accuracy", "Test_Accuracy", "Time"])
    
    parent_path = Path.cwd()
    output_folder = os.path.join(parent_path, "temp_models")

    for filename in dataset_list: # 處理每個資料集
        print(f"開始生成並保存模型，處理資料集: {filename}...")
        file_path = os.path.join(data_folder, filename + '.csv')
        target_column = "class"    # 類別欄位設為'class'

        # 訓練 PSO 與 Bagging 模型
        exec_time = generate_and_save_models(file_path, target_column, k, num_base_models, num_particles, bounds, max_iter, output_folder, selection_model_nums)
        # 測試 PSO 與 Bagging 混合模型
        training_accuracy, test_accuracy = test_save_models(file_path, target_column, k, output_folder, num_base_models, bagging_ratio)


        # 將兩個準確率 寫入 csv（使用 append 模式，避免被覆蓋）
        with open(csv_filename, mode='a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, training_accuracy, test_accuracy, "Nan", exec_time] )



