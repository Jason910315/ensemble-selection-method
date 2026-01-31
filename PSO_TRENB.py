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

"""
1. 此程式的目的為訓練所有基本模型，記錄訓練集與測試集正確率(未集成挑選)
2. 並執行資料過濾，最後儲存模型資訊
"""

def random_naive_model(feature_value_counts, class_nums):
    """
    隨機生成簡易貝氏的先驗機率與似然機率
    Args:
        feature_value_counts: 一個列表，包含每個特徵的可能值數量
        class_nums: 類別數量
    """
    feature_nums = len(feature_value_counts)  
    # 1. 使用 dirichlet() 生成隨機的先驗機率，其總和為 1
    random_prior = np.random.dirichlet(np.ones(class_nums))

    # 2. 生成 likelihood (用 List 儲存不同形狀的陣列)
    random_likelihood = []

    for v_count in feature_value_counts:
        # 生成一個形狀為 (class_nums, v_count) 的隨機矩陣
        l_matrix = np.random.dirichlet(np.ones(v_count), size=class_nums)
        random_likelihood.append(l_matrix)
    return random_prior, random_likelihood

def predict(X, prior, likelihood):
    """
    根據機率，預測樣本類別 (多類別)
    Args:
        X: 樣本特徵矩陣 (N, F)
        prior: (C,)
        likelihood: List[array-like]，長度 F，每個元素的形狀為 (class_nums, v_count)
    Return:
        1. 樣本的預測結果，shape = (N, C)，裡面每個元素為 1d array，裡面有 C 個元素，代表 C 個類別的預測與否，1/0
        2. 所有樣本實際預測類別，shape = (N, )
    """
    N = len(X)
    F = len(likelihood) 
    # 預測每個樣本，先初始 posterior
    posterior = np.tile(prior, (N, 1))   # 重複貼上 N 次 prior，形成一個 (N, C) 的矩陣

    for f in range(F):
        # likelihood[f] 是第 f 個特徵的似然機率矩陣，(class_nums, v_count)
        # X[:,f] 是所有樣本在第 f 個特徵的值，(N, )
        # likelihood[f][:, X[:,f]] 會取出所有樣本在第 f 個特徵中，所有類別在該特徵值下的機率，(class_nums, N)
        posterior *= likelihood[f][:, X[:,f]].T 

    # 多類別投票一樣取最大值，且為了解決平手問題，每個票數都加上隨機雜訊
    max_prob_index = np.argmax(posterior, axis = 1)

    # 生成預測的向量
    pred_vector = np.zeros_like(posterior, dtype=int)
    pred_vector[np.arange(len(pred_vector)), max_prob_index] = 1   # 將每個樣本機率最大的位置設為 1，其他位置設為 0

    # 實際預測的類別結果
    pred_class = max_prob_index
    return np.array(pred_vector), np.array(pred_class)
    

# 每個粒子都代表一個模型，帶有兩個參數 - prior、likelihood
def PSO(X, y, feature_value_counts, class_nums, pso_config):
    """
    定義粒子群優化演算法，優化使用隨機生成產生的簡易貝氏模型
    Return:
        一個基本模型
    """
    # 從 pso_config 中解析超參數
    num_particles = pso_config['num_particles']
    max_iter = pso_config['max_iter']
    bounds = pso_config['bounds']
    w, c1, c2 = pso_config['w'], pso_config['c1'], pso_config['c2']

    # 初始化 fitness function 物件，後續用此可以挑選不同的 fitness function
    fitness_function = config.FitnessFunction()
    feature_nums = len(feature_value_counts)

    # 初始化粒子位置和速度
    positions = []
    velocities = []
    for i in range(num_particles):
        # 隨機初始化粒子先驗機率、似然機率 (做為粒子的位置參數)
        particle_prior, particle_likelihood = random_naive_model(feature_value_counts, class_nums)
        positions.append((particle_prior, particle_likelihood))

        # 隨機初始化先驗機率、似然機率的「速度」，速度的總和不需為 1
        prior_veclocity = np.random.uniform(bounds[0], bounds[1], size=class_nums) * 0.1  # 產生一個大小為 size 的隨機數陣列
        likelihood_velocity = []
        for v_count in feature_value_counts:
            l_vel = np.random.uniform(bounds[0], bounds[1], size=(class_nums, v_count)) * 0.1
            likelihood_velocity.append(l_vel)

        velocities.append((prior_veclocity, likelihood_velocity))

    positions = np.array(positions, dtype=object)
    velocities = np.array(velocities, dtype=object)

    # 使用每個粒子的初始機率預測，並計算初始適應值
    # predict 回傳兩個值，tuple : (pred_vector, pred_class)，[1] 是 pred_class
    pred_class_list = [predict(X, p[0], p[1])[1] for p in positions]
    fitness = np.array([fitness_function.cal_micro_accuracy(y, pred_class) for pred_class in pred_class_list])
    # 初始化個體最佳位置、適應值
    personal_best_positions = copy.deepcopy(positions)
    personal_best_fitness = fitness.copy()

    # 找出初始最佳適應值的粒子
    best_idx = np.argmax(fitness)
    global_best_positions = personal_best_positions[best_idx]
    global_best_fitness = personal_best_fitness[best_idx]

    # PSO 主循環
    for _ in range(max_iter):
        # 更新每個粒子的速度與位置
        for j in range(num_particles):
            # 取得第 j 個粒子的所有資訊
            prior, likelihood = positions[j]
            pbest_prior, pbest_likelihood = personal_best_positions[j]
            gbest_prior, gbest_likelihood = global_best_positions
            vel_prior, vel_likelihood = velocities[j]

            r1, r2 = np.random.random(2)

            # 更新 prior
            velocities[j][0] = w * vel_prior + c1 * r1 * (pbest_prior - prior) + c2 * r2 * (gbest_prior - prior)
            # 計算新 prior 位置，並正規化，使其總和為 1
            new_prior = np.clip(prior + velocities[j][0], bounds[0], bounds[1])
            new_prior /= new_prior.sum()

            # 更新 likelihood
            new_likelihood = []
            # 每個特徵個別更新
            for f in range(feature_nums):
                curr_l_pos = likelihood[f]
                new_vel_l = w * vel_likelihood[f] + c1 * r1 * (pbest_likelihood[f] - curr_l_pos) + c2 * r2 * (gbest_likelihood[f] - curr_l_pos)
                velocities[j][1][f] = new_vel_l  # 更新第 f 個特徵的 likelihood 速度

                # 更新特徵 f 的似然機率位置，並處理邊界
                new_l_pos = np.clip(curr_l_pos + new_vel_l, bounds[0], bounds[1])
                # 正規化
                new_l_pos /= new_l_pos.sum()
                new_likelihood.append(new_l_pos)

            # 更新粒子位置
            positions[j] = (new_prior, new_likelihood)
            # 使用新粒子預測，並計算適應值
            new_fitness = fitness_function.cal_micro_accuracy(y, predict(X, new_prior, new_likelihood)[1])

            if new_fitness > personal_best_fitness[j]:
                personal_best_fitness[j] = new_fitness
                personal_best_positions[j] = (new_prior, new_likelihood)

        # 若目前最佳群體適應值有大於歷史最佳群體適應值，更新
        best_idx = np.argmax(personal_best_fitness)
        if personal_best_fitness[best_idx] > global_best_fitness:
            global_best_fitness = personal_best_fitness[best_idx]
            global_best_positions = personal_best_positions[best_idx]

    return global_best_positions[0], global_best_positions[1]


def cv_with_ensemble_selection(file_path, target_column, model_config, pso_config, dataset_name):
    """
    進行五折交叉驗證訓練，以下每個步驟都是在每一折內各自進行
    1. 每一折訓練產生 25 個基本模型 (使用隨機生成結合粒子群優化)
    2. 每個基本模型預測每個訓練樣本建構資料過濾表
    3. 進行資料過濾
    5. 使用全部的基本模型進行集成，預測測試集樣本
    """
    # 從 model_config 中解析模型參數
    k = model_config['k_folds']
    num_base_models = model_config['num_base_models']
    selection_model_nums = model_config['selection_model_nums']

    # 載入資料，並區分特徵與類別
    data = pd.read_csv(file_path)
    X = data.drop(columns = [target_column]).values
    y = data[target_column].values
    class_nums = len(np.unique(y))  # 計算類別數量
    feature_value_counts = []

    # 對 X 做 LabelEncode，防止空箱問題產生
    for i in range(X.shape[1]): # 針對每個特徵
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i]) 
        # 記錄每個特徵的可能值數量，le.classes_ 會取出 label 的數量
        feature_value_counts.append(len(le.classes_))
    
    start_time = time.time()  # 計時開始

    # 分割訓練與測試
    kf = KFold(n_splits = k, shuffle = True, random_state = 42) 
    training_accuracies = []    # 儲存每個 fold 基本模型對 training set 的預測準確率
    test_accuracies = []        # 儲存每個 fold 集成模型對 test set 的預測準確率
    data_filter_table = {f"fold_{fold + 1}": [] for fold in range(k)}  # 儲存每個基本模型對每個訓練樣本的預測向量

    X_reserved = {f"fold_{fold + 1}": [] for fold in range(k)}  # 儲存資料過濾結果

    fold_models = {}  # 儲存每個折數的模型 

    start_time = time.time()

    # 進行五折交叉驗證訓練
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fold_data 儲存模型資訊
        fold_data = {}
        pso_models = []  # 記錄訓練出的模型

        # 每個基本模型對測試集樣本的預測
        model_test_predictions = np.zeros((len(y_test), num_base_models), dtype=int)    # 單一折數中基本模型的預測結果，(樣本數, 模型數)

        nums = 0
        # 生成 25 個模型
        with tqdm(total = num_base_models, desc = f"Fold {fold + 1} - Building Models") as pbar:
            while nums < num_base_models:     
                try:
                    # 生成一個簡易貝氏基本模型，global_best_fitness 為基本模型對於訓練集的正確率
                    prior, likelihood = PSO(
                        X_train, y_train, feature_value_counts, class_nums, pso_config
                    )
                    pso_models.append((prior, likelihood))
                    pred_vector, pred_class = predict(X_train, prior, likelihood)  # 用剛剛生成的基本模型去預測訓練集
                    # 建構多類別資料過濾表，shape = (B, N, C)，後續要轉置
                    data_filter_table[f"fold_{fold + 1}"].append(pred_vector)

                except Exception as e:
                    error_detail = traceback.format_exc()
                    tqdm.write(f"[WARN] PSO failed for fold {fold + 1} model {nums + 1}: {error_detail}")
                    return None, None

                training_accuracies.append(np.mean(pred_class == y_train))  # 一個 fold 裡的一個基本模型的訓練集準確率

                # 測試集預測
                pred_vector, pred_class = predict(X_test, prior, likelihood)
                model_test_predictions[:,nums] = pred_class  # 基本模型 i 對所有測試樣本的預測結果
                nums += 1
                pbar.update(1)
            
        fold_data["PSO_TRENB"] = pso_models  # 記錄所有 PSO 基本模型資訊

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

    path = config.PATH.get("PSO_TRENB")

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
    path = config.PATH.get("PSO_TRENB")
    
    # 處理多類別資料
    data_folder = "datasets/離散化資料集/多類別" # 使用離散化後的資料
    dataset_list = config.DATASET_LIST
    df_res_path = path["data_filter_result_path"]

    # 建立 csv 檔案，用以儲存 PSO_TRENB 的 training 和 test 準確率
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

    for filename in dataset_list: # 處理每個資料集
        print(f"處理資料集: {filename}")
        file_path = os.path.join(data_folder,filename + '.csv')
        target_column = "class"    # 類別欄位設為'class'

        # 進行 PSO_TRENB 訓練，得到五折交叉驗證後的訓練集
        training_accuracy, test_accuracy, exec_time  = cv_with_ensemble_selection(file_path, target_column, model_config, pso_config, filename)
            
        # 將兩個準確率 寫入 csv（使用 append 模式，避免被覆蓋）
        with open(path["log_file"], mode='a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, training_accuracy, test_accuracy, exec_time] )
            
            parent_path = Path.cwd()
            data_filter_var_path = os.path.join(parent_path, "data_filter_var.xlsx")

            # 檢查資料過濾正確性，寫過濾結果
            check_data_filter.check_filtering_result(data_filter_var_path, model_config, filename, df_res_path)

