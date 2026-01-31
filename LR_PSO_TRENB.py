import pickle
import copy, re, traceback
import numpy as np
import pandas as pd
import os, time, json, csv
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from scipy.special import expit
from scipy.stats import norm
import math
import check_data_filter
import data_filtering
import config

"""
1. 此程式的目的為訓練所有基本模型，記錄訓練集與測試集正確率(未集成挑選)
2. 並執行資料過濾，最後儲存模型資訊
3. 使用羅吉斯回歸模型
"""

def sigmoid(z):
    """定義 sigmoid function 以轉換機率"""
    return expit(z)  # 避免 overflow

def random_lr_model(X):
    """
    隨機生成多類別羅吉斯回歸模型的權重 θ*
    Args:
        X: 樣本特徵矩陣 (N, F)
    Returns:
        weights: List[array-like]，長度 C，每個元素的形狀為 (F,)，代表每個類別的權重向量
    """
    n_samples, n_features = X.shape
    weights = []
    
    # 隨機生成一般迴歸係數 θ' 的乘法係數
    eps = 1e-12
    Sj = np.var(X, axis=0, ddof=1)  # 計算每個特徵的樣本方差
    Sj = np.maximum(Sj, eps)  # 避免 Sj 為 0
    coef = Sj / (np.pi / math.sqrt(3))
        
    # 1. 隨機生成數值介於 0~1 的實數 p
    p_list = np.random.uniform(0, 1, size=(n_features,))
    # 2. 代入反向標準常態函數 Φ⁻¹(p)
    norm_ppf_p = norm.ppf(p_list)
    # 3. 產生隨機標準化係數 (權重)
    random_weights = norm_ppf_p * coef
    return random_weights

def predict(X, weights):
    N = len(X)
    C = len(weights)
    
    # 計算每個類別的 logit (未正規化的機率)
    logits = np.zeros((N, C))
    for c in range(C):
        logits[:, c] = np.dot(X, weights[c])
    
    # 使用 sigmoid 轉換為機率 
    posterior = sigmoid(logits)
    # 正規化使每個樣本的機率總和為 1
    posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
    
    # 多類別投票取最大值
    max_prob_index = np.argmax(posterior, axis=1)
    
    # 生成預測的向量
    pred_vector = np.zeros_like(posterior, dtype=int)
    pred_vector[np.arange(len(pred_vector)), max_prob_index] = 1
    
    # 實際預測的類別結果
    pred_class = max_prob_index
    return np.array(pred_vector), np.array(pred_class)


# 每個粒子都代表一個模型，帶有權重參數
def PSO(X, y, class_nums, pso_config):
    """
    定義粒子群優化演算法，優化使用隨機生成產生的羅吉斯回歸模型
    Return:
        一個基本模型的權重
    """
    # 從 pso_config 中解析超參數
    num_particles = pso_config['num_particles']
    max_iter = pso_config['max_iter']
    w, c1, c2 = pso_config['w'], pso_config['c1'], pso_config['c2']
    
    # 羅吉斯回歸權重的邊界
    weight_bounds = (-10.0, 10.0)
    
    # 初始化 fitness function 物件
    fitness_function = config.FitnessFunction()
    n_features = X.shape[1]
    
    # 初始化粒子位置和速度
    positions = []
    velocities = []
    for i in range(num_particles):
        # 隨機初始化粒子權重 (做為粒子的位置參數)
        particle_weights = random_lr_model(X, class_nums)
        positions.append(particle_weights)
        
        # 隨機初始化權重的「速度」
        weights_velocity = []
        for c in range(class_nums):
            w_vel = np.random.uniform(weight_bounds[0], weight_bounds[1], size=(n_features,)) * 0.1
            weights_velocity.append(w_vel)
        velocities.append(weights_velocity)
    
    positions = np.array(positions, dtype=object)
    velocities = np.array(velocities, dtype=object)
    
    # 使用每個粒子的初始權重預測，並計算初始適應值
    pred_class_list = [predict(X, p)[1] for p in positions]
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
            weights = positions[j]
            pbest_weights = personal_best_positions[j]
            gbest_weights = global_best_positions
            vel_weights = velocities[j]
            
            r1, r2 = np.random.random(2)
            
            # 更新每個類別的權重
            new_weights = []
            for c in range(class_nums):
                # 更新權重速度
                velocities[j][c] = w * vel_weights[c] + c1 * r1 * (pbest_weights[c] - weights[c]) + c2 * r2 * (gbest_weights[c] - weights[c])
                # 計算新權重位置，並處理邊界
                new_w = np.clip(weights[c] + velocities[j][c], weight_bounds[0], weight_bounds[1])  # 限制權重範圍避免過大
                new_weights.append(new_w)
            
            # 更新粒子位置
            positions[j] = new_weights
            # 使用新粒子預測，並計算適應值
            new_fitness = fitness_function.cal_micro_accuracy(y, predict(X, new_weights)[1])
            
            if new_fitness > personal_best_fitness[j]:
                personal_best_fitness[j] = new_fitness
                personal_best_positions[j] = new_weights
        
        # 若目前最佳群體適應值有大於歷史最佳群體適應值，更新
        best_idx = np.argmax(personal_best_fitness)
        if personal_best_fitness[best_idx] > global_best_fitness:
            global_best_fitness = personal_best_fitness[best_idx]
            global_best_positions = personal_best_positions[best_idx]
    
    return global_best_positions


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
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    class_nums = len(np.unique(y))  # 計算類別數量
    
    # 對 X 做 LabelEncode，防止空箱問題產生
    for i in range(X.shape[1]):  # 針對每個特徵
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
    
    start_time = time.time()  # 計時開始
    
    # 分割訓練與測試
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
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
        
        # 進行特徵標準化，避免不同特徵間的範圍差距過大
        scaler_fold = StandardScaler()
        X_train = scaler_fold.fit_transform(X_train)
        X_test = scaler_fold.transform(X_test)
        
        # fold_data 儲存模型資訊
        fold_data = {
            "test_index": test_index.tolist(),
            "train_class": y_train.tolist(),
            "class_nums": class_nums
            # 注意：scaler 不保存在模型中，因為每個 fold 都需要重新標準化
        }
        pso_models = []  # 記錄訓練出的模型
        
        # 每個基本模型對測試集樣本的預測
        model_test_predictions = np.zeros((len(y_test), num_base_models), dtype=int)    # 單一折數中基本模型的預測結果，(樣本數, 模型數)
        
        nums = 0
        # 生成 num_base_models 個模型
        with tqdm(total=num_base_models, desc=f"Fold {fold + 1} - Building Models") as pbar:
            while nums < num_base_models:
                try:
                    # 生成一個羅吉斯回歸基本模型
                    weights = PSO(X_train, y_train, class_nums, pso_config)
                    pso_models.append(weights)
                    pred_vector, pred_class = predict(X_train, weights)  # 用剛剛生成的基本模型去預測訓練集
                    # 建構多類別資料過濾表，shape = (B, N, C)，後續要轉置
                    data_filter_table[f"fold_{fold + 1}"].append(pred_vector)
                    
                except Exception as e:
                    error_detail = traceback.format_exc()
                    tqdm.write(f"[WARN] PSO failed for fold {fold + 1} model {nums + 1}: {error_detail}")
                    return None, None
                
                # 正常情況下，收集基本模型資訊
                training_accuracies.append(np.mean(pred_class == y_train))  # 一個 fold 裡的一個基本模型的訓練集準確率
                
                # 測試集預測
                pred_vector, pred_class = predict(X_test, weights)
                model_test_predictions[:, nums] = pred_class  # 基本模型 i 對所有測試樣本的預測結果
                nums += 1
                pbar.update(1)
        
        fold_data["LR_PSO_TRENB"] = pso_models  # 記錄所有 PSO 基本模型資訊
        
        # 保存該折模型與樣本資訊
        fold_models[f"fold_{fold + 1}"] = fold_data
        
        # 集成所有基本模型的預測成為最終預測 (投票)
        ensemble_test_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=class_nums).argmax(),
            axis=1, arr=model_test_predictions
        )
        test_accuracy = np.mean(y_test == ensemble_test_predictions)
        test_accuracies.append(test_accuracy)  # 儲存該 fold 的集成模型測試集準確率
        
        # 將 fold_{fold + 1} 的資料過濾表轉置，shape = (N, B, C)
        data_filter_table[f"fold_{fold + 1}"] = np.array(data_filter_table[f"fold_{fold + 1}"]).transpose(1, 0, 2).tolist()
        
        # 計算資料過濾所需變數
        X_reserved[f"fold_{fold + 1}"] = data_filtering.process_filtering(y_train, data_filter_table[f"fold_{fold + 1}"], num_base_models, selection_model_nums, fold)
    end_time = time.time()
    exec_time = end_time - start_time  # 訓練結束，計算總時間
    
    path = config.PATH.get("PSO_TRENB")  # 使用相同的路徑配置，或可以新增 LR_PSO_TRENB 的路徑

    """
    下方程式皆為儲存實驗結果，若要使用則將註解取消
    """
    
    # 將所有預測資訊寫入
    # write_json_data(path["training_accuracy_path"], dataset_name, training_accuracies)  # 將五折交叉驗證中的訓練集樣本預測正確率寫入
    # write_json_data(path["training_pred_vector_path"], dataset_name, data_filter_table)  # 將五折交叉驗證中的訓練集樣本預測向量寫入
    # write_json_data(path["data_filter_reserved_path"], dataset_name, X_reserved)
    
    # 保存模型至文件
    # output_path = os.path.join(path["model_path"], f"{dataset_name}_models.pkl")
    # with open(output_path, 'wb') as f:
    #     pickle.dump(fold_models, f)
    
    # print(f"模型已保存至 {output_path}")
    
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
    model_config = config.MODEL_CONFIG
    pso_config = config.PSO_CONFIG
    path = config.PATH.get("PSO_TRENB")  # 使用相同的路徑配置
    
    # 處理多類別資料
    data_folder = "datasets/原始資料集/二類別"  # 使用離散化後的資料
    dataset_list = config.DATASET_LIST
    df_res_path = path["data_filter_result_path"]

    """
    下方程式皆為儲存實驗結果，若要使用則將註解取消
    """
    
    # 建立 csv 檔案，用以儲存 LR_PSO_TRENB 的 training 和 test 準確率
    # with open(path["log_file"], mode='w', encoding='utf-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Dataset", "Training_Accuracy", "Test_Accuracy", "Time"])
    
    # # 建儲存資料過濾筆數與比例
    # with open(df_res_path, mode='w', encoding='utf-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Dataset", "avg filtered count", "avg filtered rate"])
    
    # # 建立儲存集成挑選後的模型索引，以及集成挑選後的測試集正確率
    # with open(path["es_result_path"], mode='w', encoding='utf-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Dataset"] + ["fold"] + [str(i+1) for i in range(25)] + ["Obj"] + ["Test Accuracy"])
    
    for filename in dataset_list:  # 處理每個資料集
        print(f"處理資料集: {filename}")
        file_path = os.path.join(data_folder, filename + '.csv')
        target_column = "class"    # 類別欄位設為'class'
        
        # 進行 LR_PSO_TRENB 訓練，得到五折交叉驗證後的訓練集
        training_accuracy, test_accuracy, exec_time = cv_with_ensemble_selection(file_path, target_column, model_config, pso_config, filename)
        print(f"Training Accuracy: {training_accuracy}, Test Accuracy: {test_accuracy}, Execution Time: {exec_time}")
        """
        下方程式皆為儲存實驗結果，若要使用則將註解取消
        """
        # # 將兩個準確率 寫入 csv（使用 append 模式，避免被覆蓋）
        # with open(path["log_file"], mode='a', encoding='utf-8', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([filename, training_accuracy, test_accuracy, exec_time])
            
        #     parent_path = Path.cwd()
        #     data_filter_var_path = os.path.join(parent_path, "data_filter_var.xlsx")
            
        #     # 檢查資料過濾正確性，寫過濾結果
        #     check_data_filter.check_filtering_result(data_filter_var_path, model_config, filename, df_res_path)
