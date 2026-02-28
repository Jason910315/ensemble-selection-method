
import math, traceback, csv
import config
from pathlib import Path
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import bagging, PSO_TRENB

class COMEP:
    def __init__(self):
        self.model_config = config.MODEL_CONFIG
        self.parent_path = Path.cwd()

    # 載入基本模型，靜態方法，不需要實例化
    @staticmethod
    def load_models(method, dataset_name):
        path = config.PATH
        model_path = os.path.join(path[method]["model_path"], f"{dataset_name}_models.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        return models

    # 載入預測向量
    def load_pred_vector(self, dataset_name, ensemble_method):
        """
        return:
            pred_vector: 該向量代表的是所有基本模型對於 dataset_name 且使用 ensemble_method 集成方法的預測結果，有 5 個 folds，每個 fold 的 shape = (N, B, C)
                - N: 訓練集樣本數
                - B: 基本模型數
                - C: 類別數
        """
        es_method_path = config.PATH.get(ensemble_method)

        # 讀取訓練集預測結果
        with open(es_method_path["training_pred_vector_path"], 'r', encoding='utf-8') as f:
            result = json.load(f)
            pred_vector = result[dataset_name]

        return pred_vector

    # --- 以下開始定義 COMEP 的計算方法 ---
    def H_entropy(self, class_list):
        """
        計算基本模型對於資料集的預測結果 / 真實類別列表的 entropy，shape = (N, C)
        """
        entropy = 0.0

        # 將預測的類別向量轉換為單一類別結果
        N = len(class_list)  # 總樣本數

        # classes 可能不為所有類別 (因為可能是預測出來的)
        classes = np.unique(class_list)
        for c in classes:
            count = np.sum(class_list == c)
            # 計算每個類別的 entropy
            entropy -= (count / N) * np.log(count / N)

        return entropy

    def H_joint_entropy(self, class_list_A, class_list_B):
        """
        兩個基本模型對於資料集的預測結果 / 真實類別列表的 joint entropy
        """
        joint_entropy = 0.0
        N = len(class_list_A)

        # 分別統計兩個基本模型預測出來的類別種類數
        a_classes = np.unique(class_list_A)
        b_classes = np.unique(class_list_B)

        for c1 in a_classes:
            for c2 in b_classes:
                # 類別 c1 和 c2 同時出現的次數
                count_c1_c2 = np.sum((class_list_A == c1) & (class_list_B == c2))
                if count_c1_c2 > 0:  # 避免從未出現過的組合 (除以 0)
                    # 計算 joint entropy
                    joint_entropy -= (count_c1_c2 / N) * np.log(count_c1_c2 / N)

        return joint_entropy
    
    # 計算 MI (下面都是計算一個基本模型對真實類別的 MI)
    def cal_normalized_information(self, pred_vector, class_list):
        """
        計算基本模型的預測結果對於真實類別列表的 normalized mutual information (有正規化的)
        """
        entropy_A = self.H_entropy(pred_vector)
        entropy_B = self.H_entropy(class_list)
        joint_entropy = self.H_joint_entropy(pred_vector, class_list)
        # 互資訊 (未正規化前)
        mutual_information = entropy_A + entropy_B - joint_entropy

        # 正規化
        if entropy_A > 0 and entropy_B > 0:   # 避免除以 0 
            mormalized_MI = mutual_information / math.sqrt(entropy_A * entropy_B)
        else:
            mormalized_MI = 0.0
        
        return mormalized_MI
    
    # 計算 VI (下面都是計算兩個基本模型間的 VI)
    def cal_variation_information(self, pred_vector_A, pred_vector_B):
        entropy_A = self.H_entropy(pred_vector_A)
        entropy_B = self.H_entropy(pred_vector_B)
        joint_entropy = self.H_joint_entropy(pred_vector_A, pred_vector_B)
        # 互資訊 (未正規化前)
        mutual_information = entropy_A + entropy_B - joint_entropy

        if joint_entropy > 0:   # 避免除以 0 
            variation_information = 1 - (mutual_information / joint_entropy)
        else:
            variation_information = 0.0
        
        return variation_information

    def run_comep_selection(self, dataset_name, ensemble_method, lambda_param=0.5):
        """
        執行 COMEP 最佳化集成挑選 (five fold)
        Args:
            lamba_param: 權重參數，用於平衡 normalized information 和 variation information
        """
        print(f"處理 {dataset_name} 的 COMEP selection...")

        # --- 參數定義 ---
        # bagging 模型會存其他額外的訓練測試資訊，所以也要載入
        bagging_models = self.load_models("Bagging", dataset_name)

        # 讀取該訓練集預測結果 (包含所有 fold)
        pred_vector = self.load_pred_vector(dataset_name, ensemble_method)

        # 輸入的基本模型總數與想要挑選的模型數
        m = self.model_config["num_base_models"]
        b = self.model_config['selection_model_nums']

        selected_results = {f"fold_{fold}": [] for fold in range(1, 6)}

        # --- 執行 COMEP ---
        try:
            # 每個 key: value 是 "fold_1": []...，陣列 shape = (N, B, C)
            for fold, fold_pred_vector in pred_vector.items():

                models_info = bagging_models[fold]     # 裡面儲存該 fold 的模型資訊
                fold_pred_vector = np.array(fold_pred_vector)
                selected_models = []

                # 隨機選擇一個初始模型進入挑選
                first_model = np.random.randint(0, m - 1)
                selected_models.append(first_model)

                true_class = models_info["train_class"]   # 一個 list，代表樣本實際的類別
        
                # 逐步選擇剩餘的模型 (也就是迴圈要再跑 b-1 次)
                while len(selected_models) < b:
                    best_tdac = 0.0

                    for i in range(b):
                        # 每次都是從還沒被挑到的模型內選選一個最好的
                        if i in selected_models:
                            continue

                        pred_vector_i = fold_pred_vector[:, i]    # 第 i 個基本模型對該訓練集的預測結果 (向量)
                        pred_class_list_i = np.argmax(pred_vector_i, axis=1)  # 轉換成實際預測類別

                        total_tdac = 0.0  # 針對不同模型 i 會重置
                        
                        # 計算 tdac 總和 (遍歷每個已經被選擇的模型)
                        for j in selected_models:

                            pred_vector_j = fold_pred_vector[:, j]    # 第 j 個基本模型對該訓練集的預測結果 (向量)
                            pred_class_list_j = np.argmax(pred_vector_j, axis=1)  # 轉換成實際預測類別

                            # 計算基本模型 i / j 與真實類別間的 MI
                            mi_i = self.cal_normalized_information(pred_class_list_i, true_class)
                            mi_j = self.cal_normalized_information(pred_class_list_j, true_class)
                            # 計算基本模型 i / j 間的 VI
                            vi_i_j = self.cal_variation_information(pred_class_list_i, pred_class_list_j)

                            tdac_i_j = lambda_param * vi_i_j + (1 - lambda_param) * ((mi_i + mi_j) / 2)

                            total_tdac += tdac_i_j
                        
                        if total_tdac > best_tdac:
                            best_tdac = total_tdac
                            selected_models.append(i)  # 將基本模型 i 加入挑選

                selected_results[f"fold_{fold}"] = selected_models

                # 進行測試集評估
                test_accuracy = self.eval_ensemble_selection(dataset_name, fold, selected_models, ensemble_method)
                print(f"Fold {fold} : {selected_models}, Test Accuracy: {test_accuracy}")

        except Exception as e:
            error_detail = traceback.format_exc()
            print(f"Error in run_comep_selection: {error_detail}")
            return None

        print("Comep selection 完成\n------------------------------------------------------")

        return selected_results

    def eval_ensemble_selection(self, dataset_name, fold, selected_models, ensemble_method):
        """
        負責預測單折的測試樣本
        """

        path = config.PATH.get(ensemble_method)
        # 載入基本模型，bagging 模型會存其他額外的訓練測試資訊，所以也要載入
        models = self.load_models(ensemble_method, dataset_name)
        bagging_models = self.load_models("Bagging", dataset_name)

        # 載入原始資料集
        df = pd.read_csv(os.path.join(self.parent_path, "datasets", "離散化資料集", "多類別", f"{dataset_name}.csv"))
        attributes = df.columns[:-1]

        fold_model = models[fold]
        base_models = fold_model[ensemble_method]
        bagging_fold_model = bagging_models[fold]

        # 取得 X 資訊，drop 掉 class 欄位
        X = df.drop(columns = ["class"]).values
        # 對 X 做 LabelEncode，防止空箱問題產生
        for i in range(X.shape[1]): # 針對每個特徵
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i]) 
            
        # 取出測試集資訊，資訊都存在 bagging 模型中
        test_index = bagging_fold_model["test_index"]
        X_test = X[test_index]
        y_test = df["class"].iloc[test_index]
        class_nums = bagging_fold_model["class_nums"]

        # bagging 預測要額外定義
        test_data = pd.DataFrame(X_test, columns=attributes)
        test_data['class'] = y_test

        # 每個基本模型對測試集樣本的預測
        model_test_predictions = np.zeros((len(y_test), self.model_config['selection_model_nums']), dtype=int)   # 單一折數中基本模型的預測結果，(樣本數, 模型數)
        # 使用集成挑選後的模型預測
        for i, select_idx in enumerate(selected_models):
            # 不同集成方法要用不同預測手段
            if ensemble_method == "PSO_TRENB":
                # 被挑選到的基本模型
                prior, likelihood = base_models[select_idx]
                pred_vector, pred_class = PSO_TRENB.predict(X_test, prior, likelihood)
                model_test_predictions[:, i] = pred_class  # 基本模型 i 對所有測試樣本的預測結果
            
            elif ensemble_method == "Bagging":
                prior, p_Xi_Cj_dict, att_value_counts, nj = base_models[select_idx]
                pred_vector, pred_class = bagging.predict(nj, test_data, prior, p_Xi_Cj_dict, att_value_counts)
                model_test_predictions[:, i] = pred_class  # 基本模型 i 對所有測試樣本的預測結果

            elif ensemble_method == "PSO_Bagging":
                # 代表被挑到的是 PSO 的模型 (參數量不同)
                if len(base_models[select_idx]) == 2:
                    prior, likelihood = base_models[select_idx]
                    pred_vector, pred_class = PSO_TRENB.predict(X_test, prior, likelihood)
                # 代表被挑到的是 Bagging 的模型
                elif len(base_models[select_idx]) == 4:
                    prior, p_Xi_Cj_dict, att_value_counts, nj = base_models[select_idx]
                    pred_vector, pred_class = bagging.predict(nj, test_data, prior, p_Xi_Cj_dict, att_value_counts)
                model_test_predictions[:, i] = pred_class 
                
            # 集成所有基本模型的預測成為最終預測 (投票)
            ensemble_test_predictions = np.apply_along_axis(
                lambda x : np.bincount(x.astype(int), minlength = class_nums).argmax(),
                axis = 1, arr = model_test_predictions
            )
            test_accuracy = np.mean(y_test == ensemble_test_predictions)
        return test_accuracy

if __name__ == "__main__":
    dataset_list = config.DATASET_LIST

    comep = COMEP()

    for dataset in dataset_list:
        result = comep.run_comep_selection(dataset, "PSO_TRENB")





            
