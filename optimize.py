import json, os, csv
from pathlib import Path
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gurobipy import *
import numpy as np
import config
import bagging, PSO_TRENB
import traceback

class OptimizeModel:
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

    def run_optimize_model(self, dataset_name, ensemble_method):
        """
        執行最佳化集成挑選
        Args:
            ensemble_method: 集成方法名稱
        """
        path = config.PATH.get(ensemble_method)

        # 載入基本模型，bagging 模型會存其他額外的訓練測試資訊，所以也要載入
        if ensemble_method == "Bagging":
            models = self.load_models("Bagging", dataset_name)
        else:
            models = self.load_models(ensemble_method, dataset_name)
        bagging_models = self.load_models("Bagging", dataset_name)

        # 讀取訓練集預測結果
        with open(path["training_pred_vector_path"], 'r', encoding='utf-8') as f:
            pred_vector = json.load(f)

        # 讀取資料過濾結果
        with open(path["data_filter_reserved_path"], 'r', encoding='utf-8') as f:
            filter_res = json.load(f)

        results = []  # 存結果，方便後續計算平均
        print(f"------------------------------------------------------\nStart optimizing {dataset_name} model...")
        # 取出該資料集的第 fold 折的預測向量，shape = (N, model)
        for fold, pred_array in pred_vector[dataset_name].items():
            fold_model = bagging_models[fold]
            m = self.model_config["num_base_models"]
            b = self.model_config['selection_model_nums']
            num_samples = len(pred_array)
            num_classes = fold_model["class_nums"]
            train_class = fold_model["train_class"]

            model = Model(f"optimize_{dataset_name}_fold_{fold}")
            model.setParam('OutputFlag', 0)
            time_limit = self.model_config["time_limit_per_fold"]
            # gurobi 中設定求解時間，若超過則會返回當下最佳解
            model.setParam('TimeLimit', time_limit)
            # --- 建立決策變數 ---

            # 1. x[i]: alpha_i，基本模型 i 是否被選中 (1/0)
            x = {}
            for i in range(m):
                x[i] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{i}")

            # 2. beta_var[j]: beta_j，樣本 j 是否被正確預測
            beta_var = {}
            for j in range(num_samples):
                beta_var[j] = model.addVar(vtype=GRB.BINARY, name=f"beta_{j}")
            
            # 更新模型，確保變數已註冊到模型中
            model.update()

            """
            建立限制式
            """
            # 1. 模型限制數量
            model.addConstr(quicksum(x[i] for i in range(m)) == b, name="Model_Count_Limit")

            # 2. 針對保留的訓練集樣本設定多類別正確性限制 
            fold_filter_res = filter_res[dataset_name][fold]

            # j 被保留下來的樣本在訓練資料集中的索引
            for j in fold_filter_res:
                true_class = train_class[j]

                # 對正確類別外的所有類別進行限制
                for class_idx in range(num_classes):
                    if class_idx == true_class:
                        continue
                    # 1. 計算票數 
                    correct_votes = quicksum(pred_array[j][i][true_class] * x[i] for i in range(m))
                    wrong_votes = quicksum(pred_array[j][i][class_idx] * x[i] for i in range(m))
            
                    # 2. 使用指示符 (Indicator) 解決非線性
                    # 當 beta_var[j] == 1 時，後面這個限制式才必須成立
                    # 當 beta_var[j] == 0 時，這個限制式會被直接忽略
                    model.addGenConstrIndicator(
                        beta_var[j],               # 開關變數 (beta_j)
                        1,                  # 當變數為 1 時觸發
                        correct_votes - wrong_votes >= 1, # 要執行的限制式
                        name=f'indicator_{j}_{true_class}_vs_{class_idx}'
                    )

            """
            設定目標函數
            """
            # 最大化被正確分類的樣本數
            model.setObjective(quicksum(beta_var[j] for j in fold_filter_res), GRB.MAXIMIZE)

            # 求解
            model.optimize()

            # 處理求解結果：支援 OPTIMAL 和 TIME_LIMIT 狀態 (兩者都有可行解)
            selected_models = None
            gap = 0.0

            if model.status == GRB.OPTIMAL:
                selected_models = [i for i in range(m) if x[i].x > 0.5]
                # 用挑選後的基本模型集成預測
                test_accuracy = self.eval_ensemble_selection(dataset_name, fold, selected_models, ensemble_method)

            elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
                selected_models = [i for i in range(m) if x[i].x > 0.5]
                test_accuracy = self.eval_ensemble_selection(dataset_name, fold, selected_models, ensemble_method)
                gap = model.MIPGap
                
            else:
                test_accuracy = 0.0
            
            # 這裡 results 是一個 list，但後面有對 results 做 DataFrame(results)
            # 如果你把 results 變成 DataFrame，再呼叫 append 就會報錯
            # 正確用法：確保 results 是 list，在這裡加 dict 到 list
            results.append({
                "fold": fold,
                "selected": selected_models,
                "obj": int(model.ObjVal) if (model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT) else 0.0,
                "sample_nums": len(fold_filter_res),
                "test_accuracy": test_accuracy,
                "gap": gap
            })
            runtime = model.Runtime
            print(f"Optimization runtime: {runtime:.4f} seconds")

        results_df = pd.DataFrame(results)
        avg_obj = results_df['obj'].mean()
        avg_sample_nums = results_df['sample_nums'].mean()
        avg_test_accuracy = results_df['test_accuracy'].mean()
        avg_gap = results_df['gap'].mean()

        with open(path["es_result_path"], mode='a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)    
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                writer.writerow([dataset_name, avg_obj, avg_sample_nums, avg_test_accuracy, avg_gap])
            else:
                writer.writerow([dataset_name, 0.0, 0.0, 0.0, 0.0])

        # 印出所有結果
        for result in results:
            print(f"{result['fold']}, Selected Models: {result['selected']}, Obj: {result['obj']}/{result['sample_nums']}, Test Accuracy: {result['test_accuracy']}, Gap: {result['gap']}")

        return True
    
    def eval_ensemble_selection(self, dataset_name, fold, selected_models, ensemble_method):
        """負責預測單折的測試樣本"""

        path = config.PATH.get(ensemble_method)
        # 載入基本模型
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
        model_test_predictions = np.zeros((len(y_test), self.model_config['selection_model_nums']), dtype=int)    # 單一折數中基本模型的預測結果，(樣本數, 模型數)
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
                # 代表被挑到的是 PSO 的模型
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

    # 初始化最佳化模型物件
    OptimizeModel = OptimizeModel()
    path = config.PATH.get("PSO_TRENB")
    with open(path["es_result_path"], mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)    
        writer.writerow(["Dataset", "Obj", "Sample Nums", "Test Accuracy", "Gap"])

    for dataset in dataset_list:
        try:
            result = OptimizeModel.run_optimize_model(dataset, "PSO_TRENB")
        except Exception as e:
            error_detail = traceback.format_exc()
            print(f"資料集 {dataset} 最佳化模型求解發生錯誤: {error_detail}")
            continue






    



    

