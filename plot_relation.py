from pathlib import Path
import config
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

"""本程式想要捕捉特徵與類別數對於預測準確率的關聯"""

class Plot:
    def __init__(self):
        self.dataset_list = config.DATASET_LIST
        self.path = config.PATH  # 儲存所有方法的實驗結果路徑
    
    # 計算每個資料集的特徵數量、類別數量、樣本數量
    def cal_dataset_info(self):
        data_folder = "datasets/離散化資料集/多類別" # 使用離散化後的資料
        num_classes = []
        num_features = []
        for dataset in self.dataset_list:
            file_path = os.path.join(data_folder,dataset + '.csv')
            df = pd.read_csv(file_path)
            # 計算特徵數量與類別數量
            num_samples = df.shape[0]
            num_features.append(df.shape[1] - 1)
            num_classes.append(len(df['class'].unique()))
        return num_samples, num_features, num_classes

    # 繪製特徵數與類別數對於準確率差值的關聯圖
    def plot_2dim_relation(self, feature_nums, class_nums, acc_diff, method1, method2):
        # 設定兩張子圖
        fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- 左圖 : 特徵數 v.s 準確率差值 ---
        ax1.scatter(feature_nums, acc_diff, alpha=0.7, c='royalblue', edgecolor='w', s=80)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1.3)
        ax1.set_xlabel('Number of Features', fontsize=12)
        ax1.set_ylabel('Accuracy Difference', fontsize=12)
        ax1.set_title('Trend of Feature Count')

        # ----- 右圖 : 類別數 v.s 準確率差值 ---
        ax2.scatter(class_nums, acc_diff, alpha=0.7, c='crimson', edgecolor='w', s=80)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1.3)
        ax2.set_xlabel('Number of Classes', fontsize=12)
        ax2.set_ylabel('Accuracy Difference', fontsize=12)
        ax2.set_title('Trend of Class Count')
        fig_2d.suptitle(f'2D Accuracy Difference Trend: {method1} - {method2}', fontsize=14, y=1.0)

        plt.tight_layout()
        
        save_path = os.path.join(Path.cwd(), "accuracy_trend")
        save_name = os.path.join(save_path, f"Accuracy_Trend_2D.png")
        plt.savefig(save_name, dpi=300)
        print(f"趨勢圖已儲存至: {save_name}")
        plt.show()

        # 繪製單獨的準確率差值趨勢圖
        fig, ax = plt.subplots(figsize=(10, 6))
        # 計算特徵數與類別數的交互複雜度
        total_complextity = np.array(feature_nums) * np.array(class_nums)
        ax.scatter(total_complextity, acc_diff, alpha=0.7, c='royalblue', edgecolor='w', s=80)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.3)
        ax.set_xlabel('Total Complexity(Feature * Class)', fontsize=12)
        ax.set_ylabel('Accuracy Difference', fontsize=12)
        ax.set_title(f'2D Accuracy Difference Trend: {method1} - {method2}') 
        save_name = os.path.join(save_path, f"Accuracy_Trend_2D_Total_Complexity.png")
        plt.savefig(save_name, dpi=300)
        print(f"趨勢圖已儲存至: {save_name}")
        plt.show()

        return

    # 繪製 3D 關聯圖
    def plot_3dim_relation(self, feature_nums, class_nums, acc_diff, method1, method2):
        fig_3d = plt.figure(figsize=(10,8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # 繪製 3d 點
        # c=acc_diffs: 讓點的顏色根據差值變化
        # cmap='coolwarm': 冷暖色調 (藍色是負值，紅色是正值)
        scatter = ax_3d.scatter(
            feature_nums, class_nums, acc_diff, c=acc_diff, cmap='coolwarm', alpha=0.9, edgecolor='k', s=75
        )

        #  設定軸標籤
        ax_3d.set_xlabel('Number of Features', fontsize=11, labelpad=10)
        ax_3d.set_ylabel('Number of Classes', fontsize=11, labelpad=10)
        ax_3d.set_zlabel('Accuracy Difference', fontsize=11, labelpad=10)
        ax_3d.set_title(f'3D relationship : Feature vs Class vs Accuracy Difference {method1} - {method2}',  fontsize=13, pad=15)

        # 加入 colorbar
        cbar = plt.colorbar(scatter, ax=ax_3d, pad=0.1, shrink=0.7)
        cbar.set_label('Accuracy Difference')

        save_path = os.path.join(Path.cwd(), "accuracy_trend")
        save_name = os.path.join(save_path, "Accuracy_Trend_3D.png")
        plt.savefig(save_name, dpi=300)
        print(f"3D 關聯圖已儲存至: {save_name}")
        plt.show()

        return


if __name__ == "__main__":
    plot = Plot()

    """
    繪製所有方法的準確率差值趨勢圖   
    """
    # 讀取 log 檔
    method1 = "PSO_TRENB"
    method2 = "Bagging"
    log_path1 = plot.path.get(method1)["log_file"]
    log_path2 = plot.path.get(method2)["log_file"]
    log_info1 = pd.read_csv(log_path1)
    log_info2 = pd.read_csv(log_path2)
    
    # 計算每個資料集的樣本數、特徵數、類別數
    sample_nums, feature_nums, class_nums = plot.cal_dataset_info()
    # 計算兩種集成方法在測試集上的準確率差值
    acc_diff = list(log_info1['Test_Accuracy'] - log_info2['Test_Accuracy'])
        
    # 繪製 2d 關聯圖
    plot.plot_2dim_relation(feature_nums, class_nums, acc_diff, method1, method2)
    # 繪製 3d 關聯圖
    plot.plot_3dim_relation(feature_nums, class_nums, acc_diff, method1, method2)






