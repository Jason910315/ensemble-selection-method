import subprocess
import os
import optimize
import config

# 模型訓練程式清單
scripts = ["PSO_Bagging.py"]

# 逐一執行
for script in scripts:
    print(f"正在透過 uv 執行: {script}")
    subprocess.run(["uv", "run", script], check=True)

