import torch
import time

# 設定裝置為 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
print(torch.cuda.is_available())  # 應該輸出 True
print(torch.version.cuda)         # 應顯示 CUDA 版本
# 檢查是否成功使用 GPU
if device.type == "cuda":
    print("正在使用 GPU 計算")

    # 生成一個大 Tensor 並移到 GPU
    x = torch.randn(10000, 10000, device=device)

    # 進行矩陣乘法運算以增加 GPU 負載
    start_time = time.time()
    for _ in range(10):  # 重複運算多次以增加 GPU 占用率
        y = x @ x  # 矩陣乘法
    end_time = time.time()

    print("運算完成，耗時:", end_time - start_time, "秒")
else:
    print("未偵測到 GPU，請檢查 CUDA 安裝或硬體相容性。")