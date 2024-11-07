# Reinforcement learning for Unity maze game
![image](https://github.com/georgehsu2/Unity-Maze-DQN/blob/main/maze.png)
# 簡介  
使用DQN演算法訓練AI走迷宮

# 遊戲環境基本資料:  
動作空間: 上下左右  
觀察空間: 所有物件包括腳色、牆壁、終點的座標(在main的obs_to_array中轉換成13x13的迷宮array)

# Reward Function 設定:  
距離獎勵: (previousDistanceToGoal - currentDistanceToGoal) * 0.5f  
每走一步: -0.05 Reward  
走重複走過的位置: -0.5 Reward  
嘗試往牆壁移動: -1 Reward  
到達終點: +100 Reward

# 用法  
1.用python -m venv創建虛擬環境在這個資料夾並用終端機啟動(python=3.7.9)  
2.安裝requirements.txt  
3.執行GPU_test來確保使用GPU訓練(我的版本:CUDA Version: 12.3, Driver Version: 546.80, torch: 1.13.1+cu117)  
4.執行main.py開始訓練  
5.在終端機(需先啟動虛擬環境)中輸入tensorboard --logdir=./runs，把跑出的URL輸入到瀏覽器，可以實時觀察訓練狀態  
6.訓練完成的模型會儲存成dqn_maze_model.pth  
7.執行run_model

