#基於https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import torch.optim
import torch.types
# from torch.utils.tensorboard import SummaryWriter
# 初始化Unity環境和Gym Wrapper
unity_env = UnityEnvironment(file_name="C:/Users/user/Desktop/Games/8_21_3", no_graphics=False)
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 訓練參數
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.7 #(1-EPSILON)*100%採取隨機動作
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 4000000
N_ACTIONS = env.action_space.nvec
N_STATES = env.observation_space[0].shape[0]
MAX_STEPS = 10000

#神經網路
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out1 = nn.Linear(128, N_ACTIONS[0])
        self.out2 = nn.Linear(128, N_ACTIONS[1])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_valueX = self.out1(x) #X軸移動
        action_valueZ = self.out2(x) #Z軸移動

        return action_valueX, action_valueZ

class DQN(object):
    def __init__(self):
        self.eval_net = Net().to(device)
        self.target_net = Net().to(device)
        # self.writer = SummaryWriter("./runs") 
        self.learn_step_counter = 0 #step數量
        self.memory_counter = 0 #memory數量
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3)) #initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        if np.random.uniform() < EPSILON:   # greedy
            action_values = self.eval_net(x)
            action_x = torch.max(action_values[0], 2)[1].squeeze().data.cpu().numpy()  # X軸動作
            # print(action_values)
            # print(action_values[0])
            # print(torch.max(action_values[0], 2))
            # print(torch.max(action_values[0], 2)[1])
            # print(torch.max(action_values[0], 2)[1].squeeze())
            # print(torch.max(action_values[0], 2)[1].squeeze().data.numpy())
            # print(action_x)
            action_z = torch.max(action_values[1], 2)[1].squeeze().data.cpu().numpy()  # Z軸動作
            action = np.array([action_x, action_z])  # 組合動作

        else: #隨機動作
            # print("random")
            action = np.array([np.random.randint(0, N_ACTIONS[0]), np.random.randint(0, N_ACTIONS[1])])
        return action

    def store_transition(self, s, a, r, s_):

        transition = np.concatenate([np.ravel(s), a, [r], np.ravel(s_)])
        # transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        #target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+2]).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+2:N_STATES+3]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        q_eval = self.eval_net(b_s)
        q_eval = torch.cat([q_eval[0].gather(1, b_a[:, 0].unsqueeze(1)), 
                            q_eval[1].gather(1, b_a[:, 1].unsqueeze(1))], dim=1)

        #獲取下一個state的Q值
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * torch.cat([q_next[0].max(1)[0].unsqueeze(1), 
                                                 q_next[1].max(1)[0].unsqueeze(1)], dim=1)

        loss = self.loss_func(q_eval, q_target)#計算Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

#訓練循環          
for i_episode in range(41):
    s = env.reset()
    episode_reward = 0
    step_count = 0
    reward_given_cp1 = False
    reward_given_cp2 = False
    visited_distances = []  # 用於追蹤每個episode中已經獲得獎勵的距離
    while True:
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a) #執行action
        step_count += 1
        episode_reward += r
        # print(step_count)
        if step_count >= MAX_STEPS:    
            done = True

        ################################  #reward function

        Player_X = s_[0][0]
        Player_Z = s_[0][2]

        Goal_X = 0.5
        Goal_Z = -4
        current_distance = round(np.sqrt((Player_X - Goal_X) ** 2 + (Player_Z - Goal_Z) ** 2))
        if current_distance < 7 and current_distance not in visited_distances:
            r += 10
            print(f"+10 reward for being closer to the goal (distance: {current_distance})")
            visited_distances.append(current_distance)  # 記錄已獲得獎勵的距離
        if 0 <= Player_X <= 1 and -4 <= Player_Z <= -3: #終點
            r += 100
            print("+100 reward (Goal)")
            done = True
        else:
            if not reward_given_cp1 and -2 <= Player_X <= -1 and 1 <= Player_Z <= 2: #check point 1
                r += 20
                print("+20 reward (cp1)")
                reward_given_cp1 = True
            if not reward_given_cp2 and 2 <= Player_X <= 3 and -2 <= Player_Z <= -1: #check point 2
                r += 50
                print("+50 reward (cp2)")
                reward_given_cp2 = True

        ################################

        dqn.store_transition(s, a, r, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print(f'Episode: {i_episode}, Reward: {episode_reward}')

        if done:
            break
        s = s_

        #保存模型
    if i_episode % 10 == 0:  # 每10個episode保存一次
        torch.save(dqn.eval_net.state_dict(), f'C:/Users/user/Desktop/Project/models/dqn_eval_net_episode_{i_episode}.pth')
        torch.save(dqn.target_net.state_dict(), f'C:/Users/user/Desktop/Project/models/dqn_target_net_episode_{i_episode}.pth')
        print(f'Model saved at episode {i_episode}')