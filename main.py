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
from torch.utils.tensorboard import SummaryWriter

# 初始化Unity環境和Gym Wrapper
unity_env = UnityEnvironment(file_name="C:/Users/user/Desktop/Games/9_16", no_graphics=False)
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 訓練參數
BATCH_SIZE = 32
LR = 0.01

EPSILON_START = 1.0  # 初始值
EPSILON_END = 0.02  # 最小值
EPSILON_DECAY_STEPS = 10  # 多少步內將 epsilon 從 EPSILON_START 減少到 EPSILON_END
EPSILON = EPSILON_START

# EPSILON = 0.7 #EPSILON*100%採取隨機動作
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 1000000
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space[0].shape[0]
MAX_STEPS = 10000
# print(env.action_space)
# print(env.observation_space)
print(N_STATES)

writer = SummaryWriter('./runs')
total_steps = 0

def get_epsilon(step):
    if step >= EPSILON_DECAY_STEPS:
        return EPSILON_END
    else:
        # 線性插值計算 epsilon 值
        return EPSILON_START - (EPSILON_START - EPSILON_END) * (step / EPSILON_DECAY_STEPS)
    
#神經網路
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        # print("x in:", x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # print("x out:", x)
        action_value = self.out(x)
        # print("action value:", action_value)
        return action_value

class DQN(object):
    def __init__(self):
        self.eval_net = Net().to(device)
        self.target_net = Net().to(device)
        self.learn_step_counter = 0 #step數量
        self.memory_counter = 0 #memory數量
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) #initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = np.array(x)
        x = torch.FloatTensor(x).to(device)
        if np.random.uniform() > EPSILON:   # greedy
            action_values = self.eval_net(x)
            print(action_values)
            action = torch.max(action_values, 1)
            print(action)
            action = action[1].squeeze().data.cpu().numpy()
            print(action)
            print("action chosen by network:", action)

        else: #隨機動作
            action = np.random.randint(0, N_ACTIONS)
            action = action
            # print("random action", action)
        return action

    def store_transition(self, s, a, r, s_):

        # transition = np.concatenate([np.ravel(s), a, [r], np.ravel(s_)])
        transition = np.hstack((np.ravel(s), [a, r], np.ravel(s_)))

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
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1]).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)  # 評估網絡的Q值

        #獲取下一個state的Q值
        q_next = self.target_net(b_s_).detach().max(1)[0].unsqueeze(1)  # 目標網絡的Q值
        q_target = b_r + GAMMA * q_next  # 計算目標Q值

        loss = self.loss_func(q_eval, q_target)#計算Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # writer.add_scalar('Loss/train', loss.item(), self.learn_step_counter)

dqn = DQN()

#訓練循環          
for i_episode in range(21):
    s = env.reset()
    episode_reward = 0
    step_count = 0
    reward_given_cp1 = False
    reward_given_cp2 = False
    visited_distances = []  # 用於追蹤每個episode中已經獲得獎勵的距離
    previous_distance = float('inf')
    while True:
        # 更新 epsilon 根據當前步數
        EPSILON = get_epsilon(total_steps)
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a) #執行action
        print(s)
        step_count += 1
        total_steps += 1
        if step_count >= MAX_STEPS:    
            done = True

        ################################  #reward function

        Player_X = s_[0][0]
        Player_Z = s_[0][2]

        Goal_X = 0.5
        Goal_Z = -4
        current_distance = round(np.sqrt((Player_X - Goal_X) ** 2 + (Player_Z - Goal_Z) ** 2))

        r -= 0.0001
        
        # 當current_distance比之前的小時，給予0.1的reward
        if current_distance < previous_distance:
            r += 0.1
    
        # 更新previous_distance為當前的距離，用於下一個step的比較
        previous_distance = current_distance
        
        # if current_distance < 7 and current_distance not in visited_distances:
        #     r += 10
        #     print(f"+10 reward for being closer to the goal (distance: {current_distance})")
        #     visited_distances.append(current_distance)  # 記錄已獲得獎勵的距離
        if 0 <= Player_X <= 1 and -4 <= Player_Z <= -3: #終點
            r += 100
            print("+100 reward (Goal)")
            done = True
        # else:
        #     if not reward_given_cp1 and -2 <= Player_X <= -1 and 1 <= Player_Z <= 2: #check point 1
        #         r += 20
        #         print("+20 reward (cp1)")
        #         reward_given_cp1 = True
        #     if not reward_given_cp2 and 2 <= Player_X <= 3 and -2 <= Player_Z <= -1: #check point 2
        #         r += 50
        #         print("+50 reward (cp2)")
        #         reward_given_cp2 = True

        ################################
        episode_reward += r
        dqn.store_transition(s, a, r, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            print(f'Episode: {i_episode}, Reward: {episode_reward}')
            writer.add_scalar('Reward/episode', episode_reward, i_episode)
            break

        s = s_

        #保存模型
    if i_episode % 100 == 0:  # 每100個episode保存一次
        torch.save(dqn.eval_net.state_dict(), f'C:/Users/user/Desktop/Project/models/dqn_eval_net_episode_{i_episode}.pth')
        torch.save(dqn.target_net.state_dict(), f'C:/Users/user/Desktop/Project/models/dqn_target_net_episode_{i_episode}.pth')
        print(f'Model saved at episode {i_episode}')

writer.close()