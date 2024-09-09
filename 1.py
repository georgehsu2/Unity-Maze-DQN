import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

# 初始化Unity環境和Gym Wrapper
unity_env = UnityEnvironment(file_name="C:/Users/user/Desktop/Games/8_16")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
print(env.observation_space)
print(env.action_space)

class DQN(nn.Module):
    def __init__(self, input_dim, branch_sizes):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.branch_outputs = nn.ModuleList([nn.Linear(128, size) for size in branch_sizes])
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = [branch(x) for branch in self.branch_outputs]
        return q_values

# 訓練參數
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
batch_size = 64
memory_size = 10000
num_episodes = 1000

# 初始化DQN和目標網絡
state_dim = env.observation_space[0].shape[0]
branch_sizes = env.action_space.nvec
q_network = DQN(state_dim, branch_sizes)
target_network = DQN(state_dim, branch_sizes)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# 動作選擇函數
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = q_network(state)
        actions = [torch.argmax(branch_q_values).item() for branch_q_values in q_values]
        return actions

# 模型優化函數
def optimize_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    actions = torch.LongTensor(actions).transpose(0, 1)
    current_q_values = [q_network(states)[i].gather(1, actions[i].unsqueeze(1)).squeeze(1) for i in range(len(actions))]
    next_q_values = [target_network(next_states)[i].max(1)[0] for i in range(len(actions))]

    target_q_values = [rewards + (gamma * next_q_value * (1 - dones)) for next_q_value in next_q_values]

    loss = sum([F.mse_loss(current_q, target_q) for current_q, target_q in zip(current_q_values, target_q_values)])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 訓練循環
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        
        optimize_model()
    
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
