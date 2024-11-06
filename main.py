import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 用Gym Wrapper把Unity遊戲轉成Gym環境
unity_env = UnityEnvironment(file_name="./game", no_graphics=True)
env = UnityToGymWrapper(unity_env, flatten_branched=True, allow_multiple_obs=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把gym的觀察空間轉成迷宮array
def obs_to_array(coordinates):
    coordinates = coordinates[0]
    agent = coordinates[:2]
    goal = coordinates[2:4]
    walls = coordinates[4:]

    maze = np.zeros((13, 13), dtype=int)

    def coord_to_index(x, z):
        x_idx = int(np.round((x + 6) * 12 / 12))
        z_idx = 12 - int(np.round((z + 6) * 12 / 12))
        return max(0, min(12, x_idx)), max(0, min(12, z_idx))

    for i in range(0, len(walls), 2):
        x, z = walls[i], walls[i+1]
        idx_x, idx_z = coord_to_index(x, z)
        maze[idx_z, idx_x] = 3

    agent_x, agent_z = coord_to_index(agent[0], agent[1])
    maze[agent_z, agent_x] = 1

    goal_x, goal_z = coord_to_index(goal[0], goal[1])
    maze[goal_z, goal_x] = 2

    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 3
    return maze

# 模型架構
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.to(device)# 把計算移動到device(GPU)
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def _get_conv_out(self, shape):

        o = torch.zeros(1, 1, *shape).to(self.conv1.weight.device)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x): # 向前傳播

        x = x.to(self.conv1.weight.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 經驗回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 訓練方法、參數
class DQNAgent:
    def __init__(self, state_shape, n_actions, num_episodes, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.num_episodes = num_episodes
        self.epsilon_decay_episodes = int(num_episodes * 2 / 3)
        
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
    
    def update_epsilon(self, episode):
        # epsilon deacy方法，在前2/3的訓練循環中，epsilon會從epsilon_start線性遞減到epsilon_final
        if episode < self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_final) * (episode / self.epsilon_decay_episodes)
        else:
            self.epsilon = self.epsilon_final
            
    def select_action(self, state):

        state = state.to(self.device)
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0]).to(self.device)
        action_batch = torch.cat(batch[1]).to(self.device)
        reward_batch = torch.cat(batch[2]).to(self.device)
        next_state_batch = torch.cat(batch[3]).to(self.device)
        done_batch = torch.cat(batch[4]).to(self.device)
        
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch.float()))
        
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss

# 訓練迴圈
def train(agent, env, num_episodes, batch_size):
    writer = SummaryWriter()
    
    for episode in range(num_episodes):
        state = env.reset()
        state = obs_to_array(state)

        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = obs_to_array(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward.item()
            
            loss = agent.optimize_model(batch_size)
            if loss is not None:
                writer.add_scalar('Loss/train', loss.item(), episode) # 儲存訓練Loss值用來在tensorboard畫圖
        
        agent.update_epsilon(episode)
        writer.add_scalar('Reward/train', total_reward, episode)# 儲存訓練Reward值用來在tensorboard畫圖
        
        if episode % 10 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/dqn_maze_model_episode_{episode}.pth") # 每10個episodes會儲存一次模型
            print(f"Episode {episode}, Model saved. Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    writer.close()

if __name__ == "__main__":
    state_shape = (13, 13)
    n_actions = 4  # Up, down, left, right
    num_episodes = 200  # 訓練次數
    agent = DQNAgent(state_shape, n_actions, num_episodes)
    batch_size = 32
    
    train(agent, env, num_episodes, batch_size)
    
    torch.save(agent.policy_net.state_dict(), "models/dqn_maze_model.pth") # 儲存最終模型

    env.close()