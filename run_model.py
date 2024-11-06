import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

# 定义obs_to_array函数
def obs_to_array(coordinates):
    if isinstance(coordinates, list) and len(coordinates) == 1:
        coordinates = coordinates[0]
    coordinates = np.array(coordinates)
    
    agent = coordinates[:2]
    goal = coordinates[2:4]
    walls = coordinates[4:]
    
    maze = np.zeros((13, 13), dtype=int)
    
    def coord_to_index(x, z):
        x_idx = int(np.round((x + 6) * 12 / 12))
        z_idx = 12 - int(np.round((z + 6) * 12 / 12))
        return max(0, min(12, x_idx)), max(0, min(12, z_idx))
    
    for i in range(0, len(walls), 2):
        if i + 1 < len(walls):
            x, z = walls[i], walls[i+1]
            idx_x, idx_z = coord_to_index(x, z)
            maze[idx_z, idx_x] = 3
    
    agent_x, agent_z = coord_to_index(agent[0], agent[1])
    maze[agent_z, agent_x] = 1
    
    goal_x, goal_z = coord_to_index(goal[0], goal[1])
    maze[goal_z, goal_x] = 2
    
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 3
    
    return maze

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, 1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 主程序：加载模型并运行
if __name__ == "__main__":
    # 初始化Unity环境
    unity_env = UnityEnvironment(file_name="./game", no_graphics=False)
    env = UnityToGymWrapper(unity_env, flatten_branched=True, allow_multiple_obs=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义状态空间和动作空间大小
    state_shape = (13, 13)
    n_actions = 4  # 上下左右
    
    # 初始化网络并加载已训练好的模型
    policy_net = DQN(state_shape, n_actions).to(device)
    policy_net.load_state_dict(torch.load("models/dqn_maze_model.pth"))
    policy_net.eval()  # 设置为评估模式，避免更新模型参数
    for param_tensor in policy_net.state_dict():
        print(param_tensor, "\t", policy_net.state_dict()[param_tensor])
    # 进行推理（不需要训练）
    for episode in range(10):  # 运行10个测试episode
        state = env.reset()
        state = obs_to_array(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                output = policy_net(state)
                print("Model output:", output)
                # 选择模型认为最优的动作
                action = policy_net(state).max(1)[1].view(1, 1).item()
                print(action)

            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_state = obs_to_array(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()