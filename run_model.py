import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import numpy as np
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

# 定義 DQN 和 Net 類別
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_value = self.out(x)
        return action_value

class DQN(object):
    def __init__(self):
        self.eval_net = Net()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        action_values = self.eval_net(x)
        action = torch.max(action_values, 1)[1].squeeze().data.cpu().numpy()
        action = action[0]
        return action

# 初始化Unity環境和Gym Wrapper
unity_env = UnityEnvironment(file_name="C:/Users/user/Desktop/Games/9_20")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

# 訓練參數
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space[0].shape[0]

# 創建 DQN 物件
dqn = DQN()

# 載入模型
model_path = "C:/Users/user/Desktop/Project/models/dqn_eval_net_episode_400.pth"
dqn.eval_net.load_state_dict(torch.load(model_path))
dqn.eval_net.eval()  # 設定為評估模式

#測試
for i_episode in range(10):  # 測試10個 episode
    s = env.reset()
    episode_reward = 0
    step_count = 0

    while True:
        env.render()
        a = dqn.choose_action(s)
        print(a)
        s_, r, done, info = env.step(a)
        step_count += 1
        episode_reward += r

        if done:
            print(f'Episode {i_episode} finished with reward: {episode_reward}')
            break
        
        s = s_ 

env.close()