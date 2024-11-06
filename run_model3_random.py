import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import random

# Define obs_to_array function
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

# Define DQN network
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

# Main function: load model and run inference
if __name__ == "__main__":
    # Initialize Unity environment
    unity_env = UnityEnvironment(file_name="C:/Users/user/Desktop/Games/10_30", no_graphics=False)
    env = UnityToGymWrapper(unity_env, flatten_branched=True, allow_multiple_obs=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define state space and action space size
    state_shape = (13, 13)
    n_actions = 4  # up, down, left, right
    
    # Initialize network and load trained model
    policy_net = DQN(state_shape, n_actions).to(device)
    policy_net.load_state_dict(torch.load("models/dqn_maze_model_episode_70.pth"))
    policy_net.eval()  # Set to evaluation mode to avoid updating model parameters

    for episode in range(10):  # Run 10 test episodes
        state = env.reset()
        state = obs_to_array(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        done = False
        total_reward = 0
        
        while not done:
            with torch.no_grad():
                if np.random.rand() < 0.03:  # 1% chance to perform a random action
                    action = random.randint(0, n_actions - 1)
                    print("Random action selected:", action)
                else:
                    output = policy_net(state)
                    action = output.max(1)[1].view(1, 1).item()
                    print("Model action selected:", action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            next_state = obs_to_array(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()