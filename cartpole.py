import torch
import numpy as np
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import random
import gym

env = gym.make("CartPole-v1")

n_episode = 5000
n_time_step = 1000
EPSILON_DECAY = 10000 #在訓練開始的前EPSILON_DECAY步內遞減動作的機率
EPSILON_START = 1.0
EPSILON_END = 0.02
REWARD_BUFFER = np.empty(shape=n_episode) #創建一個空陣列用於儲存每個episode的reward

for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(n_episode * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])#在0到EPSILON_DECAY的步數內隨機動作的機率(epsilon)從EPSILON_START遞減到EPSILON_END
        random_sample = random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()#隨機動作
        else:
            a = agent.online_net.act() #TODO
        
        s_, r, done, info =  env.step(a) #從env返回next state, reward, done, info
        agent.memo.add_memo(s, a, r, done, s_)# TODO
        s = s_
        episode_reward += r

        if done:
            s = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break
        
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample() #從

        # 
