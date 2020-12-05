# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:20:47 2020

@author: 77433
"""
from newdqnlander import Agent
import numpy as np
import gym
from sklearn.preprocessing import StandardScaler
env = gym.make('LunarLander-v2')
observation_samples = []
# play a bunch of games randomly and collect observations
for n in range(500):
    observation = env.reset()
    observation_samples.append(observation)
    done = False
    while not done:
        action = np.random.randint(0, env.action_space.n)
        observation, reward, done, _ = env.step(action)
        observation_samples.append(observation)
        
observation_samples = np.array(observation_samples)
scaler = StandardScaler()
scaler.fit(observation_samples)

dqnagent = Agent(4,scaler,name = 'LunarLander1')
dqnagent.runonetime()
#dqnagent.runtimes(10)
