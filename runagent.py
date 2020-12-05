# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:27:49 2020

@author: 77433
"""
from newdqnlander1 import Agent

import gym

env = gym.make('LunarLander-v2')

dqnagent = Agent(4,name = 'LunarLander829')
dqnagent.runonetime()
#dqnagent.runtimes(10)