# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:28:38 2020

@author: 77433
"""
import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Activation,Input
from keras.optimizers import SGD, RMSprop, Adam, Adamax

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
class Agent:
    def __init__(self,actions,name='newLunarLander',):
        self.actions = actions
        self.name = name
        self.env = gym.make('LunarLander-v2')
        self.learningrate = 0.001 #0.001better
        self.model = self.getmodel()
        self.epsilon = 1
        self.gama = 0.99
        self.timestep = 0 
        self.loss = 0
        self.explore = 1
    
    def getmodel(self):
        if os.path.exists(f'{self.name}.h5'):
            m = load_model(f'{self.name}.h5')
            print('load history model!')
            return m
        else:
            print('building a new model')
            return self.buildmodel()
    
    def buildmodel(self):
        IN = Input((8,))
        x = Dense(64,activation=('relu'))(IN)
        x = Dense(128,activation=('tanh'))(x) #use tanh activation better,because tanh has negactive value
        #x = Dense(64,activation=('tanh'))(x)
        out = Dense(self.actions,activation=('linear'))(x)
        adam = Adam(lr = self.learningrate)
        m = Model(inputs= IN ,outputs = out)
        m.compile(optimizer = adam,loss='mse')
        m.summary()
        return m
    
    def trainmodel(self,batch):
        ob_current = batch[0]
        action = batch[1]
        reward = batch[2]
        done = batch[3]
        ob_next = batch[4]
        ob_x = np.array([ob_current])
        ob_next = np.array([ob_next])
        Qvalue = self.model.predict([ob_x])
        
        if not done:
            Qvalue[0][action] = reward + self.gama* np.max(self.model.predict(ob_next))
        else:
            Qvalue[0][action] = reward
        Qvalue = np.array(Qvalue)
        self.loss = self.model.train_on_batch(ob_x,Qvalue)
        
        #self.model.fit(ob_x,Qvalue,batch_size = 16,epochs=50)
        if self.timestep%5000 == 0 :
            self.model.save(f'{self.name}.h5')
            print('model saved!')

    def starttrain(self,n = 3000):
        t = 0 
        record = np.empty(n+10)
        while(1):
            ob_current = self.env.reset()
            t+=1
            done = False
            if t > n:
                print('t:',t)
                print('train done!')
                return
            r = 0
            frames = 0
            self.explore = self.explore + 1
            while not done:
                a = self.getactions(ob_current)
                ob_next , reward, done ,_ = self.env.step(a)
                unit = (ob_current,a,reward,done,ob_next)
                ob_current = ob_next
                r = r+reward
                frames+=1
                #self.store(unit)
                self.timestep=self.timestep+1
                self.change_epsl()
                self.trainmodel(unit)
            record[t] = r
            print('episode:',t,' ',frames,self.epsilon, 'reward:',r,'average reward last 100 times:',record[max(0, t-100):(t+1)].mean())
            if record[max(0, t-100):(t+1)].mean() >200:
                break

    
        
    def change_epsl(self):
        self.epsilon = 1/np.sqrt(self.explore)
    
    def getactions(self,observation):
        observation = np.atleast_2d(observation)
        t = np.random.random()
        if t< self.epsilon:
            action = np.random.randint(0,self.actions-1)
            #print('--random action-- ')
        else:
            r = self.model.predict(observation)
            #print('observation',observation,'  ',r)
            action = np.argmax(r)
        #print(action)
        return action
    
    def getactions1(self,observation):
        observation = np.atleast_2d(observation)
        r = self.model.predict(observation)
        action = np.argmax(r)
        #print(' output ',r)
        return action
    
    def runonetime(self):
        done = False
        env = gym.wrappers.Monitor(self.env, 'mydqn', force=True)
        ob = env.reset()
        r = 0
        t = 0
        while not done:
            env.render()
            action = self.getactions1(ob)
            ob,reward,done,_ = env.step(action)
            r+= reward
            t+=1
        self.env.close()
        print('reward:', r,'--frames:',t)
        return r
            
        
    def runtimes(self,n):
        ar = 0
        for i in range(n):
            r=0
            t=0
            ob = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.getactions1(ob)
                ob,reward,done,_ = self.env.step(action)
                r+= reward
                t+= 1
            print('reward:', r,'--frames:',t)
            ar =ar +r
        self.env.close()
        print(f'average reward of {n} time: {ar/n}' )
    
    
if __name__ =='__main__':
    
    a = Agent(4)
    a.starttrain(n=3000)
    a.runtimes(10)