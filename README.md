# keraslunarlander  
##  on brief
An implementation of dqn to tackle the problem of a gym problem lunarlander-v2
 using keras to build a simple neraul network as the agent.
## Requirements
In order to run the code, you need Python3 and the following libraries:
* tensorflow-gpu:1.10.0
 
* keras:2.1.3
 
* gym: 0.17.3
 
 
run _runagent.py_ to see how the trained model perform

run _newdqnlander.py_ to train your own model

##  Notice ##
  Learning rate is a very important hyper-parameter. When I set it as 0.1, the trained network is totally rubbish. After try and try ,the learning rate should be in 0.001-0.002. The **LunarLander2000.h5** files  is a model of 2000 times iteration with learning rate 0.001 ,the **LunarLander3000.h5** is the model of 2000 times iteration with learning rate 0.0015. The latter one performed better. You can change the name of the model in _runagent.py_ to see how these model perform. No guarantee for solving the problem every time, sometimes it also failed.

Maybe changing hyper parameters and network structure help to improve . Sometimes I feel it is just like randomly guess.

Here using standard scaler may cause the performance not stable, because every time you run the code the scaler is different.This problem can be solved by save the scaler which was used during training.

The video shows the agent running ,the **train.png** indicates the training process.
