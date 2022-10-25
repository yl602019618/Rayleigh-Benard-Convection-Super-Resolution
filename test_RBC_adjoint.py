import numpy as np
from os import path
import math

import torch
import random
import math
import torch.nn as nn
from RBC_adjoint import RBC
import Buffer1 as bf
print('ZH220102')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device is ' , device)
buffer = bf.ReplayBuffer(obs_dim =[32*2+1,33] , size =200,device = device)
simulator = RBC()
simulator.reset()
for i in range(100):
    temp , velo , p , episode_over,a1, b1 = simulator.step()
    print(a1,b1)
    # temp = torch.Tensor(temp).to(device)
    # velo = torch.Tensor(velo).to(device)
    # p =  torch.Tensor(p).to(device)
    # buffer.store(temp  =temp ,p = p, velo = velo , done = episode_over)

# solver.plot_temp()
# solver.get_observation()
# pp = buffer.sample_batch_FNO(batch_size= 2)
# velo = pp['temp']
# print(velo[0],velo[0].shape)