import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
#from
#  main_PIPOMDP import heat_eqn
from POMDP_rbc import RBC_SR
#from POMDP import heat_eqn
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device is ' , device)

r = 16
model = RBC_SR(exp_step =r*100 , total_step = 10000,update_freq_model = 10 , update_freq_policy = 10, device = device,n_step =4)
model.init_variable()

#print(model.optimizer.defaults['lr'])
#model.exp_step()
#model.data_real.save_model()
#print('load model done')
#model.train_SFNO(epoch=20000,batch_size=10)
#model.save_model()
model.data_real.load_model()
model.test_gen_traj()


