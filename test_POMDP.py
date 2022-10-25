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
# model.exp_step()
# model.data_real.save_model()
model.data_real.load_model()
print('load model done')
#model.test()

#model.train_SFNO(epoch=20,batch_size=5)
#model.save_model()
model.error_traj()
#model.save_model()













# grad_en,grad_enp1,grid= model.test_loss()
# print(grad_enp1.shape)
#print(grad_en.vector()[:].shape,grad_enp1.vector()[:].shape,grad_en.vector()[:],grad_enp1.vector()[:])

# fig = plt.figure()
# plt.axis('equal')
# plt.contourf(meshgrid[0],meshgrid[1],p, 200, cmap='jet')
# plt.colorbar()
# plt.savefig('temp2.png')

# fig = plt.figure()
# plt.axis('equal')
# plt.contourf(grid[0],grid[1],a-b, 200, cmap='jet')
# plt.colorbar()
# plt.savefig('temp3.png')

# fig = plt.figure()
# plt.axis('equal')
# plt.contourf(grid[0],grid[1],temp2-b, 200, cmap='jet')
# plt.colorbar()
# plt.savefig('temp4.png')


#temp = model.test_loss()


# input1, input2, grid= model.test_loss()
# input1 = input1.detach().cpu().numpy()
# print('loss',np.max(np.abs(input1 - input2)),input1, input2)


# ig = plt.figure()
# plt.axis('equal')
# plt.contourf(grid[0],grid[1],input1-input2, 200, cmap='jet')
# plt.colorbar()
# plt.savefig('temp1.png')
#model.train_SFNO(epoch = 1,batch_size =4)
#model.train_SFNO(epoch=10,batch_size=10)