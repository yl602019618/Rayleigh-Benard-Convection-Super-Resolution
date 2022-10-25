import torch
import numpy as np
from torch.utils.data import Dataset
import gym



class ReplayBuffer(object):
    """
    First In First Out experience replay buffer agents.
    """

    def __init__(self, obs_dim =[128+1,64+1] , size =250, device=None):
        '''
        obs_dim: observation data dimension
        act_dim :action_ dimension
        size: size of a buffer which control the sample number that restored in the buffer
        ptr : curren step ,which runs recurrently in the buffer
        '''

        super(ReplayBuffer,self).__init__()
        self.device = device
        self.obs_dimx = obs_dim[1]
        self.obs_dimy = obs_dim[0]
        self.temp_buf = torch.zeros((size, self.obs_dimy,self.obs_dimx), dtype=torch.float) #zeros(size,obs_dim)
        self.p_buf = torch.zeros((size, self.obs_dimy,self.obs_dimx), dtype=torch.float) #zeros(size,obs_dim)
        self.velo_buf = torch.zeros((size, self.obs_dimy,self.obs_dimx, 2), dtype=torch.float)
        self.done_buf = torch.zeros(size, dtype=torch.float)  #zeros(done)
        self.const_buf = torch.zeros(size, dtype=torch.float)
        self.amp_buf = torch.zeros(size, dtype=torch.float)  
        if device is not None:
            self.device = device
            self.temp_buf.to(device)
            self.p_buf.to(device)
            self.velo_buf.to(device)
            self.done_buf.to(device)
            self.const_buf.to(device)
            self.amp_buf.to(device)
        self.ptr, self.size, self.max_size = 0, 0, size
    def save_model(self):
        state = {'temp_buf':self.temp_buf,'p_buf':self.p_buf,'velo_buf':self.velo_buf,'done_buf':self.done_buf,'const_buf':self.const_buf,'amp_buf':self.amp_buf}
        torch.save(state,'data.pth')
    def load_model(self):
        ckpt = torch.load('data.pth')
        self.temp_buf = ckpt['temp_buf'].to(self.device)
        self.p_buf = ckpt['p_buf'].to(self.device)
        self.velo_buf = ckpt['velo_buf'].to(self.device)
        self.done_buf = ckpt['done_buf'].to(self.device)
        self.const_buf = ckpt['const_buf'].to(self.device)
        self.amp_buf = ckpt['amp_buf'].to(self.device)
        self.size = self.temp_buf.shape[0]
        self.max_size = self.size
        self.ptr = self.size
    def store(self, temp, p, velo, done,const ,amp,store_size=1):
        self.temp_buf[self.ptr:self.ptr+store_size] = temp
        self.p_buf[self.ptr:self.ptr+store_size] = p
        self.velo_buf[self.ptr:self.ptr+store_size] = velo
        self.done_buf[self.ptr:self.ptr+store_size] = done
        self.const_buf[self.ptr:self.ptr+store_size] = const
        self.amp_buf[self.ptr:self.ptr+store_size] = amp
        self.ptr = (self.ptr+store_size) % self.max_size  # 如果超过maxsize就重写
        self.size = min(self.size+store_size, self.max_size) #现在的数据量 


    def sample_batch_FNO(self,batch_size,start = 0 , end = int(1e8),n_step = 3):
        '''
        sample from start to end, each sample contains n_step obs-action and final obs2
        obs : obs0 ,... ,obs_(n_step-1)
        act : act0,act1,... , act(n_step-2)
        done: done0 , done1 , ... , done_(n_step-1)                 
        '''
        idxs = torch.randint(start, min(self.size-n_step+1,end), size=(4*batch_size,)).to(self.device) # batch_size*2
        idxs = idxs.unsqueeze(-1)# batch_size*2,1
        idx = idxs.clone()# batch_size*2,1
        for i in range(n_step-1):
            idxs = torch.cat((idxs,idx+i+1),dim = 1)
        # idxs batchsize ,n_step
        idx_obs  = idxs.reshape(-1)# 2*batchsize *n_step
        done_before_select=self.done_buf[idx_obs].reshape(4*batch_size,n_step)
        done_before_select = torch.sum(done_before_select,dim = 1)
        idxs = idxs[done_before_select<1,:]
        idxs = idxs[:batch_size,:]
        idx_obs = idxs.reshape(-1)

        temp = self.temp_buf[idx_obs].reshape(-1,n_step,self.obs_dimy,self.obs_dimx)
        p = self.p_buf[idx_obs].reshape(-1,n_step,self.obs_dimy,self.obs_dimx)
        velo = self.velo_buf[idx_obs].reshape(-1,n_step,self.obs_dimy,self.obs_dimx,2)
        const = self.const_buf[idx_obs].reshape(-1,n_step)
        amp = self.amp_buf[idx_obs].reshape(-1,n_step)
        return  dict(temp=temp, p=p,velo = velo,const = const , amp = amp)

    def sample_traj(self,start = 0 , end = int(1e8),n_step = 3):
        '''
        sample from start to end, each sample contains n_step obs-action and final obs2
        obs : obs0 ,... ,obs_(n_step-1)
        act : act0,act1,... , act(n_step-2)
        done: done0 , done1 , ... , done_(n_step-1)                 
        '''
        n = 100
        idx = torch.arange(15*n-1,16*n-n_step-1).to(self.device) 
        idxs = idx.unsqueeze(-1)# n-n_step,1
        idx = idxs.clone()# n-n_step,1
        for i in range(n_step-1):
            idxs = torch.cat((idxs,idx+i+1),dim = 1)
        # idxs  : n-n_step,n_step
        idx_obs  = idxs.reshape(-1).detach().cpu().numpy().astype('float32')# (n-n_step)*n_step
       
        temp = self.temp_buf[idx_obs].reshape(-1,n_step,self.obs_dimy,self.obs_dimx)
        p = self.p_buf[idx_obs].reshape(-1,n_step,self.obs_dimy,self.obs_dimx)
        velo = self.velo_buf[idx_obs].reshape(-1,n_step,self.obs_dimy,self.obs_dimx,2)
        const = self.const_buf[idx_obs].reshape(-1,n_step)
        amp = self.amp_buf[idx_obs].reshape(-1,n_step)
        return  dict(temp=temp, p=p,velo = velo,const = const , amp = amp)

