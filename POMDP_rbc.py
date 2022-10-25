import torch
import torch.nn as nn
#from model import FNO2d, Heat_forward
from model_rbc import FNO2d
import gym
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import Buffer1 as bf
from RBC_env import RBC
#import wandb
#wandb.init(project="POMDP")
import matplotlib.pyplot as plt
from scipy.io import savemat

print(1)
plt.rcParams['figure.figsize']=(12.8, 7.2)


class RBC_SR():
    def __init__(self,exp_step = 3500, total_step = 3000,update_freq_model = 10 , update_freq_policy = 10,n_step = 3, device = None):
        '''
        current_step
        current_episode
        exp_step_num : number of step in exploration period
        total_step : number of step in real environment in the whole training procedure
        update_freq_model : number of step in whole 2ed period takes for 1 step transition model training 
        update_freq_policy: number of step in whole 2ed period takes for 1 step policy model training
        obs_dim : the dimension of observation data
        mode1 : first kind modes number of SFNO
        mode2 : second kind modes number of SFNO
        width : width of SFNO
        resolution : the dimension size of hidden state recovered by SFNO in x\y axis
        dt : match dt in fenics simulation setting
        '''
        
        self.current_step = 0
        self.current_episode = 0
        self.exp_step_num = exp_step
        self.total_step = total_step
        self.update_freq_model = update_freq_model
        self.update_freq_policy = update_freq_policy
        self.start = 50
        self.end = 150
        self.obs_dim = [17,33]
        self.mode1 = 5
        self.mode2 = 9
        self.mode3 = 17
        self.mode4 = 17
        self.width = 20
        self.device = device
        self.resolution = [(self.obs_dim[0]-1)*4+1 ,(self.obs_dim[1]-1)*4+1]   # 33,65 
        self.scale =4
        # if n_step = 3 then we take out 3 step data and divide it into 2 2
        self.n_step = n_step 

        #self.n = n_step+1
        
        self.SFNO = FNO2d(self.mode1 , self.mode2,self.mode3,self.mode4,self.width, self.resolution,self.n_step).to(self.device)
        self.dt = 0.125
        #self.step_forward = Heat_forward(n_x =self.resolution ,dt = self.dt, alpha =1/16,device = self.device).to(self.device)
        self.beta_loss =  [1,0.002,10000]
        self.env =  RBC()
        self.episode = 0
            
        #wandb.config.n_step = n_step   
        self.data_real = bf.ReplayBuffer(obs_dim = self.resolution , size = self.exp_step_num  ,device = device)
        self.total_loss_log = []
        self.data_loss_log = []
        self.phy_loss_log = []
        self.boundary_loss_log =[]
        self.HR_loss_log =[]
        self.x1 = np.linspace(0,2,128+1)
        self.sinx1 =torch.Tensor(np.sin(2*np.pi*self.x1)).to(self.device)


    def init_variable(self):
        self.env.reset()
        self.optimizer = optim.Adam(self.SFNO.parameters(), lr=1e-3)
        self.mesh = self.env.solver.meshgrid
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.998)

    def exp_step(self):
        #observation_new = self.env.get_value()  #get observe of step 0
        for i in range(self.start):
            self.env.step()
        for n in range(self.exp_step_num):
            temp , velo , p , episode_over,const ,amp = self.env.step()
            temp = torch.Tensor(temp).to(self.device)
            velo = torch.Tensor(velo).to(self.device)
            p =  torch.Tensor(p).to(self.device)
            const = torch.Tensor(const) .to (self.device)
            amp = torch.Tensor(amp) .to (self.device)
            self.data_real.store(temp  = temp ,p = p, velo = velo , done = episode_over,const = const , amp = amp)
            if n % 20 == 0:
                print('current extrapolation step is :',n)
            if episode_over:
                print('episode_over!!!!')
                self.env.reset()
                self.episode +=1
                for i in range(self.start):
                    self.env.step()

    def downsample(self,input,var):
        if var == 1:
            return input[:,:,::4,::4,:]
        elif var == 0:
            return input[...,::4,::4]    
        else:
            return input[:,::4,::4,:]

    def loss_gen(self,output0,output1, truth ,const,amp):
        '''
        output0/1 of SFNO is (batch_size ,y,x,4)
        truth is (batchsize, 2, y, x, 4) which represent the corresponding point of HR data
        top batchsize , n_step
        bottom batchsize , n_step
        beta_loss is the weight of data loss, boundary_loss and physic-informed loss
        '''
        MSE_loss = nn.MSELoss()
        self.r = 4 # downsample factor
        #compute data_loss
        
        self.data_loss = MSE_loss(self.downsample(input = output0,var = 2) ,self.downsample(truth[:,0,:,:,:],var = 2))+MSE_loss(self.downsample(input = output1,var = 2) ,self.downsample(truth[:,1,:,:,:],var = 2))
        '''
        boundary loss should be treated carefully
        the output is under order u,v,p,t
        on the left and right boundary , u,v,p,t is periodic 
        on the top boundary u v is 0 ,T is top
        on the bottom boundary u v is  0 , T is bottom 
        top  is batchsize,n_step
        bottom is batchsize , n_stpe
        '''

        self.boundary_loss = self.bnd_loss(output0,const,amp) + self.bnd_loss(output1,const,amp) 
        output_fem  = self.env.step_forward(output0,const,amp).to(self.device)
        self.phy_loss = MSE_loss(output_fem,output1)

        self.loss = self.data_loss*self.beta_loss[0] + self.boundary_loss*self.beta_loss[1]+ self.phy_loss*self.beta_loss[2]
        
        #grad_en,grad_enp1,self.J = self.env.adjoint_forward(output0,output1,a1,b1)
        #wandb.log({'PHY_loss': self.J})
        return self.loss ,output_fem
    # def test_forward(self,output0,output1,top,bottom):
    #     return self.env.adjoint_forward(output0,output1,top,bottom)
    def bnd_loss(self,input,const,amp):
        '''
        input shape is (batch_size ,y,x,4) ,top batchsize 1 ,bottom batchsize 1
        order is u,v,p,t
        on the top boundary u v is 0 ,T is top where y = 1
        top and bottom should be batch_size,1
        the periodic boundary should be treated as the norm of difference of column 0 and -1 in x axis 
        '''
        
        top_loss = (torch.norm(input[:,-1,:,0]) +  torch.norm(input[:,-1,:,1]) +torch.norm(input[:,-1,:,3] ))/3
        bottom = const[:,0:1]+amp[:,0:1]*self.sinx1.unsqueeze(0).repeat(input.shape[0],1)
        
        bottom_loss = (torch.norm(input[:,0,:,0]) + torch.norm(input[:,0,:,1]) + torch.norm(input[:,0,:,3] - bottom))/3
        
        boundary_loss = top_loss + bottom_loss 
        return boundary_loss
        
    def train_SFNO(self,epoch,batch_size):
        '''
        1. sample batch from data_real
        2. Training SFNO via physic-informed loss and data loss
        epoch is the number of training step
        n_step is the n_step in sampling batch
        '''
        # wandb.config.batchsize = batch_size
        # wandb.config.epoch =epoch

        print_every = 100
        plot_every = 40

        for i in range(epoch):
            sample = self.data_real.sample_batch_FNO(batch_size= batch_size,start = 0 , end = self.data_real.size,n_step = self.n_step)
            velo = sample['velo'] #(batch_size,n_step ,grid_x,grid_y,2)
            p = sample['p']  #(batch_size , n _step , x, y )
            temp = sample['temp'] #(batch_size , n _step , x, y )
            # the input is concate as batchsize ,n_step ,x, y,4
            const = sample['const'].to(self.device)
            amp  =sample ['amp'].to(self.device)

            input = torch.cat((velo,p.unsqueeze(-1)),dim = 4)
            input  = torch.cat((input , temp.unsqueeze(-1)),dim = 4).to(self.device)# batch ,n_step ,y,x,4

            output0 = self.SFNO(self.downsample(input[:,:-1,:,:,:],var=1).permute(0, 2, 3, 1,4))# batch ,  y, x ,nstep, 4 -- ---- # batch ,  y, x , 4
            output1 = self.SFNO(self.downsample(input[:,1:,:,:,:],var = 1).permute(0, 2, 3, 1,4)) # batch ,  y, x ,nstep, 4-- ---- # batch ,  y, x , 4
            
            '''
            to compute the loss ,we should offer the information output0,output1 , input, top , bottom
            '''

            self.loss,output_fem= self.loss_gen(output0 = output0 ,output1 = output1, truth = input[:,-2:,:,:,:],const = const , amp = amp)
            self.loss.backward(retain_graph = True)
            self.optimizer.step()
            self.optimizer.zero_grad()


            if i in [4000,7000,43000,35000]:
                for params in self.optimizer.param_groups:
                    params['lr'] /= 2
                    #wandb.log({'lr': params['lr']})
            if i in [10000,15000, 20000,30000,40000,45000]:
                for params in self.optimizer.param_groups:
                    params['lr'] /= 4
                    #wandb.log({'lr': params['lr']})

            if (i+1) % print_every == 0:
                print('Epoch :%d ; Loss:%.8f;phy_loss %.8f;data_loss %.8f; boundary_loss %.8f'  % (i+1, self.loss_dpb.item()+self.J*self.beta_loss[2],self.J,self.data_loss.item(),self.boundary_loss.item()))
            if (i+1)% plot_every  == 0:    
                self.plot_train(input[1,-2:],output0[1],output1[1],i)
                #wandb.log({'HR_loss': HR_loss})

        
    def save_model(self):
        torch.save(self.SFNO.state_dict(), 'SFNO.pt')
    
    def load_model(self):
        SFNO_state_dict = torch.load('SFNO.pt')
        self.SFNO.load_state_dict(SFNO_state_dict)

    # def test_loss(self):
    #     sample = self.data_real.sample_batch_FNO(batch_size= 3,start = 0 , end = self.data_real.size,n_step = self.n_step)
    #     velo = sample['velo']#(batch_size,n_step ,grid_x,grid_y,2)
    #     p = sample['p'] #(batch_size , n _step , x, y )
    #     temp = sample['temp'] #(batch_size , n _step , x, y )
    #     a1 = sample['a1'].to(self.device)
    #     b1  =sample ['b1'].to(self.device)
    #     input = torch.cat((velo,p.unsqueeze(-1)),dim = 4)
    #     input  = torch.cat((input , temp.unsqueeze(-1)),dim = 4).to(self.device)# batch ,n_step ,y,x,4
        
    #     grad_en,grad_enp1,meshgrid= self.env.adjoint_forward(input[:,0,:,:],input[:,1,:,:],a1,b1)
    #     #grad_dof = self.env.adjoint_forward(input[0:1,0,:,:],input[0:1,1,:,:],top,bottom)
    #     return grad_en,grad_enp1,meshgrid
    

    def test_gen_traj(self):
        sample = self.data_real.sample_traj(n_step = self.n_step)
        velo = sample['velo'].detach().cpu().numpy()#(n-n_step,n_step ,grid_x,grid_y,2)
        p = sample['p'].detach().cpu().numpy() #(n-n_step, n _step , x, y )
        temp = sample['temp'].detach().cpu().numpy() #(n-n_step, n _step , x, y )
        const = sample['const'].detach().cpu().numpy()
        amp  =sample ['amp'].detach().cpu().numpy()
        mesh = self.env.solver.meshgrid
        for i in range(velo.shape[0]):
            self.plot_all(temp[i,-1,:,:],velo[i,-1,:,:,:],p[i,-1,:,:],mesh,i)
            print(const[i,:],amp[i,:])
    
    def error_traj(self):
        self.load_model()
        self.data_real.load_model()
        sample = self.data_real.sample_traj(n_step = self.n_step)
        velo = sample['velo']#(n-n_step,n_step ,grid_x,grid_y,2)
        p = sample['p'] #(n-n_step, n _step , x, y )
        temp = sample['temp'] #(n-n_step, n _step , x, y )
        input = torch.cat((velo,p.unsqueeze(-1)),dim = 4)
        input  = torch.cat((input , temp.unsqueeze(-1)),dim = 4).to(self.device)# batch ,n_step ,y,x,4

        output0 = self.SFNO(self.downsample(input[:,:-1,:,:,:],var=1).permute(0, 2, 3, 1,4))# batch ,  y, x ,nstep, 4 -- ---- # batch ,  y, x , 4
        output1 = self.SFNO(self.downsample(input[:,1:,:,:,:],var = 1).permute(0, 2, 3, 1,4)) # batch ,  y, x ,nstep, 4-- ---- # batch ,  y, x , 4
        line = np.linspace(0,1,output0.shape[0])
        # computing the error wrt uv ,p, t and overall  
        error_unnormed = torch.norm(output0 - input[:,-2,:,:,:],p=2,dim =[1,2,3] ).detach().cpu().numpy()
        error_uv_unnormed =  torch.norm(output0[:,:,:,0:2] - input[:,-2,:,:,0:2],p=2,dim =[1,2,3] ).detach().cpu().numpy()
        error_p_unnormed =  torch.norm(output0[:,:,:,2] - input[:,-2,:,:,2],p=2,dim =[1,2] ).detach().cpu().numpy()
        error_t_unnormed =  torch.norm(output0[:,:,:,3] - input[:,-2,:,:,3],p=2,dim =[1,2] ).detach().cpu().numpy()
        fig = plt.figure()
        plt.plot(line,error_unnormed,label = 'error_all')
        plt.plot(line,error_uv_unnormed,label = 'error_uv')
        plt.plot(line,error_p_unnormed,label = 'error_p')
        plt.plot(line,error_t_unnormed,label = 'error_t')
        plt.title('Error without Norm')
        plt.legend(loc = 'best')
        plt.savefig('error_unnormed.png')
        # computing the norm wrt uv, p , t and overall
        norm = torch.norm( input[:,-2,:,:,:],p=2,dim =[1,2,3] ).detach().cpu().numpy()
        norm_uv =  torch.norm( input[:,-2,:,:,0:2],p=2,dim =[1,2,3] ).detach().cpu().numpy()
        norm_p =  torch.norm( input[:,-2,:,:,2],p=2,dim =[1,2] ).detach().cpu().numpy()
        norm_t =  torch.norm( input[:,-2,:,:,3],p=2,dim =[1,2] ).detach().cpu().numpy()
        fig = plt.figure()
        plt.plot(line,norm,label = 'norm_all')
        plt.plot(line,norm_uv,label = 'norm_uv')
        plt.plot(line,norm_p,label = 'norm_p')
        plt.plot(line,norm_t,label = 'norm_t')
        plt.title('Norm')
        plt.legend(loc = 'best')
        plt.savefig('Norm.png')
        # computing the relative error wrt uv, p, t and overall
        error_normed = error_unnormed/norm
        error_uv_normed = error_uv_unnormed/norm_uv
        error_p_normed = error_p_unnormed/norm_p
        error_t_normed = error_t_unnormed/norm_t
        fig = plt.figure()
        plt.plot(line,error_normed,label = 'error_all')
        plt.plot(line,error_uv_normed,label = 'error_uv')
        plt.plot(line,error_p_normed,label = 'error_p')
        plt.plot(line,error_t_normed,label = 'error_t')
        plt.title('Error with Norm')
        plt.legend(loc = 'best')
        plt.savefig('error_normed.png')

        
    def plot_train(self,input,output0,output1,epoch):
        '''
        input 2,y,x,4
        output0 y,x,4
        output1 y,x,4
        '''
        input = input.detach().cpu().numpy()
        output0 = output0.detach().cpu().numpy()
        output1 = output1.detach().cpu().numpy()
        url_input0 = './result/train_plot_input0'
        url_input1 = './result/train_plot_input1'
        url_output0 = './result/train_plot_output0'
        url_output1 = './result/train_plot_output1'
        self.plot_all(input[0,:,:,3],input[0,:,:,0:2],input[0,:,:,2],self.mesh,epoch = epoch,url = url_input0)
        self.plot_all(input[1,:,:,3],input[1,:,:,0:2],input[1,:,:,2],self.mesh,epoch = epoch,url = url_input1)
        self.plot_all(output0[:,:,3],output0[:,:,0:2],output0[:,:,2],self.mesh,epoch = epoch,url = url_output0)
        self.plot_all(output1[:,:,3],output1[:,:,0:2],output1[:,:,2],self.mesh,epoch = epoch,url = url_output1)

    def plot_all(self,temp,velo,p,meshgrid,epoch,url = None):  
        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(meshgrid[0],meshgrid[1],temp, 200, cmap='jet')
        plt.colorbar()
        if url  != None:
            plt.savefig(url+'/temp_pic-{}.png'.format(epoch))
        else:
            plt.savefig('./img_temp/pic-{}.png'.format(epoch))
        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(meshgrid[0],meshgrid[1],p, 200, cmap='jet')
        plt.colorbar()
        if url  != None:
            plt.savefig(url+'/pressure_pic-{}.png'.format(epoch))
        else:
            plt.savefig('./img_pressure/pic-{}.png'.format(epoch))
        xl, xh  = self.env.solver.geometry.min_x, self.env.solver.geometry.max_x
        yl, yh = self.env.solver.geometry.min_y, self.env.solver.geometry.max_y
        fig, ax = plt.subplots(figsize=(12,9))
        ax.axis('equal')
        ax.set(xlim=(xl, xh), ylim=(yl, yh))
        ax.quiver(meshgrid[0],meshgrid[1], velo[:,:,0],velo[:,:,1] , temp)
        if url  != None:
            plt.savefig(url+'/velo_pic-{}.png'.format(epoch))
        else:
            plt.savefig('./img_velo/pic-{}.png'.format(epoch))



    def test(self):
        batch_size =15
        sample = self.data_real.sample_batch_FNO(batch_size= batch_size,start = 0 , end = self.data_real.size,n_step = self.n_step)
        velo = sample['velo'] #(batch_size,n_step ,grid_x,grid_y,2)
        p = sample['p']  #(batch_size , n _step , x, y )
        temp = sample['temp'] #(batch_size , n _step , x, y )
        # the input is concate as batchsize ,n_step ,x, y,4
        const = sample['const'].to(self.device)
        amp  =sample ['amp'].to(self.device)

        input = torch.cat((velo,p.unsqueeze(-1)),dim = 4)
        input  = torch.cat((input , temp.unsqueeze(-1)),dim = 4).to(self.device)# batch ,n_step ,y,x,4

        output0 = self.SFNO(self.downsample(input[:,:-1,:,:,:],var=1).permute(0, 2, 3, 1,4))# batch ,  y, x ,nstep, 4 -- ---- # batch ,  y, x , 4
        output1 = self.SFNO(self.downsample(input[:,1:,:,:,:],var = 1).permute(0, 2, 3, 1,4)) # batch ,  y, x ,nstep, 4-- ---- # batch ,  y, x , 4

        output_fem  = self.env.step_forward(input[:,1,:,:,:],const,amp).to(self.device)
        error = torch.norm(input[:,2,:,:,3] - output_fem[:,:,:,3],p = 2 , dim  =[1,2])
        print(error)
        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.mesh[0],self.mesh[1],input[4,1,:,:,3], 200, cmap='jet')
        plt.colorbar()
        plt.savefig('input0.png')

        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.mesh[0],self.mesh[1],input[4,2,:,:,3], 200, cmap='jet')
        plt.colorbar()
        plt.savefig('input1.png')

        
        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.mesh[0],self.mesh[1],output_fem[4,:,:,3], 200, cmap='jet')
        plt.colorbar()
        plt.savefig('output1.png')

        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.mesh[0],self.mesh[1],input[5,1,:,:,3], 200, cmap='jet')
        plt.colorbar()
        plt.savefig('input01.png')

        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.mesh[0],self.mesh[1],input[5,2,:,:,3], 200, cmap='jet')
        plt.colorbar()
        plt.savefig('input11.png')

        
        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.mesh[0],self.mesh[1],output_fem[5,:,:,3], 200, cmap='jet')
        plt.colorbar()
        plt.savefig('output11.png')
        
        
        
        
