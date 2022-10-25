import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
import torchvision.transforms.functional as vtF
from timeit import default_timer

#torch.manual_seed(0)
np.random.seed(0)
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SuperConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2,resolution):
        super(SuperConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.resolution = resolution
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.resolution[0] , self.resolution[1]//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, self.modes1:2*self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights3)
        out_ft[:, :, -2*self.modes1:-self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights4)
        
        out_ft[:, :, 2*self.modes1:3*self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights5)
        out_ft[:, :, -3*self.modes1:-2*self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights6)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.resolution[0], self.resolution[1]))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,modes3,modes4,width,resolution,step):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (u(x, y),v(x,y),p(x,y),T(x,y), x, y)
        input shape: (batchsize, x=s, y=s, c=5)
        output: the solution (u,c,p,t)
        output shape: (batchsize, x=s, y=s, c=4)
        """
        self.resolution = resolution #[33,65]   
        self.modes1 = modes1 #5
        self.modes2 = modes2 #
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        
        self.step = (step-1)*6 # 
        self.last_layer = 128
        self.fc0 = nn.Linear(self.step, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SuperConv2d(self.width, self.width, self.modes1, self.modes2,self.resolution)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes3, self.modes4)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes3, self.modes4)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes3, self.modes4)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        #self.ln1 = nn.LayerNorm([self.resolution,self.resolution,self.last_layer])
        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.width)
        #self.bn3 = nn.BatchNorm2d(self.last_layer)
        
        self.fc1 = nn.Linear(self.width, self.last_layer)
        
        self.fc2 = nn.Linear(self.last_layer, 4)


    def forward(self, x):
        '''
        x batchsize y,x,n_step-1 ,4
        '''

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1).reshape(x.shape[0],x.shape[1],x.shape[2],self.step)  #x batchsize y,x,n_step-1 ,6-->#x batchsize x,y,(n_step-1)*6
        
        
        x = self.fc0(x) # batchsize y ,x ,width 
        x = x.permute(0, 3, 1, 2) # batchsize ,width ,y ,x
        x = self.bn0(x) # batchsize ,width , y, x
        
        x1 = self.conv0(x) # batchsize ,width , ysuper, xsuper
        x0 = vtF.resize(x, size=[self.resolution[0],self.resolution[1]],interpolation=2)
        x2 = self.w0(x0)
        x = x1 + x2
        x = F.gelu(x)    
        
        x = self.bn1(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x) 
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x = self.bn2(x)
        x = x.permute(0,2,3,1) # batchsize , ysuper , xsuper , self.width
        x = self.fc1(x) # batchsize , ysuper , xsuper ,self.last layer
        #x = self.bn3(x).permute(0,2,3,1)
        x =  F.gelu(x)
        
        x = self.fc2(x) #batchsize  , y super , xsuper , 4
        x = torch.cat((x[:,:,-1:,:],x[:,:,1:,:]),dim = 2)
        
        return x
    def get_grid(self, shape, device):
        batchsize, size_x, size_y,n_step = shape[0], shape[1], shape[2],shape[3]
        gridx = torch.tensor(np.linspace(0, 3, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1,1, 1).repeat([batchsize, 1, size_y, 1,1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y,1, 1).repeat([batchsize, size_x, 1, 1,1])
        return torch.cat((gridx, gridy), dim=-1).repeat([1,1,1,n_step,1]).to(device)





