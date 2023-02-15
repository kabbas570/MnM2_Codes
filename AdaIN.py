import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
      
        
      

import numpy as np
import torch

temp1 = np.ones([1024,8,8])
temp1[:,0,0] = 1
temp1[:,0,1] = 2
temp1[:,1,0] = 3
temp1[:,1,1] = 4

temp1= np.expand_dims(temp1, axis=0)
temp = torch.tensor(temp1)


temp1 = np.ones([1024,8,8])
temp1[:,0,0] = 1
temp1[:,0,1] = 2
temp1[:,1,0] = 3
temp1[:,1,1] = 4

temp1= np.expand_dims(temp1, axis=0)
temp1 = torch.tensor(temp1)




def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])


def sigma(x):
    return torch.sqrt((torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))


mean = mu(temp1)
print(mean.shape)

std = sigma(temp1)
print(std.shape)
