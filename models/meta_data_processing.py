import numpy as np
import torch

def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

def sigma(x):
    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
    across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
    the permutations are required for broadcasting"""
    
    return torch.sqrt((torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))



temp = np.zeros([1,2,2])
temp[:,0,0] = 1
temp[:,0,1] = 1
temp[:,1,0] = 1
temp[:,1,1] = -1
temp = np.expand_dims(temp, axis=0)
temp = torch.tensor(temp)

# mean = mu(temp)
# print(mean)
# std = sigma(temp)
# print(std)

temp1 = np.zeros([1,2,2])
temp1[:,0,0] = 1
temp1[:,0,1] = 2
temp1[:,1,0] = 3
temp1[:,1,1] = 4

temp1= np.expand_dims(temp1, axis=0)
temp1 = torch.tensor(temp1)

# mean = mu(temp1)
# print(mean)
# std = sigma(temp1)
# print(std)




import torch.nn.functional as F
import torch.nn as nn
class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])
    def sigma(self, x):    
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))
    def forward(self, x, y):
        #a2 = ((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) 
        x = x.permute([2,3,0,1])
        a2 = x + self.mu(y)
        a2 = self.sigma(y)*a2
        a2 = a2.permute([2,3,0,1])
        return a2
    
    
ada = AdaIN() 
temp3 = ada(temp1,temp)
Rel = nn.ReLU(inplace=True)
print(temp3)
print(Rel(temp3))

print(temp3.shape)


# mean = mu(temp3)
# print(mean)
# std = sigma(temp3)
# print(std)

# a =temp3[0,:,:,:].numpy()
# print(a)



# def mu(x):
#     """ Takes a (n,c,h,w) tensor as input and returns the average across
#     it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
#     return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

# def sigma(x):
#     """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
#     across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
#     the permutations are required for broadcasting"""
    
#     return torch.sqrt((torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))


# temp1 = torch.tensor(temp1)

# mean = mu(temp1)
# mean = mean.numpy()
# print(mean)


# temp = np.zeros([1024,2,2])
# temp[:,0,0] = .1
# temp[:,0,1] = .2
# temp[:,1,0] = .3
# temp[:,1,1] = .4


# temp = np.zeros([1024,2,2])
# temp[:,0,0] = .1
# temp[:,0,1] = .02
# temp[:,1,0] = .03
# temp[:,1,1] = .4

# temp = np.zeros([1024,2,2])
# temp[:,0,0] = 3.8
# temp[:,0,1] = 0
# temp[:,1,0] = 0
# temp[:,1,1] = 0

# temp = np.expand_dims(temp, axis=0)

# temp = torch.tensor(temp)

# std = sigma(temp)
# std = std.numpy()
# print(std)

# mean = mu(temp)
# mean = mean.numpy()
# print(mean)

# .5 --> .2165  and .125 
# 1.8 --> 0.779 and 0.45
# 3.8 --> 1.64  and.95



# def Generate_Meta_(vendors_,scanners_,diseases_): 
#     temp = np.zeros([1024,2,2])
#     if vendors_=='GE MEDICAL SYSTEMS':
#         temp[:,0,0] = 0.5
#         temp[:,0,1] = 0
#         temp[:,1,0] = 0
#         temp[:,1,1] = 0
#     if vendors_=='SIEMENS':
#         temp[:,0,0] = 1.8
#         temp[:,0,1] = 0
#         temp[:,1,0] = 0
#         temp[:,1,1] = 0
#     if vendors_=='Philips Medical Systems':
#         temp[:,0,0] = 3.8
#         temp[:,0,1] = 0
#         temp[:,1,0] = 0
#         temp[:,1,1] = 0
#     return temp

import numpy as np
import torch

def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

def sigma(x):
    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
    across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
    the permutations are required for broadcasting"""
    
    return torch.sqrt((torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))



temp = np.zeros([1,2,2])
temp[:,0,0] = .1
temp[:,0,1] = .2
temp[:,1,0] = -3
temp[:,1,1] = 4

# temp[:,1,0] = .1
# temp[:,1,1] = .2
# temp[:,2,1] = .3
# temp[:,3,1] = .4

temp = np.expand_dims(temp, axis=0)
temp = torch.tensor(temp)

# mean = mu(temp)
# print(mean)
# std = sigma(temp)
# print(std)

temp1 = np.zeros([1,8,8])
temp1[:,0,0] = 1
temp1[:,0,1] = 2
temp1[:,1,1] = 3
temp1[:,1,1] = 4

temp1= np.expand_dims(temp1, axis=0)
temp1 = torch.tensor(temp1)

# mean = mu(temp1)
# print(mean)
# std = sigma(temp1)
# print(std)

# print(temp1[0,4,:,:])



import torch.nn.functional as F
import torch.nn as nn
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
       
        a2 = ((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) 
        
        #print(a2[:,:,0,0])
        
        a2 = a2 + self.mu(y)
        a2 = self.sigma(y)*a2
        
        
        
        a2 = a2.permute([2,3,0,1])
        #print(a2.shape)
        #print(a2)
        
        #return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
        
        return a2
ada = AdaIN() 
temp3 = ada(temp1,temp)
print(temp3[0,0,:,:])
print(temp3.shape)


mean = mu(temp3)
print(mean)
std = sigma(temp3)
print(std)

a =temp3[0,:,:,:].numpy()
print(a)



def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

def sigma(x):
    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
    across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
    the permutations are required for broadcasting"""
    
    return torch.sqrt((torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))


temp1 = torch.tensor(temp1)

mean = mu(temp1)
mean = mean.numpy()
print(mean)


# temp = np.zeros([1024,2,2])
# temp[:,0,0] = .1
# temp[:,0,1] = .2
# temp[:,1,0] = .3
# temp[:,1,1] = .4


# temp = np.zeros([1024,2,2])
# temp[:,0,0] = .1
# temp[:,0,1] = .02
# temp[:,1,0] = .03
# temp[:,1,1] = .4

# temp = np.zeros([1024,2,2])
# temp[:,0,0] = 3.8
# temp[:,0,1] = 0
# temp[:,1,0] = 0
# temp[:,1,1] = 0

# temp = np.expand_dims(temp, axis=0)

# temp = torch.tensor(temp)

# std = sigma(temp)
# std = std.numpy()
# print(std)

# mean = mu(temp)
# mean = mean.numpy()
# print(mean)

# .5 --> .2165  and .125 
# 1.8 --> 0.779 and 0.45
# 3.8 --> 1.64  and.95



# def Generate_Meta_(vendors_,scanners_,diseases_): 
#     temp = np.zeros([1024,2,2])
#     if vendors_=='GE MEDICAL SYSTEMS':
#         temp[:,0,0] = 0.5
#         temp[:,0,1] = 0
#         temp[:,1,0] = 0
#         temp[:,1,1] = 0
#     if vendors_=='SIEMENS':
#         temp[:,0,0] = 1.8
#         temp[:,0,1] = 0
#         temp[:,1,0] = 0
#         temp[:,1,1] = 0
#     if vendors_=='Philips Medical Systems':
#         temp[:,0,0] = 3.8
#         temp[:,0,1] = 0
#         temp[:,1,0] = 0
#         temp[:,1,1] = 0
#     return temp









# y_mu = np.zeros([1,1])
# y_mu[0,0] = 5

# print(y_mu)
# data = np.ones([8,8,1,10])

# data = torch.tensor(data)
# y_mu = torch.tensor(y_mu)

# print(y_mu.shape)
# print(data.shape)

# data = data+y_mu
# print(data)

# a1 = data.numpy()






