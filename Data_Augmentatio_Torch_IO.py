import torch
import torchio as tio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np


def same_depth(img):
    temp = np.zeros([img.shape[0],1,256,256])
    temp[:,0:img.shape[1],:,:] = img
    return temp  

img = nib.load(r'C:\My_Data\M2M Data\data\train\002\002_LA_ES.nii.gz')
gt = nib.load(r'C:\My_Data\M2M Data\data\train\002\002_LA_ES_gt.nii.gz')



img = img.get_fdata()
img = img.transpose(2,1,0)
img = np.expand_dims(img, axis=0)

gt = gt.get_fdata()
gt = gt.transpose(2,1,0)

gt = np.expand_dims(gt, axis=0)

img = same_depth(img)
gt = same_depth(gt)



d = {}
print(img.shape)
print(gt.shape)
d['Image'] = tio.Image(tensor=img, type=tio.INTENSITY)
d['Mask'] = tio.Image(tensor=gt, type=tio.LABEL)
sample = tio.Subject(d)



plt.figure()
plt.imshow(img[0,0,:,:])

plt.figure()
plt.imshow(gt[0,0,:,:])


#transforms_all = tio.Compose({
#       tio.RandomBiasField(): .15,  ## axis [0,1] or [1,2]
#       tio.RandomGhosting(axes=([1,2])): 0.15,
#       tio.RandomFlip(axes=([1,2])): .15,  ## axis [0,1] or [1,2]
#       tio.RandomFlip(axes=([0,1])): .15,  ## axis [0,1] or [1,2]
#       tio.RandomAffine(degrees=(40,0,0)): 0.3, ## for 2D rotation 
#       tio.RandomMotion(degrees =(30) ):0.15 ,
#       tio.RandomBlur(): 0.15,
#       tio.RandomGamma(): 0.15,   
#       tio.RandomNoise(mean=0.1,std=0.1):0.15,
#})

transforms_all = tio.OneOf({
        tio.RandomBiasField(): .3,  ## axis [0,1] or [1,2]
        tio.RandomGhosting(axes=([1,2])): 0.3,
        tio.RandomFlip(axes=([1,2])): .3,  ## axis [0,1] or [1,2]
        tio.RandomFlip(axes=([0,1])): .3,  ## axis [0,1] or [1,2]
        tio.RandomAffine(degrees=(30,0,0)): 0.3, ## for 2D rotation 
        tio.RandomMotion(degrees =(30) ):0.3 ,
        tio.RandomBlur(): 0.3,
        tio.RandomGamma(): 0.3,   
        tio.RandomNoise(mean=0.1,std=0.1):0.20,
})

transformed_tensor = transforms_all(sample)

img = transformed_tensor['Image'].data
gt = transformed_tensor['Mask'].data


plt.figure()
plt.imshow(img[0,0,:,:])

plt.figure()
plt.imshow(gt[0,0,:,:])



import torch
import torchio as tio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np


def same_depth(img):
    temp = np.zeros([img.shape[0],1,256,256])
    temp[:,0:img.shape[1],:,:] = img
    return temp  

img = nib.load(r'C:\My_Data\M2M Data\data\train\002\002_LA_ES.nii.gz')
gt = nib.load(r'C:\My_Data\M2M Data\data\train\002\002_LA_ES_gt.nii.gz')



img = img.get_fdata()
img = img.transpose(2,1,0)
img = np.expand_dims(img, axis=0)

gt = gt.get_fdata()
gt = gt.transpose(2,1,0)

gt = np.expand_dims(gt, axis=0)

img = same_depth(img)
gt = same_depth(gt)



d = {}
print(img.shape)
print(gt.shape)
d['Image'] = tio.Image(tensor=img, type=tio.INTENSITY)
d['Mask'] = tio.Image(tensor=gt, type=tio.LABEL)
sample = tio.Subject(d)



plt.figure()
plt.imshow(img[0,0,:,:])

plt.figure()
plt.imshow(gt[0,0,:,:])


transforms_all = tio.Compose({
        tio.RandomBiasField(): .4,  ## axis [0,1] or [1,2]
        tio.RandomGhosting(axes=([1,2])): 0.3,
        #tio.RandomGhosting(axes=([0,1])): 0.3,
        tio.RandomFlip(axes=([1,2])): .4,  ## axis [0,1] or [1,2]
        tio.RandomFlip(axes=([0,1])): .4,  ## axis [0,1] or [1,2]
        tio.RandomAffine(degrees=(40,0,0)): 0.3, ## for 2D rotation 
        tio.RandomMotion(degrees =(30) ):0.5 ,
        tio.RandomBlur(): 0.3,
        tio.RandomGamma(): 0.3,   
        tio.RandomNoise(mean=10,std=20):0.5,
})



transformed_tensor = transforms_all(sample)

img = transformed_tensor['Image'].data
gt = transformed_tensor['Mask'].data


plt.figure()
plt.imshow(img[0,0,:,:])

plt.figure()
plt.imshow(gt[0,0,:,:])





transforms_spatial = tio.Compose({
        tio.RandomFlip(axes=([1,2])): .4,  ## axis [0,1] or [1,2]
        tio.RandomFlip(axes=([0,1])): .4,  ## axis [0,1] or [1,2]
        #tio.RandomElasticDeformation(num_control_points=(7,7,7),locked_borders=2,image_interpolation ='nearest'): 0.3,
        tio.RandomAffine(degrees=(40,0,0)): 0.3, ## for 2D rotation 
       # tio.RandomAffine(scales=(0.5,0,0.5)): 0.3,  
})



transforms_intensity = tio.Compose({
        tio.RandomMotion(degrees =(30) ):0.5 ,
        tio.RandomBlur(): 0.3,
        tio.RandomGamma(): 0.3,   
        tio.RandomNoise(mean=114,std=190):0.5,
})
transforms_mri_specific = tio.Compose({
        tio.RandomBiasField(): .4,  ## axis [0,1] or [1,2]
        tio.RandomGhosting(axes=([1,2])): 0.3,
        tio.RandomGhosting(axes=([0,1])): 0.3,
})
