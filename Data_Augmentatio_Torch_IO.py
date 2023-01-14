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



transforms_1 = tio.OneOf({
        # tio.RandomFlip(): 0.2,
        # tio.RandomElasticDeformation(num_control_points=7,locked_borders=2,): 0.3,
        # tio.RandomAffine(scales=(0.9, 1.2),degrees=15): 0.3,
        #tio.RandomBiasField(): 0.3,
        #tio.RandomBlur(): 0.3,
        #tio.RandomGamma(): 0.3,   
        #tio.RandomAnisotropy():0.3,
        #tio.RandomMotion():0.3,
})

transformed_tensor = transforms_1(sample)

img = transformed_tensor['Image'].data
gt = transformed_tensor['Mask'].data


plt.figure()
plt.imshow(img[0,0,:,:])

plt.figure()
plt.imshow(gt[0,0,:,:])
