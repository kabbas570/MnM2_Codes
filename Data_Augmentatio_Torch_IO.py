import torch
import torchio as tio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np

img = nib.load(r'C:\My_Data\M2M Data\data\train\001\001_LA_ES.nii.gz')
img1 = nib.load(r'C:\My_Data\M2M Data\data\train\002\002_LA_ES_gt.nii.gz')

img = img.get_fdata()
img = np.expand_dims(img, axis=0)
                     
img1 = img1.get_fdata()
img1 = np.expand_dims(img1, axis=0)


d = {}
d['Image'] = tio.Image(tensor=img, type=tio.INTENSITY)
d['Mask'] = tio.Image(tensor=img1, type=tio.LABEL)

sample = tio.Subject(d)




plt.figure()
plt.imshow(img[0,:,:,0])

plt.figure()
plt.imshow(img1[0,:,:,0])


transforms_1 = tio.OneOf({
        tio.RandomFlip(): 0.2,
        tio.RandomElasticDeformation(): 0.2,

})

transformed_tensor = transforms_1(sample)


img = transformed_tensor['Image'].data
img1 = transformed_tensor['Mask'].data


plt.figure()
plt.imshow(img[0,:,:,0])

plt.figure()
plt.imshow(img1[0,:,:,0])
