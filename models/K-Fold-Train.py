import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import torch
import albumentations as A
import cv2
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
   
def Generate_Meta_(vendors_,scanners_,diseases_): 
    temp = np.zeros([1,2,2])
    if vendors_=='GE MEDICAL SYSTEMS':
        temp[:,0,0] = 0.1
        temp[:,0,1] = 0.2
        temp[:,1,0] = 0.3
        temp[:,1,1] = 0.4
    if vendors_=='SIEMENS':
        temp[:,0,0] = 0.1
        temp[:,0,1] = 0.2
        temp[:,1,0] = 0.3
        temp[:,1,1] = 4.0
    if vendors_=='Philips Medical Systems':
        temp[:,0,0] = 0.01
        temp[:,0,1] = 0.02
        temp[:,1,0] = -0.1
        temp[:,1,1] = +0.1
    return temp

def same_depth(img):
    temp = np.zeros([img.shape[0],17,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
    
def LA_to_SA(SA_img,LA_img):
    # Get sizes
    SA_size = (SA_img.GetSize())   ## --> [H,W,C]
    LA_size = (LA_img.GetSize())
    
    # Create a new short axis image the same size as the SA stack.
    new_SA_img = sitk.Image(SA_size, sitk.sitkFloat64)
    
    # Loop over every pixel in the LA image, and put into into the new SA image
    for x in range(0, LA_size[0]):
        for y in range(0, LA_size[1]):
            # Determine the physical location of the LA pixel
            point = LA_img.TransformIndexToPhysicalPoint([x, y, 0])
    
            # Find which index this position maps to in the SA image
            index_SA = SA_img.TransformPhysicalPointToIndex(point)
    
            # Check if the pixel is outside the bounds of the SA image
            if index_SA[0] - 1 < 0 or index_SA[0] + 1 >= SA_img.GetSize()[0]:
                continue
            if index_SA[1] - 1 < 0 or index_SA[1] + 1 >= SA_img.GetSize()[1]:
                continue
            if index_SA[2] - 1 < 0 or index_SA[2] + 1 >= SA_img.GetSize()[2]:
                continue
    
            # Assign the LA pixel to the voxel location in the new SA image
            new_SA_img[index_SA[0], index_SA[1], index_SA[2]] = LA_img[x, y, 0]
    
            # Dilate the intensity (optional)
            new_SA_img[index_SA[0] - 1, index_SA[1], index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1] - 1, index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1], index_SA[2] - 1] = LA_img[x, y, 0]
            new_SA_img[index_SA[0] + 1, index_SA[1], index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1] + 1, index_SA[2]] = LA_img[x, y, 0]
            new_SA_img[index_SA[0], index_SA[1], index_SA[2] + 1] = LA_img[x, y, 0]
    return new_SA_img


def resample_image_SA(itk_image):
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing=(1.25, 1.25, original_spacing[2])
    
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / original_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
    

def resample_image_LA(itk_image):

    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing=(1.25, 1.25, original_spacing[2])
    
    # calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # instantiate resample filter with properties and execute it
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
    
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 
    
def Normalization_2(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def generate_label_1(gt,org_dim3):
        temp_ = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[2,:,:,:][np.where(gt==3)]=1
        temp_[3,:,:,:][np.where(gt==0)]=1
        return temp_
        
def generate_label_2(gt,org_dim3):
    
        temp_ = np.zeros([6,org_dim3,DIM_,DIM_])
    
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[1,:,:,:][np.where(gt==3)]=1
        
        temp_[2,:,:,:][np.where(gt==2)]=1
        temp_[3,:,:,:][np.where(gt==1)]=1
        temp_[3,:,:,:][np.where(gt==3)]=1
        
        temp_[4,:,:,:][np.where(gt==3)]=1
        temp_[5,:,:,:][np.where(gt==2)]=1
        temp_[5,:,:,:][np.where(gt==1)]=1

        return temp_

def generate_label_3(gt,org_dim3):
        temp_ = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[2,:,:,:][np.where(gt==3)]=1
        temp_[3,:,:,:][np.where(gt==0)]=1
        return temp_


def generate_label_4(gt,org_dim3):
        temp_ = np.zeros([1,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[0,:,:,:][np.where(gt==2)]=1
        temp_[0,:,:,:][np.where(gt==3)]=1
        return temp_
    
    
    
transform = A.Compose([
    #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.2),
    A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_NEAREST, always_apply=False, p=0.2),
    A.Rotate(limit=30, always_apply=False, p=0.2),
    A.GaussNoise(always_apply=False, p = 0.2),
    A.ElasticTransform(alpha=1, sigma=9,border_mode=cv2.BORDER_REPLICATE,interpolation=cv2.INTER_LINEAR, always_apply=False, p = 0.2)
])

class Dataset_(Dataset): 
    def __init__(self, df, images_folder,transformations=transform):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        ## sa_es_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        
       ### Augmentation for img_SA ####
        img_SA = np.transpose(img_SA, (1,2,0))  ## to bring channel as last dimenssion 
        img_SA_gt = np.transpose(img_SA_gt, (1,2,0))  ## to bring channel as last dimenssion 
        
        if self.transformations is not None:
            augmentations = self.transformations(image = img_SA, mask = img_SA_gt)
            img_SA = augmentations["image"]
            img_SA_gt = augmentations["mask"]
            img_SA = img_SA.copy()  # this is a workaround to fix the negative stride bug
            img_SA_gt = img_SA_gt.copy()
        img_SA = np.transpose(img_SA, (2,0,1))  ## to bring channel first 
        img_SA_gt = np.transpose(img_SA_gt, (2,0,1))  ## to bring channel first 

        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)

        temp_SA_ES = generate_label_3(img_SA_gt,org_dim3)        
        temp_SA_ES = same_depth(temp_SA_ES)
        
   
        #####    LA Images #####
        ## la_es_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ES.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ES = Normalization_1(img_LA_ES)

        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        temp_LA_ES = generate_label_3(img_LA_gt,org_dim3)
        temp_LA_ES = temp_LA_ES[:,0,:,:]
        
        all_three_ES = generate_label_4(img_LA_gt,org_dim3)
        all_three_ES = all_three_ES[:,0,:,:]
        
        
        img_LA_ES = np.transpose(img_LA_ES, (1,2,0))  ## to bring channel as last dimenssion 
        temp_LA_ES = np.transpose(temp_LA_ES, (1,2,0))  ## to bring channel as last dimenssion 
        if self.transformations is not None:
            augmentations = self.transformations(image = img_LA_ES, mask = temp_LA_ES)
            img_LA_ES = augmentations["image"]
            temp_LA_ES = augmentations["mask"]
            img_LA_ES = img_LA_ES.copy()  # this is a workaround to fix the negative stride bug
            temp_LA_ES = temp_LA_ES.copy()
        img_LA_ES = np.transpose(img_LA_ES, (2,0,1))  ## to bring channel first 
        temp_LA_ES = np.transpose(temp_LA_ES, (2,0,1))  ## to bring channel first 
        
        
        ## ED images ##
        ## sa_eD_img ####
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        
        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        
        ### Augmentation for img_SA ####
        img_SA = np.transpose(img_SA, (1,2,0))  ## to bring channel as last dimenssion 
        img_SA_gt = np.transpose(img_SA_gt, (1,2,0))  ## to bring channel as last dimenssion 
         
        if self.transformations is not None:
             augmentations = self.transformations(image = img_SA, mask = img_SA_gt)
             img_SA = augmentations["image"]
             img_SA_gt = augmentations["mask"]
             img_SA = img_SA.copy()  # this is a workaround to fix the negative stride bug
             img_SA_gt = img_SA_gt.copy()
        img_SA = np.transpose(img_SA, (2,0,1))  ## to bring channel first 
        img_SA_gt = np.transpose(img_SA_gt, (2,0,1))  ## to bring channel first 
         
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)

        temp_SA_ED = generate_label_3(img_SA_gt,org_dim3)        
        temp_SA_ED = same_depth(temp_SA_ED)

        #####    LA Images #####
        ## la_ed_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ED.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ED = Normalization_1(img_LA_ED)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        temp_LA_ED = generate_label_3(img_LA_gt,org_dim3)
        temp_LA_ED = temp_LA_ED[:,0,:,:]
        
        all_three_ED = generate_label_4(img_LA_gt,org_dim3)
        all_three_ED = all_three_ED[:,0,:,:]
        

        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        
        M = Generate_Meta_(vendors_,scanners_,diseases_)
        
        img_LA_ED = np.transpose(img_LA_ED, (1,2,0))  ## to bring channel as last dimenssion 
        temp_LA_ED = np.transpose(temp_LA_ED, (1,2,0))  ## to bring channel as last dimenssion 
        if self.transformations is not None:
            augmentations = self.transformations(image = img_LA_ED, mask = temp_LA_ED)
            img_LA_ED = augmentations["image"]
            temp_LA_ED = augmentations["mask"]
            img_LA_ED = img_LA_ED.copy()  # this is a workaround to fix the negative stride bug
            temp_LA_ED = temp_LA_ED.copy()
        img_LA_ED = np.transpose(img_LA_ED, (2,0,1))  ## to bring channel first 
        temp_LA_ED = np.transpose(temp_LA_ED, (2,0,1))  ## to bring channel first 
        return img_LA_ES,temp_LA_ES[:,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],M,all_three_ES,all_three_ED

def Data_Loader_(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_V(Dataset): 
    def __init__(self, df, images_folder,transformations=None):
        self.df = df
        self.images_folder = images_folder
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases=df['DISEASE']
        self.fields=df['FIELD']        
        self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        ## sa_es_img ####
        img_SA_path = img_path+'_SA_ES.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        
       ### Augmentation for img_SA ####
        img_SA = np.transpose(img_SA, (1,2,0))  ## to bring channel as last dimenssion 
        img_SA_gt = np.transpose(img_SA_gt, (1,2,0))  ## to bring channel as last dimenssion 
        
        if self.transformations is not None:
            augmentations = self.transformations(image = img_SA, mask = img_SA_gt)
            img_SA = augmentations["image"]
            img_SA_gt = augmentations["mask"]
            img_SA = img_SA.copy()  # this is a workaround to fix the negative stride bug
            img_SA_gt = img_SA_gt.copy()
        img_SA = np.transpose(img_SA, (2,0,1))  ## to bring channel first 
        img_SA_gt = np.transpose(img_SA_gt, (2,0,1))  ## to bring channel first 

        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)

        temp_SA_ES = generate_label_3(img_SA_gt,org_dim3)        
        temp_SA_ES = same_depth(temp_SA_ES)
        
   
        #####    LA Images #####
        ## la_es_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ES.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ES = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ES = Normalization_1(img_LA_ES)

        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        temp_LA_ES = generate_label_3(img_LA_gt,org_dim3)
        temp_LA_ES = temp_LA_ES[:,0,:,:]
        
        all_three_ES = generate_label_4(img_LA_gt,org_dim3)
        all_three_ES = all_three_ES[:,0,:,:]
        
        
        img_LA_ES = np.transpose(img_LA_ES, (1,2,0))  ## to bring channel as last dimenssion 
        temp_LA_ES = np.transpose(temp_LA_ES, (1,2,0))  ## to bring channel as last dimenssion 
        if self.transformations is not None:
            augmentations = self.transformations(image = img_LA_ES, mask = temp_LA_ES)
            img_LA_ES = augmentations["image"]
            temp_LA_ES = augmentations["mask"]
            img_LA_ES = img_LA_ES.copy()  # this is a workaround to fix the negative stride bug
            temp_LA_ES = temp_LA_ES.copy()
        img_LA_ES = np.transpose(img_LA_ES, (2,0,1))  ## to bring channel first 
        temp_LA_ES = np.transpose(temp_LA_ES, (2,0,1))  ## to bring channel first 
        
        
        ## ED images ##
        ## sa_eD_img ####
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_1(img_SA)
        
        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt)  
        
        ### Augmentation for img_SA ####
        img_SA = np.transpose(img_SA, (1,2,0))  ## to bring channel as last dimenssion 
        img_SA_gt = np.transpose(img_SA_gt, (1,2,0))  ## to bring channel as last dimenssion 
         
        if self.transformations is not None:
             augmentations = self.transformations(image = img_SA, mask = img_SA_gt)
             img_SA = augmentations["image"]
             img_SA_gt = augmentations["mask"]
             img_SA = img_SA.copy()  # this is a workaround to fix the negative stride bug
             img_SA_gt = img_SA_gt.copy()
        img_SA = np.transpose(img_SA, (2,0,1))  ## to bring channel first 
        img_SA_gt = np.transpose(img_SA_gt, (2,0,1))  ## to bring channel first 
         
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)

        temp_SA_ED = generate_label_3(img_SA_gt,org_dim3)        
        temp_SA_ED = same_depth(temp_SA_ED)

        #####    LA Images #####
        ## la_ed_img ####
        img_path = os.path.join(self.images_folder, str(self.images_name[index]).zfill(3),str(self.images_name[index]).zfill(3))
        img_LA_path=img_path+'_LA_ED.nii.gz'
        img_LA = sitk.ReadImage(img_LA_path)
        img_LA = resample_image_LA(img_LA)
        img_LA = sitk.GetArrayFromImage(img_LA)
        org_dim3 = img_LA.shape[0]
        org_dim1 = img_LA.shape[1]
        org_dim2 = img_LA.shape[2] 
        img_LA_ED = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
        img_LA_ED = Normalization_1(img_LA_ED)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt)  
        temp_LA_ED = generate_label_3(img_LA_gt,org_dim3)
        temp_LA_ED = temp_LA_ED[:,0,:,:]
        
        all_three_ED = generate_label_4(img_LA_gt,org_dim3)
        all_three_ED = all_three_ED[:,0,:,:]
        

        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        
        M = Generate_Meta_(vendors_,scanners_,diseases_)
        
        img_LA_ED = np.transpose(img_LA_ED, (1,2,0))  ## to bring channel as last dimenssion 
        temp_LA_ED = np.transpose(temp_LA_ED, (1,2,0))  ## to bring channel as last dimenssion 
        if self.transformations is not None:
            augmentations = self.transformations(image = img_LA_ED, mask = temp_LA_ED)
            img_LA_ED = augmentations["image"]
            temp_LA_ED = augmentations["mask"]
            img_LA_ED = img_LA_ED.copy()  # this is a workaround to fix the negative stride bug
            temp_LA_ED = temp_LA_ED.copy()
        img_LA_ED = np.transpose(img_LA_ED, (2,0,1))  ## to bring channel first 
        temp_LA_ED = np.transpose(temp_LA_ED, (2,0,1))  ## to bring channel first 
        return img_LA_ES,temp_LA_ES[:,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],M,all_three_ES,all_three_ED

def Data_Loader_V(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader
# val_imgs= r'C:\My_Data\M2M Data\data\val'
# val_csv_path= r'C:\My_Data\M2M Data\data\val.csv'
# df_val = pd.read_csv(val_csv_path)
# train_loader_ED = Data_Loader_(df_val,val_imgs,batch_size=1)
# a = iter(train_loader_ED)
# a1 =next(a)
# plt.figure()
# plt.imshow(a1[10][0,0,:,:])
# print(a1[9])
# img_path = r'C:\My_Data\M2M Data\data\val\163\163'
# img_SA_path = img_path+'_SA_ES_gt.nii.gz'
# img_LA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
# img_LA = resample_image_SA(img_LA )      ## --> [H,W,C]
# img_LA = sitk.GetArrayFromImage(img_LA)   ## --> [C,H,W]
# org_dim3 = img_LA.shape[0]
# org_dim1 = img_LA.shape[1]
# org_dim2 = img_LA.shape[2] 
# img_LA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA)
# #img_LA = Normalization_1(img_LA)
# #img_LA = Normalization_2(img_LA)
# plt.figure()
# plt.imshow(img_LA[5,:,:])





   #### Specify all the paths here #####
   
# train_imgs='/data/scratch/acw676/MnM/data_2/train/'
# val_imgs='/data/scratch/acw676/MnM/data_2/val/'
# #test_imgs='/data/scratch/acw676/MnM/test/'

# train_csv_path='/data/scratch/acw676/MnM/train.csv'
# val_csv_path='/data/scratch/acw676/MnM/val.csv'
# #test_csv_path='/data/scratch/acw676/MnM/test.csv'


### 3 - this function will save the check-points 
    
        #### Specify all the Hyperparameters\image dimenssions here #####

batch_size = 4
Max_Epochs=100
LEARNING_RATE=0.0001
Patience = 5

        #### Import All libraies used for training  #####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
#import pandas as pd
from Early_Stopping import EarlyStopping
            ### Data_Generators ########
            
            #### The first one will agument and Normalize data and used for training ###
            #### The second will not apply Data augmentaations and only prcocess the data during validation ###



# train_loader = Data_Loader_V(df_train,train_imgs,batch_size)

# val_loader = Data_Loader_V(df_val,val_imgs,batch_size)
#    ### Load the Data using Data generators and paths specified #####
#    #######################################
   
# print(len(train_loader)) ### this shoud be = Total_images/ batch size
# print(len(val_loader))   ### same here
# #print(len(test_loader))   ### same here

### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses1 = []   # losses of all training epochs
avg_valid_losses1 = []  #losses of all training epochs
avg_valid_DS1 = []  # all training epochs

### Next we have all the funcitons which will be called in the main for training ####

Actual_ = 0.5
Not_ = 0.5  

LA_ = 0.5
SA_ = 0.5

RV_ = 0.4
LV_ = 0.3
MYO_ = 0.3 

def generate_pred_label_1(gt):
        gt = torch.argmax(gt,dim =1)    
        temp_ = torch.zeros([gt.shape[0],4,DIM_,DIM_])
        temp_ = temp_.to(device=DEVICE,dtype=torch.float) 
        temp_[:,0,:,:][torch.where(gt==0)]=1
        temp_[:,1,:,:][torch.where(gt==1)]=1
        temp_[:,2,:,:][torch.where(gt==2)]=1
        temp_[:,3,:,:][torch.where(gt==3)]=1
        return temp_
    
### 2- the main training fucntion to update the weights....
def train_fn(loader_train1,loader_valid1,model1, optimizer1,loss_fn1, scaler):  ### Loader_1--> ED and Loader2-->ES
    train_losses1 = [] # loss of each batch
    valid_losses1 = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.train()
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED) in enumerate(loop):
        
        img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)  
        temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
        img_SA_ES = img_SA_ES.to(device=DEVICE,dtype=torch.float)  
        temp_SA_ES = temp_SA_ES.to(device=DEVICE,dtype=torch.float)
        
        img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)  
        temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
        img_SA_ED = img_SA_ED.to(device=DEVICE,dtype=torch.float)  
        temp_SA_ED = temp_SA_ED.to(device=DEVICE,dtype=torch.float)

        M = M.to(device=DEVICE,dtype=torch.float)
        all_three_ES = all_three_ES.to(device=DEVICE,dtype=torch.float)  
        all_three_ED = all_three_ED.to(device=DEVICE,dtype=torch.float)

        with torch.cuda.amp.autocast():
            ou1_ES,out2_ES = model1(img_LA_ES,M)  
            ou1_ED,out2_ED = model1(img_LA_ED,M)  
            
            loss1 = loss_fn1(ou1_ES,all_three_ES)
            loss2 = loss_fn1(ou1_ED,all_three_ED)
            loss3 = loss_fn1(out2_ES,temp_LA_ES)
            loss4 = loss_fn1(out2_ED,temp_LA_ED)
            
            
        # backward
        loss = (loss1 + loss2+loss3 + loss4)/4
        optimizer1.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer1)

        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses1.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED) in enumerate(loop_v):
        
        img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)  
        temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
        img_SA_ES = img_SA_ES.to(device=DEVICE,dtype=torch.float)  
        temp_SA_ES = temp_SA_ES.to(device=DEVICE,dtype=torch.float)
        
        img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)  
        temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
        img_SA_ED = img_SA_ED.to(device=DEVICE,dtype=torch.float)  
        temp_SA_ED = temp_SA_ED.to(device=DEVICE,dtype=torch.float)

        M = M.to(device=DEVICE,dtype=torch.float)
        all_three_ES = all_three_ES.to(device=DEVICE,dtype=torch.float)  
        all_three_ED = all_three_ED.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 
            
            ou1_ES,out2_ES = model1(img_LA_ES,M)  
            ou1_ED,out2_ED = model1(img_LA_ED,M)  
            
            loss1 = loss_fn1(ou1_ES,all_three_ES)
            loss2 = loss_fn1(ou1_ED,all_three_ED)
            loss3 = loss_fn1(out2_ES,temp_LA_ES)
            loss4 = loss_fn1(out2_ED,temp_LA_ED)
            
            
        # backward
        loss = (loss1 + loss2+loss3 + loss4)/4
        loop_v.set_postfix(loss = loss.item())
        valid_losses1.append(float(loss))

    train_loss_per_epoch1 = np.average(train_losses1)
    valid_loss_per_epoch1 = np.average(valid_losses1)
    ## all epochs
    avg_train_losses1.append(train_loss_per_epoch1)
    avg_valid_losses1.append(valid_loss_per_epoch1)
    
    return train_loss_per_epoch1, valid_loss_per_epoch1


    ### 4 - It will check the Dice-Score on each epoch for validation data 
def check_Dice_Score(loader, model1, device=DEVICE):
    Dice_score_LA_RV_ES = 0
    Dice_score_LA_MYO_ES = 0
    Dice_score_LA_LV_ES = 0
    
    Dice_score_LA_RV_ED = 0
    Dice_score_LA_MYO_ED = 0
    Dice_score_LA_LV_ED = 0
    
    Three_ES = 0
    Three_ED = 0

    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED) in enumerate(loop):
        
        img_LA_ES = img_LA_ES.to(device=DEVICE,dtype=torch.float)  
        temp_LA_ES = temp_LA_ES.to(device=DEVICE,dtype=torch.float)
        img_SA_ES = img_SA_ES.to(device=DEVICE,dtype=torch.float)  
        temp_SA_ES = temp_SA_ES.to(device=DEVICE,dtype=torch.float)
        
        img_LA_ED = img_LA_ED.to(device=DEVICE,dtype=torch.float)  
        temp_LA_ED = temp_LA_ED.to(device=DEVICE,dtype=torch.float)
        img_SA_ED = img_SA_ED.to(device=DEVICE,dtype=torch.float)  
        temp_SA_ED = temp_SA_ED.to(device=DEVICE,dtype=torch.float)

        M = M.to(device=DEVICE,dtype=torch.float)
        all_three_ES = all_three_ES.to(device=DEVICE,dtype=torch.float)  
        all_three_ED = all_three_ED.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 

            ou1_ES,out2_ES = model1(img_LA_ES,M)  
            ou1_ED,out2_ED = model1(img_LA_ED,M)  
                        
            #out_LA_ES = generate_pred_label_1(out_LA_ES) # --- [B,C,H,W]
            out_LA_ES = (out2_ES > 0.5)* 1
            
            out_LA_ES_LV = out_LA_ES[:,0:1,:,:]
            out_LA_ES_MYO = out_LA_ES[:,1:2,:,:]
            out_LA_ES_RV = out_LA_ES[:,2:3,:,:]

            #out_LA_ED = generate_pred_label_1(out_LA_ED) # --- [B,C,H,W]
            out_LA_ED = (out2_ED > 0.5)* 1
            
            out_LA_ED_LV = out_LA_ED[:,0:1,:,:]
            out_LA_ED_MYO = out_LA_ED[:,1:2,:,:]
            out_LA_ED_RV = out_LA_ED[:,2:3,:,:]
            
            
            ou1_ES = (ou1_ES > 0.5)* 1
            ou1_ED = (ou1_ED > 0.5)* 1
            
            
            ## Dice Score for ES ###

            single_Three_ES = (2 * (ou1_ES * all_three_ES).sum()) / (
                (ou1_ES + all_three_ES).sum() + 1e-8)
            Three_ES +=single_Three_ES

            ## Dice Score for ED ###
            single_Three_ED = (2 * (ou1_ED * all_three_ED).sum()) / (
                (ou1_ED + all_three_ED).sum() + 1e-8)
            Three_ED +=single_Three_ED
            
            
            ## Dice Score for ES ###

            single_LA_LV_ES = (2 * (out_LA_ES_LV * temp_LA_ES[:,0:1,:,:]).sum()) / (
                (out_LA_ES_LV + temp_LA_ES[:,0:1,:,:]).sum() + 1e-8)
            
            Dice_score_LA_LV_ES +=single_LA_LV_ES
            
            single_LA_MYO_ES = (2 * (out_LA_ES_MYO*temp_LA_ES[:,1:2,:,:]).sum()) / (
    (out_LA_ES_MYO + temp_LA_ES[:,1:2,:,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ES += single_LA_MYO_ES

            single_LA_RV_ES = (2 * (out_LA_ES_RV* temp_LA_ES[:,2:3,:,:]).sum()) / (
        (out_LA_ES_RV + temp_LA_ES[:,2:3,:,:]).sum() + 1e-8)
            Dice_score_LA_RV_ES += single_LA_RV_ES
        

            # print(f"single_ID_  : {single_ID_}")
            # print(f"single_LA_LV_ES  : {single_LA_LV_ES}")
            # print(f"single_LA_MYO_ES  : {single_LA_MYO_ES}")
            # print(f"single_LA_RV_ES  : {single_LA_RV_ES}")
            # print("                          ")
            
            ## Dice Score for ED ###

            single_LA_LV_ED = (2 * (out_LA_ED_LV * temp_LA_ED[:,0:1,:,:]).sum()) / (
                (out_LA_ED_LV + temp_LA_ED[:,0:1,:,:]).sum() + 1e-8)
            
            Dice_score_LA_LV_ED +=single_LA_LV_ED
            
            single_LA_MYO_ED = (2 * (out_LA_ED_MYO*temp_LA_ED[:,1:2,:,:]).sum()) / (
    (out_LA_ED_MYO + temp_LA_ED[:,1:2,:,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ED += single_LA_MYO_ED

            single_LA_RV_ED = (2 * (out_LA_ED_RV* temp_LA_ED[:,2:3,:,:]).sum()) / (
        (out_LA_ED_RV + temp_LA_ED[:,2:3,:,:]).sum() + 1e-8)
            Dice_score_LA_RV_ED += single_LA_RV_ED
        

            # print(f"single_ID_  : {single_ID_}")
            # print(f"single_LA_LV_ED  : {single_LA_LV_ED}")
            # print(f"single_LA_MYO_ED  : {single_LA_MYO_ED}")
            # print(f"single_LA_RV_ED  : {single_LA_RV_ED}")
            # print("                          ")
     
    Dice_RV = (Dice_score_LA_RV_ES + Dice_score_LA_RV_ED)/2
    Dice_MYO = (Dice_score_LA_MYO_ES + Dice_score_LA_MYO_ED)/2
    Dice_LV = (Dice_score_LA_LV_ES + Dice_score_LA_LV_ED)/2
    Three_Overall = (Three_ES + Three_ED)/2
    
    print(f"Dice_RV  : {Dice_RV/len(loader)}")
    print(f"Dice_MYO  : {Dice_MYO/len(loader)}")
    print(f"Dice_LV  : {Dice_LV/len(loader)}")
    print(f"Three_Overall  : {Three_Overall/len(loader)}")

    Overall_Dicescore =( Dice_RV + Dice_MYO + Dice_LV + Three_Overall) /4

    return Overall_Dicescore/len(loader)
    
### 6 - This is Focal Tversky Loss loss function ### 

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))
           
        
## 7- This is the main Training function, where we will call all previous functions
       
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=Patience, verbose=True)


from unets import UNet5_4
model_1 = UNet5_4()

#from unets import UNet4
#model_1 = UNet4()

train_imgs='/data/scratch/acw676/MnM/data_2/val/'
train_csv_path='/data/scratch/acw676/MnM/val.csv'
df_train = pd.read_csv(train_csv_path)

k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

all_data = Dataset_V(train_csv_path,train_imgs)

path_to_save_Learning_Curve='/data/home/acw676/MM/weights/'+'/5_fOLD'

def save_checkpoint(state,fold):
    print("=> Saving checkpoint")
    path_to_save_check_points = '/data/home/acw676/MM/weights/'+'/fold_number_'+str(fold)
    filename = path_to_save_check_points + "_.pth.tar"
    torch.save(state, filename)
    

def main():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
            ## Fine Tunnning Part ###
#    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
#    weights_paths= "/data/home/acw676/MM/weights/a_3_three.pth.tar"
#    checkpoint = torch.load(weights_paths,map_location=DEVICE)
#    model.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])

    loss_fn1 = DiceLoss()
    optimizer1 = optim.AdamW(model1.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    #optimizer1 = optim.SGD(model1.parameters(),momentum=0.99,lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(all_data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        
        train_loader = DataLoader(all_data, batch_size=4, sampler=train_subsampler)
        valid_loader = DataLoader(all_data, batch_size=4, sampler=valid_subsampler)
        
        print(len(train_loader))
        print(len(valid_loader))
    
        for epoch in range(Max_Epochs):
            train_loss,valid_loss=train_fn(train_loader,valid_loader, model1, optimizer1, loss_fn1,scaler)
            
            print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')
            
            print(print_msg)
    
            dice_score = check_Dice_Score(valid_loader, model1, device=DEVICE)
            
            
            avg_valid_DS1.append(dice_score.detach().cpu().numpy())
            
            early_stopping(valid_loss, dice_score)
            if early_stopping.early_stop:
                print("Early stopping Reached at  :",epoch)
                
                ### save model    ######
                checkpoint = {
                    "state_dict": model1.state_dict(),
                    "optimizer":optimizer1.state_dict(),
                }
                save_checkpoint(checkpoint,fold)
                
                ### This part of the code will generate the learning curve ......

                # visualize the loss as the network trained
                fig = plt.figure(figsize=(10,8))
                plt.plot(range(1,len(avg_train_losses1)+1),avg_train_losses1, label='Training Loss')
                plt.plot(range(1,len(avg_valid_losses1)+1),avg_valid_losses1,label='Validation Loss')
                plt.plot(range(1,len(avg_valid_DS1)+1),avg_valid_DS1,label='Validation DS')

                # find position of lowest validation loss
                minposs = avg_valid_losses1.index(min(avg_valid_losses1))+1 
                plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

                font1 = {'size':20}

                plt.title("Learning Curve Graph",fontdict = font1)
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.ylim(0, 1) # consistent scale
                plt.xlim(0, len(avg_train_losses1)+1) # consistent scale
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
                fig.savefig(path_to_save_Learning_Curve+'lC_Fold_Num_'+str(fold)+'_.png', bbox_inches='tight')
                
                break

if __name__ == "__main__":
    main()

