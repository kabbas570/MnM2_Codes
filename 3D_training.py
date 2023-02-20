import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
#from typing import List, Union, Tuple
import torch
import albumentations as A
import cv2
from torch.utils.data import SubsetRandomSampler
import torchio as tio
from sklearn.model_selection import KFold
           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
   

def Generate_Meta_(vendors_,scanners_,diseases_): 
    temp = np.zeros([2,17,8,8])
    if vendors_=='GE MEDICAL SYSTEMS': 
        temp[0,:,0:4,0:4] = 0.1
        temp[0,:,4:9,4:9] = 0.2
        temp[0,:,0:4,4:9] = 0.3
        temp[0,:,4:9,0:4] = 0.4
    if vendors_=='SIEMENS':
        temp[0,:,0:4,0:4] = 0.7
        temp[0,:,4:9,4:9] = 0.3
        temp[0,:,0:4,4:9] = 0.1
        temp[0,:,4:9,0:4] = 0.5
    if vendors_=='Philips Medical Systems':
        temp[0,:,0:4,0:4] = 0.8
        temp[0,:,4:9,4:9] = 0.6
        temp[0,:,0:4,4:9] = 0.9
        temp[0,:,4:9,0:4] = 0.1
    if scanners_=='Symphony':
        temp[1,:,0:4,0:4] = 0
        temp[1,:,4:9,0:4] = 0
        temp[1,:,0:4,4:9] = 1.6
        temp[1,:,5:9,5:9] = 0.9
    if scanners_=='SIGNA EXCITE':
        temp[1,:,0:3,0:8] = -0.1
        temp[1,:,3:8,0:2] = .3
        temp[1,:,6:8,6:8] = 1.9
    if scanners_=='Signa Explorer':
        temp[1,:,1:8,1:5] = 1.1
        temp[1,:,0:8,5:8] = .8
    if scanners_=='SymphonyTim':
        temp[1,:,0:3,:] = 1.6
        temp[1,:,5:8,:] = 1.1
    if scanners_=='Avanto Fit':
        temp[1,:,:,0:3] = -0.8
        temp[1,:,5:8] = 0.9
    if scanners_=='Avanto':
        temp[1,:,0:2,:] = -0.9
        temp[1,:,:,0:2] = 1.8
        temp[1,:,:,6:8] = -0.9
        temp[1,:,6:8,2:6] = 1.2
    if scanners_=='Achieva':
        temp[1,:,0:4,0:4] = 0.8
        temp[1,:,4:9,4:9] = 0.6
    if scanners_=='Signa HDxt':
        temp[1,:,0:4,4:9] = 0.2
        temp[1,:,4:9,0:4] = 0.4
    if scanners_=='TrioTim':
        temp[1,:,0:4,0:4] = 1.2
        temp[1,:,4:9,0:4] = 1.8
    
    return temp
           
def same_depth(img):
    temp = np.zeros([img.shape[0],17,DIM_,DIM_])
    temp[:,0:img.shape[1],:,:] = img
    return temp  
    
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

transforms_all = tio.OneOf({
        tio.RandomBiasField(): .3,  ## axis [0,1] or [1,2]
        tio.RandomGhosting(axes=([1,2])): 0.3,
        #tio.RandomFlip(axes=([1,2])): .3,  ## axis [0,1] or [1,2]
        #tio.RandomFlip(axes=([0,1])): .3,  ## axis [0,1] or [1,2]
        #tio.RandomAffine(degrees=(30,0,0)): 0.3, ## for 2D rotation 
        #tio.RandomMotion(degrees =(30) ):0.3 ,
        tio.RandomBlur(): 0.3,
        tio.RandomGamma(): 0.3,   
        #tio.RandomNoise(mean=0.1,std=0.1):0.20,
})

def Normalization_LA_ES(img):
        img = (img-114.8071)/191.2891
        return img 
def Normalization_LA_ED(img):
        img = (img-114.7321)/189.8573
        return img 
    
#def Normalization_SA_ES(img):
#        img = (img-92.6559)/170.3444
#        return img 
#def Normalization_SA_ED(img):
#        img = (img-93.2379)/170.5417
#        return img 
    
def Normalization_SA_ES(img):
        img = (img-62.5983)/147.4826
        return img 
def Normalization_SA_ED(img):
        img = (img-62.9529)/147.6579
        return img 
    
class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=None):  ## If I apply Data Augmentation here, the validation loss becomes None. 
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
        img_SA = Normalization_SA_ES(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ES = same_depth(img_SA_gt)
       ### Augmentation for img_SA ####
        
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ES, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ES = transformed_tensor['Image'].data
            temp_SA_ES = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ES[0,:]  ## expand dim is removed here
        temp_SA_ES = generate_label_3(temp_SA_gt,17) 
        
        all_three_ES_SA = generate_label_4(temp_SA_gt,17)
        

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
        img_LA_ES = Normalization_LA_ES(img_LA_ES)
        img_LA_ES = np.expand_dims(img_LA_ES, axis=0)
        ## la_es_gt ####
        
        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
                
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ES = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        

        img_LA_ES = img_LA_ES[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ES = generate_label_3(img_LA_gt,1)
        temp_LA_ES = temp_LA_ES[:,0,:,:]
        
        all_three_ES = generate_label_4(img_LA_gt,1)
        all_three_ES = all_three_ES[:,0,:,:]
        
        ## ED images ##
        ## sa_ED_img ####
        
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ED(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)
        

        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ED = same_depth(img_SA_gt)
        
        ### Augmentation for img_SA ####
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ED, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ED = transformed_tensor['Image'].data
            temp_SA_ED = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ED[0,:]  ## expand im is removed here
        temp_SA_ED = generate_label_3(temp_SA_gt,17) 
        
        all_three_ED_SA = generate_label_4(temp_SA_gt,17)
        
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
        img_LA_ED = Normalization_LA_ED(img_LA_ED)
        img_LA_ED = np.expand_dims(img_LA_ED, axis=0)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
        
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ED = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        
        img_LA_ED = img_LA_ED[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ED = generate_label_3(img_LA_gt,1)
        temp_LA_ED = temp_LA_ED[:,0,:,:]
        
        all_three_ED = generate_label_4(img_LA_gt,1)
        all_three_ED = all_three_ED[:,0,:,:]
        
        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        M = Generate_Meta_(vendors_,scanners_,diseases_)
        return img_LA_ES,temp_LA_ES[:,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],M,all_three_ES,all_three_ED,all_three_ES_SA,all_three_ED_SA

def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
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
        img_SA = Normalization_SA_ES(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ES = same_depth(img_SA)
        
        ## sa_es_gt ####
        img_SA_gt_path = img_path+'_SA_ES_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ES = same_depth(img_SA_gt)
       ### Augmentation for img_SA ####
        
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ES, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ES = transformed_tensor['Image'].data
            temp_SA_ES = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ES[0,:]  ## expand dim is removed here
        temp_SA_ES = generate_label_3(temp_SA_gt,17) 
        
        all_three_ES_SA = generate_label_4(temp_SA_gt,17)
        

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
        img_LA_ES = Normalization_LA_ES(img_LA_ES)
        img_LA_ES = np.expand_dims(img_LA_ES, axis=0)
        ## la_es_gt ####
        
        img_LA_gt_path = img_path+'_LA_ES_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
                
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ES, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ES = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        

        img_LA_ES = img_LA_ES[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ES = generate_label_3(img_LA_gt,1)
        temp_LA_ES = temp_LA_ES[:,0,:,:]
        
        all_three_ES = generate_label_4(img_LA_gt,1)
        all_three_ES = all_three_ES[:,0,:,:]
        
        ## ED images ##
        ## sa_ED_img ####
        
        img_SA_path = img_path+'_SA_ED.nii.gz'
        img_SA = sitk.ReadImage(img_SA_path)    ## --> [H,W,C]
        img_SA = resample_image_SA(img_SA )      ## --> [H,W,C]
        img_SA = sitk.GetArrayFromImage(img_SA)   ## --> [C,H,W]
        org_dim3 = img_SA.shape[0]
        org_dim1 = img_SA.shape[1]
        org_dim2 = img_SA.shape[2] 
        img_SA = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA)
        img_SA = Normalization_SA_ED(img_SA)
        # img_SA = Normalization_1(img_SA)
        img_SA = np.expand_dims(img_SA, axis=0)
        img_SA_ED = same_depth(img_SA)
        

        ## sa_ed_gt ####
        img_SA_gt_path = img_path+'_SA_ED_gt.nii.gz'
        img_SA_gt = sitk.ReadImage(img_SA_gt_path)
        img_SA_gt = resample_image_SA(img_SA_gt)
        img_SA_gt = sitk.GetArrayFromImage(img_SA_gt)   ## --> [C,H,W]
        img_SA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_SA_gt) 
        img_SA_gt = np.expand_dims(img_SA_gt, axis=0)
        temp_SA_ED = same_depth(img_SA_gt)
        
        ### Augmentation for img_SA ####
        d = {}
        d['Image'] = tio.Image(tensor = img_SA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = temp_SA_ED, type=tio.LABEL)
        sample = tio.Subject(d)


        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_SA_ED = transformed_tensor['Image'].data
            temp_SA_ED = transformed_tensor['Mask'].data
        
        temp_SA_gt = temp_SA_ED[0,:]  ## expand im is removed here
        temp_SA_ED = generate_label_3(temp_SA_gt,17) 
        
        all_three_ED_SA = generate_label_4(temp_SA_gt,17)
        
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
        img_LA_ED = Normalization_LA_ED(img_LA_ED)
        img_LA_ED = np.expand_dims(img_LA_ED, axis=0)

        img_LA_gt_path = img_path+'_LA_ED_gt.nii.gz'
        img_LA_gt = sitk.ReadImage(img_LA_gt_path)
        img_LA_gt = resample_image_LA(img_LA_gt)
        img_LA_gt = sitk.GetArrayFromImage(img_LA_gt)
        img_LA_gt = Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_LA_gt) 
        img_LA_gt = np.expand_dims(img_LA_gt, axis=0)
        
        ### Augmentation for img_LA #### 
        
        d = {}
        d['Image'] = tio.Image(tensor = img_LA_ED, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = img_LA_gt, type=tio.LABEL)
        sample = tio.Subject(d)
        
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img_LA_ED = transformed_tensor['Image'].data
            img_LA_gt = transformed_tensor['Mask'].data
        
        img_LA_ED = img_LA_ED[0,:]
        img_LA_gt = img_LA_gt[0,:]
        
        temp_LA_ED = generate_label_3(img_LA_gt,1)
        temp_LA_ED = temp_LA_ED[:,0,:,:]
        
        all_three_ED = generate_label_4(img_LA_gt,1)
        all_three_ED = all_three_ED[:,0,:,:]
        
        ## meta data ##
        vendors_ = self.vendors[index]
        scanners_ = self.scanners[index]
        diseases_ = self.diseases[index]
        M = Generate_Meta_(vendors_,scanners_,diseases_)
        return img_LA_ES,temp_LA_ES[:,:,:],img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED[:,:,:],img_SA_ED,temp_SA_ED,self.images_name[index],M,all_three_ES,all_three_ED,all_three_ES_SA,all_three_ED_SA

def Data_Loader_V(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_V(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# val_imgs = r'C:\My_Data\M2M Data\data\data_2/val'
# val_csv_path = r'C:\My_Data\M2M Data\data\val.csv'
# df_val = pd.read_csv(val_csv_path)
# train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
# #train_loader = Data_Loader_V(df_val,val_imgs,batch_size = 1)
# a = iter(train_loader)
# a1 =next(a)
# gt1=a1[7][0,2,4,:,:]
# plt.figure()
# plt.imshow(gt1)
# gt2=a1[13][0,0,4,:,:]
# plt.figure()
# plt.imshow(gt2)



   #### Specify all the paths here #####
   
train_imgs='/data/scratch/acw676/MnM/data_2/train/'
val_imgs='/data/scratch/acw676/MnM/data_2/val/'
#test_imgs='/data/scratch/acw676/MnM/test/'

train_csv_path='/data/scratch/acw676/MnM/train.csv'
val_csv_path='/data/scratch/acw676/MnM/val.csv'
#test_csv_path='/data/scratch/acw676/MnM/test.csv'

path_to_save_Learning_Curve='/data/home/acw676/MM/weights/'+'/3D_ADAM_NO_AUG'
path_to_save_check_points='/data/home/acw676/MM/weights/'+'/3D_ADAM_NO_AUG'
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
        #### Specify all the Hyperparameters\image dimenssions here #####

batch_size = 3
Max_Epochs = 100
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


df_train = pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)

train_loader = Data_Loader_io_transforms(df_train,train_imgs,batch_size)
val_loader = Data_Loader_V(df_val,val_imgs,batch_size)
   ### Load the Data using Data generators and paths specified #####
   #######################################
   
print(len(train_loader)) ### this shoud be = Total_images/ batch size
print(len(val_loader))   ### same here
#print(len(test_loader))   ### same here

### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses1 = []   # losses of all training epochs
avg_valid_losses1 = []  #losses of all training epochs
avg_valid_DS1 = []  # all training epochs

### Next we have all the funcitons which will be called in the main for training ####

  
### 2- the main training fucntion to update the weights....
def train_fn(loader_train1,loader_valid1,model1, optimizer1,loss_fn1, scaler):  ### Loader_1--> ED and Loader2-->ES
    train_losses1 = [] # loss of each batch
    valid_losses1 = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.train()
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED,three_ES_SA,three_ED_SA) in enumerate(loop):
        
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
        
        three_ES_SA = three_ES_SA.to(device=DEVICE,dtype=torch.float)
        three_ED_SA = three_ED_SA.to(device=DEVICE,dtype=torch.float)

        with torch.cuda.amp.autocast():
            ou1_ES,out2_ES = model1(img_SA_ES,M)  
            ou1_ED,out2_ED = model1(img_SA_ED,M)  
            
            loss1 = loss_fn1(ou1_ES,three_ES_SA)
            loss2 = loss_fn1(ou1_ED,three_ED_SA)
            loss3 = loss_fn1(out2_ES,temp_SA_ES)
            loss4 = loss_fn1(out2_ED,temp_SA_ED)
            
            # backward
        loss = (loss1 + loss2 + loss3 + loss4)/4
        optimizer1.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer1)

        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses1.append(float(loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED,three_ES_SA,three_ED_SA) in enumerate(loop_v):
        
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
        
        three_ES_SA = three_ES_SA.to(device=DEVICE,dtype=torch.float)
        three_ED_SA = three_ED_SA.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 
            
            ou1_ES,out2_ES = model1(img_SA_ES,M)  
            ou1_ED,out2_ED = model1(img_SA_ED,M)  
            
            loss1 = loss_fn1(ou1_ES,three_ES_SA)
            loss2 = loss_fn1(ou1_ED,three_ED_SA)
            loss3 = loss_fn1(out2_ES,temp_SA_ES)
            loss4 = loss_fn1(out2_ED,temp_SA_ED)
            
            
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
    
    for batch_idx, (img_LA_ES,temp_LA_ES,img_SA_ES,temp_SA_ES,img_LA_ED,temp_LA_ED,img_SA_ED,temp_SA_ED,label,M,all_three_ES,all_three_ED,three_ES_SA,three_ED_SA) in enumerate(loop):
        
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
        
        three_ES_SA = three_ES_SA.to(device=DEVICE,dtype=torch.float)
        three_ED_SA = three_ED_SA.to(device=DEVICE,dtype=torch.float)
        
        with torch.no_grad(): 

            ou1_ES,out2_ES = model1(img_SA_ES,M)  
            ou1_ED,out2_ED = model1(img_SA_ED,M)  
                        
            #out_LA_ES = generate_pred_label_1(out_LA_ES) # --- [B,C,H,W]
            out_LA_ES = (out2_ES > 0.5)* 1
            
            out_LA_ES_LV = out_LA_ES[:,0:1,:,:,:]
            out_LA_ES_MYO = out_LA_ES[:,1:2,:,:,:]
            out_LA_ES_RV = out_LA_ES[:,2:3,:,:,:]

            #out_LA_ED = generate_pred_label_1(out_LA_ED) # --- [B,C,H,W]
            out_LA_ED = (out2_ED > 0.5)* 1
            
            out_LA_ED_LV = out_LA_ED[:,0:1,:,:,:]
            out_LA_ED_MYO = out_LA_ED[:,1:2,:,:,:]
            out_LA_ED_RV = out_LA_ED[:,2:3,:,:,:]
            
            
            ou1_ES = (ou1_ES > 0.5)* 1
            ou1_ED = (ou1_ED > 0.5)* 1
            
            
            ## Dice Score for ES ###

            single_Three_ES = (2 * (ou1_ES * three_ES_SA).sum()) / (
                (ou1_ES + three_ES_SA).sum() + 1e-8)
            Three_ES +=single_Three_ES

            ## Dice Score for ED ###
            single_Three_ED = (2 * (ou1_ED * three_ED_SA).sum()) / (
                (ou1_ED + three_ED_SA).sum() + 1e-8)
            Three_ED +=single_Three_ED
            
            
            ## Dice Score for ES ###

            single_LA_LV_ES = (2 * (out_LA_ES_LV * temp_SA_ES[:,0:1,:,:,:]).sum()) / (
                (out_LA_ES_LV + temp_SA_ES[:,0:1,:,:,:]).sum() + 1e-8)
            
            Dice_score_LA_LV_ES +=single_LA_LV_ES
            
            single_LA_MYO_ES = (2 * (out_LA_ES_MYO*temp_SA_ES[:,1:2,:,:,:]).sum()) / (
    (out_LA_ES_MYO + temp_SA_ES[:,1:2,:,:,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ES += single_LA_MYO_ES

            single_LA_RV_ES = (2 * (out_LA_ES_RV* temp_SA_ES[:,2:3,:,:,:]).sum()) / (
        (out_LA_ES_RV + temp_SA_ES[:,2:3,:,:,:]).sum() + 1e-8)
            Dice_score_LA_RV_ES += single_LA_RV_ES
        

            # print(f"single_ID_  : {single_ID_}")
            # print(f"single_LA_LV_ES  : {single_LA_LV_ES}")
            # print(f"single_LA_MYO_ES  : {single_LA_MYO_ES}")
            # print(f"single_LA_RV_ES  : {single_LA_RV_ES}")
            # print("                          ")
            
            ## Dice Score for ED ###

            single_LA_LV_ED = (2 * (out_LA_ED_LV * temp_SA_ED[:,0:1,:,:,:]).sum()) / (
                (out_LA_ED_LV + temp_SA_ED[:,0:1,:,:,:]).sum() + 1e-8)
            
            Dice_score_LA_LV_ED +=single_LA_LV_ED
            
            single_LA_MYO_ED = (2 * (out_LA_ED_MYO*temp_SA_ED[:,1:2,:,:,:]).sum()) / (
    (out_LA_ED_MYO + temp_SA_ED[:,1:2,:,:,:]).sum() + 1e-8)
            Dice_score_LA_MYO_ED += single_LA_MYO_ED

            single_LA_RV_ED = (2 * (out_LA_ED_RV* temp_SA_ED[:,2:3,:,:,:]).sum()) / (
        (out_LA_ED_RV + temp_SA_ED[:,2:3,:,:,:]).sum() + 1e-8)
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

import torch
#W_1 = torch.tensor([1.5,1,1,1])
#W_1 = torch.tensor([1,1.5,1,1])
#W_1 = torch.tensor([1,1,1.5,1])
#W_1 = torch.tensor([1.5,1.5,1.5,1])
#W_1 = torch.tensor([0.2,0.2,0.4,0.2])
#W_1 = torch.tensor([1,1,1.5,1])
W_1 = torch.tensor([1,1,1.5,1])  
W_1 = W_1.to(device=DEVICE,dtype=torch.float)
class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=W_1, normalization='sigmoid'):
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

    def __init__(self,weight=W_1, normalization='sigmoid'):
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

    
def compute_per_channel_dice(input, target, epsilon=1e-6, weight=W_1):
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


from model2 import UNet5_4_M_CONV
model_1 = UNet5_4_M_CONV()

#from unets import UNet4
#model_1 = UNet4()


def main():
    model1 = model_1.to(device=DEVICE,dtype=torch.float)
    
            ## Fine Tunnning Part ###
#    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
#    weights_paths= "/data/home/acw676/MM/weights/a_3_three.pth.tar"
#    checkpoint = torch.load(weights_paths,map_location=DEVICE)
#    model.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])

    loss_fn1 = DiceLoss()
    optimizer1 = optim.Adam(model1.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    #optimizer1 = optim.SGD(model1.parameters(),momentum=0.99,lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model1, optimizer1, loss_fn1,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)

        dice_score = check_Dice_Score(val_loader, model1, device=DEVICE)
        
        
        avg_valid_DS1.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            
            ### save model    ######
            checkpoint = {
                "state_dict": model1.state_dict(),
                "optimizer":optimizer1.state_dict(),
            }
            save_checkpoint(checkpoint)

            break

if __name__ == "__main__":
    main()

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
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
