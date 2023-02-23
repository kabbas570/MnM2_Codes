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
    def __init__(self, df, images_folder,transformations=transforms_all):  ## If I apply Data Augmentation here, the validation loss becomes None. 
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


val_imgs = r'C:\My_Data\M2M Data\data\data_2/val' ## path to images
val_csv_path = r'C:\My_Data\M2M Data\data\val.csv'  ## path to csv file
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)
a1 =next(a)
gt1=a1[7][0,2,4,:,:]
plt.figure()
plt.imshow(gt1)
gt2=a1[13][0,0,4,:,:]
plt.figure()
plt.imshow(gt2)
