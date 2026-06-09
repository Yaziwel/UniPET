import torch
import os
import SimpleITK as sitk 
import pickle 
from tqdm import tqdm
# import pydicom
import numpy as np
import datetime
import pandas as pd
import json 
from torch.nn import functional as F 
class transformPET:
    def __init__(self, data_range, mode="linear"):
        self.r = data_range
        self.m = mode

    def cut(self,x):
        thresh=self.r
        x[x>thresh]=thresh
        x[x<=0]=0.0
        _, _, d,h,w=x.shape
        x = x[:, :, 4:d-4, 16:h-16, 16:w-16]
        return x 

    def clip(self,x):

        x[x>self.r]=self.r
        x[x<=0]=0.0
        return x 
    
    def get_error_mask(self, low, full, error_thresh=0.01):
        error = torch.abs(full-low) 
        mask = (error>error_thresh).type(torch.FloatTensor).to(full.device)
        return mask
    
    def normalize(self, img, cut=False):
        img = self.clip(img)
        if self.m == "exp":
            c = torch.log(torch.Tensor([self.r+1])).to(img.device)
            img = torch.log(1+img)/c
        else:
            img = img/self.r
        
        if cut:
            img = self.cut(img)
        return img
    def denormalize(self, img):
       	if self.m=='exp':
       		c = torch.log(torch.Tensor([self.r+1])).to(img.device)
       		img = torch.exp(c*img)-1
       	else:
       		img*=self.r
       	img = self.clip(img)
        return img
            
@torch.no_grad()
def synthesisOneAxial(model, img, kernel_size=64, stride=32, crop_size = 3):
    model.eval()
    B, C, D, H, W = img.shape
    nz = int(D//stride)
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in range(nz):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride==D:
            break
        elif idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ######
        G_patches = model(patches)
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
                else:
                    result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
    return result

# @torch.no_grad()
# def merge_patch(model, img, kernel_size=64, stride=32,crop_size = 3):
#     model.eval()
#     B, C, D, H, W = img.shape
#     nz = int(D//stride)
#     nx = int(H//stride)-1
#     ny = int(W//stride)-1
#     result = torch.zeros((B, C, D, H, W)).type(torch.FloatTensor).to(img.device)
#     flag=True
#     for k in range(nz):
#         idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
#         if idz+crop_size+stride==D:
#             break
#         elif idz+crop_size+stride>D:
#             flag=False
#         x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
#         patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
#         patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
#         ######
#         #Synthesis
#         ###### 
#         # import pdb 
#         # pdb.set_trace()
#         G_patches = patches.clone().detach() 
#         for i in range(len(patches)):
#             G_patches[i] = model(patches[[i],:,:,:,:])
#         for i in range(nx):
#             idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
#             for j in range(ny):
#                 idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
#                 if flag:
#                     result[:,:,idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
#                 else:
#                     result[:,:,idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*nx+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:].unsqueeze(0).unsqueeze(0)
#     return result 

@torch.no_grad()
def merge_patch(model, img, kernel_size=64, stride=32, crop_size = 3, batch_compute=False): 
    # img = input_img.clone()
    B, C, D, H, W = img.shape
    # Check if H and W are greater than kernel_size
    # if D < kernel_size or H < kernel_size or W < kernel_size:
    #     raise ValueError("D, H and W must be greater than {}".format(kernel_size)) 
    
    pad_d = (stride - D % stride) if D % stride != 0 else 0
    pad_h = (stride - H % stride) if H % stride != 0 else 0
    pad_w = (stride - W % stride) if W % stride != 0 else 0 


    # If padding is needed, calculate symmetric padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Symmetric padding for depth (front, back)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        # Symmetric padding for height (top, bottom)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        # Symmetric padding for width (left, right)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding
        padding = (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)  # (left, right, top, bottom, front, back)
        img = F.pad(img, padding, mode='constant', value=0) 
    else: 
        padding = (0, 0, 0, 0, 0, 0) 
    

    
    # Process input with Model
    B, C, D, H, W = img.shape
    nz = int(D//stride)-1
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in tqdm(list(range(nz))):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ###### 
        
        G_patches = patches.clone().detach() 
        if batch_compute: 
            G_patches = model(patches) 
        else:
            for i in range(len(patches)):
                G_patches[i] = model(patches[[i],:,:,:,:]) 
                
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*ny+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:]
                else:
                    result[idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*ny+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:] 

    # Remove padding 
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding 
    
    # import pdb 
    # pdb.set_trace()
    # result = result[pad_front:-pad_back, pad_top:-pad_bottom, pad_left:-pad_right]
    if pad_back > 0:
        result = result[:-pad_back, :, :]
    if pad_front > 0:
        result = result[pad_front:, :, :]
    if pad_bottom > 0:
        result = result[:, :-pad_bottom, :]
    if pad_top > 0:
        result = result[:, pad_top:, :]
    if pad_right > 0:
        result = result[:, :, :-pad_right]
    if pad_left > 0:
        result = result[:, :, pad_left:] 
        

    return result.unsqueeze(0).unsqueeze(0)

class dataIO:
    def __init__(self):
        self.reader = {
            '.img':self.load_itk,
            '.gz':self.load_itk,
            '.bin':self.load_bin, 
            '.txt':self.load_txt, 
            '.json':self.load_json
            
            }
        self.writer = {
            '.img':self.save_itk, 
            '.gz':self.save_itk,
            '.bin':self.save_bin,
            '.csv':self.save_csv,
            '.txt':self.save_txt, 
            '.txt':self.save_json
            }
    
    
    def save_itk(self, data, path):
        sitk.WriteImage(sitk.GetImageFromArray(data), path) 
        
    def load_itk(self,path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
        
    def load_bin(self,path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data 
    def load_txt(self, path):
        with open(path, "r") as f:
            data = f.read() 
        return data

    def save_bin(self,data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_json(self, data, path):
        with open(path, "w", encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 

    def load_json(self, path):
        with open(path, encoding='utf8') as f:
            data = json.load(f) 
        return data 

    def save_csv(self, data_dict, path):
        result=pd.DataFrame({ key:pd.Series(value) for key, value in data_dict.items() })
        result.to_csv(path)
    
    def save_txt(self, s, path):
        with open(path,'w') as f:
            f.write(s) 
            

        
    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex
    
    def load(self, path):
        ex = self.getFileEX(path)
        return self.reader[ex](path)
    def save(self, data, path):
        ex = self.getFileEX(path)
        return self.writer[ex](data, path)




