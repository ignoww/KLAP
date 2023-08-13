from torch.utils.data import Dataset
import torch
import glob
from torchvision import transforms
import os
from PIL import Image
from glob import glob
import json
import random
import numpy as np
import cv2
import os
import os.path
import random
import pickle
import matplotlib.pyplot as plt
    

    
class DatasetForTrain_sup_self(Dataset):
    def __init__(self,mode,img_index):
        self.img_index = img_index
        self.image_size = 240
        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((self.image_size, self.image_size))

        self.mode = mode
        self.datasets = []
        main_path = "./data/"
        file_paths = os.listdir(os.path.join(main_path))
        self.file_train = []

        file_paths.sort()

        for file_path in file_paths:
            file_path_snow = os.path.join(main_path,file_path)
            self.file_train.append(file_path_snow)
    
    
        
        
        self.image_size = 240
        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((self.image_size, self.image_size))
    
    
        
    def __len__(self):
        return len(self.file_train)
    
    def __getitem__(self, index):
        index = self.img_index
        with open(self.file_train[index],"rb") as fr:
            img = pickle.load(fr)
        # random path and aug
        img = self.random_patch(img)
        img = self.augment(img)
        
        ## Demosaicing
        if self.mode == "bayer":
            ran_mode = 1
        elif self.mode == "quad":
            ran_mode = 2
        elif self.mode == "nano":
            ran_mode = 3
        elif self.mode == "qxq":
            ran_mode = 4
        elif self.mode == "all":
            ran_mode = random.randint(1, 4)
            
        # add strong noise
        img_n = self.add_noise(img)
        img_de, img_n2self, img_mask = self.demosaicing(img_n,ran_mode)
        
        ## torch to numpy
        img_de, img = self.np2Tensor(img_de,img)
        img_n2self, img_mask = self.np2Tensor(img_n2self,img_mask)
        
        return img, img_de, img_n2self, img_mask, ran_mode-1   
    
    def np2Tensor(self,img_de, img, rgb_range=255):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(1 / 255)

            return tensor

        return _np2Tensor(img_de), _np2Tensor(img)

    def augment(self, img, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

            return img

        return _augment(img)

    def add_noise(self,image, shot_noise=0.004, read_noise=0.008**2):
        image = image/255
        variance = image * shot_noise + read_noise
        sigma=np.sqrt(variance)
        noise=sigma *np.random.normal(0,1,(np.shape(image)[0],np.shape(image)[1],3))
        out      = image + noise
        out=np.maximum(0.0,np.minimum(out,1.0))
        return out.astype(np.float32)*255

    def random_patch(self,img):
        ih, iw = img.shape[:2]
            
        ix = random.randrange(0, iw - self.image_size + 1)
        iy = random.randrange(0, ih - self.image_size + 1)
        img = img[iy:iy + self.image_size, ix:ix + self.image_size, :]
        return img
    
    def demosaicing(self,rgb,sh_mode):    
        mosaic_img = np.zeros((rgb.shape[0], rgb.shape[1],1), dtype=rgb.dtype)
        mosaic_img_n2self = np.zeros((rgb.shape[0], rgb.shape[1],3), dtype=rgb.dtype)
        mosaic_img_mask = np.zeros((rgb.shape[0], rgb.shape[1],3), dtype=rgb.dtype)

        indx = 0
        indy = 0
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = 1
        indx = sh_mode
        indy = 0
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 2:3]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,2:3] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 2:3]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,2:3] = 1
                
        indx = 0
        indy = sh_mode
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 0:1]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,0:1] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 0:1]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,0:1] = 1
        indx = sh_mode
        indy = sh_mode
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = 1
        return mosaic_img, mosaic_img_n2self, mosaic_img_mask
    
class DatasetForTest_sup_self(Dataset):
    def __init__(self,mode,img_index):
        ####################

        # bayer, quad, nona, qxq
        #if self.mode == "all":
            
        
        self.img_index = img_index
        self.image_size = 240
        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((self.image_size, self.image_size))
#        fake_path = '/home/hyun/demosaic/data/Hynix_data/Hi4821QxQ_objects_Exp48_VCM355.raw'
        
#        fid = open(fake_path,"rb")
#        raw = np.fromfile(fid, dtype='<H')
#        blc_offset = np.full(6000*8000, 64, dtype=np.int16)
#        raw = np.where(raw < 1023, raw, 1023).astype(np.uint16)
#        raw = np.where(raw.astype(np.int16) - blc_offset > 0, raw.astype(np.int16) - blc_offset, 0).astype(np.uint16)
#        fake_img = (raw.reshape((6000, 8000))).astype(np.float32)
#        fake_img = (fake_img/4).astype(np.float32)
#        self.fake_img =fake_img
        
        self.mode = mode
        self.datasets = []
        main_path = "./data/"
        file_paths = os.listdir(os.path.join(main_path))
        self.file_train = []

        file_paths.sort()

        for file_path in file_paths:
            file_path_snow = os.path.join(main_path,file_path)
            self.file_train.append(file_path_snow)
    
    
        
        
        self.image_size = 240
        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((self.image_size, self.image_size))
    
    
        
    def __len__(self):
        return len(self.file_train)
    
    def __getitem__(self, index):
        index = self.img_index
        with open(self.file_train[index],"rb") as fr:
            img = pickle.load(fr)
        # random path and aug
        img = img[:1200,:1200,:]
#        img = self.random_patch(img)
#        img = self.augment(img)
        
        ## Demosaicing
        if self.mode == "bayer":
            ran_mode = 1
        elif self.mode == "quad":
            ran_mode = 2
        elif self.mode == "nano":
            ran_mode = 3
        elif self.mode == "qxq":
            ran_mode = 4
        elif self.mode == "all":
            ran_mode = random.randint(1, 4)
        img_n = self.add_noise(img)
        img_de, img_n2self, img_mask = self.demosaicing(img_n,ran_mode)
        ## noise add
        
        ## torch to numpy
        img_de, img = self.np2Tensor(img_de,img)
        img_n2self, img_mask = self.np2Tensor(img_n2self,img_mask)
        
        return img, img_de, ran_mode-1 , img_n2self, img_mask
    
    def np2Tensor(self,img_de, img, rgb_range=255):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(1 / 255)

            return tensor

        return _np2Tensor(img_de), _np2Tensor(img)
    
    def np2Tensor_fake(self, fake_img, rgb_range=255):
        tensor = torch.from_numpy(fake_img).float()
        tensor.mul_(1 / 255)
        return tensor
        
    def augment(self, img, hflip=True, rot=True):
        
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

            return img

        return _augment(img)

    def add_noise(self,image, shot_noise=0.004, read_noise=0.008**2):
        image = image/255
        variance = image * shot_noise + read_noise
        sigma=np.sqrt(variance)
        noise=sigma *np.random.normal(0,1,(np.shape(image)[0],np.shape(image)[1],3))
        out      = image + noise
        out=np.maximum(0.0,np.minimum(out,1.0))
        return out.astype(np.float32)*255

    def random_patch(self,img):
        ih, iw = img.shape[:2]
            
        ix = random.randrange(0, iw - self.image_size + 1)
        iy = random.randrange(0, ih - self.image_size + 1)
        img = img[iy:iy + self.image_size, ix:ix + self.image_size, :]
        return img
    
    def demosaicing(self,rgb,sh_mode):    
        mosaic_img = np.zeros((rgb.shape[0], rgb.shape[1],1), dtype=rgb.dtype)
        mosaic_img_n2self = np.zeros((rgb.shape[0], rgb.shape[1],3), dtype=rgb.dtype)
        mosaic_img_mask = np.zeros((rgb.shape[0], rgb.shape[1],3), dtype=rgb.dtype)

        indx = 0
        indy = 0
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = 1
        indx = sh_mode
        indy = 0
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 2:3]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,2:3] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 2:3]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,2:3] = 1
                
        indx = 0
        indy = sh_mode
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 0:1]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,0:1] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 0:1]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,0:1] = 1
        indx = sh_mode
        indy = sh_mode
        for sh_idx in range(sh_mode):
            for sh_idy in range(sh_mode):
                mosaic_img[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,:] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_n2self[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = rgb[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2, 1:2]
                mosaic_img_mask[indx+sh_idx::sh_mode*2, indy+sh_idy::sh_mode*2,1:2] = 1
        return mosaic_img, mosaic_img_n2self, mosaic_img_mask
