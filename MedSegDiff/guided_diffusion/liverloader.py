import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from glob import glob
from sklearn.model_selection import train_test_split

class LiverDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):

        print("loading data from the directory :",data_path)
        path=data_path
        images = sorted(glob(os.path.join(path, "images/*.png")))
        masks = sorted(glob(os.path.join(path, "masks/0/*.png")))
        self.name_list = images
        self.label_list = masks
        self.data_path = path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(mask_name)

        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(msk_path,cv2.IMREAD_GRAYSCALE)
        #img = Image.open(img_path)
        #mask = Image.open(msk_path).convert('L')
        _,mask=cv2.threshold(mask,5,255,cv2.THRESH_BINARY)
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transforms.ToTensor()(transformed["image"])
            mask = transforms.ToTensor()(transformed["mask"])
            
        return (img, mask, name)
