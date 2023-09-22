""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from glob import glob
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
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage
from sklearn.model_selection import train_test_split


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }
        
class LiTS2017(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training', prompt = 'click', plane = False, img_ext = '.png', msk_ext = '.png', num_classes = 1):
        """
        Args:
            data_path: Image/Mask base file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
            mode (str): 'Training' or 'Test'. Defaults to 'Training'.
            prompt (str): Mouse click. 'click' if autoclicker on (default, requires ground truth). 
            plane (bool): No use at all for this dataset.
            num_classes (int): Number of classes in dataset. Defaults to 1.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        img_ids = glob(os.path.join(data_path, 'images', '*' + img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        
        self.img_ids = train_img_ids if mode == 'Training' else val_img_ids
        
        self.img_dir = os.path.join(data_path, 'images')
        self.mask_dir = os.path.join(data_path, 'masks')
        self.img_ext = img_ext
        self.mask_ext = msk_ext
        self.num_classes = num_classes
        self.transform = transform
        self.prompt = prompt
        self.pt_mean = np.zeros(3)
        self.pt_cnt = 0

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext), -1)
        if img.ndim == 2:
        	img = img[..., None]

        mask = []
        for i in range(self.num_classes):
            mask_pre=cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
            _,mask_post=cv2.threshold(mask_pre,5,255,cv2.THRESH_BINARY)
            mask.append(mask_post[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        point_label = 1 if np.sum(mask) > 0 else 0
        inout = 1 # 1 si click dentro del higado, 0 si afuera (default: adentro)
        
        if self.prompt == 'click':
            if point_label == 1:
                div = np.max(mask)
                pt = random_click(np.array(mask) / div, point_label, inout)
                # Para las imágenes sin hígado (point_label = 0), usar promedio
                # de los valores anteriores como clicker
                self.pt_mean = ((self.pt_mean * self.pt_cnt + np.array(pt)) / (self.pt_cnt + 1)).astype(np.int16)
                self.pt_cnt += 1
            else:
                pt = self.pt_mean
        
        image_meta_dict = {'filename_or_obj':img_id}
        
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt[1:],
            'image_meta_dict':image_meta_dict,
            'clicker': self.pt_mean
        }
