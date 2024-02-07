import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import cv2
import argparse
import random
from datetime import datetime
import shutil
from glob import glob
import albumentations as A
from albumentations.core.composition import Compose
from collections import OrderedDict
import pandas as pd
from metrics import iou_score, dice_coef
from utils import AverageMeter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="../../../Data/liver_only",help='path data validation')
    parser.add_argument('--out_path', default="../../../Data/results_val",help='path checkpoints')
   #parser.add_argument('--check_path', default="./work_dir/other_train/medsam_model_latest.pth",help='path checkpoints')
    parser.add_argument('--check_path', default="./work_dir/latests_model/medsam_model_latest.pth",help='path checkpoints')
    parser.add_argument('--num_classes', default=1,help='classes')
    parser.add_argument('--input_channels', default=1,help='input channels')
    args = parser.parse_args()

    return args


val_transform = Compose([
    A.Resize(512,512),
])



class LiverDataset(Dataset):
    def __init__(self, data_root, transform = None, mode = 'Training', img_ext = '.png', msk_ext = '.png',num_classes = 1):

        self.data_root = data_root
        img_ids = glob(os.path.join(data_root, 'images', '*'+img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        self.img_ids = train_img_ids if mode == 'Training' else val_img_ids
        self.img_ext = img_ext
        self.mask_ext = msk_ext
        self.img_path = os.path.join(data_root, 'images')
        self.gt_path = os.path.join(data_root, 'masks')
        self.num_classes = num_classes
        self.transform = transform
        print(f"number of images: {len(self.img_ids)}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = cv2.imread(os.path.join(self.img_path, img_id + self.img_ext), -1)
        if img.ndim == 2:
            img = img[..., None]
        mask = []
        for i in range(self.num_classes):
            mask_pre=cv2.imread(os.path.join(self.gt_path, str(i),
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

        # load npy image (1024, 1024, 3), [0,1]
        # convert the shape to (3, H, W)
        gt2D = mask # only one label, (256, 256)
        #assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        _, H, W = gt2D.shape
        bboxes = np.array([0, 0, W, H]) #[x_min, y_min, x_max, y_max]
        return (
            torch.tensor(img).float(),
            torch.tensor(gt2D).long(),
            torch.tensor(bboxes).float(),
            img_id,
        )


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def model_MedSAM(args):
    model_type="vit_b"
    checkpoint="SAM/sam_vit_b_01ec64.pth"
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint,img_size=512,in_chans=args.input_channels)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).cuda()
    medsam_model.load_state_dict(torch.load(args.check_path)["model"])
    print("epoch: ",torch.load(args.check_path)["epoch"])
    return medsam_model


def main():

    args=parse_args()

    print("=> creating model MedSAM")

    if not os.path.isdir(f"{args.out_path}/msk_pred/MedSAM"):
        os.mkdir(f"{args.out_path}/msk_pred/MedSAM")


    model=model_MedSAM(args)


    lits_val_dataset = LiverDataset(args.path, transform = val_transform, mode = 'Test')

    model.eval()
    
    nice_val_loader = torch.utils.data.DataLoader(
        lits_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False)

    avg_meter = AverageMeter()
    avg_meter_CSM = AverageMeter()
    avg_meter_KITS = AverageMeter()
    avg_meter_dice = AverageMeter()
    avg_meter_dice_CSM = AverageMeter()
    avg_meter_dice_KITS = AverageMeter()

    log = OrderedDict([
        ('name', []),
        ('iou', []),
    ])


    with torch.no_grad():
        for step, (image, gt2D, boxes, name) in enumerate(tqdm(nice_val_loader)):
            boxes_np = boxes.detach().cpu().numpy()
            image, target = image.cuda(), gt2D.cuda()
            output = model(image, boxes_np)
            meta=name[0].split("_")[1]
            iou = iou_score(output, target)
            avg_meter_val.update(iou, image.size(0))
            pred_img=output.detach().cpu().numpy()[0][0,:,:]
            pred_img[pred_img>0.5]=255
            log['name'].append(name[0])
            log['iou'].append(iou)
            cv2.imwrite(f"{args.out_path}/msk_pred/MedSAM/{name[0]}_Normal.png",target.detach().cpu().numpy()[0][0,:,:])
            cv2.imwrite(f"{args.out_path}/msk_pred/MedSAM/{name[0]}_pred.png",output.detach().cpu().numpy()[0][0,:,:])

 #   pd.DataFrame(log).to_csv(f'{args.out_path}/csv/log_MedSAM.csv', index=False)

    print('IoU val: %.4f' % avg_meter_val.avg)


if __name__ == "__main__":
    print("Working")
    main()
