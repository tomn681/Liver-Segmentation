# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
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
from metrics import iou_score
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            os.path.join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

class LiverDataset(Dataset):
    def __init__(self, data_root, transform = None, mode = 'Training', img_ext = '.png', msk_ext = '.png',num_classes = 1):

        self.data_root = data_root
        img_ids = glob(os.path.join(data_root, 'images', '*'+img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        
        self.img_ids = train_img_ids if mode == 'Training' else val_img_ids
        #self.img_ids = train_img_ids if mode == 'Training' else img_ids

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
    	return (
            torch.tensor(img).float(),
            torch.tensor(gt2D).long(),
            img_id,
        )

# %% sanity test of dataset class
tr_dataset = LiverDataset(data_root="../../../Data/liver_only", mode = 'Test')
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
for step, (image, gt, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    #show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    #show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="../../Data/liver_only",
    help="path to training image files; two subfolders: mask/0 and imgs",
)
parser.add_argument(
    "--data_val_path",
    type=str,
    default="../../Data/liver_only_val",
    help="path to validaion image files; two subfolders: mask/0 and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
#
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument("-img_size", type=int, default=512)
parser.add_argument("-in_channels", type=int, default=1)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument("-results_folder", type=str, default="", help="folder for output results")
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        
        for param in image_encoder.parameters():
        	param.requires_grad=False

        for i in range(6,12):
        	for param in image_encoder.blocks[i].parameters():
        		param.requires_grad=True

        for param in image_encoder.neck.parameters():
        	param.requires_grad=True
        
        #for name,param in image_encoder.named_parameters():
        #	print(name,param.requires_grad)

        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        # do not compute gradients for encoder and prompt encoder
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        with torch.no_grad():
            #box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            #if len(box_torch.shape) == 2:
            #    box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                #boxes=box_torch,
                boxes=None,
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


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, os.path.join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint,img_size=args.img_size,in_chans=args.in_channels)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
   
    
    medsam_model.train()
    #print(medsam_model)
    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    best_iou=0

    transform_train_lits = Compose([
	    A.Resize(args.img_size, args.img_size),
	    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
	    A.HorizontalFlip(p=0.5),
	    A.Rotate (limit=10, interpolation=1, border_mode=1, value=None, crop_border=False, always_apply=False, p=0.5),
	    #trns.Normalize(),
	])

    transform_val_lits = Compose([
	    A.Resize(args.img_size, args.img_size),
	    #trns.Normalize(),
	])


    if args.results_folder and not os.path.exists(args.results_folder):
    	os.mkdir(args.results_folder)



    lits_train_dataset = LiverDataset(args.data_path, transform = transform_train_lits, mode = 'Training')
    lits_val_dataset = LiverDataset(args.data_path, transform = transform_val_lits, mode = 'Test')

    nice_train_loader = DataLoader(lits_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    nice_val_loader = DataLoader(lits_val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Number of training samples: ", len(lits_train_dataset))
    print("Number of validation samples: ", len(lits_val_dataset))

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])


    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    print(f"start trainning , number of epochs:  {num_epochs}")
    for epoch in range(start_epoch, num_epochs):
    	print(f"Epoch: {epoch}")
    	medsam_model.train()
    	epoch_loss = 0
    	ious=[]
    	for step, (image, gt2D, _) in enumerate(tqdm(nice_train_loader)):
            optimizer.zero_grad()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                ious.append(iou_score(medsam_pred, gt2D))
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                ious.append(iou_score(medsam_pred, gt2D))
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1
    	val_ious=[]
    	val_epoch_loss=0
    	medsam_model.eval()
    	name_folder_results=os.path.join(args.results_folder,str(epoch))
    	if args.results_folder and not os.path.exists(name_folder_results):
    		os.mkdir(name_folder_results)


    	with torch.no_grad():
	        for step, (image, gt2D, img_id) in enumerate(tqdm(nice_val_loader)):
	            image, gt2D = image.to(device), gt2D.to(device)
	            medsam_pred = medsam_model(image)
	            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
	            val_epoch_loss+=loss.item()
	            val_ious.append(iou_score(medsam_pred, gt2D))
	            if args.results_folder:
	            	cv2.imwrite(os.path.join(name_folder_results,img_id[0])+".png",medsam_pred.detach().cpu().numpy()[0][0,:,:])

    	epoch_loss /= step
    	val_epoch_loss/=step
    	losses.append(epoch_loss)
    	epoch_iou=np.mean(ious)
    	val_epoch_iou=np.mean(val_ious)
    	log["epoch"].append(epoch)
    	log["loss"].append(epoch_loss)
    	log["iou"].append(epoch_iou)
    	log["val_loss"].append(val_epoch_loss)
    	log["val_iou"].append(val_epoch_iou)
    	print("iou: ",val_epoch_iou)
    	pd.DataFrame(log).to_csv(os.path.join(model_save_path, "log.csv"), index=False)
    	if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
    	print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}, IoU: {epoch_iou}, valIoU: {val_epoch_iou}'
        )
        ## save the latest model
    	checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
    	torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
    	if val_epoch_iou > best_iou:
            print("NEW BEST IOU : ",val_epoch_iou)
            best_iou = val_epoch_iou
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
    	
        #plt.plot(losses)
        #plt.title("Dice + Cross Entropy Loss")
        #plt.xlabel("Epoch")
        #plt.ylabel("Loss")
        #plt.savefig(os.path.join(model_save_path, args.task_name + "train_loss.png"))
        #plt.close())


if __name__ == "__main__":
    main()
