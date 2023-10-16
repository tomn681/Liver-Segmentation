import numpy as np
import torch
import torch.nn.functional as F
import cv2

def connectPoints(points,size):
    cont = np.zeros(size).astype(np.uint8)
    points=points.astype('int')
    lenght=len(points)
    points2=np.append(points[1:],[points[0]],axis=0)
    diffs=points2-points
    maxs=np.max(np.abs(diffs), axis=1)
    for i in range(lenght):
        x, y = points[i]
        x, y = int(x), int(y)
        cont[y-2:y+2, x-2:x+2] = 255
        #cont[points[i][1],points[i][0]]=255
        for j in range(np.abs(maxs[i])):
            xi=points[i][0]+(diffs[i][0]*j)//maxs[i]
            yi=points[i][1]+(diffs[i][1]*j)//maxs[i]
            cont[yi,xi]=255

    mask_inv=cont.copy()
    cv2.floodFill(mask_inv,None,(0,0),255)
    mask=cv2.bitwise_not(mask_inv)
    
    return mask | cont  

import random
def iou_score(output_points, target_points,size):
    smooth = 1e-5
    output=connectPoints(output_points*size[0],size)
    target=connectPoints(target_points*size[0],size)
    intersection = (output & target).sum()
    union = (output | target).sum()
    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
