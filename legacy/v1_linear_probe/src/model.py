import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image, ImageOps

PREPROCESS_VERSION='rgb-letterbox224-lab-clahe2-grid8-imagenet-v1'

def preprocess(image, clahe=False, size=224):
    image=ImageOps.exif_transpose(image).convert('RGB')
    # CLAHE on luminance only; preserve chromatic vessel information.
    image=ImageOps.contain(image,(size,size),Image.Resampling.BILINEAR)
    arr=np.asarray(image).copy()
    if clahe:
        lab=cv2.cvtColor(arr,cv2.COLOR_RGB2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
        arr=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    canvas=np.zeros((size,size,3),dtype=np.uint8)
    h,w=arr.shape[:2];canvas[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w]=arr
    x=torch.from_numpy(canvas.copy()).permute(2,0,1).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def backbone(name,pretrained=True):
    if name=='resnet18':
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None);dim=m.fc.in_features;m.fc=nn.Identity()
    elif name=='mobilenet_v3_small':
        m=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None);dim=m.classifier[0].in_features;m.classifier=nn.Identity()
    else:raise ValueError(name)
    return m,dim

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
