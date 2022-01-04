#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from fastai import *
from fastai.vision.all import *
from fastai.imports import *
import cv2, os
from torchvision import transforms as T
from torchvision.transforms import functional as F
import albumentations as A
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

class ImagenetDataset(torch.utils.data.Dataset):

    def __init__(self, path, df, label_dict, dim = 256, transforms=None):
        #define our vars
        self.df = df
        self.transforms = transforms
        self.image_paths = get_image_files(path)
        self.label_dict = label_dict
        self.dim = dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #get image path for each image by idx
        img_path = self.image_paths[idx]
        # read image
        img = cv2.imread(str(img_path))
        #make sure its RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #Get the labels from our dict
        label = self.label_dict[self.df[self.df["image"] == img_path.name]["label"].values[0]]
        #resize if need be
        img = A.Resize(self.dim, self.dim)(image=img)["image"]

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        #convert back to tensor for use
        img = F.to_tensor(img)
        label = torch.as_tensor(label)
        label = torch.nn.functional.one_hot(label,len(self.df["label"].unique()))

        return img, label.float()

