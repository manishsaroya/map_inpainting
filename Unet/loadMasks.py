import pandas as pd
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

class CustomMasks(Dataset):

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + str('{num:06d}'.format(num=x)) + img_ext)).all(), \
            "Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = 0

    def __getitem__(self, index):
        img = Image.open(self.img_path + str('{num:06d}'.format(num=self.X_train[index])) + self.img_ext)
        #img = img.convert('RGB')
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        
        label = 0
        return img

    def __len__(self):
        return len(self.X_train.index)