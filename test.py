import argparse
import torch
from torchvision import transforms
from get_data import dataset
import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from util.io import get_state_dict_on_cpu
import os

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
#parser.add_argument('--snapshot', type=str, default='snapshots/adaptivelongsize32/ckpt/995000.pth')
parser.add_argument('--snapshot', type=str, default='snapshots/toploss24variable/ckpt/500000.pth')
#parser.add_argument('--snapshot', type=str, default='/home/subt/Desktop/Important_top_trained_network/500000.pth')
parser.add_argument('--image_size', type=int, default=24)
parser.add_argument('--output_dir',type=str, default='snapshots/toploss24variable/output/')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

#dataset_val = Places2(args.root, img_transform, mask_transform, 'val')
dataset_val = dataset('test',args.image_size)
model = PConvUNet(layer_size=3, input_channels=1).to(device)
load_ckpt(args.snapshot, [('model', model)])
#model.load_state_dict(torch.load(args.snapshot, map_location='cpu'))
model.eval()
evaluate(model, dataset_val, device, args.output_dir,num_=20, if_save=True)
