import argparse
import torch
from torchvision import transforms
from get_data import dataset
import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='./snapshots/default/ckpt/250000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

#dataset_val = Places2(args.root, img_transform, mask_transform, 'val')
dataset_val = torch.tensor(dataset('test',args.grid_size))
model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])
#model.load_state_dict(torch.load('mapinpainting_10000.pth'))
model.eval()
evaluate(model, torch.tensor(dataset_val), device, 'result.jpg',True)
