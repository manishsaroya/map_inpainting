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

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--snapshot', type=str, default='mapinpainting_adaptive_mask.pth')
parser.add_argument('--image_size', type=int, default=32)
args = parser.parse_args()

device = torch.device('cpu')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

#dataset_val = Places2(args.root, img_transform, mask_transform, 'val')
dataset_val = torch.tensor(dataset('test',args.image_size))
model = PConvUNet(layer_size=3).to(device)
model.load_state_dict(torch.load(args.snapshot, map_location='cpu'))
model.eval()
evaluate(model, torch.tensor(dataset_val), device, 'result.jpg',True)
