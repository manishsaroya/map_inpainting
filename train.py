import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
debug = 0
import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from places2 import Places2
from util.io import load_ckpt
from util.io import save_ckpt
from get_data import dataset
class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/size32')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--log_dir_val', type=str, default='./logs/default_val')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=400000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=5000)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--grid_size', type=int, default=32)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)
writer_val = SummaryWriter(log_dir=args.log_dir_val)

print("image size in ArgumentParser",args.image_size,"number of threads",args.n_threads)
size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

#dataset_train = Places2(args.root, args.mask_root, img_tf, mask_tf, 'train')
#dataset_val = Places2(args.root, args.mask_root, img_tf, mask_tf, 'val')

dataset_train = dataset('train', args.grid_size)
dataset_val = dataset('val',args.grid_size)
#total_white = 0
#for i in range(len(dataset_train)):
#    total_white = total_white + (dataset_train[i][0].sum()) / 3.0
#    
#total_white = total_white /(len(dataset_train) * 64 *64)

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))

iterator_val = iter(data.DataLoader(
    dataset_val, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_val)),
    num_workers=args.n_threads))

print(len(dataset_train))
model = PConvUNet(layer_size=3).to(device)

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, args.max_iter)):
    model.train()

    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]   
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % args.log_interval ==0:
        image, mask, gt = [x.to(device) for x in next(iterator_val)]
        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)
        loss = 0.0
        for key, coef in opt.LAMBDA_DICT.items():
            value = coef * loss_dict[key]   
            loss += value
            writer_val.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
        
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        #print("Going to evaluate")
        evaluate(model, torch.tensor(dataset_val), device,
                 '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))
    #if (i+1)%500 == 0:
    #    print(i+1," iterations completed","loss is ",loss.item())
dataset_test = torch.tensor(dataset('test',args.grid_size))
torch.save(model.state_dict(), 'mapinpainting_10000.pth')
model.eval()
evaluate(model, dataset_test,device,'result.jpg',True)

writer.close()
writer_val.close()
