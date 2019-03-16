import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from net import PConv2d
import PIL
import opt
from tensorboardX import SummaryWriter
from matplotlib import pyplot
from loadMasks import CustomMasks
from loss_biased import InpaintingLoss
from net import VGG16FeatureExtractor
from FullReconstructionNetworkDropout128 import Net
from random import randint
import numpy as np

# Device configuration
if torch.cuda.is_available():
    print('yes!')
else:
    print('no!')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 40000
num_classes = 10
batch_size = 1
learning_rate = 0.03

IMG_PATH = 'dataLinearMasks128/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'train.csv'


transform1=transforms.Compose([transforms.Resize(size=128),transforms.ToTensor()])
# MNIST dataset
#,transforms.Normalize(mean=opt.MEAN, std=opt.STD)

mask_train_dataset = CustomMasks(TRAIN_DATA,IMG_PATH,IMG_EXT,transform1)

train_dataset = torchvision.datasets.MNIST(root='MnistData/',
                                           train = True,
                                           transform=transform1,
                                           download=True)
#STL10
test_dataset = torchvision.datasets.MNIST(root='MnistData/',
                                          train = False,
                                          transform=transform1)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

mask_train_loader = torch.utils.data.DataLoader(dataset=mask_train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

writer = SummaryWriter()

net =Net()

net.load_state_dict(torch.load('LinearMaskWeightsOceansDropout128'))
data = np.load('trainingSet.npy')
data = np.reshape(data,(data.shape[0],1,data.shape[1],data.shape[2],data.shape[3]))
avgData = np.mean(data[:,:,0,:,:])
np.random.shuffle(data)
stdData = np.std(data[:,:,0,:,:])
data = (data[:,:,:1,:,:]-avgData)/stdData
data = torch.from_numpy(data)



optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):

    running_loss = 0.0

    for i, images in enumerate(data):
        input_mask = []
        #input_mask = mask_train_dataset.__getitem__(0)
        #input_mask = input_mask.view(1, 3, 512, 512)
        #for input_masks, nothing in mask_train_loader:

        for j, (input_masks) in enumerate(mask_train_loader):
            input_mask = input_masks

        #torchvision.utils.save_image(images, 'input_mask_only.jpg')



        xLoc = randint(10,118)
        yLoc = randint(10,118)


        #input_mask_only = PIL.ImageChops.offset(PIL.Image.fromarray(np.reshape(input_mask.cpu().data.numpy(),(128,128))),64-yLoc,64-xLoc)
        #input_mask_only = np.reshape(np.array(input_mask.mask.convert('F'),dtype=np.float32),(len(images),1,128,128))
        #input_mask_only = torch.from_numpy(input_mask)
        input_mask = input_mask.to(device)


        #images = PIL.ImageChops.offset(PIL.Image.fromarray(np.reshape(images.cpu().data.numpy(),(128,128))),64-yLoc,64-xLoc)
        #images = np.reshape(np.array(images.convert('F'),dtype=np.float32),(1,1,128,128))
        #images = torch.from_numpy(images)
        images = images.to(device)

        #torchvision.utils.save_image(images, 'input_mask_only.jpg')

        input_images = images*input_mask
        input_images = input_images.to(device)
        input_mask = input_mask.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        output = net(input_images,input_mask)
        loss_dict = criterion(input_images,input_mask,output,images)

        loss = 0.0
        for key, coef in opt.LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value
            #if (i + 1) % 500 == 0:
            #    writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:  # print every 2000 mini-batches
            torch.save(net.state_dict(), 'LinearMaskWeightsOceansDropout128')
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0



    #torch.save(net.state_dict(),'trainedModel')
    #the_model.load_state_dict(torch.load(PATH))
