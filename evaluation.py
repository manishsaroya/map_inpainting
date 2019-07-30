import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize

import matplotlib.pyplot as plt
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
import pdb

def evaluate(model, dataset, device, filename,if_save=False):

    #print("Inside evaluate..." ," and filename is",filename)
    
    image, mask, gt, z, f = dataset[0]
    #image, mask, gt, z, f = zip(*[dataset[i] for i in range(0,8)])
    image  = torch.as_tensor(image)
    mask = torch.as_tensor(mask)
    gt = torch.as_tensor(gt)
    #image = torch.stack(image)
    #mask = torch.stack(mask)
    #gt = torch.stack(gt)
    image = image.unsqueeze(0).unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    gt = gt.unsqueeze(0).unsqueeze(0)
    z = z.unsqueeze(1)
    f = f.unsqueeze(1)

    #pdb.set_trace()
    #print(len(image),len(mask),len(gt))
    #print(image.shape,mask.shape,gt.shape)
    #print(image[0].shape,mask[0].shape,gt[0].shape)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))

    output_comp = mask * image + (1 - mask) * output
    #print("output and output_comp shapes",output.shape,output_comp.shape)
    #grid = make_grid(
    #        torch.cat((unnormalize(image), mask, unnormalize(output),
    #                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    #save_image(grid, filename)

    image = image[0][0] #.permute(1,2,0)
    gt = gt[0][0] #.permute(1,2,0)
    mask = mask[0][0] #.permute(1,2,0)
    output = output[0][0] #.permute(1,2,0)
    #print("permuted shapes",image.shape,mask.shape,gt.shape,output.shape)
    if if_save== True:
        fig = plt.figure(figsize=(15,10))

        fig.add_subplot(2,3,1)
        plt.imshow(gt.numpy())
        plt.title("Ground Truth Map")
        #plt.ylabel('y')
        #plt.xlabel('x')


        fig.add_subplot(2,3,4)
        plt.imshow(numpy.stack([image.numpy(), image.numpy(), mask.numpy()],axis=-1))
        plt.title("Mask")
        #plt.ylabel('mask_y')
        #plt.xlabel('mask_x')


        fig.add_subplot(2,3,2)
        title = "70% explored Map"
        plt.imshow(image.numpy())
        plt.title(title)
        #plt.ylabel('masked_y')
        #plt.xlabel('masked_x')
        plt.savefig("figure_8.png")

        fig.add_subplot(2,3,3)
        plt.imshow(output.numpy())
        plt.title("Predicted Map")
        #plt.ylabel('output_y')
        #plt.xlabel('output_x')

        fig.add_subplot(2,3,5)
        plt.imshow(1.0/(1.0 + numpy.exp(-output.numpy())) > 0.55)
        #print(output.numpy().max())
        #print(output.numpy().min())
        #plt.imshow(numpy.stack([output.numpy(), image.numpy(), mask.numpy()],axis=-1),cmap='hot')
        plt.title("output image")
        plt.ylabel('output_y')
        plt.xlabel('output_x')

        fig.add_subplot(2,3,6)
        a = 1.0/(1.0 + numpy.exp(-output.numpy()))
        plt.imshow((a * abs(mask.numpy() - 1)) > 0.55)
        plt.title("Predicted Map")

        #plt.show()
        plt.savefig(filename)