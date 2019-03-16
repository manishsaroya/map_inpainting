#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:21:46 2019

VGG class load the pretrained VGG16 network parameters

@author: Manish Saroya
@contact: saroya@oregonstate.edu

"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision import models 

class VGG16(nn.Module):
    def __init__(self):
        """ Initialize the VGG16 network
        """
        # Call Base class constructor
        super().__init__()
        # Get VGG16 pretrained model
        self.vgg16 = models.vgg16(pretrained=True)
                
        
            
