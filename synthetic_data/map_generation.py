#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:06:14 2019

@author: manish

This module generates database of maps. The database is split into training and
test sets and stored into a pickle file.
"""

import pickle
import numpy as np
from x_map_gen import Exploration
from mask_generation import Mask
import matplotlib.pyplot as plt
######## Parameters for generating the database #############
GRID_SIZE = 32
numPOI = 18
trainRatio = 0.8
totalData = 500
validRatio = 0.1
testRatio = 0.1
#############################################################

######### Exploration Parameters #############
filterRatio = 0.7
##############################################

explore = Exploration(GRID_SIZE, numPOI, filterRatio)
mask = Mask()
#gridDimension = [GRID_SIZE, GRID_SIZE]
test_func = None
def generate(ratio,totalData,tpe):
	print("Generating",tpe,"data...")
	groundTruthData = []
	tunnelMapData = []
	maskData = []
	for i in range(int(ratio * totalData)):
		groundTruth = explore.generate_map()
		tunnelMap, frontierVector = explore.flood_fill_filter()
		global test_func
		test_func= frontierVector
		mask.set_map(tunnelMap, frontierVector)
		masking = mask.get_mask()

        #store all the elements
		groundTruthData.append(np.float32(groundTruth))
		tunnelMapData.append(np.float32(tunnelMap))
		maskData.append(np.float32(masking))
		print(
        '\r[Generating Data {} of {}]'.format(
            i,
            int(ratio * totalData),
        ),
        end=''
        )
	print('')
	return groundTruthData, tunnelMapData, maskData

groundTruthData = {}
tunnelMapData = {}
maskData = {}

groundTruthData["train"], tunnelMapData["train"], maskData["train"]  = generate(trainRatio,totalData,"training")
groundTruthData["validation"], tunnelMapData["validation"], maskData["validation"] = generate(validRatio,totalData,"validation")
groundTruthData["test"], tunnelMapData["test"], maskData["test"] = generate(testRatio,totalData,"testing")

with open('test/ground_truth_dataset_{}.pickle'.format(GRID_SIZE), 'wb') as handle:
	pickle.dump(groundTruthData, handle)

with open('test/image_dataset_{}.pickle'.format(GRID_SIZE), 'wb') as handle:
    pickle.dump(tunnelMapData, handle)
with open('test/mask_dataset_{}.pickle'.format(GRID_SIZE), 'wb') as handle:
    pickle.dump(maskData, handle)

fig = plt.figure(figsize=(6,8))
image,masking ,gt = tunnelMapData['train'][0], maskData['train'][0] , groundTruthData['train'][0]
fig.add_subplot(2,3,1)
plt.imshow(gt)
plt.title("True image")
plt.ylabel('y')
plt.xlabel('x')

fig.add_subplot(2,3,2)
plt.imshow(image)
plt.title("explored")
plt.ylabel('explored_y')
plt.xlabel('explored_x')

fig.add_subplot(2,3,3)
for p in test_func:			# Why doing this things, comment or remove it.
	image[p[0],p[1]] = 0.5
plt.imshow(image)
plt.title("explored with forntiers")
plt.ylabel('explored_y')
plt.xlabel('explored_x')

fig.add_subplot(2,3,4)
plt.imshow(masking)
title = "Masked for PATCH_SIZE = " + str(0)
plt.title(title)
plt.ylabel('masked_y')
plt.xlabel('masked_x')
plt.show()
plt.savefig("figure_8.png")