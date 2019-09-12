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
import sys
sys.path.append("../synthetic_data/")

from x_map_gen import Exploration
from mask_generation import Mask
import matplotlib.pyplot as plt
import os
import pdb
######## Parameters for generating the database #############
GRID_SIZE = 24
numPOI = 4
trainRatio = 0
totalData = 100
validRatio = 0
testRatio = 1
#############################################################

######### Exploration Parameters #############
filterRatio = 0.7
data_dir = "./for_test_gaussian"
##############################################

if not os.path.exists(data_dir):
	os.makedirs(data_dir)


explore = Exploration(GRID_SIZE, numPOI)
mask = Mask()
#gridDimension = [GRID_SIZE, GRID_SIZE]
test_func = None

def sanityCheckMasknExplored(ground, image, mask):
	# check that the explored image is equivalent to ground when mask is greater than one.
	# if failure occures check implementation of % explored maps
	count = 0 
	for i in range(len(ground)):
		#pdb.set_trace()
		if not (image[i] == ground[i] * mask[i]).all():
			print("Sanity test failed at index", i)
			count = count + 1
	print(".....done testing.....")
	print("number of failures", count)
	return count

def generate(ratio,totalData,tpe):
	print("Generating",tpe,"data...")
	groundTruthData = []
	tunnelMapData = []
	maskData = []
	frontierVectorData = []
	#adaptivemaskData = []
	#percent_exploredData = []
	for i in range(int(ratio * totalData)):
		# generate a map 
		groundTruth = explore.generate_map()
		# randomly select percent_explored
		percent_explored = np.random.uniform(0.1,0.2)
		# get percent explored map which will be image or network input later on
		tunnelMap, frontierVector = explore.flood_fill_filter(percent_explored)
		frontierVectorData.append(frontierVector)
		global test_func
		test_func= frontierVector
		mask.set_map(tunnelMap, frontierVector)
		masking = mask.get_mask()
        #store all the elements
		#groundTruth = groundTruth * np.logical_not(mask.get_adaptive_mask(masking)) + tunnelMap
		#explore.set_occupancy_map(groundTruth)
		#groundTruth, _ = explore.flood_fill_filter(1.0)

		groundTruthData.append(np.float32(groundTruth))
		tunnelMapData.append(np.float32(tunnelMap))
		maskData.append(np.float32(mask.get_adaptive_mask(masking)))
		#adaptivemaskData.append(np.float32(mask.get_adaptive_mask(masking)))
		print(
        '\r[Generating Data {} of {}]'.format(
            i,
            int(ratio * totalData),
        ),
        end=''
        )
	print('')
	return groundTruthData, tunnelMapData, maskData, frontierVectorData

groundTruthData = {}
tunnelMapData = {}
maskData = {}
frontierVectorData = {}

groundTruthData["train"], tunnelMapData["train"], maskData["train"], frontierVectorData["train"]  = generate(trainRatio,totalData,"training")
groundTruthData["validation"], tunnelMapData["validation"], maskData["validation"], frontierVectorData["validation"] = generate(validRatio,totalData,"validation")
groundTruthData["test"], tunnelMapData["test"], maskData["test"], frontierVectorData["test"] = generate(testRatio,totalData,"testing")


a = sanityCheckMasknExplored(groundTruthData["train"], tunnelMapData["train"], maskData["train"])
b = sanityCheckMasknExplored(groundTruthData["validation"], tunnelMapData["validation"], maskData["validation"])
c = sanityCheckMasknExplored(groundTruthData["test"], tunnelMapData["test"], maskData["test"])

if a+b+c ==0 or a+b+c>=0:
	with open('{:s}/ground_truth_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
		pickle.dump(groundTruthData, handle)
	# Normalize Data
	#tunnelMapData["train"] = (tunnelMapData["train"]-np.mean(tunnelMapData["train"]))/ np.std(tunnelMapData["train"])
	#tunnelMapData["validation"] = (tunnelMapData["validation"]-np.mean(tunnelMapData["train"]))/ np.std(tunnelMapData["train"])
	#tunnelMapData["test"] = (tunnelMapData["test"]-np.mean(tunnelMapData["train"]))/ np.std(tunnelMapData["train"])
	with open('{:s}/image_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
	    pickle.dump(tunnelMapData, handle)
	with open('{:s}/mask_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
	    pickle.dump(maskData, handle)
	with open('{:s}/frontier_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
	    pickle.dump(frontierVectorData, handle)
else:
	print("SANITY CHECK FAILED- NOT SAVING ANY DATA")


fig = plt.figure(figsize=(10,10))
image, masking,gt = tunnelMapData['test'][0], maskData['test'][0] , groundTruthData['test'][0]
fig.add_subplot(2,2,1)
#pdb.set_trace()
plt.imshow(np.stack([gt,gt,gt],axis=-1))
plt.title("True image")
plt.ylabel('y')
plt.xlabel('x')

fig.add_subplot(2,2,2)
title = "70% explored Map"
plt.imshow(np.stack([image, image, image],axis=-1))
plt.title(title)
plt.ylabel('y')
plt.xlabel('x')

fig.add_subplot(2,2,3)
plt.imshow(np.stack([image, image, masking],axis=-1))
title = "Mask"
plt.title(title)
plt.ylabel('y')
plt.xlabel('x')

fig.add_subplot(2,2,4)
#for p in test_func:			# Why doing this things, comment or remove it.
#	image[p[0],p[1]] = 0.5
plt.imshow(abs(groundTruthData["test"][0] * maskData["test"][0] - tunnelMapData["test"][0]))
plt.title("explored with forntiers")
plt.ylabel('explored_y')
plt.xlabel('explored_x')
plt.show()
#plt.savefig("sample_map.png")