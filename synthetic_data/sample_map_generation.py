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
import os
import datetime
import numpy
import pdb

######## Parameters for generating the database #############
GRID_SIZE = 24
numPOI = 36
trainRatio = 0.8
totalData = 30
file_path = "./sample/"+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'/'
#############################################################

######### Exploration Parameters #############
filterRatio = 0.7
##############################################

explore = Exploration(GRID_SIZE, numPOI, filterRatio)
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

groundTruthData["sample"], tunnelMapData["sample"], maskData["sample"]  = generate(trainRatio,totalData,"sample")
if not os.path.exists(file_path):
    os.makedirs('{:s}'.format(file_path))

sanityCheckMasknExplored(groundTruthData["sample"], tunnelMapData["sample"], maskData["sample"])

for i in range(len(groundTruthData['sample'])):

	fig = plt.figure(figsize=(10,10))
	image, masking,gt = tunnelMapData['sample'][i], maskData['sample'][i] , groundTruthData['sample'][i]
	fig.add_subplot(2,2,1)
	#pdb.set_trace()
	plt.imshow(numpy.stack([gt,gt,gt],axis=-1))
	plt.title("True image")
	plt.ylabel('y')
	plt.xlabel('x')

	fig.add_subplot(2,2,2)
	title = "70% explored Map"
	plt.imshow(numpy.stack([image, image, image],axis=-1))
	plt.title(title)
	plt.ylabel('y')
	plt.xlabel('x')

	fig.add_subplot(2,2,3)
	plt.imshow(numpy.stack([image, image, masking],axis=-1))
	title = "Mask"
	plt.title(title)
	plt.ylabel('y')
	plt.xlabel('x')

	fig.add_subplot(2,2,4)
	#for p in test_func:			# Why doing this things, comment or remove it.
	#	image[p[0],p[1]] = 0.5
	plt.imshow(abs(groundTruthData["sample"][i] * maskData["sample"][i] - tunnelMapData["sample"][i]))
	plt.title("explored with forntiers")
	plt.ylabel('explored_y')
	plt.xlabel('explored_x')
	#plt.show()
	plt.savefig(file_path+"sample_{}.png".format(i))