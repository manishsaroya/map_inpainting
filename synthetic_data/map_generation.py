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
from matplotlib import rcParams
import matplotlib.patches as patches
import os
import pdb
######## Parameters for generating the database #############
GRID_SIZE = 24
numPOI = 1
trainRatio = 0.8
totalData = 10
validRatio = 0.1
testRatio = 0.1
#############################################################

######### Exploration Parameters #############
filterRatio = 0.7
data_dir = "./variable"
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
		percent_explored = np.random.uniform(0.2,0.8)
		# get percent explored map which will be image or network input later on
		tunnelMap, frontierVector = explore.flood_fill_filter(percent_explored)
		frontierVectorData.append(frontierVector)
		global test_func
		test_func= frontierVector
		mask.set_map(tunnelMap, frontierVector)
		masking = mask.get_mask()
        #store all the elements
		groundTruth = groundTruth * np.logical_not(mask.get_adaptive_mask(masking)) + tunnelMap
		explore.set_occupancy_map(groundTruth)
		groundTruth, _ = explore.flood_fill_filter(1.0)

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

if a+b+c ==0:
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
image, masking,gt = tunnelMapData['train'][0], maskData['train'][0] , groundTruthData['train'][0]
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
plt.imshow(abs(groundTruthData["train"][0] * maskData["train"][0] - tunnelMapData["train"][0]))
plt.title("explored with forntiers")
plt.ylabel('explored_y')
plt.xlabel('explored_x')
plt.show()
plt.pause(0.001)
#plt.savefig("sample_map.png")


# Awesome plotter
tunnel_map, updated_artifact_locations, explored_map, frontiers_indicator = groundTruthData['train'][0], \
	groundTruthData['train'][0] - tunnelMapData['train'][0], tunnelMapData['train'][0], frontierVectorData['train'][0]
######### Class constructor 
_y_dim, _x_dim = tunnel_map.shape
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
fig.canvas.set_window_title("Sub-T Simulator")

#### INIT ##########
plt.style.use('seaborn-dark')
plt.tight_layout()
ax.imshow(tunnel_map)#, cmap=plt.get_cmap('gist_gray'))
plt.ion()
#plt.show(fig)

#with open('./case_53_{:s}/step_{:d}_{:s}.pickle'.format(value_dist, steps, value_dist),'rb') as tf:
#	data = pickle.load(tf)


#_artifact_locations = updated_artifact_locations
ax.cla()
ax.imshow(tunnel_map) #cmap=plt.get_cmap('gist_gray'))

#plt.imshow(frontier_indicator-0.5)
#plt.pause(0.0001)

# robot start location
rect = patches.Rectangle((12 - 0.5, 0 - 0.5), 1, 1, linewidth=2, edgecolor='darkorange', hatch="", facecolor='red')
ax.add_patch(rect)


# Plot artifact locations
artifact_indices = np.nonzero(updated_artifact_locations)
artifactVector = []

for i in range(len(artifact_indices[0])):
	artifactVector.append([artifact_indices[0][i], artifact_indices[1][i]])
for artifact in artifactVector:
	rect = patches.Rectangle((artifact[1] - 0.5, artifact[0] - 0.5), 1, 1, linewidth=1.5, linestyle='--', joinstyle='round', edgecolor='black', hatch='', facecolor='white')
	#rect = patches.Patch((artifact[0] - 0.5, artifact[1] - 0.5), linewidth=0.001, edgecolor='y', hatch='/', facecolor='none')
	ax.add_patch(rect)


# Plot frontier locations
# f_indices = np.nonzero(frontiers_indicator)
# frontierVector = []
# for i in range(len(f_indices[0])):
# 	frontierVector.append([f_indices[0][i], f_indices[1][i]])
for f in frontiers_indicator: # frontierVector:
	rect = patches.Rectangle((f[1] - 0.5, f[0] - 0.5), 1, 1, linewidth=2, edgecolor='mediumorchid', hatch='*', facecolor='Orange')
	ax.add_patch(rect)
#pdb.set_trace()

plt.xticks([])
plt.yticks([])


#ax.invert_yaxis()
plt.savefig("partially_explored.eps")
plt.show(fig)
pdb.set_trace()

# Plot explored map
for y in range(_y_dim):
	for x in range(_x_dim):
		if explored_map[x][y]:
			rect = patches.Circle((x, y), 0.1, linewidth=1, edgecolor='b', facecolor='b')
			ax.add_patch(rect)

ax.plot()
# self.ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.draw()

plt.show(fig)
#plt.pause(.001)
print(b)
#pdb.set_trace()