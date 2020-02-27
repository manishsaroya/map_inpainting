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
numPOI = 5
#trainRatio = 0.8
totalData = 2
#validRatio = 0.1
testRatio = 1
#############################################################

######### Exploration Parameters #############
#filterRatio = 0.7
data_dir = "./mine_data_chilean_brunton2"
read_data = ['chilean.txt', 'bruceton2.txt']
##############################################

if not os.path.exists(data_dir):
	os.makedirs(data_dir)


explore = Exploration(GRID_SIZE, numPOI)
mask = Mask()
#gridDimension = [GRID_SIZE, GRID_SIZE]
test_func = None

def readFile(filename):
	"""Read the file and return the indices as list of lists."""
	#filename = 'bruceton.txt'
	with open(filename) as file:
		array2d = [[float(digit) for digit in line.split()] for line in file]
	#pdb.set_trace()
	array2d = np.flip(np.array(array2d), 0)
	return array2d

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
		#pdb.set_trace()
		#groundTruth = explore.generate_map()
		groundTruth = readFile(read_data[i]) #explore.generate_map()
		explore.set_occupancy_map(groundTruth)
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
		# adaptivemaskData.append(np.float32(mask.get_adaptive_mask(masking)))
		#pdb.set_trace()
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

#groundTruthData["train"], tunnelMapData["train"], maskData["train"], frontierVectorData["train"]  = generate(trainRatio,totalData,"training")
#groundTruthData["validation"], tunnelMapData["validation"], maskData["validation"], frontierVectorData["validation"] = generate(validRatio,totalData,"validation")
groundTruthData["test"], tunnelMapData["test"], maskData["test"], frontierVectorData["test"] = generate(testRatio,totalData,"testing")


a = 0 #sanityCheckMasknExplored(groundTruthData["train"], tunnelMapData["train"], maskData["train"])
b = 0 #sanityCheckMasknExplored(groundTruthData["validation"], tunnelMapData["validation"], maskData["validation"])
c = 0 #sanityCheckMasknExplored(groundTruthData["test"], tunnelMapData["test"], maskData["test"])

if a+b+c ==0:
	with open('{:s}/ground_truth_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
		pickle.dump(groundTruthData, handle)
	with open('{:s}/image_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
		pickle.dump(tunnelMapData, handle)
	with open('{:s}/mask_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
		pickle.dump(maskData, handle)
	with open('{:s}/frontier_dataset_{:d}.pickle'.format(data_dir, GRID_SIZE), 'wb') as handle:
		pickle.dump(frontierVectorData, handle)
else:
	print("SANITY CHECK FAILED- NOT SAVING ANY DATA")

# Awesome plotter

import sys,tty,termios

class _Getch:
	def __call__(self):
			fd = sys.stdin.fileno()
			old_settings = termios.tcgetattr(fd)
			try:
				tty.setraw(sys.stdin.fileno())
				ch = sys.stdin.read(3)
			finally:
				termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
			return ch

#### Note for matrix x is facing down and y is pointing righth
#### for patches y is down and x is pointing right.
# Need to swap y and x 
tunnel_map, updated_artifact_locations, explored_map, frontiers_indicator, final_mask = groundTruthData['test'][0], \
	groundTruthData['test'][0] - tunnelMapData['test'][0], tunnelMapData['test'][0], frontierVectorData['test'][0], maskData['test'][0]
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
display_map = 2

for p in range(display_map):
	tunnel_map, updated_artifact_locations, explored_map, frontiers_indicator, final_mask = groundTruthData['test'][p], \
	groundTruthData['test'][p] - tunnelMapData['test'][p], tunnelMapData['test'][p], frontierVectorData['test'][p], maskData['test'][p]
	ax.cla()
	ax.imshow(tunnel_map)

	##################### Plot mask ########
	mask_indices = np.nonzero(abs(final_mask - 1))
	maskVector = []
	for i in range(len(mask_indices[0])):
		maskVector.append([mask_indices[0][i], mask_indices[1][i]])
	#pdb.set_trace()
	
	for m in maskVector:
		rect = patches.Rectangle((m[1] - 0.5, m[0] - 0.5), 1, 1, linewidth=0.0001, edgecolor='lightgoldenrodyellow', hatch='', facecolor='lightgoldenrodyellow')
		ax.add_patch(rect)


	# robot start location
	rect = patches.Rectangle((GRID_SIZE//2 - 0.5, 0 - 0.5), 1, 1, linewidth=2, edgecolor='darkorange', hatch="", facecolor='red')
	ax.add_patch(rect)


	# Plot artifact locations
	artifact_indices = np.nonzero(updated_artifact_locations)
	artifactVector = []

	for i in range(len(artifact_indices[0])):
		artifactVector.append([artifact_indices[0][i], artifact_indices[1][i]])
	for artifact in artifactVector:
		rect = patches.Rectangle((artifact[1] - 0.5, artifact[0] - 0.5),1, 1,linewidth=1.5, linestyle='--', joinstyle='round', edgecolor='green', hatch='', facecolor='green')
		#rect = patches.Patch((artifact[0] - 0.5, artifact[1] - 0.5), linewidth=0.001, edgecolor='y', hatch='/', facecolor='none')
		ax.add_patch(rect)


	# Plot frontier locations
	for f in frontiers_indicator: # frontierVector:
		rect = patches.Rectangle((f[1] - 0.5, f[0] - 0.5), 1, 1, linewidth=2, edgecolor='mediumorchid', hatch='*', facecolor='Orange')
		ax.add_patch(rect)
	#pdb.set_trace()


	plt.xticks([])
	plt.yticks([])



	# To save in eps uncomment below lines.

	ax.invert_yaxis()
	plt.savefig("./output_maps/partially_explored{:d}.jpg".format(p))
	#plt.show(fig)
	plt.draw()
	plt.pause(0.2)
