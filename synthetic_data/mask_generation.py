#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:22:18 2019

@author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
from x_map_gen import Exploration
import numpy as np
import matplotlib.pyplot as plt
import pdb

class Mask:
	"""Creates a mask over a explored map """
	def __init__(self):
		self.exploredMap = None
		self.frontierVector = None
		self.dirs_motion = [
		    lambda x, y: (x-1, y),  # up
		    lambda x, y: (x+1, y),  # down
		    lambda x, y: (x, y - 1),  # left
		    lambda x, y: (x, y + 1),  # right
		    # adding diagonal nbrs
		    #lambda x, y: (x+1, y+1), # down-right 
		    #lambda x, y: (x+1, y-1), # down-left
		    #lambda x, y: (x-1, y+1), # up-right
		    #lambda x, y: (x-1, y-1), # up-left
		]

	def set_map(self, exploredMap, frontierVector):
		self.exploredMap = exploredMap
		self.frontierVector = frontierVector

	def free_nbrs(self, point):
		nbrsVector = []
		for d in self.dirs_motion:
			nx, ny = d(point[0], point[1])
			if 0 <= nx < len(self.exploredMap) and 0 <= ny < len(self.exploredMap[0]):
				# only adding non occupied neighbor
				if self.exploredMap[nx, ny]==0:
					nbrsVector.append([nx, ny])
		return nbrsVector

	def nbrs(self, point):
		nbrsVector = []
		for d in self.dirs_motion:
			nx, ny = d(point[0], point[1])
			if 0 <= nx < len(self.exploredMap) and 0 <= ny < len(self.exploredMap[0]):
				nbrsVector.append([nx, ny])
		return nbrsVector

	def get_mask(self):
		mask = np.zeros((len(self.exploredMap),len(self.exploredMap[0])))
		for x in range(len(self.exploredMap)):
			for y in range(len(self.exploredMap[0])):
				if self.exploredMap[x,y]==1:
					mask[x,y] = 1
					if [x,y] in self.frontierVector:
						nbrsVector = []
					else:
						nbrsVector = self.free_nbrs([x,y])
					for p in nbrsVector:
						mask[p[0],p[1]] = 1

		for p in self.frontierVector:
			mask[p[0],p[1]] = 1
		return mask

	def get_adaptive_mask(self, predict_falsemask):
		predict_truemask = np.zeros((len(self.exploredMap),len(self.exploredMap[0])))
		depth = int(self.exploredMap.shape[0]/3) 
		#depth = 2
		# for f in self.frontierVector:
		# 	latentPoints = []
		# 	latentPoints.append(f)
		# 	for d in range(depth):
		# 		layer =[]
		# 		check_mat = np.zeros((len(self.exploredMap),len(self.exploredMap[0])))
		# 		while(len(latentPoints)>0):
		# 			latent = latentPoints.pop(0)
		# 			predict_truemask[latent[0],latent[1]] = 1
		# 			check_mat[latent[0],latent[1]] = 1
		# 			#pdb.set_trace()
		# 			nbrsVector = self.nbrs(latent)
		# 			#Append to layer 
		# 			for i in nbrsVector:
		# 				if check_mat[i[0],i[1]]==0:
		# 					layer.append(i)
		# 		latentPoints = layer

		for f in self.frontierVector:
			for x in range(-depth, depth+1):
				for y in range(-depth, depth+1):
					if 0 <= f[0]+x < len(self.exploredMap) and 0 <= f[1]+y < len(self.exploredMap[0]):
						predict_truemask[f[0]+x,f[1]+y] = 1
			#plt.imshow(predict_truemask)
			#plt.pause(0.001)
			#pdb.set_trace()
		# plt.imshow(predict_truemask)
		# plt.show()
		# plt.imshow()
		#plt.show()
		#pdb.set_trace()
		#m = np.ones((len(self.exploredMap),len(self.exploredMap[0]))) * np.bitwise_xor(predict_falsemask>0.5, predict_truemask>0.5)
		m = np.ones((len(self.exploredMap),len(self.exploredMap[0]))) * np.logical_not(np.logical_and(np.logical_not(predict_falsemask), predict_truemask))
		#for i in self.frontierVector:
		#	m[i[0],i[1]] = 0
		return m

		# startp = [0, int(self.gridDimension[1]/2)]

		# exploredMap = np.zeros((self.gridDimension[0],self.gridDimension[1]))
		# exploredMap[startp[0],startp[1]] = 1

		# latentPoints = []
		# latentPoints.append(startp)

		# while(len(latentPoints)>0):
		# 	latent = latentPoints.pop(0)
		# 	frontierVector = self.frontier(latent,exploredMap)
		# 	latentPoints = latentPoints + frontierVector
		# 	if (np.sum(exploredMap) > self.filterRatio * np.sum(self.occupancy_map)):
		# 		return exploredMap, latentPoints
		# return exploredMap, latentPoints

		# nbrsVector = []
		# for d in self.dirs_motion:
		# 	nx, ny = d(point[0], point[1])
		# 	if 0 <= nx < len(self.exploredMap) and 0 <= ny < len(self.exploredMap[0]):
		# 		# only adding non occupied neighbor
		# 		if self.exploredMap[nx, ny]==0:
		# 			nbrsVector.append([nx, ny])
		# return nbrsVector
		
