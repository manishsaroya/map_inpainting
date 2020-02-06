#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:32:42 2019

@author: manish
"""

from flood_fill import getTiles
import numpy as np
#import matplotlib.pyplot as plt

class Exploration:
	"""Filters X% exploration map form completely known map
	   with appropriate flood fill based masking"""

	def __init__(self, gridSize, numPOI):
		self.gridSize = gridSize
		self.gridDimension = [self.gridSize, self.gridSize]
		self.numPOI = numPOI
		self.occupancy_map = None

	def generate_map(self):
		feature_map, self.occupancy_map = getTiles(self.gridDimension, self.numPOI)
		return self.occupancy_map

	def set_occupancy_map(self, map_):
		self.occupancy_map = map_

	def frontier(self, point, exploredMap):
		dirs_motion = [
		    lambda x, y: (x-1, y),  # up
		    lambda x, y: (x+1, y),  # down
		    lambda x, y: (x, y - 1),  # left
		    lambda x, y: (x, y + 1),  # right
		    # adding diagonal nbrs
		    # lambda x, y: (x+1, y+1), # down-right 
		    # lambda x, y: (x+1, y-1), # down-left
		    # lambda x, y: (x-1, y+1), # up-right
		    # lambda x, y: (x-1, y-1), # up-left
		]
		frontierVector = []
		for d in dirs_motion:
			nx, ny = d(point[0], point[1])
			if 0 <= nx < len(self.occupancy_map) and 0 <= ny < len(self.occupancy_map[0]):
				if self.occupancy_map[nx, ny] == 1 and exploredMap[nx, ny]==0:
					exploredMap[nx, ny] = 1
					frontierVector.append([nx, ny])
		return frontierVector

	def flood_fill_filter(self, filterRatio=0.7):
		self.filterRatio = filterRatio
		# adapted from map generation procedure
		startp = [0, int(self.gridDimension[1]/2)]

		exploredMap = np.zeros((self.gridDimension[0],self.gridDimension[1]))
		exploredMap[startp[0],startp[1]] = 1

		latentPoints = []
		latentPoints.append(startp)

		while(len(latentPoints)>0):
			latent = latentPoints.pop(0)
			frontierVector = self.frontier(latent,exploredMap)
			latentPoints = latentPoints + frontierVector
			if (np.sum(exploredMap) > self.filterRatio * np.sum(self.occupancy_map)):
				return exploredMap, latentPoints
		return exploredMap, latentPoints


########## Exploration Parameters #############
#GRID_SIZE = 32
#numPOI = 20
#filterRatio = 0.7
###############################################
#
#explore = Exploration(GRID_SIZE, numPOI, filterRatio)
#explore.generate_map()
#map, frontierVector = explore.flood_fill_filter()
#for p in frontierVector:
#	map[p[0],p[1]] = 0.5
#print map
#plt.imshow(map)
