#!/usr/bin/env python3

"""
Frontier Algorithm to choose which direction to explore next

Author: Anna Nickelson
"""

import numpy as np
import a_star
from math import sqrt
import matplotlib.pyplot as plt
import pdb
# This file has a class, but there is no need for that class, just a function would work perfectly fine.
# To the best of my knowlege the get_next_frontier returns a path to the next best forntier based on some criterias,  
class Frontier:


    def get_next_frontier(self, current, observed_map, frontiers, frontiers_indicator, value_dist):
        """ Input: 
            current: current position of the robot.
            observed_map: Ideal- a np arrat which informs whether the position is explored or not, 0 indicates unexplored.(Needs a fix)
						  Current- has fidilidy nearby robots travel but once on initially explored. Check robot class to fix this. TODO
						  Non zero observation is considered as open space and astar path can be computed through it. 
            
            frontiers: a matrix with importance values on frontier location, zeros otherwise (fix this right now the frontiers only has value 1) TODO
            value_dist: type of measure method to choose a frontier. 

            Output: path to the best frontier according to the given measurement method, with random tie break.
        """
        frontier_values = []
        fronts = []
        paths = []
        #closest_frontier_value = []
        # Find coordinates of each frontier
        #frontier_indices = np.array(np.nonzero(frontiers))
        frontier_indices = np.array(np.nonzero(frontiers_indicator))

        # Creates a list of current values of the frontiers.
        # This gets updated each time because frontier values change
        for i in range(frontier_indices.shape[1]):
            # Find shortest path to the next frontier and the length of that path
            try:
                path = np.array(a_star.getPath(observed_map, current, frontier_indices[:, i]))
            except KeyError:
                print("Cannot find path to the goal frontier", frontier_indices[:, i])
                pdb.set_trace()
            path = np.flip(path, 0)
            path_length = np.shape(path)[0]

            # Append the frontier value at that point, scaled based on the cost to get there.
            # Current point is included in path, so length - 1
            #pdb.set_trace()
            if value_dist == 'normal':
                #fronts.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
                #print("path", path)
                #print("current position", current)
                #print("x,y of frontier:", frontier_indices[0,i], frontier_indices[1,i])
                #frontier_values.append(100 - (path_length - 1))
                frontier_values.append((frontiers[frontier_indices[0, i]][frontier_indices[1, i]]+0.5) / (path_length-1))
                #closest_frontier_value.append(100 - (path_length - 1))
                # print("Scaled frontier:", frontiers[frontier_indices[0, i]][frontier_indices[1, i]] / (path_length-1))
            elif value_dist == 'sqrt':
                #fronts.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
                frontier_values.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]] / sqrt(path_length-1))
            elif value_dist == 'value':
                #fronts.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
                frontier_values.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
            elif value_dist == 'quarter':
                #fronts.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
                frontier_values.append((frontiers[frontier_indices[0, i]][frontier_indices[1, i]]+0.5) / (path_length-1)) # TODO: change back to 1/4
            elif value_dist == 'closest':
                #fronts.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
                frontier_values.append(100 - (path_length - 1))
            # Keep track of all paths to all frontiers
            paths.append(path)

        # Finds the largest value in the frontier values and returns the index
        # print("Frontiers:", fronts)
        # print("Frontier Values scaled", frontier_values)
        # random tie break
        #print("frontiers")
        #pdb.set_trace()
        
        # if value_dist=='normal':
        # 	prediction_value = []
        # 	if frontier_values.count(np.array(frontier_values).max())>=2:
        # 		for i in np.flatnonzero(np.array(frontier_values) == np.array(frontier_values).max()):
        # 			prediction_value.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]])
        # 		#pdb.set_trace()
        # 		choice_index = np.flatnonzero(np.array(frontier_values) == np.array(frontier_values).max())[prediction_value.index(np.amax(prediction_value))]
        # 	else:
        # 		choice_index = frontier_values.index(np.amax(frontier_values))
        # elif value_dist=='closest':
        # 	choice_index = np.random.choice(np.flatnonzero(np.array(frontier_values) == np.array(frontier_values).max()))
        #pdb.set_trace()
        #if np.array(frontier_values).max()!=0:
        choice_index = np.random.choice(np.flatnonzero(np.array(frontier_values) == np.array(frontier_values).max()))
        #else:
        #	choice_index = np.random.choice(np.flatnonzero(np.array(closest_frontier_value) == np.array(closest_frontier_value).max()))
        #choice_index = frontier_values.index(np.amax(frontier_values))
        #pdb.set_trace()
        # print("Robot Position:", current)
        # print("Chosen path", paths[choice_index])

        return paths[choice_index]
