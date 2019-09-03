#!/usr/bin/env python3

""" Robot Module

This module initializes the robot class

Author: Abhijeet Agnihotri
"""
# Please not that the simulator has x and y swaped as compared to map generation module.


import numpy as np
import pdb

class Robot:

	def __init__(self, x_dim, y_dim, neural_input, fidelity=None):
		self._x_dim = x_dim
		self._y_dim = y_dim
		self._tunnel_grid = np.zeros((self._x_dim, self._y_dim))
		# initializing explored map.
		#pdb.set_trace()
		# matrix containing fidility value at frontiers 
		self._frontiers = np.zeros_like(self._tunnel_grid)
		self._frontiers_indicator = np.zeros_like(self._tunnel_grid)
		old_version = False
		if old_version:
			self._explored_map = np.zeros_like(self._tunnel_grid)
			self._observed_map = np.zeros_like(self._tunnel_grid)
			self._observation_indicator = np.zeros_like(self._tunnel_grid)
		else:
			for p in neural_input[3]:
				self._frontiers[p[1],p[0]] = 1   # Note the transpose action
				self._frontiers_indicator[p[1],p[0]] = 1
			self._explored_map = np.transpose(neural_input[0]) - self._frontiers_indicator #np.zeros_like(self._tunnel_grid)
			self._frontiers = self._frontiers_indicator * fidelity
			#self._frontiers = self._frontiers * fidelity
			# Keeps track of fidelity values in spaces that have been observed but not explored
			self._observed_map = np.transpose(neural_input[0]) #np.zeros_like(self._tunnel_grid)
			self._observation_indicator = np.transpose(neural_input[0])
		# Definition of entry point can be changed subject to map generation
		# Note: state = (x,y)
		self._current_position = [int(self._x_dim/2), 0]
		self._update_explored_map()
		# Actions without a "none" option
		#self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3, "down-right": 4, "down-left": 5, "up-right": 6, "up-left": 7}
		#self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
		#self._action_list = ["up", "right", "down", "left", "down-right", "down-left", "up-right", "up-left"]

		self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]
		self._action_list = ["up", "right", "down", "left"] 
		self._reward = 0

	def _recursive_prediction_update(self,fidelity):
		#pdb.set_trace()
		self._frontiers = self._frontiers_indicator * fidelity
		#self._frontiers = (self._frontiers > 0) * fidelity

	def _get_current_location(self):
		# returns (x, y)
		return self._current_position

	def _next_action(self, goal, allowed_actions):
		""" Input: 
				goal, allowed_actions
			Output: 
				select action according to goal from allowed_actions
				# NOTE DOES NOT take in account the diagonal actions  
		"""
		#pdb.set_trace()
		try:
			action = self._action_coords.index(((goal-self._current_position)[0], (goal-self._current_position)[1]))
			return self._action_list[action]
		except ValueError:
			# This breaks if there are no valid actions
			print("Allowed actions", allowed_actions)
			print("Robot position", self._current_position)
			print("Goal:", goal)
			print("SOMETHING HAS GONE TERRIBLY WRONG")
			return False

		# if self._current_position[0] < goal[0] and "right" in allowed_actions:
		# 	return "right"
		# elif self._current_position[0] > goal[0] and "left" in allowed_actions:
		# 	return "left"
		# elif self._current_position[1] < goal[1] and "down" in allowed_actions:
		# 	return "down"
		# elif self._current_position[1] > goal[1] and "up" in allowed_actions:
		# 	return "up"
		# else:
		# 	# This breaks if there are no valid actions
		# 	print("Allowed actions", allowed_actions)
		# 	print("Robot position", self._current_position)
		# 	print("Goal:", goal)
		# 	print("SOMETHING HAS GONE TERRIBLY WRONG")
		# 	return False

	def _give_action(self, action):
		# print(action)
		new_state = (self._current_position[0] + self._action_coords[self._action_dict[action]][0], self._current_position[1] + self._action_coords[self._action_dict[action]][1])
		self._update_location(new_state)

	def _get_explored_map(self):
		return self._explored_map

	def _get_total_reward(self):
		return self._reward

	def _update_explored_map(self):
		""" if the current position is not previously explored, add current position to explored map and remove frontier from current position 
		"""
		if self._explored_map[self._current_position[0], self._current_position[1]] == 0:
			self._explored_map[self._current_position[0], self._current_position[1]] = 1
			# Remove explored cells from the frontiers map
			self._frontiers[self._current_position[0], self._current_position[1]] = 0
			self._frontiers_indicator[self._current_position[0], self._current_position[1]] = 0

	# Observed map keeps track of the observed areas
	# Frontier map keeps track of areas that have been observed but not explored
	def update_observed_map(self, observation, freespace_indicator, radius):
		state = self._get_current_location()
		tunnel_grid_size = self._tunnel_grid.shape
		for y in range(radius * 2 + 1):
			# y coordinate of state in loop
			state_y = state[1] + y - radius

			# If out of bounds, continue to next loop
			if state_y < 0 or state_y >= tunnel_grid_size[1]:
				continue

			for x in range(radius * 2 + 1):
				# Similar logic for x coordinate
				state_x = state[0] + x - radius

				if state_x < 0 or state_x >= tunnel_grid_size[0]:
					continue

				# print("Observed Map", self._observed_map.shape)
				# print("State x, state y", state_x, state_y)
				self._observed_map[state_x][state_y] = observation[x][y]
				if freespace_indicator[x][y]==1:
					self._observation_indicator[state_x][state_y] = 1
				# If the robot has not yet explored that area, add it to the observation map and frontiers map
				if self._explored_map[state_x][state_y] == 0:
					self._frontiers[state_x][state_y] = observation[x][y]
					if freespace_indicator[x][y]==1:
						self._frontiers_indicator[state_x][state_y] = 1
						#try:
						#	self._action_coords.index(((np.array([state_x, state_y])-np.array(state))[0], (np.array([state_x, state_y])-np.array(state))[1]))
						
						#except ValueError:
						#	self._frontiers_indicator[state_x][state_y] = 0

	def _update_location(self, state):
		self._current_position = state
		self._update_explored_map()

	def _update_reward(self, found_artifact):
		if found_artifact:
			self._reward += 100
			return True
		else:
			self._reward -= 1
			return False
