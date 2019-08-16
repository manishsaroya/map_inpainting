#!/usr/bin/env python3

""" Robot Module

This module initializes the robot class

Author: Abhijeet Agnihotri
"""
# Please not that the simulator has x and y swaped as compared to map generation module.


import numpy as np
import pdb

class Robot:

	def __init__(self, x_dim, y_dim, neural_input):
		self._x_dim = x_dim
		self._y_dim = y_dim
		self._tunnel_grid = np.zeros((self._x_dim, self._y_dim))
		# initializing explored map.
		pdb.set_trace()
		self._explored_map = np.transpose(neural_input[0]) #np.zeros_like(self._tunnel_grid)
		# Keeps track of fidelity values in spaces that have been observed but not explored
		self._observed_map = np.zeros_like(self._tunnel_grid)
		self._frontiers = np.zeros_like(self._tunnel_grid)
		for p in neural_input[3]:
			self._frontiers[p[1],p[0]] = 1   # Note the transpose action

		# Definition of entry point can be changed subject to map generation
		# Note: state = (x,y)
		self._entry_point = [int(self._x_dim/2), 0]
		self._current_position = self._entry_point
		self._update_explored_map()
		# self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		# self._action_coords = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
		# Actions without a "none" option
		self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]
		self._reward = 0

	def _get_current_location(self):
		# returns (x, y)
		return self._current_position

	def _next_action(self, goal, allowed_actions):
		if self._current_position[0] < goal[0] and "right" in allowed_actions:
			return "right"
		elif self._current_position[0] > goal[0] and "left" in allowed_actions:
			return "left"
		elif self._current_position[1] < goal[1] and "down" in allowed_actions:
			return "down"
		elif self._current_position[1] > goal[1] and "up" in allowed_actions:
			return "up"
		else:
			# This breaks if there are no valid actions
			print("Allowed actions", allowed_actions)
			print("Robot position", self._current_position)
			print("Goal:", goal)
			print("SOMETHING HAS GONE TERRIBLY WRONG")
			return False

	def _give_action(self, action):
		# print(action)
		new_state = (self._current_position[0] + self._action_coords[self._action_dict[action]][0], self._current_position[1] + self._action_coords[self._action_dict[action]][1])
		self._update_location(new_state)

	def _get_explored_map(self):
		return self._explored_map

	def _get_total_reward(self):
		return self._reward

	def _update_explored_map(self):
		if self._explored_map[self._current_position[0], self._current_position[1]] == 0:
			self._explored_map[self._current_position[0], self._current_position[1]] = 1
			# Remove explored cells from the frontiers map
			self._frontiers[self._current_position[0], self._current_position[1]] = 0

	# Observed map keeps track of the observed areas
	# Frontier map keeps track of areas that have been observed but not explored
	def update_observed_map(self, observation, radius):
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
				# If the robot has not yet explored that area, add it to the observation map and frontiers map
				if self._explored_map[state_x][state_y] == 0:
					self._frontiers[state_x][state_y] = observation[x][y]

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
