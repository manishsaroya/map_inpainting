#!/usr/bin/env python3

""" Visualize Module

This module contains methods to visualize the underground tunnel system

Author: Abhijeet Agnihotri
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pdb
import pickle


class Visualize:

	def __init__(self, tunnel_filename):
		self._tunnel_map = tunnel_filename
		self._y_dim, self._x_dim = self._tunnel_map.shape
		self.fig, self.ax = plt.subplots()
		self.fig.set_size_inches(10,10)
		self.fig.canvas.set_window_title("Sub-T Simulator")
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]

	def _initialise_visualization(self, artifact_locations):
		self._artifact_locations = artifact_locations
		plt.tight_layout()
		self.ax.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))
		plt.ion()
		plt.show()

	def _check_state_in_tunnel(self, state):
		# state = (x, y)
		if state[0] < 0 or state[1] < 0 or state[0] >= self._x_dim or state[1] >= self._y_dim:
			return 0
		else:
			return self._tunnel_map[state[1]][state[0]]

	def _keep_visualizing(self, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map,frontiers_indicator, steps, value_dist):
		############# Dump some data to pickle ###############
		# data = [self._tunnel_map, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map,frontiers_indicator, steps]
		# with open('./case_53/step_{:d}_{:s}.pickle'.format(steps, value_dist), 'wb') as handle:
		# 	pickle.dump(data, handle)


		# TODO: Adapt for multiple robots
		self._artifact_locations = updated_artifact_locations
		self.ax.cla()
		self.ax.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))

		#plt.imshow(frontier_indicator-0.5)
		#plt.pause(0.0001)
		observation_radius = len(current_observation[0])//2
		#pdb.set_trace()

		if len(self._artifact_locations) > 0:
			# Draw heat map
			plt.imshow(fidelity_map, cmap=plt.get_cmap('hot'), interpolation='nearest')

		# Plot observation of the robot
		for y in range(observation_radius*2 + 1):
			for x in range(observation_radius*2 + 1):
				_i_state = (robot_states[0] + x - observation_radius, robot_states[1] + y - observation_radius)
				if self._check_state_in_tunnel(_i_state):
					try:
						self._action_coords.index(((np.array(_i_state)-np.array(robot_states))[0], (np.array(_i_state)-np.array(robot_states))[1]))
						rect = patches.Rectangle((_i_state[0] - 0.5, _i_state[1] - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='g')
						self.ax.add_patch(rect)
					except ValueError:
						pass

		# Plot current robot locations
		rect = patches.Rectangle((robot_states[0] - 0.5, robot_states[1] - 0.5), 1, 1, linewidth=2, edgecolor='c', facecolor='m')
		self.ax.add_patch(rect)

		# Plot artifact locations
		for artifact in self._artifact_locations:
			rect = patches.Rectangle((artifact[0] - 0.5, artifact[1] - 0.5), 1, 1, linewidth=2, edgecolor='b', hatch='x', facecolor='none')
			self.ax.add_patch(rect)

		# Plot frontier locations
		f_indices = np.nonzero(frontiers_indicator)
		frontierVector = []
		for i in range(len(f_indices[0])):
			frontierVector.append([f_indices[0][i], f_indices[1][i]])
		for f in frontierVector:
			rect = patches.Rectangle((f[0] - 0.5, f[1] - 0.5), 1, 1, linewidth=2, edgecolor='b', hatch='*', facecolor='Orange')
			self.ax.add_patch(rect)

		# Plot explored map
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				if explored_map[x][y]:
					rect = patches.Circle((x, y), 0.1, linewidth=1, edgecolor='b', facecolor='b')
					self.ax.add_patch(rect)

		self.ax.plot()
		# self.ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
		plt.draw()
		plt.pause(.0001)
