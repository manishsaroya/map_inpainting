#!/usr/bin/env python3

""" Visualize Module

This module contains methods to visualize the underground tunnel system

Author: Abhijeet Agnihotri
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class Visualize:

	def __init__(self, tunnel_filename):
		self._tunnel_map = tunnel_filename
		self._y_dim, self._x_dim = self._tunnel_map.shape
		self.fig, self.ax = plt.subplots()
		self.fig.set_size_inches(10,10)
		self.fig.canvas.set_window_title("Sub-T Simulator")

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

	def _keep_visualizing(self, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map):
		# TODO: Adapt for multiple robots
		self._artifact_locations = updated_artifact_locations
		self.ax.cla()
		self.ax.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))
		observation_radius = len(current_observation[0])//2

		if len(self._artifact_locations) > 0:
			# Draw heat map
			plt.imshow(fidelity_map, cmap=plt.get_cmap('hot'), interpolation='nearest')

		# Plot observation of the robot
		for y in range(observation_radius*2 + 1):
			for x in range(observation_radius*2 + 1):
				_i_state = (robot_states[0] + x - observation_radius, robot_states[1] + y - observation_radius)
				if self._check_state_in_tunnel(_i_state):
					rect = patches.Rectangle((_i_state[0] - 0.5, _i_state[1] - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='g')
					self.ax.add_patch(rect)

		# Plot current robot locations
		rect = patches.Rectangle((robot_states[0] - 0.5, robot_states[1] - 0.5), 1, 1, linewidth=2, edgecolor='c', facecolor='m')
		self.ax.add_patch(rect)

		# Plot artifact locations
		for artifact in self._artifact_locations:
			rect = patches.Rectangle((artifact[0] - 0.5, artifact[1] - 0.5), 1, 1, linewidth=2, edgecolor='b', hatch='x', facecolor='none')
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
		plt.pause(.01)
