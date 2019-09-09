#!/usr/bin/env python3

""" Testing Module for loading the trained model

This module tests all the submodule and calls the visualizer

Author: Abhijeet Agnihotri
"""

import random
import atexit
from visualize import Visualize
from underground import Underground
from robot import Robot
from frontier2 import Frontier
import numpy as np
import pdb


GRID_SIZE = 32

def shutdown():
	print('\nGoodbye')


def tick(tunnel, wall_e, action):
	wall_e._give_action(action)
	state = wall_e._get_current_location()
	rewards_bool_list = tunnel._found_artifact(state)
	for i in rewards_bool_list:
		#wall_e._update_reward(tunnel._found_artifact(state))
		wall_e._update_reward(i)
	return state, rewards_bool_list


def main(value_dist, TUNNEL_FILE, ARTIFACT_FILE, neural_input, visualize=True):
	# atexit.register(shutdown)
	# Instantiate the environment
	tunnel = Underground(GRID_SIZE, TUNNEL_FILE, ARTIFACT_FILE, neural_input, value_dist)
	# print("Number of artifacts", len(tunnel._updated_artifact_locations))
	x_dim, y_dim = tunnel._x_dim, tunnel._y_dim

	steps = 0
	budget = 500

	# Store value at the start because it changes over time
	num_artifacts = len(tunnel._updated_artifact_locations)

	# Tracks time steps when rewards are found
	points_found = []
	score_list = np.ones(budget-1)

	# Introduce a robot, only one for now
	#if value_dist == 'closest':
	wall_e = Robot(x_dim, y_dim, neural_input,fidelity=tunnel._get_predicted_artifact_fidelity_map()) #, fidelity= tunnel._get_predicted_artifact_fidelity_map)

	# Initialize frontier
	frontier = Frontier()

	if visualize:
		# To visualize
		graph = Visualize(TUNNEL_FILE)
		graph._initialise_visualization(tunnel._get_predicted_artifact_locations())

	# Set current state
	state = wall_e._get_current_location()
	#print("state", state)
	# Get matrix of observed frontier values around wall-e and update observed map

	observation, freespace_indicator = tunnel._get_observation(state)
	wall_e.update_observed_map(observation, freespace_indicator, tunnel._observation_radius)
	
	# Runs until Wall-e runs out of budget or there are no frontiers left to explore
	try:
		while steps < budget and len(tunnel._updated_artifact_locations) > 0:
			# rest_halt = input("enter input")
			# Get matrix of observed frontier values around wall-e and update observed map
			observation, freespace_indicator = tunnel._get_observation(state)
			wall_e.update_observed_map(observation, freespace_indicator, tunnel._observation_radius)
			#pdb.set_trace()
			# print("real artifacts\t: {}".format(len(tunnel._updated_artifact_locations)))
			# print("predicted artifacts\t:{}".format(len(tunnel._updated_predicted_artifact_locations)))
			
			# Recursive Prediction
			if value_dist=='normal': 
				nth_prediction = 0 #int(steps/10)
				tunnel._recursive_predict(wall_e._observation_indicator, wall_e._frontiers_indicator, state, nth_prediction=nth_prediction)
				wall_e._recursive_prediction_update(tunnel._get_predicted_artifact_fidelity_map())

			if visualize:
				# Update visualization
				predicted_artifacts = np.zeros_like(tunnel._tunnel_map)
				for p in tunnel._get_predicted_artifact_locations():
					predicted_artifacts[p[1],p[0]] = 1 # Note the transpose action
				graph._keep_visualizing(state, tunnel._get_artifact_locations(), observation, wall_e._get_explored_map(), predicted_artifacts)

			# Pick the next frontier and get a path to that point
			path = frontier.get_next_frontier(state, wall_e._observation_indicator, tunnel._get_predicted_artifact_fidelity_map(), wall_e._frontiers_indicator, value_dist)
			#path = frontier.get_next_frontier(state, wall_e._observation_indicator, wall_e._frontiers, wall_e._frontiers_indicator, value_dist)

			# Loop through the path and update the robot at each step
			for point in path:
				# print("Next point", point)
				distance = abs(wall_e._get_current_location()[0] - point[0]) + abs(wall_e._get_current_location()[1] - point[1])

				# While loop continues to move robot until point has been reached
				while distance > 0:

					# Find allowed actions
					allowed_actions = tunnel._get_allowed_actions(state)
					# Get the action that will take the robot to the next point
					action = wall_e._next_action(point, allowed_actions)
					# Move robot and update world
					state, reward_bool = tick(tunnel, wall_e, action)
					steps += 1

					if steps >= budget:
						break

					# If a POI was found, mark the time step
					for i in reward_bool:
						if i:
							points_found.append(steps)

					# Normalizes to be percent of total artifacts found
					score_list[steps - 1] = len(points_found) / num_artifacts

					if visualize:
						predicted_artifacts = np.zeros_like(tunnel._tunnel_map)
						for p in tunnel._get_predicted_artifact_locations():
							predicted_artifacts[p[1],p[0]] = 1 # Note the transpose action
						graph._keep_visualizing(state, tunnel._get_artifact_locations(), observation,
												wall_e._get_explored_map(), predicted_artifacts)
					# Update the distance to the next point
					distance = abs(wall_e._get_current_location()[0] - point[0]) + abs(wall_e._get_current_location()[1] - point[1])

				if steps >= budget:
					break

	except ValueError:
		print("Wall-e has not captured all POIs, but has run out of frontiers")
	return steps, wall_e._reward, score_list, points_found


if __name__ == "__main__":

	tunnel_num = 30
	# available value functions: 'value', 'quarter', 'closest', 'sqrt', 'normal'
	value_dist = 'closest'
	TUNNEL_FILE = './maps_{}/tunnel_{}.npy'.format(GRID_SIZE, tunnel_num)
	ARTIFACT_FILE = './maps_{}/artifacts_{}.npy'.format(GRID_SIZE, tunnel_num)

	try:
		print('Started exploring\n')
		steps, reward, score_list, points_found = main(value_dist, TUNNEL_FILE, ARTIFACT_FILE, True)
		print("Tunnel {}".format(tunnel_num))
		print("Steps", steps)
		print("Reward", reward)
		print("POIs found", len(points_found))
	except (KeyboardInterrupt, SystemExit):
		raise


