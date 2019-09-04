#!/usr/bin/env python3

""" Underground (environment) Module

This module initialises the main environment class

Author: Abhijeet Agnihotri, Manish Saroya
RESOLVE TODO
"""

import torch
from torchvision import transforms
import numpy 
import sys 
import matplotlib.pyplot as plt
import a_star
#print(sys.path)
sys.path.append('../')
sys.path.append("../synthetic_data/")
from mask_generation import Mask

#from neural_network.artifact_prediction_cuda import Net
from net import PConvUNet
import opt
from util.io import load_ckpt, get_state_dict_on_cpu
import copy
import pdb

class Underground:

	def __init__(self, grid_size, tunnel_filename, artifact_filename, neural_input, value_dist):
		# tunnel map is (y, x)
		# partially explored initial map and mask is neural input.
		self._neural_input = neural_input
		# tunnel map is the ground tunnel 
		self._tunnel_map = tunnel_filename  # TODO: Change the variable name, its not a file name now.
		self._observation_radius = 1
		self._grid_size = grid_size
		# list of artifacts predicted by neural network.
		self._updated_predicted_artifact_locations = []
		self._thresholding = 0.55

		# update  _updated_predicted_artifact_locations and stores the fidility map as transpose of network output
		#self._predict_artifact(self._neural_input)
		
		# TRANSPOSE
		self._updated_artifact_locations = [(x[1], x[0]) for x in artifact_filename.tolist()] # Transpose action on the artifact_locations
		# TRANSPOSE
		self._y_dim, self._x_dim = self._tunnel_map.shape
		#self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3, "down-right": 4, "down-left": 5, "up-right": 6, "up-left": 7}
		#self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
		#self._action_list = ["up", "right", "down", "left", "down-right", "down-left", "up-right", "up-left"]
		self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}  # Actions without a "none" option
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Actions without a "none" option
		self._action_list = ["up", "right", "down", "left"]
		self._artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)
		
		# fidility maps are created from scratch from zero matrix.
		self._update_artifact_fidelity_map()
		#self._update_predicted_artifact_fidelity_map()
		###############################
		ob = numpy.transpose(neural_input[0])  # Not transpose action
		ft = numpy.zeros((self._x_dim, self._y_dim))
		for p in neural_input[3]:
			ft[p[1],p[0]] = 1   # Note the transpose action
		self._recursive_predict(ob, ft, [int(self._x_dim/2), 0])

	def _recursive_predict(self, observation_indicator, frontiers_indicator, state, nth_prediction=0):
		# Tranpose action
		image = numpy.transpose(numpy.float32(observation_indicator))
		f_indices = numpy.nonzero(frontiers_indicator)
		frontierVector = []
		for i in range(len(f_indices[0])):
			frontierVector.append([f_indices[1][i], f_indices[0][i]])  # Transpose action
		mask = Mask()
		mask.set_map(image, frontierVector)
		masking = mask.get_adaptive_mask(mask.get_mask())
		#pdb.set_trace()
		self._neural_input[0] = numpy.float32(image)
		self._neural_input[1] = numpy.float32(masking)
		self._neural_input[3] = frontierVector

		self._updated_predicted_artifact_locations = []
		#pdb.set_trace()
		self._predict_artifact(self._neural_input, nth_prediction=nth_prediction)
		self._connected_fidelity_map(observation_indicator, frontiers_indicator, state)
		#self._update_predicted_artifact_fidelity_map()

	def run_network(self, model, dataset, device, if_save=False, nth_prediction=0):
	    """ Input:
	        model: network model.
	        dataset: image, mask, gt, frontiers
	        device: cuda or cpu

	        Output:
	        prediction: just the prediction, does not include partial input explored area. 
	    """
	    #pdb.set_trace()

	    #####################################################
	    image, mask, gt, frontierVector = dataset #zip(*[dataset])
	    image  = torch.as_tensor(image)
	    mask = torch.as_tensor(mask)
	    gt = torch.as_tensor(gt)

	    image = image.unsqueeze(0).unsqueeze(0)
	    mask = mask.unsqueeze(0).unsqueeze(0)
	    gt = gt.unsqueeze(0).unsqueeze(0)

	    #with torch.no_grad():
	    #    output, _ = model(image.to(device), mask.to(device))
	    #output = output.to(torch.device('cpu'))

	    #output_comp = mask * image + (1 - mask) * output

	    image = image[0][0]
	    gt = gt[0][0]
	    mask = mask[0][0]
	    #output = output[0][0]

	    #a = 1.0/(1.0 + numpy.exp(-output.numpy()))
	    #prediction = (a * abs(mask.numpy() - 1))
	    
	    ################## To be changed ###################
	    prediction = gt - image
	    for p in frontierVector:
	        prediction[p[0],p[1]] = 1.0
	    ####################################################
	    if if_save== True:
	    	self._save_output(gt,image,mask,output, frontierVector, prediction, nth_prediction=nth_prediction)
	    return prediction

	def _predict_artifact(self, dataset_val, nth_prediction=0):
		""" Input:
				dataset_val: image, mask, gt, frontiers
			Output: 
				modifies: _updated_predicted_artifact_locations
				modifies: _predicted_artifact_fidelity_map as transpose of network output
		"""
		device = torch.device('cpu')
		size = (self._grid_size, self._grid_size)

		model = PConvUNet(layer_size=3, input_channels=1).to(device)
		#pdb.set_trace()
		load_ckpt('../snapshots/toploss24variable/ckpt/500000.pth', [('model', model)])
		model.eval()
		# network output is just the prediction, does not include the input partial explored area. 
		self.network_output = self.run_network(model, dataset_val, device,if_save=False, nth_prediction=nth_prediction)

		_predicted_artifact_locations = []
		self._predicted_artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)
		for x in range(len(self.network_output)):
			for y in range(len(self.network_output[0])):
				if self.network_output[x][y] > self._thresholding:
					# TRANSPOSE			
					_predicted_artifact_locations.append((y,x))
					# TRANSPOSE
					# These intensities are not being used. TODO
					self._predicted_artifact_fidelity_map[y][x] += self.network_output[x][y]
		
		self._updated_predicted_artifact_locations = _predicted_artifact_locations[:]
		#pdb.set_trace()
		#plt.imshow(self._predicted_artifact_fidelity_map)
		#plt.pause(5.0)

	def _connected_fidelity_map(self, observation_indicator, frontiers_indicator, state):
		artifact_locs = self._updated_predicted_artifact_locations
		# append frontiers to artifacts as well.

		observ = numpy.zeros((self._x_dim, self._y_dim))
		for i in artifact_locs:
			observ[i[0]][i[1]] = 1
		observ = observ + frontiers_indicator

		frontier_indices = numpy.array(numpy.nonzero(frontiers_indicator))
		front_dist = []
		for i in range(frontier_indices.shape[1]):
			try:
				path = numpy.array(a_star.getPath(observation_indicator, state, frontier_indices[:, i]))
				front_dist.append(len(path)-1)
			except KeyError:
				print("Cannot find path to the goal frontier", frontier_indices[:, i])

		# note the shape should be quivalent to tunnel_map transpose TODO
		self._predicted_artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)
		for i in range(frontier_indices.shape[1]):
			for a in artifact_locs:
				try:
					path = numpy.array(a_star.getPath(observ, a, frontier_indices[:, i]))
					self._predicted_artifact_fidelity_map[frontier_indices[0, i]][frontier_indices[1, i]] +=  1.0 / (len(path)+front_dist[i])
				except KeyError:
					pass
					#print("Cannot find path to the goal frontier", frontier_indices[:, i])
		#pdb.set_trace()



	def _update_artifact_fidelity_map(self):
		"""	Input: None
			_artifact_fidelity_map: creates artifact fidility map based on the ground truth artifacts.
				As ground artifacts fidelity map will only exists on the true tunnel points it filters out the other locations fidelity. 
		"""
		self._artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)

		for artifact in self._updated_artifact_locations:
			self._add_artifact_fidelity(artifact[0], artifact[1])
		#print("max fidility value", numpy.max(self._artifact_fidelity_map))
		#pdb.set_trace()
		# filtering process
		#self._artifact_fidelity_map = numpy.multiply(self._artifact_fidelity_map, self._tunnel_map)

	def _update_predicted_artifact_fidelity_map(self):
		# update it corresponding to predicted map fidelity value
		self._predicted_artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)

		for artifact in self._updated_predicted_artifact_locations:
			self._add_predicted_artifact_fidelity(artifact[0], artifact[1])
		#pdb.set_trace()
		# unnecessary step
		#self._predicted_artifact_fidelity_map = numpy.multiply(self._predicted_artifact_fidelity_map, self._tunnel_map)

	def _add_predicted_artifact_fidelity(self, artifact_x, artifact_y):
		"""Input: artifact point
		modifies: _predicted_artifact_fidelity_map += 5/(euclidean_dist + 1)
		"""
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				# self._artifact_fidelity_map[y][x] += (self._x_dim + self._y_dim) - (numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				self._predicted_artifact_fidelity_map[y][x] += 5.0/(numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				#self._predicted_artifact_fidelity_map[y][x] += 5.0/(((y - artifact_y)**2 + (x - artifact_x)**2) + 1)



	def _add_artifact_fidelity(self, artifact_x, artifact_y):
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				# self._artifact_fidelity_map[y][x] += (self._x_dim + self._y_dim) - (numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				self._artifact_fidelity_map[y][x] += 5.0/(numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				#self._artifact_fidelity_map[y][x] += 5.0/(((y - artifact_y)**2 + (x - artifact_x)**2) + 1)

	def _check_state_in_tunnel(self, state):
		""" Input: State 
			Output: 1 if the state in within the ground truth else 0
		"""
		# state = (x, y)
		#pdb.set_trace()
		if state[0] < 0 or state[1] < 0 or state[0] >= self._x_dim or state[1] >= self._y_dim:
			return 0
		else:
			# TRANSPOSE action as state is transpose of _tunnel_map
			return self._tunnel_map[state[1]][state[0]]

	def _get_allowed_actions(self, state):
		""" Input: State
			Output: Allowed actions for the state based on the ground truth(_tunnel_map)
		"""
		allowed_actions = []
		for key, value in self._action_dict.items():
			new_state = [state[0] + self._action_coords[value][0], state[1] + self._action_coords[value][1]]
			if self._check_state_in_tunnel(new_state):
				allowed_actions.append(key)
		return allowed_actions

	def _found_artifact(self, state):
		""" Input: state of the robot
			Output: remove visited artifact from the predicted count. 
		"""
		if self._updated_predicted_artifact_locations.count(state):
			self._updated_predicted_artifact_locations.remove(state)
			#self._update_predicted_artifact_fidelity_map()
			# removing this for now but its good for visualization
		if self._updated_artifact_locations.count(state):
			self._updated_artifact_locations.remove(state)
			self._update_artifact_fidelity_map()
			# print('Artifact found at: {}'.format(state))
			return True
		else:
			return False

	def _get_observation(self, state):
		""" Input: (state): position of the robot. 
			return: (observation):matrix with artifact fidility value stored in free space other wise 0 for obstacles, 
								  matrix demension depend on observation radius.
								  Diagonal values are included in the observations.
		"""

		# state and current observation are (x, y)
		self._current_observation = numpy.zeros((self._observation_radius*2 + 1, self._observation_radius*2 + 1))
		self._freespace_indicator = numpy.zeros((self._observation_radius*2 + 1, self._observation_radius*2 + 1))
		# update current state i.e. the center
		self._current_observation[self._observation_radius][self._observation_radius] = self._predicted_artifact_fidelity_map[state[1]][state[0]]
		self._freespace_indicator[self._observation_radius][self._observation_radius] = 1

		for y in range(self._observation_radius*2 + 1):
			for x in range(self._observation_radius*2 + 1):
				_i_state = (state[0] + x - self._observation_radius, state[1] + y - self._observation_radius)
				if self._check_state_in_tunnel(_i_state):
					# Return the fidelity value at the observed points
					# TODO
					#self._current_observation[x][y] = self._predicted_artifact_fidelity_map[_i_state[1]][_i_state[0]]
					try:
						self._action_coords.index(((numpy.array(_i_state)-numpy.array(state))[0], (numpy.array(_i_state)-numpy.array(state))[1]))
						self._freespace_indicator[x][y] = 1
						self._current_observation[x][y] = self._predicted_artifact_fidelity_map[_i_state[1]][_i_state[0]]
					except ValueError:
						self._freespace_indicator[x][y] = 0
				else:
					self._current_observation[x][y] = 0
					self._freespace_indicator[x][y] = 0
		return self._current_observation, self._freespace_indicator

	def _save_output(self, gt, image, mask,output,frontierVector, prediction, nth_prediction=0):
		fig = plt.figure(figsize=(15,10))

		fig.add_subplot(2,3,1)
		plt.imshow(gt.numpy())
		plt.title("Ground Truth Map")

		fig.add_subplot(2,3,4)
		plt.imshow(numpy.stack([image.numpy(), image.numpy(), mask.numpy()],axis=-1))
		plt.title("Mask")

		fig.add_subplot(2,3,2)
		title = "x% explored Map"
		explored_map = copy.deepcopy(image)
		for i in frontierVector:
			explored_map[i[0]][i[1]] = 0.5
		plt.imshow(explored_map.numpy())
		plt.title(title)

		fig.add_subplot(2,3,3)
		plt.imshow(output.numpy())
		plt.title("Predicted Map")

		fig.add_subplot(2,3,5)
		plt.imshow(1.0/(1.0 + numpy.exp(-output.numpy())) > self._thresholding)
		print(output.numpy().max())
		print(output.numpy().min())
		plt.title("output image")
		plt.ylabel('output_y')
		plt.xlabel('output_x')

		fig.add_subplot(2,3,6)
		#a = 1.0/(1.0 + numpy.exp(-output.numpy()))
		#prediction = (a * abs(mask.numpy() - 1))
		plt.imshow(prediction > self._thresholding)
		plt.title("Predicted Map")

		#plt.show()
		plt.savefig("./r_prediction/all_images{:d}.png".format(nth_prediction))


	def _get_predicted_artifact_locations(self):
		return self._updated_predicted_artifact_locations

	def _get_artifact_locations(self):
		return self._updated_artifact_locations

	def _get_artifact_fidelity_map(self):
		return self._artifact_fidelity_map

	def _get_predicted_artifact_fidelity_map(self):
		return self._predicted_artifact_fidelity_map

