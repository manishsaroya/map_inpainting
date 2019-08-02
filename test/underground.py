#!/usr/bin/env python3

""" Underground (environment) Module

This module initialises the main environment class

Author: Abhijeet Agnihotri
"""

import torch
from torchvision import transforms
import numpy 
import sys 
import matplotlib.pyplot as plt
print(sys.path)
sys.path.append('../')
#from neural_network.artifact_prediction_cuda import Net
from net import PConvUNet
import opt
from util.io import load_ckpt
import pdb

class Underground:

	def __init__(self, grid_size, tunnel_filename, artifact_filename, neural_input):
		# tunnel map is (y, x)
		self._neural_input = neural_input
		self._tunnel_map = tunnel_filename  # TODO: Change the variable name, its not a file name now.
		self._observation_radius = 1
		self._grid_size = grid_size
		# artifact locations = (x, y)
		self._predict_artifact(self._tunnel_map, self._neural_input)
		self._artifact_locations = [(x[1], x[0]) for x in artifact_filename.tolist()]
		self._updated_artifact_locations = self._artifact_locations[:]
		self._y_dim, self._x_dim = self._tunnel_map.shape
		# self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		# self._action_coords = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
		self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}  # Actions without a "none" option
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Actions without a "none" option
		self._artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)
		self._update_artifact_fidelity_map()
		self._update_predicted_artifact_fidelity_map()
		# This helps debug. Only need to run once per tunnel map
		# np.savetxt("artifact_fidelity.csv", self._artifact_fidelity_map, delimiter=',')

	def run_network(self, model, dataset, device, filename,if_save=False):


        #print("Inside evaluate..." ," and filename is",filename)
	    image, mask, gt = zip(*[dataset])
        #print(dataset[0])
	    image  = torch.as_tensor(image)
	    mask = torch.as_tensor(mask)
	    gt = torch.as_tensor(gt)
	    #pdb.set_trace()
	    image = image.unsqueeze(0)
	    mask = mask.unsqueeze(0)
	    gt = gt.unsqueeze(0)

	    #image = torch.stack(image)

	    #mask = torch.stack(mask)
	    #gt = torch.stack(gt)

	    with torch.no_grad():
	        output, _ = model(image.to(device), mask.to(device))
	    output = output.to(torch.device('cpu'))

	    output_comp = mask * image + (1 - mask) * output

	    image = image[0][0]
	    gt = gt[0][0]
	    mask = mask[0][0]
	    output = output[0][0]
	    #pdb.set_trace()
	    a = 1.0/(1.0 + numpy.exp(-output.numpy()))
	    prediction = (a * abs(mask.numpy() - 1))
	    #plt.show(plt.imshow(prediction))
	    if if_save== True:
	        fig = plt.figure(figsize=(15,10))

	        fig.add_subplot(2,3,1)
	        plt.imshow(gt.numpy())
	        plt.title("Ground Truth Map")
	        #plt.ylabel('y')
	        #plt.xlabel('x')


	        fig.add_subplot(2,3,4)
	        plt.imshow(numpy.stack([image.numpy(), image.numpy(), mask.numpy()],axis=-1))
	        plt.title("Mask")
	        #plt.ylabel('mask_y')
	        #plt.xlabel('mask_x')


	        fig.add_subplot(2,3,2)
	        title = "70% explored Map"
	        plt.imshow(image.numpy())
	        plt.title(title)
	        #plt.ylabel('masked_y')
	        #plt.xlabel('masked_x')
	        plt.savefig("figure_8.png")

	        fig.add_subplot(2,3,3)
	        plt.imshow(output.numpy())
	        plt.title("Predicted Map")
	        #plt.ylabel('output_y')
	        #plt.xlabel('output_x')

	        fig.add_subplot(2,3,5)
	        plt.imshow(1.0/(1.0 + numpy.exp(-output.numpy())) > 0.55)
	        print(output.numpy().max())
	        print(output.numpy().min())
	        #plt.imshow(numpy.stack([output.numpy(), image.numpy(), mask.numpy()],axis=-1),cmap='hot')
	        plt.title("output image")
	        plt.ylabel('output_y')
	        plt.xlabel('output_x')

	        fig.add_subplot(2,3,6)
	        #a = 1.0/(1.0 + numpy.exp(-output.numpy()))
	        #prediction = (a * abs(mask.numpy() - 1))
	        plt.imshow(prediction > 0.55)
	        plt.title("Predicted Map")

	        plt.show()
	        plt.savefig("all_images.png")
	    return prediction

	def _predict_artifact(self, image, data_set):
		device = torch.device('cuda')
		size = (self._grid_size, self._grid_size)
		img_transform = transforms.Compose(
		    [transforms.Resize(size=size), transforms.ToTensor(),
		     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
		mask_transform = transforms.Compose(
		    [transforms.Resize(size=size), transforms.ToTensor()])

		#dataset_val = Places2(args.root, img_transform, mask_transform, 'val')
		#dataset_val = torch.tensor(dataset('test',args.image_size))
		dataset_val = data_set
		model = PConvUNet(layer_size=3, input_channels=1).to(device)
		load_ckpt('../snapshots/toploss32test/ckpt/500000.pth', [('model', model)])
		#model.load_state_dict(torch.load('../snapshots/toploss32test/ckpt/500000.pth', map_location='cuda'))
		#model.load_state_dict(torch.load('mapinpainting_10000.pth'))
		model.eval()
		network_output = self.run_network(model, dataset_val, device, 'resulttoploss.jpg',False)

		#network_output = network_output.reshape(self._grid_size, self._grid_size).detach().numpy()
		self._predicted_artifact_locations = []
		self._predicted_artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)
		for x in range(len(network_output)):
			for y in range(len(network_output[0])):
				if network_output[x][y] > 0.55:
					self._predicted_artifact_locations.append((y,x))
					self._predicted_artifact_fidelity_map[y][x] += network_output[x][y]
		self._updated_predicted_artifact_locations = self._predicted_artifact_locations[:]
		#plt.imshow(self._predicted_artifact_fidelity_map)
		#plt.pause(5.0)

	def _get_predicted_artifact_locations(self):
		return self._predicted_artifact_locations

	def _get_artifact_locations(self):
		return self._updated_artifact_locations

	def _get_artifact_fidelity_map(self):
		return self._artifact_fidelity_map

	def _get_predicted_artifact_fidelity_map(self):
		return self._predicted_artifact_fidelity_map

	def _update_artifact_fidelity_map(self):
		self._artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)

		for artifact in self._updated_artifact_locations:
			self._add_artifact_fidelity(artifact[0], artifact[1])
		self._artifact_fidelity_map = numpy.multiply(self._artifact_fidelity_map, self._tunnel_map)

	def _update_predicted_artifact_fidelity_map(self):
		self._predicted_artifact_fidelity_map = numpy.zeros_like(self._tunnel_map)

		for artifact in self._updated_predicted_artifact_locations:
			self._add_predicted_artifact_fidelity(artifact[0], artifact[1])
		self._predicted_artifact_fidelity_map = numpy.multiply(self._predicted_artifact_fidelity_map, self._tunnel_map)

	def _add_predicted_artifact_fidelity(self, artifact_x, artifact_y):
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				# self._artifact_fidelity_map[y][x] += (self._x_dim + self._y_dim) - (numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				self._predicted_artifact_fidelity_map[y][x] += 5.0/(numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)

	# def _add_artifact_fidelity_2(self, artifact_x, artifact_y):
	# 	for y in range(self._y_dim):
	# 		for x in range(self._x_dim):
	# 			self._artifact_fidelity_map[y][x] += 5.0/(numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
	# 			if x == artifact_x and y == artifact_y:
	# 				self._artifact_fidelity_map[y][x] += 5
	# 			elif numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) <= 1:
	# 				# print("Square Root Term", numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2))
	# 				# print("artifact x y", artifact_x, artifact_y)
	# 				# print('x y', x, y)
	# 				self._artifact_fidelity_map += 0.5

	def _add_artifact_fidelity(self, artifact_x, artifact_y):
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				# self._artifact_fidelity_map[y][x] += (self._x_dim + self._y_dim) - (numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				self._artifact_fidelity_map[y][x] += 5.0/(numpy.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)

	def _check_state_in_tunnel(self, state):
		# state = (x, y)
		if state[0] < 0 or state[1] < 0 or state[0] >= self._x_dim or state[1] >= self._y_dim:
			return 0
		else:
			return self._tunnel_map[state[1]][state[0]]

	def _get_allowed_actions(self, state):
		allowed_actions = []
		for key, value in self._action_dict.items():
			new_state = [state[0] + self._action_coords[value][0], state[1] + self._action_coords[value][1]]
			if self._check_state_in_tunnel(new_state):
				allowed_actions.append(key)
				# print('action: {} and new_state: {} and key: {}'.format(self._check_state_in_tunnel(new_state), new_state, key))
		# print('allowed action: {} and state {}'.format(allowed_actions, state))
		return allowed_actions

	def _found_artifact(self, state):
		if self._updated_predicted_artifact_locations.count(state):
			self._updated_predicted_artifact_locations.remove(state)
			self._update_predicted_artifact_fidelity_map()
		if self._updated_artifact_locations.count(state):
			self._updated_artifact_locations.remove(state)
			self._update_artifact_fidelity_map()
			# print('Artifact found at: {}'.format(state))
			return True
		else:
			return False

	def _get_observation(self, state):
		# state and current observation are (x, y)
		self._current_observation = numpy.zeros((self._observation_radius*2 + 1, self._observation_radius*2 + 1))
		for y in range(self._observation_radius*2 + 1):
			for x in range(self._observation_radius*2 + 1):
				_i_state = (state[0] + x - self._observation_radius, state[1] + y - self._observation_radius)
				if self._check_state_in_tunnel(_i_state):
					# Return the fidelity value at the observed points
					self._current_observation[x][y] = self._artifact_fidelity_map[_i_state[1]][_i_state[0]]
				else:
					self._current_observation[x][y] = 0
		return self._current_observation

