import numpy as np
import pdb
import pickle
def dataset(tp, grid_size):
	with open('synthetic_data/variable/ground_truth_dataset_{}.pickle'.format(grid_size),'rb') as tf:
		gt = pickle.load(tf)
	with open('synthetic_data/variable/mask_dataset_{}.pickle'.format(grid_size),'rb') as rf:
		masks = pickle.load(rf)
	with open('synthetic_data/variable/image_dataset_{}.pickle'.format(grid_size),'rb') as f:
		images = pickle.load(f)
	with open('synthetic_data/variable/ground_truth_dataset_peristenceDgm_z_{}.pickle'.format(grid_size),'rb') as f:
		persistence_z = pickle.load(f)
	with open('synthetic_data/variable/ground_truth_dataset_peristenceDgm_f_{}.pickle'.format(grid_size),'rb') as f:
		persistence_f = pickle.load(f)
	d = []
	if tp == 'train':
		for i  in range(len(gt['train'])):
			image = images['train'][i]
			mask = masks['train'][i]
			ground_truth = gt['train'][i]
			z = persistence_z['train'][i]
			f = persistence_f['train'][i]
			#pdb.set_trace()
			d.append([image, mask, ground_truth, z, f])
			#d.append([convert(image),convert(mask),convert(ground_truth)])

	elif tp == 'val':
		for i in range(len(gt['validation'])):
			image = images['validation'][i]
			mask = masks['validation'][i]
			ground_truth = gt['validation'][i]
			z = persistence_z['validation'][i]
			f = persistence_f['validation'][i]
			d.append([image, mask, ground_truth, z, f])
			#d.append([convert(image),convert(mask),convert(ground_truth)])


	elif tp == 'test':
		for i in range(len(gt['test'])):
			image = images['test'][i]
			mask = masks['test'][i]
			ground_truth = gt['test'][i]
			z = persistence_z['test'][i]
			f = persistence_f['test'][i]
			d.append([image, mask, ground_truth, z, f])
			#d.append([convert(image),convert(mask),convert(ground_truth)])

	return d
def convert(image):
	return np.array([image,image,image])
