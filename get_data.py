import numpy as np
import pickle
def dataset(tp, grid_size):
	with open('ground_truth_dataset_{}.pickle'.format(grid_size),'rb') as tf:
		gt = pickle.load(tf)
	with open('mask_dataset_{}.pickle'.format(grid_size),'rb') as rf:
		masks = pickle.load(rf)
	with open('image_dataset_{}.pickle'.format(grid_size),'rb') as f:
		images = pickle.load(f)
	d = []
	if tp == 'train':
		for i  in range(len(gt['train'])):
			image = images['train'][i]
			mask = masks['train'][i]
			ground_truth = gt['train'][i]
			d.append([convert(image),convert(mask),convert(ground_truth)])

	elif tp == 'val':
		for i in range(len(gt['validation'])):
			image = images['validation'][i]
			mask = masks['validation'][i]
			ground_truth = gt['validation'][i]
			d.append([convert(image),convert(mask),convert(ground_truth)])
			

	elif tp == 'test':
		for i in range(len(gt['test'])):
			image = images['test'][i]
			mask = masks['test'][i]
			ground_truth = gt['test'][i]
			d.append([convert(image),convert(mask),convert(ground_truth)])
			
	return d
def convert(image):
	return np.array([image,image,image])
