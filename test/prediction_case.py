import pickle
import pdb
import numpy as np
import sys
sys.path.append("../synthetic_data/")
from mask_generation import Mask

from matplotlib import rcParams

from matplotlib import pyplot as plt
import matplotlib.patches as patches
rcParams['legend.handlelength'] = 2
rcParams['legend.handleheight'] = 2



steps = 22
value_dist = "normal"
#data = [tunnel_map, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map,frontiers_indicator]
with open('./case_53_{:s}/step_{:d}_{:s}.pickle'.format(value_dist, steps, value_dist),'rb') as tf:
		data = pickle.load(tf)

#pdb.set_trace()

tunnel_map, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map,frontiers_indicator = data

######### Class constructor 
_y_dim, _x_dim = tunnel_map.shape
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
fig.canvas.set_window_title("Sub-T Simulator")
_action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def _check_state_in_tunnel(state):
	# state = (x, y)
	if state[0] < 0 or state[1] < 0 or state[0] >= _x_dim or state[1] >= _y_dim:
		return 0
	else:
		return tunnel_map[state[1]][state[0]]


#### INIT ##########
plt.style.use('seaborn-dark')
plt.tight_layout()
ax.imshow(explored_map)#tunnel_map)#, cmap=plt.get_cmap('gist_gray'))
plt.ion()
#plt.show(fig)

with open('./case_53_{:s}/step_{:d}_{:s}.pickle'.format(value_dist, steps, value_dist),'rb') as tf:
	data = pickle.load(tf)

tunnel_map, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map,frontiers_indicator = data
_artifact_locations = updated_artifact_locations
ax.cla()
ax.imshow(np.transpose(explored_map))#tunnel_map) #cmap=plt.get_cmap('gist_gray'))

#plt.imshow(frontier_indicator-0.5)
#plt.pause(0.0001)
observation_radius = len(current_observation[0])//2
#pdb.set_trace()

################## Uncomment for adding predictions in the image
#if value_dist=="normal": Dont bother about it now
# if len(_artifact_locations) > 0:
# 	# Draw heat map
# 	plt.imshow(fidelity_map, cmap=plt.get_cmap('gist_gray'), interpolation='nearest')

################################



################## Uncomment to see the observation of the robot
# Plot observation of the robot
# for y in range(observation_radius*2 + 1):
# 	for x in range(observation_radius*2 + 1):
# 		_i_state = (robot_states[0] + x - observation_radius, robot_states[1] + y - observation_radius)
# 		if _check_state_in_tunnel(_i_state):
# 			try:
# 				_action_coords.index(((np.array(_i_state)-np.array(robot_states))[0], (np.array(_i_state)-np.array(robot_states))[1]))
# 				rect = patches.Rectangle((_i_state[0] - 0.5, _i_state[1] - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='g')
# 				ax.add_patch(rect)
# 			except ValueError:
# 				pass
################## Observation ##################################


# robot start location
rect = patches.Rectangle((12 - 0.5, 0 - 0.5), 1, 1, linewidth=2, edgecolor='darkorange', hatch="", facecolor='red')
ax.add_patch(rect)


# Plot artifact locations   ############### GROUND TRUTH MADE AS BACK GROUND ############
# for artifact in _artifact_locations:
# 	rect = patches.Rectangle((artifact[0] - 0.5, artifact[1] - 0.5), 1, 1, linewidth=2, joinstyle='round', edgecolor='white', hatch='', facecolor='white')
# 	#rect = patches.Patch((artifact[0] - 0.5, artifact[1] - 0.5), linewidth=0.001, edgecolor='y', hatch='/', facecolor='none')
# 	ax.add_patch(rect)


# Plot frontier locations
f_indices = np.nonzero(frontiers_indicator)
frontierVector = []
for i in range(len(f_indices[0])):
	frontierVector.append([f_indices[0][i], f_indices[1][i]])

#################### Plot  Frontiers ##############################
for f in frontierVector:
	rect = patches.Rectangle((f[0] - 0.5, f[1] - 0.5), 1, 1, linewidth=2, edgecolor='mediumorchid', hatch='*', facecolor='Orange')
	ax.add_patch(rect)
###################################################################

################## MASK GENERATION ##############################

mask = Mask()
mask.set_map(explored_map, frontierVector)
masking = mask.get_mask()
final_mask = np.float32(mask.get_adaptive_mask(masking))
mask_indices = np.nonzero(abs(final_mask - 1))
maskVector = []
for i in range(len(mask_indices[0])):
	maskVector.append([mask_indices[0][i], mask_indices[1][i]])

##################### Plot mask ########
for m in maskVector:
	rect = patches.Rectangle((m[0] - 0.5, m[1] - 0.5), 1, 1, linewidth=0.0001, edgecolor='lightgoldenrodyellow', hatch='', facecolor='lightgoldenrodyellow')
	ax.add_patch(rect)


##################################################################



################# Plot Robot locations
rect = patches.Rectangle((robot_states[0] - 0.5, robot_states[1] - 0.5), 1, 1, linewidth=4, edgecolor='b',hatch="", facecolor='deepskyblue', label="robot")
ax.add_patch(rect)
cir = patches.Circle((robot_states[0], robot_states[1]), 0.1, linewidth=1, edgecolor='b', facecolor='mediumorchid', label="robot")
ax.add_patch(cir)
plt.xticks([])
plt.yticks([])
################################################

################# Plot PREDICTIONS #############
prediction_indices = np.nonzero(fidelity_map)

predictionVector = []
for i in range(len(prediction_indices[0])):
	predictionVector.append([prediction_indices[1][i], prediction_indices[0][i]])
for a in predictionVector:
	rect = patches.Rectangle((a[0] - 0.5, a[1] - 0.5), 1, 1,linewidth=1.5, linestyle='--', joinstyle='round', edgecolor='green', hatch='', facecolor='lightgoldenrodyellow') # linewidth=1.5, edgecolor='mediumorchid', hatch='', facecolor='grey')
	#linewidth=1.5, linestyle='--', joinstyle='round', edgecolor='black', hatch='', facecolor='white'
	ax.add_patch(rect)



#################################################

#plt.legend(handles=[rect])
ax.invert_yaxis()
plt.savefig("network_predictions.eps")
plt.show(fig)
#pdb.set_trace()


########################## GARBAGE ################
# Plot explored map
for y in range(_y_dim):
	for x in range(_x_dim):
		if explored_map[x][y]:
			rect = patches.Circle((x, y), 0.1, linewidth=1, edgecolor='b', facecolor='b')
			ax.add_patch(rect)

ax.plot()
# self.ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.draw()

plt.show(fig)
#plt.pause(.001)
#print(b)
#pdb.set_trace()















# for b in range(0,500):
# 	with open('./case_53_{:s}/step_{:d}_{:s}.pickle'.format(value_dist, b, value_dist),'rb') as tf:
# 		data = pickle.load(tf)

# 	tunnel_map, robot_states, updated_artifact_locations, current_observation, explored_map, fidelity_map,frontiers_indicator = data
# 	_artifact_locations = updated_artifact_locations
# 	ax.cla()
# 	ax.imshow(tunnel_map, cmap=plt.get_cmap('gist_gray'))

# 	#plt.imshow(frontier_indicator-0.5)
# 	#plt.pause(0.0001)
# 	observation_radius = len(current_observation[0])//2
# 	#pdb.set_trace()
# 	#if value_dist=="normal":
# 	if len(_artifact_locations) > 0:
# 		# Draw heat map
# 		plt.imshow(fidelity_map, cmap=plt.get_cmap('gist_gray'), interpolation='nearest')

# 	# Plot observation of the robot
# 	for y in range(observation_radius*2 + 1):
# 		for x in range(observation_radius*2 + 1):
# 			_i_state = (robot_states[0] + x - observation_radius, robot_states[1] + y - observation_radius)
# 			if _check_state_in_tunnel(_i_state):
# 				try:
# 					_action_coords.index(((np.array(_i_state)-np.array(robot_states))[0], (np.array(_i_state)-np.array(robot_states))[1]))
# 					rect = patches.Rectangle((_i_state[0] - 0.5, _i_state[1] - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='g')
# 					ax.add_patch(rect)
# 				except ValueError:
# 					pass

# 	# Plot current robot locations
# 	rect = patches.Rectangle((robot_states[0] - 0.5, robot_states[1] - 0.5), 1, 1, linewidth=2, edgecolor='c', facecolor='m')
# 	ax.add_patch(rect)

# 	# Plot artifact locations
# 	for artifact in _artifact_locations:
# 		rect = patches.Rectangle((artifact[0] - 0.5, artifact[1] - 0.5), 1, 1, linewidth=2, edgecolor='b', hatch='x', facecolor='none')
# 		ax.add_patch(rect)

# 	# Plot frontier locations
# 	f_indices = np.nonzero(frontiers_indicator)
# 	frontierVector = []
# 	for i in range(len(f_indices[0])):
# 		frontierVector.append([f_indices[0][i], f_indices[1][i]])
# 	for f in frontierVector:
# 		rect = patches.Rectangle((f[0] - 0.5, f[1] - 0.5), 1, 1, linewidth=2, edgecolor='b', hatch='*', facecolor='Orange')
# 		ax.add_patch(rect)

# 	# Plot explored map
# 	for y in range(_y_dim):
# 		for x in range(_x_dim):
# 			if explored_map[x][y]:
# 				rect = patches.Circle((x, y), 0.1, linewidth=1, edgecolor='b', facecolor='b')
# 				ax.add_patch(rect)

# 	ax.plot()
# 	# self.ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
# 	plt.draw()
# 	#plt.show(fig)
# 	plt.pause(.001)
# 	print(b)
# 	#pdb.set_trace()