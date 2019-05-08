# Run testing.py
# Return location of each POI found

import testing_NN
import csv
from numpy import concatenate
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
def shutdown():
    print('\nGoodbye')

def get_prediction_and_ground_truth(grid_size):
    with open('ground_truth_dataset_{}.pickle'.format(grid_size),'rb') as tf:
        gt = pickle.load(tf)
    with open('mask_dataset_{}.pickle'.format(grid_size),'rb') as rf:
        masks = pickle.load(rf)

    true_prediction = []
    ground_truth = []
    for i in range(len(gt['test'])):
        true_prediction.append(gt['test'][i] * abs(masks['test'][i] - 1))
        ground_truth.append(gt['test'][i])

    return true_prediction, ground_truth

def test_dataset(grid_size):
    with open('ground_truth_dataset_{}.pickle'.format(grid_size),'rb') as tf:
        gt = pickle.load(tf)
    with open('mask_dataset_{}.pickle'.format(grid_size),'rb') as rf:
        masks = pickle.load(rf)
    with open('image_dataset_{}.pickle'.format(grid_size),'rb') as f:
        images = pickle.load(f)
    d = []
    for i in range(len(gt['test'])):
        image = images['test'][i]
        mask = masks['test'][i]
        ground_truth = gt['test'][i]
        d.append([convert(image),convert(mask),convert(ground_truth)])
    return d

def convert(image):
    return np.array([image,image,image])

if __name__ == "__main__":

    grid_size = 32 # 16
    num_tunnel_files = 20
    #value_distance = ['value', 'quarter', 'closest', 'sqrt', 'normal']
    value_distance = ['sqrt', 'closest', 'normal']
    #value_distance = ['normal']
    visualize = True
    true_prediction, ground_truth = get_prediction_and_ground_truth(grid_size)
    network_input = test_dataset(grid_size)

    try:
        print('Started exploring\n')
        with open('experiments_all_cumulative_score_{}.csv'.format(grid_size), mode='w') as experiments:
            experiment_writer = csv.writer(experiments, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, num_tunnel_files):
                # print('')
                print("##################")
                print("Tunnel {}".format(i))
                #if i not in [0, 1, 7, 9, 10, 14, 16, 17]:
                if i!=2:
                    print("skipping tunnel")
                else:
                    tunnel_file = ground_truth[i] #'./maps_{}/tunnel_{}.npy'.format(grid_size, i)
                    artifact_file = []
                    for x in range(len(true_prediction[i])):
                        for y in range(len(true_prediction[i][0])):
                            if true_prediction[i][x][y] ==1:
                                artifact_file.append([x,y])
                    #artifact_file = list(true_prediction[i]) #'./maps_{}/artifacts_{}.npy'.format(grid_size, i)
                    artifact_file = np.array(artifact_file)
                    #print(artifact_file)
                    time.sleep(5) 
                    for e in value_distance:
                        print("Value", e)
                        steps, reward, score_list, points_found = testing_NN.main(e, tunnel_file, artifact_file, network_input[i], visualize)
                        # print("Steps", steps)
                        # print("Reward", reward)
                        # print("POIs found", len(points_found))
                        to_write = concatenate([['tunnel_{}'.format(i)], ['method_{}'.format(e)], [steps], [reward], [sum(score_list)], score_list])
                        experiment_writer.writerow(to_write)

    except (KeyboardInterrupt, SystemExit):
        raise

