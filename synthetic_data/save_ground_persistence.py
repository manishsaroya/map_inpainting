"""
This file save the ground truth persistence homology such that we don't have to
compute it while training.

Author: Manish Saroya
Contact: saroyam@orgonstate.edu

"""
import pickle 
import numpy as np 
import torch, torch.nn as nn
from topologylayer.nn import LevelSetLayer2D
import matplotlib.pyplot as plt
import pdb
def reduceinfo(info):
    r = []
    for i in info:
        if abs(i[0]) != np.inf and abs(i[1])!= np.inf and i[0]!=i[1]:
            r.append(i.detach().numpy())
    return r

class PersistenceDgm(nn.Module):
    def __init__(self, size):
        super(PersistenceDgm, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size, sublevel=False)

    def dgmplot(self, image):
        dgminfo = self.pdfn(image)
        return dgminfo

    def filtration(self, info):
        #pdb.set_trace()
        end, start = info[:,0], info[:,1]
        end_ = torch.where(torch.abs(end)!=np.inf, torch.ones(end.shape), torch.zeros(end.shape))
        start_ = torch.where(torch.abs(start)!=np.inf, torch.ones(start.shape), torch.zeros(start.shape))
        # remove infinite values
        index = torch.nonzero(end_ * start_)
        out = torch.index_select(info, 0, torch.squeeze(index))
        # remove the y=x line features
        end, start = out[:,0], out[:,1]
        index = torch.nonzero(end - start)
        out = torch.index_select(out,0,torch.squeeze(index))
        return out

    def generatePersistence(self, data, type_):
        p_z = []
        p_f = []
        print("Computing persistence for ", type_)
        for i in range(len(data)):
            ground_t = torch.tensor(data[i], dtype=torch.float, requires_grad=False)
            dgm = self.dgmplot(ground_t)
            #pdb.set_trace()
            #z = self.filtration(dgm[0][0])
            #f = self.filtration(dgm[0][1])
            p_z.append(dgm[0][0])
            p_f.append(dgm[0][1])
        	# z = np.asarray(reduceinfo(dgm[0][0]))
        	# f = np.asarray(reduceinfo(dgm[0][1]))
            # if i%100==0:
            #     pdb.set_trace()
            print(
            '\r[Generating persistence {} of {}]'.format(
                i,
                int(len(data)),
            ),
            end=''
            )
        return p_z, p_f


size = 24
with open('ground_truth_dataset_{}.pickle'.format(size),'rb') as tf:
	groundTruthData = pickle.load(tf)

pobj = PersistenceDgm((size,size))
#dgm = pobj.dgmplot(ground_t)

persistence_z = {}
persistence_f = {}
persistence_z["train"], persistence_f["train"] = pobj.generatePersistence(groundTruthData["train"], "training")
persistence_z["validation"], persistence_f["validation"] = pobj.generatePersistence(groundTruthData["validation"], "validation")
persistence_z["test"], persistence_f["test"] = pobj.generatePersistence(groundTruthData["test"], "test")

with open('ground_truth_dataset_peristenceDgm_z_{}.pickle'.format(size), 'wb') as handle:
	pickle.dump(persistence_z, handle)

with open('ground_truth_dataset_peristenceDgm_f_{}.pickle'.format(size), 'wb') as handle:
    pickle.dump(persistence_f, handle)
#pdb.set_trace()
