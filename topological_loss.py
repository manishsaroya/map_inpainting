from scipy.optimize import linear_sum_assignment
import torch, torch.nn as nn, numpy as np
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
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
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)

    def dgmplot(self, image):
        dgminfo = self.pdfn(image)
        #pdb.set_trace()
        z = np.asarray(reduceinfo(dgminfo[0][0]))
        f = np.asarray(reduceinfo(dgminfo[0][1]))
        return z, f


class TopLoss(nn.Module):
    def __init__(self, size, skip=1):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)
        self.pdfn_g = LevelSetLayer2D(size=size, sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=skip)
        self.topfn2 = SumBarcodeLengths(dim=0)

    def correspondence(self, reduced_dgminfo, reduced_dgminfo_g):
        ################### Hungarian algorithm ###############
        #creating a cost matrix with rows containing ground persistence and columns prediction persistence
        cost = []
        for i in reduced_dgminfo:
            row = []
            for j in reduced_dgminfo_g:
                row.append(torch.norm(i-j).detach().numpy())
            cost.append(row)
        return linear_sum_assignment(cost)
        #######################################################

    def filtration(self, info):
        end, start = info[:,0], info[:,1]
        end_ = torch.where(torch.abs(end)!=np.inf, end, torch.zeros(end.shape))
        start_ = torch.where(torch.abs(start)!=np.inf, start, torch.zeros(start.shape))
        # remove infinite values
        index = torch.nonzero(end_ * start_)
        out = torch.index_select(info, 0, torch.squeeze(index))
        # remove the y=x line features
        end, start = out[:,0], out[:,1]
        index = torch.nonzero(end - start)
        out = torch.index_select(out,0,torch.squeeze(index))
        return out

    def computeloss(self, dgminfohom, dgminfohom_g):

        reduced_dgminfo = self.filtration(dgminfohom)
        reduced_dgminfo_g = self.filtration(dgminfohom_g)
        ordered_ground_truth = torch.zeros(reduced_dgminfo.shape)

        # get correspondence between persistence points
        p_ind, g_ind = self.correspondence(reduced_dgminfo, reduced_dgminfo_g)
        # fill mean in all ground truth
        for i in range(reduced_dgminfo.shape[0]):
            ordered_ground_truth[i] = torch.cat((torch.unsqueeze(torch.mean(reduced_dgminfo[i]),0), torch.unsqueeze(torch.mean(reduced_dgminfo[i]),0)))
        for i in range(len(p_ind)):
            ordered_ground_truth[p_ind[i]] = reduced_dgminfo_g[g_ind[i]]
        #pdb.set_trace()
        final_loss = torch.norm(torch.reshape(reduced_dgminfo,(-1,)) - torch.reshape(ordered_ground_truth,(-1,)))
        return final_loss

    def forward(self, beta, ground):
        loss_ = torch.tensor([])
        for i in range(3):
            for j in range(beta.shape[1]):
                dgminfo = self.pdfn(beta[i][j])
                dgminfo_g = self.pdfn_g(ground[i][j])
                ############ Code starts ##########################
                zero_loss = self.computeloss(dgminfo[0][0],dgminfo_g[0][0])
                one_loss = self.computeloss(dgminfo[0][1],dgminfo_g[0][1])
                loss_ = torch.cat((loss_, torch.unsqueeze(zero_loss + one_loss,0)))
        #pdb.set_trace()
        return torch.mean(loss_) #zero_loss + one_loss #zero_loss #self.topfn(dgminfo) + self.topfn2(dgminfo)

        #dgminfo = self.pdfn(beta)
        #dgminfo_g = self.pdfn_g(ground)
        ############ Code starts ##########################
        #zero_loss = self.computeloss(dgminfo[0][0],dgminfo_g[0][0])
        #one_loss = self.computeloss(dgminfo[0][1],dgminfo_g[0][1])
        #return zero_loss + one_loss #zero_loss #self.topfn(dgminfo) + self.topfn2(dgminfo)