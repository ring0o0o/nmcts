import numpy as np
import itertools
from scipy.spatial import cKDTree

'''
        k : list of integer or integer
        The list of k-th nearest neighbors to return.
        If k is an integer it is treated as a list of [1, ... k] (range(1, k+1)).
        Note that the counting starts from 1.
'''
class gamma_test():
    def __init__(self,x,y):
        self.x = x.values
        self.y = y.values
    def get_mse(self):
        delta = np.zeros(p)
        gamma = np.zeros(p)
        xnik = np.array([])
        ynik = np.array([])
        p = 10
        tree = cKDTree(x)
        dist, ind = tree.query(x,k=p+1,p=2) #ind.shape()=M*p
        ind = np.delete(ind,0,axis=1)
        dist = np.delete(dist,0,axis=1)
        # xnik = np.zeros((M*k,p))
        ynik = np.zeros((M,p))
        # xnik = x[ind]
        for row in range(M):
            for col in range(p):
                ynik[row,col] = y[ind[row][col]]
        # for col in range(1,p):
        #     # tmp = x[ind[:,col],:] - x[col,:].reshape(1,-1)
        #     xnik = np.hstack([ xnik,x[ind[:,col]] ])
        #     # ynik = np.vstack([ynik,(y[ind[:,col]]- y[col])])
        #     ynik = np.vstack([ynik,(y[ind[col,:]])])
        for row in range(M):
            ynik[row,:] -=y[row]

        delta =np.sum(dist,axis=0)/M
        gamma = np.sum(ynik**2,axis=0)/(2*M)
        # assert dist**2 == xnik**2

        AandT = np.polyfit(delta,gamma,1)
        return AandT[1]



