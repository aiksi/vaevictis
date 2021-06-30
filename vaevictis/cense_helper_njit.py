import sys
import numpy as np
import numba
from numba import jit
MAX_VAL = np.log(sys.float_info.max) / 2.0

np.random.seed(0)

@jit(nopython=True)
def dist_to_knn(dist) -> np.float64:
    k = dist.shape[0]
    newdist_and_knn = np.zeros((2*k,k))
    
    for i in range(k):
        ids = np.argsort(dist[i,:])
        newdist_and_knn[i,:] = dist[i,ids]
        newdist_and_knn[k+i,:] = ids

    return newdist_and_knn 

@jit(nopython=True)
def remove_asym(p, knn) -> np.float64:
    n = p.shape[0]
    k = p.shape[1]
    newp = np.zeros_like(p)
    for i in range(n):
        for j in range(k):
            neigh = int(knn[i,j])
            ind = np.where(knn[neigh,:]==i)[0]
            if ind.size != 0: # check if current point is also a neighbor of that point's current neighbor
                sym = (p[i,j] + p[neigh,ind[0]])/2
                newp[i,j] = sym
                newp[neigh, ind[0]] = sym
            else:
                newp[i,j] = 0
        
    return newp
