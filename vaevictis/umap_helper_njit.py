import sys
import numpy as np
import numba
import ipdb
from numba import jit
from scipy.optimize import curve_fit
MAX_VAL = np.log(sys.float_info.max) / 2.0

np.random.seed(0)

def find_ab_params(spread = 1., min_dist = 0.1) -> np.float64:
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params

@jit(nopython=True)
def dist_to_knn(dist) -> np.float64:
    k = dist.shape[0]
    newdist_and_knn = np.zeros((2*k,k))
    
    for i,d in enumerate(dist):
        newdist_and_knn[i,:] = d # not sorting dists for umap
        ids = np.argsort(d)
        newdist_and_knn[k+i,:] = ids

    return newdist_and_knn 

@jit(nopython=True)
def smooth_knn_dist(dist, max_iter=50, tol=1e-4) -> np.float64:
    # Compute the factors rho and sigma for each point in the batch
    k = dist.shape[0]
    target = np.log2(k) # change k to # of nearest neighbors if not all points of the batch are used
    rhos = np.zeros(k, dtype=np.float64)
    sigmas = np.zeros(k, dtype=np.float64)  

    for i in range(k):
        lo = -np.inf
        hi = np.inf
        mid = 1.0
        
        cur_d = dist[i,:]
        non_zero_d = cur_d[cur_d > 1e-7]
        rho_i = np.min(non_zero_d)
        rhos[i] = rho_i
        
        corr_d = cur_d - rho_i # d(xi, xij) - pi
        corr_d = corr_d * (corr_d > 1e-7) # max(d(xi, xij) - pi, 0), avoid floating point mistakes
        
        # Binary search for sigma_i
        S_i = np.sum(np.exp(-corr_d/mid))
        diff = abs(S_i - target)
        
        iter_i = 0
        while diff > tol and iter_i < max_iter:
            
            if(S_i > target):
                hi = mid
                if np.isfinite(lo):
                    mid = (lo+hi)/2.0
                else:
                    mid /= 2.0
                    
            else:
                lo = mid
                if np.isfinite(hi):
                    mid = (lo+hi)/2.0
                else:
                    mid *= 2.0
            S_i = np.sum(np.exp(-corr_d/mid))
            diff = abs(S_i - target)
            
            iter_i += 1
            
        sigmas[i] = mid
        
    rhos_and_sigmas = np.stack((rhos, sigmas), axis=0)
    
    return rhos_and_sigmas # 2D array rhos = rhos_and_sigmas[0,:]

@jit(nopython=True)
def compute_membership_strengths(knn_dists, knn_mat, rhos, sigmas, mix_ratio=1.) -> np.float64:
    
    n_samples = knn_mat.shape[0]
    n_neighbors = knn_mat.shape[1]    
    
    adjacency = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        neighs = knn_mat[i,:].astype(np.int64)
        for n in neighs:
            if i != n:
                adjacency[i,n] = 1 #equivalent to setting everything except current point to 1 if kneighs = batch_size
        
        for j in range(n_neighbors):
            if(adjacency[i,j] > 0):
                if not knn_mat[i,j] == i: # point must not be similar to itself                
                    corr_d = knn_dists[i,j] - rhos[i]
                    corr_d = corr_d * (corr_d > 1e-7)
                    
                    adjacency[i,j] = np.exp(-(corr_d / sigmas[i]))
                    
    tp = np.transpose(adjacency)
    prod = np.multiply(adjacency, tp)
    res = mix_ratio * (adjacency + tp - prod) + (1.0 - mix_ratio) * prod
    
    
    return res #undirected weighted graph describing the fuzzy topological representation of the dataset


@jit(nopython=True)
def simplicial_graph_from_dist(dist) -> np.float64:
    k = dist.shape[0]
    dist_and_knn = dist_to_knn(dist)
    dist = dist_and_knn[:k,:] 
    knn = dist_and_knn[k:,:]

    rhos_and_sigmas = smooth_knn_dist(dist)
    rhos = rhos_and_sigmas[0,:]
    sigmas = rhos_and_sigmas[1,:]
    
    G = compute_membership_strengths(dist, knn, rhos, sigmas)
    
    return G