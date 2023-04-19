import numpy as np
from abc import ABC
from sklearn.datasets import make_sparse_spd_matrix
import networkx as nx
from scipy.stats import multivariate_normal


class RandomGraphicalModel(ABC):
    def __init__(self, dim, density, random_state=None):
        self.dim = dim
        self.density = density
        self.precision = None
        self.covariance = None
        self.corr_model = None
        self.adj = None
        self.graph = None
        self.random_state = random_state
        
    def sample(self, n, dist=multivariate_normal):
        return dist.rvs(np.zeros(self.dim), self.covariance, size=n)
    
    def confusion(self, other_G):
        p = other_G.edges
        t = self.graph.edges
        full = nx.complete_graph(len(other_G.nodes)).edges
        
        TP = len(p & t)
        TN = len((full - p) & (full - t))
        FP = len(p & (full - t))
        FN = len((full - p) & t)
        
        return TP, TN, FP, FN
    
    def apply_metrics(self, other_G, metrics):
        TP, TN, FP, FN = self.confusion(other_G)
        return np.array([metric(TP, TN, FP, FN) for metric in metrics])
    
    def get_density(self):
        return nx.density(self.graph)
        
        
class CholPCorrModel(RandomGraphicalModel):
    def __init__(self, dim, density, random_state=None):
        super().__init__(dim, density, random_state=None)
        
        self.precision = make_sparse_spd_matrix(self.dim, alpha=self.density, norm_diag=True, random_state=random_state)
        self.covariance = np.linalg.inv(self.precision)
        
        D = np.diag(1 / np.sqrt(np.diag(self.precision)))
        
        self.pcorr = -(D @ self.precision @ D)
        np.fill_diagonal(self.pcorr, 1)
        
        self.adj = (np.abs(self.pcorr) >= 1e-6).astype(int) - np.eye(self.dim)
        self.graph = nx.from_numpy_array(self.adj)
   
        
class CholCorrModel(RandomGraphicalModel):
    def __init__(self, dim, density, random_state=None):
        super().__init__(dim, density, random_state=None)
        
        self.covariance = make_sparse_spd_matrix(self.dim, alpha=self.density, norm_diag=True, random_state=self.random_state)
        self.precision = np.linalg.inv(self.covariance)
        
        D = np.diag(1 / np.sqrt(np.diag(self.covariance)))
        
        self.corr = (D @ self.covariance @ D)
        
        self.adj = (np.abs(self.corr) >= 1e-6).astype(int) - np.eye(self.dim)
        self.graph = nx.from_numpy_array(self.adj)