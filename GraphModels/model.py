import numpy as np
from abc import ABC
from sklearn.datasets import make_sparse_spd_matrix
import networkx as nx
from scipy.stats import multivariate_normal


def corr_model_builder(cov, prec=None):
    D = np.diag(1 / np.sqrt(np.diag(cov)))
    corr = D @ cov @ D
    
    return corr, (corr >= 1e-6).astype(int) - np.eye(corr.shape[0])

def pcorr_model_builder(cov, prec=None):
    D = np.diag(1 / np.sqrt(np.diag(prec)))

    corr = -(D @ prec @ D)
    
    return corr, (corr != 0).astype(int) - np.eye(corr.shape[0])

class RandomGraphicalModel(ABC):
    def __init__(self, dim, density):
        self.dim = dim
        self.density = density
        self.precision = None
        self.covariance = None
        self.corr_model = None
        self.adj = None
        self.graph = None
        
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
        return tuple(metric(TP, TN, FP, FN) for metric in metrics)
        

class CholRandomModel(RandomGraphicalModel):
    def __init__(self, dim, density, model_builder, corr_func=np.corrcoef):
        super().__init__(dim, density, model_builder, corr_func)
        
        self.precision = make_sparse_spd_matrix(self.dim, alpha=self.density, norm_diag=True)
        self.covariance = np.linalg.inv(self.precision)
        
        self.corr_model, self.adj = self.model_builder(self.covariance, self.precision)
        self.graph = nx.from_numpy_array(self.adj)
        

class PathTestModel(RandomGraphicalModel):
      def __init__(self, dim, delta, model_builder, corr_func=np.corrcoef):
        super().__init__(dim, (dim - 1) / (dim * (dim - 1) / 2), model_builder, corr_func)
        
        self.precision = np.diag([delta for _ in range(dim - 1)], k=1) + np.diag([delta for _ in range(dim - 1)], k=-1) + np.eye(dim)
        self.covariance = np.linalg.inv(self.precision)
        
        self.corr_model, self.adj = self.model_builder(self.covariance, self.precision)
        self.graph = nx.from_numpy_array(self.adj)
        
        
class Chol