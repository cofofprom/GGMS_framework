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

    def sample(self, n, dist=multivariate_normal, **kwargs):
        return dist.rvs(np.zeros(self.dim), self.covariance, size=n, **kwargs)

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

        self.precision = make_sparse_spd_matrix(self.dim,
                                                alpha=self.density,
                                                norm_diag=True,
                                                random_state=random_state)
        self.covariance = np.linalg.inv(self.precision)
        D = np.diag(1 / np.sqrt(np.diag(self.precision)))

        self.pcorr = -(D @ self.precision @ D)
        np.fill_diagonal(self.pcorr, 1)

        self.adj = (np.abs(self.pcorr) >= 1e-6).astype(int) - np.eye(self.dim)
        self.graph = nx.from_numpy_array(self.adj)


class CholCorrModel(RandomGraphicalModel):
    def __init__(self, dim, density, random_state=None):
        super().__init__(dim, density, random_state=None)

        rs = self.random_state

        self.covariance = make_sparse_spd_matrix(self.dim,
                                                 alpha=self.density,
                                                 norm_diag=True,
                                                 random_state=rs)
        self.precision = np.linalg.inv(self.covariance)

        D = np.diag(1 / np.sqrt(np.diag(self.covariance)))

        self.corr = (D @ self.covariance @ D)

        self.adj = (np.abs(self.corr) >= 1e-6).astype(int) - np.eye(self.dim)
        self.graph = nx.from_numpy_array(self.adj)


class CholTauModel(RandomGraphicalModel):
    def __init__(self, dim, density, random_state=None):
        super().__init__(dim, density, random_state=None)

        rs = self.random_state

        self.covariance = make_sparse_spd_matrix(self.dim,
                                                 alpha=self.density,
                                                 norm_diag=True,
                                                 random_state=rs)
        self.covariance = np.sin(self.covariance * np.pi / 2)
        self.corr = self.covariance
        self.precision = np.linalg.inv(self.covariance)

        self.adj = (np.abs(self.corr) >= 1e-6).astype(int) - np.eye(self.dim)
        self.graph = nx.from_numpy_array(self.adj)

def generate_dom_diag(dim, density):
    graph = nx.gnp_random_graph(dim, density)
    adj = nx.adjacency_matrix(graph).toarray()
    
    A = np.random.uniform(0.5, 1, size=(dim, dim))
    B = np.random.choice([-1, 1], size=(dim, dim))

    covar = adj * A * B
    rowsums = np.sum(np.abs(covar), axis=1)
    rowsums[rowsums == 0] = 0.0001
    covar = covar / (1.5 * rowsums[:, None])
    covar = (covar + covar.T) / 2 + np.eye(dim)

    invA = np.linalg.inv(covar)
    D = np.diag(1 / np.sqrt(np.diag(invA)))

    covariance = D @ invA @ D
    precision = np.linalg.inv(covariance)

    pD = np.diag(1 / np.sqrt(np.diag(precision)))

    pcorr = -(pD @ precision @ pD)
    np.fill_diagonal(pcorr, 1)

    return graph, adj, covariance, precision, pcorr

class DiagDominantPcorrModel(RandomGraphicalModel):
    def __init__(self, dim, density, random_state=None):
        super().__init__(dim, density, random_state=None)

        # Generate random Erdos-Renyi graph with given probability of an edge (density)
        self.graph = nx.gnp_random_graph(self.dim, self.density)
        # Transform the graph to adjacency matrix
        self.adj = nx.adjacency_matrix(self.graph).toarray()
        
        # Generate random matrix A where elements are uniformly distributed
        A = np.random.uniform(0.5, 1, size=(self.dim, self.dim))

        # Generate random matrix B where elements are randomly selected from the set {-1, 1}
        B = np.random.choice([-1, 1], size=(self.dim, self.dim))

        # Find the matrix prec where elements are uniformly distributed on (-1, 0.5] AND [0.5, 1)
        # and if there's no edge between nodes i and j the value will be 0
        # multiplication here means element-wise multiplication
        prec = self.adj * A * B
        # Find the sum of absolute values of each row (at this point diagonal contains only zeros)
        rowsums = np.sum(np.abs(prec), axis=1)
        rowsums[rowsums == 0] = 0.0001 # handles the situation where row is completely empty

        # Scale each row by 1.5 times of corresponding row sum
        prec = prec / (1.5 * rowsums[:, None])
        # Make matrix symmetric and add ones on the diagonal
        prec = (prec + prec.T) / 2 + np.eye(self.dim)

        self.precision = prec
        # Find the covariance as matrix inverse of precision (since precision is SPD)
        self.covariance = np.linalg.inv(self.precision)

        # Calculate partial correlations (just in case)
        pD = np.diag(1 / np.sqrt(np.diag(self.precision))) 
        self.pcorr = -(pD @ self.precision @ pD)
        np.fill_diagonal(self.pcorr, 1)

class DiagDominantCorrModel(RandomGraphicalModel):
    def __init__(self, dim, density, random_state=None):
        super().__init__(dim, density, random_state)

        self.graph, self.adj, self.covariance, self.precision, self.pcorr = generate_dom_diag(dim, density)