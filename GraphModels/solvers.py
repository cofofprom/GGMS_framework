import numpy as np
from abc import ABC
from scipy.stats import t
from numba import vectorize, float64, int64

def pcorr_pvalues(r, n, N):
    dof = n - N
    stat = r * np.sqrt(dof / (1 - (r ** 2)))
    pval = 2 * t.sf(np.abs(stat), dof)
    return pval

def corr_pvalues(r, n, N=0):
    dof = n - 2
    stat = r * np.sqrt(dof / (1 - (r ** 2)))
    pval = 2 * t.sf(np.abs(stat), dof)
    return pval

def pcorrcoef(X):
    cov = np.cov(X)
    prec = np.linalg.inv(cov)
    D = np.diag(1 / np.sqrt(np.diag(prec)))
    corr = -(D @ prec @ D)
    np.fill_diagonal(corr, 1)
    
    return corr

class MHTSolver:
    def __init__(self, alpha, p_val_fun, corr_fun=np.corrcoef):
        self.alpha = alpha
        self.p_val_fun = p_val_fun
        self.corr_fun = corr_fun
        
    def fit(self, X, y=None):
        self.n_obs, self.dim = X.shape
        self.n_tests = self.dim * (self.dim - 1) // 2
        self.corr_mat = self.corr_fun(X.T)
        np.fill_diagonal(self.corr_mat, 0)
        self.p_values = np.vectorize(self.p_val_fun)(self.corr_mat, self.n_obs, self.dim)
        
    def apply_correction(self, procedure):
        if procedure == 'SI':
            return (self.p_values < self.alpha).astype(int)

        if procedure == 'B':
            return (self.p_values < (self.alpha / self.n_tests)).astype(int)
        
        if procedure == 'H':
            triu_idx = np.triu_indices_from(self.p_values, k=1)
            
            triu_idx_list = np.array(triu_idx).T
            pvals_list = self.p_values[triu_idx]
            
            perm = np.argsort(pvals_list)
            
            idx_sorted = [triu_idx_list[idx] for idx in perm]
            pvals_sorted = [pvals_list[idx] for idx in perm]
            
            adj = np.zeros((self.dim, self.dim))
            
            for k in range(1, self.n_tests + 1):
                curve_val = self.alpha / (self.n_tests + 1 - k)
                if pvals_sorted[k - 1] > curve_val:
                    break
                adj[tuple(idx_sorted[k - 1])] = 1
                adj[tuple(idx_sorted[k - 1])[::-1]] = 1
            return adj
            
        if procedure == 'BH':
            triu_idx = np.triu_indices_from(self.p_values, k=1)
            
            triu_idx_list = np.array(triu_idx).T
            pvals_list = self.p_values[triu_idx]
            
            perm = np.argsort(pvals_list)
            
            idx_sorted = [triu_idx_list[idx] for idx in perm]
            pvals_sorted = [pvals_list[idx] for idx in perm]
            
            adj = np.ones((self.dim, self.dim)) - np.eye(self.dim)
            
            for k in range(self.n_tests, 0, -1):
                curve_val = self.alpha * k / self.n_tests
                if pvals_sorted[k - 1] <= curve_val:
                    break
                adj[tuple(idx_sorted[k - 1])] = 0
                adj[tuple(idx_sorted[k - 1])[::-1]] = 0
            return adj
        
        if procedure == 'BY':
            triu_idx = np.triu_indices_from(self.p_values, k=1)
            
            triu_idx_list = np.array(triu_idx).T
            pvals_list = self.p_values[triu_idx]
            
            perm = np.argsort(pvals_list)
            
            idx_sorted = [triu_idx_list[idx] for idx in perm]
            pvals_sorted = [pvals_list[idx] for idx in perm]
            
            adj = np.ones((self.dim, self.dim)) - np.eye(self.dim)
            harm = np.log(self.n_tests) + np.euler_gamma + 1 / (2*self.n_tests)
            
            for k in range(self.n_tests, 0, -1):
                curve_val = self.alpha * k / self.n_tests / harm
                if pvals_sorted[k - 1] <= curve_val:
                    break
                adj[tuple(idx_sorted[k - 1])] = 0
                adj[tuple(idx_sorted[k - 1])[::-1]] = 0
            return adj
        
        