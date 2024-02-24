import numpy as np
from scipy.stats import kendalltau

def fechner_corr(x, y):
    ind = ((x - np.mean(x))*(y - np.mean(y)))
    res = (np.sum((ind >= 0)) - np.sum((ind < 0))) / len(x)
    return res

# Correlation matrices

def pearson_corr_mat(data):
    return np.corrcoef(data.T)

def kendall_corr_mat(data):
    dim = data.shape[1]
    matrix = np.array([[kendalltau(data[:, i], data[:, j]).statistic
                        for j in range(dim)] for i in range(dim)])

    return matrix

def fechner_corr_mat(data):
    dim = data.shape[1]
    matrix = np.array([[fechner_corr(data[:, i], data[:, j]) for j in range(dim)] for i in range(dim)])

    return matrix

# Pearson correlations via other correlations

def pearson_corr_via_kendall(data):
    kcorr_mat = kendall_corr_mat(data)
    pearson_corr = np.sin(np.pi / 2 * kcorr_mat)

    return pearson_corr

def pearson_corr_via_fechner(data):
    fcorr_mat = fechner_corr_mat(data)
    pearson_corr = np.sin(np.pi / 2 * fcorr_mat)

    return pearson_corr