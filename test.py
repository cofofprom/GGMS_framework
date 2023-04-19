from GraphModels.model import *
from GraphModels.solvers import *
from GraphModels.metrics import *
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import cProfile, pstats
from pstats import SortKey

def experiments(S_obs, metrics, density, mht_algos):
    res = {al: np.zeros(len(metrics)) for al in mht_algos}
    model = CholRandomModel(20, density, pcorr_model_builder)
    for _ in range(S_obs):
        samples = model.sample(40)
        slv = MHTSolver(0.05, pcorr_pvalues, pcorrcoef)
        slv.fit(samples)
        for corr in mht_algos:
            ms = model.apply_metrics(nx.from_numpy_array(slv.apply_correction(corr)), metrics)
            for idx in range(len(ms)):
                res[corr][idx] += ms[idx]
                
    for corr in mht_algos:
        for idx in range(len(metrics)):
            res[corr][idx] /= S_obs
            
    return res 

if __name__ == '__main__':
    densities = [0.94, 0.87, 0.85, 0.77, 0.74, 0.68, 0.64, 0.55, 0.4]
    mht_algos = ['SI', 'B', 'H', 'BH', 'BY']
    metrics = [FP, FN, TPR, FDR, F1]
    data = {d: {al: [0 for _ in metrics] for al in mht_algos} for d in densities}
    S_exp = 100
    S_obs = 10

    for density in densities:
        with mp.Pool() as pool:
            res = [pool.apply_async(experiments, (S_obs, metrics, density, mht_algos)) for _ in range(S_exp)]
            exps = [r.get() for r in res]
            parsed_result = {al: np.zeros(len(metrics)) for al in mht_algos}
            for corr in mht_algos:
                for exp in exps:
                    parsed_result[corr] += exp[corr]
                    
            data[density] = parsed_result
            
            for corr in mht_algos:        
                for idx in range(len(metrics)):
                    data[density][corr][idx] /= S_exp
            
    for m in range(len(metrics)):
        for al in mht_algos:
            x = np.arange(0.1, 1, 0.1)
            y = np.zeros_like(x)
            
            for d in range(len(densities)):
                y[d] = data[densities[d]][al][m]
            
            plt.plot(x, y, label=al)
        
        plt.legend()
        plt.title(metrics[m].__name__)
        plt.savefig(metrics[m].__name__)
        plt.clf()