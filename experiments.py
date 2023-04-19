from GraphModels.model import *
from GraphModels.solvers import *
from GraphModels.metrics import *
from networkx import from_numpy_array, draw
import multiprocessing as mp

def single_model_MHT_experiment(model, n_samples, num_iter, metrics, procedures=['SI', 'B', 'H', 'BH', 'BY']):
    experiment_data = np.zeros((len(metrics), len(procedures)))
    solver = MHTSolver(0.05, pcorr_pvalues, pcorrcoef)
    
    for _ in range(num_iter):
        samples = model.sample(n_samples)
        solver.fit(samples)
        
        for idx, procedure in enumerate(procedures):
            pred_graph = from_numpy_array(solver.apply_correction(procedure))
            
            eval_metrics = model.apply_metrics(pred_graph, metrics)
            
            experiment_data[:, idx] += eval_metrics
            
    experiment_data /= num_iter

    return experiment_data


def familywise_MHT_experiments(model_class, dim, density, n_samples, num_iter, num_repl, metrics, procedures=['SI', 'B', 'H', 'BH', 'BY'], n_jobs=None):
    experiment_data = np.zeros((len(metrics), len(procedures)))
    
    with mp.Pool(processes=n_jobs) as pool:
        waiters = [pool.apply_async(single_model_MHT_experiment, (model_class(dim, density), n_samples, num_repl, metrics, procedures)) for _ in range(num_iter)]
        results = np.stack([waiter.get() for waiter in waiters])
    
    return results.mean(axis=0)
    
    