from GraphModels.solvers import *
from GraphModels.model import *
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multivariate_t
from dataclasses import dataclass
import multiprocessing as mp
from time import perf_counter
from datetime import datetime
import pickle
import os
import argparse
import json
import data_utils

def exactRun(model, n_samples, solver, distribution=multivariate_normal, **distkwargs):
    samples = model.sample(n_samples, dist=distribution, **distkwargs)
    cov = np.cov(samples.T)
    solver.fit(cov)

    eval_metrics = model.confusion(solver.graph)
    eval_metrics = (eval_metrics[1], eval_metrics[3],
                    eval_metrics[2], eval_metrics[0])
    
    return eval_metrics

def singleModelRun(model, n_samples, num_iter, solver, distribution=multivariate_normal, **distkwargs):
    experiment_data = np.zeros((num_iter, 4, 1))
    for i in range(num_iter):
        experiment_data[i, :, 0] += exactRun(model, n_samples, solver, distribution, **distkwargs)

    return experiment_data

def multipleModelsRun(model_class, dim, density, n_samples, num_iter, num_models, solver, verbose=True, n_jobs=None, distribution=multivariate_normal, **distkwargs):
    with mp.Pool(n_jobs) as pool:
        print(f'Started experiment for dim={dim}, density={density} with {mp.cpu_count() if n_jobs == None else n_jobs} processes')
        start = perf_counter()
        waiters = []

        for _ in range(num_models):
            waiter = pool.apply_async(singleModelRun, kwds={
                'model': model_class(dim, density),
                'n_samples': n_samples,
                'num_iter': num_iter,
                'solver': solver,
                'distribution': distribution,
                **distkwargs
            })

            waiters.append(waiter)

        if verbose:
            print(f'Scheduled experiments for {num_models}. Started waiting...')

        results = np.stack([waiter.get() for waiter in waiters])

        end = perf_counter()

        elapsed = end - start
        ave_iter = elapsed / num_iter / num_models

        if verbose:
            print(f'Multiple models run with dim={dim}, density={density} is completed in {np.around(elapsed, 3)} seconds. Average iteration took: {np.around(ave_iter, 3)}')

        return results, elapsed, ave_iter


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', action='store', type=str)
    args = parser.parse_args()

    with open(args.json_path, 'r') as fi:
        json_config = json.load(fi)

    prefix_path = os.path.dirname(args.json_path)

    raw_model = json_config['model_class']
    raw_solver = json_config['solver']
    raw_distribution = json_config['distribution']

    supported_models = {
        'DiagDominantPcorrModel': DiagDominantPcorrModel,
        'CholPCorrModel': CholPCorrModel,
    }
    supported_solvers = {
        'GraphLasso': GraphLasso(json_config['reg_param'], 1000)
    }
    supported_distributions = {
        'normal': multivariate_normal,
        'student': multivariate_t,
    }

    json_config['model_class'] = supported_models[raw_model]
    json_config['solver'] = supported_solvers[raw_solver]
    json_config['distribution'] = supported_distributions[raw_distribution]
    json_config.pop('reg_param')

    experiment_params = json_config

    print('Successfully loaded config...')

    data = []
    for density in np.arange(0.1, 1, 0.1):
        run_result = multipleModelsRun(density=density, **experiment_params)
        data.append(run_result[0])

    data = np.stack(data)
    
    raw_data_path = os.path.join(prefix_path, f'data_{str(datetime.now()).replace(" ", "_")}.bin')
    with open(raw_data_path, 'wb') as fout:
        pickle.dump(data, fout)

    fdr_c, fomr_c, tnr_c, tpr_c, bacc_c, f1_c, mcc_c = data_utils.prepare_data_GL(data)
    density_list = np.arange(0.1, 1, 0.1)

    plt.subplot(2, 2, 1)
    plt.plot(density_list, tnr_c[:, 0], label='GL')
    plt.title('TNR')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(density_list, fomr_c[:, 0], label='GL')
    plt.title('FOR')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(density_list, fdr_c[:, 0], label='GL')
    plt.title('FDR')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(density_list, tpr_c[:, 0], label='GL')
    plt.title('TPR')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.legend()
    plt.gcf().set_size_inches((10, 10))
    plt.savefig(os.path.join(prefix_path, "tnr_fomr_fdr_tpr.png"))
    plt.gcf().clear()

    plt.subplot(1, 3, 1)
    plt.plot(density_list, bacc_c[:, 0], label='GL')
    plt.title('Balanced accuracy')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(density_list, f1_c[:, 0], label='GL')
    plt.title('F1 score')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(density_list, mcc_c[:, 0], label='GL')
    plt.title('MCC')
    plt.minorticks_on()
    plt.ylim((-1, 1))
    plt.legend()

    plt.gcf().set_size_inches((10, 5))
    plt.savefig(os.path.join(prefix_path, "mcc_f1_ba.png"))