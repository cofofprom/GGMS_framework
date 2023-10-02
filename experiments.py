from networkx import from_numpy_array
import multiprocessing as mp
from time import perf_counter
from scipy.stats import multivariate_normal
import numpy as np


def single_model_MHT_experiment(model, n_samples, num_iter,
                                metrics,
                                solver,
                                procedures=['SI', 'B', 'H', 'BH', 'BY'],
                                dist=multivariate_normal, **distkwargs):
    experiment_data = np.zeros((num_iter, len(metrics), len(procedures)))

    for i in range(num_iter):
        samples = model.sample(n_samples, dist=dist, **distkwargs)
        solver.fit(samples)

        for idx, procedure in enumerate(procedures):
            pred_graph = from_numpy_array(solver.apply_correction(procedure))

            eval_metrics = model.apply_metrics(pred_graph, metrics)

            experiment_data[i, :, idx] += eval_metrics

    return experiment_data


def familywise_MHT_experiments(model_class, solver, dim,
                               density,
                               n_samples, num_iter,
                               num_repl, metrics,
                               procedures=['SI', 'B', 'H', 'BH', 'BY'],
                               n_jobs=None, verbose=False,
                               dist=multivariate_normal, **distkwargs):
    start = perf_counter()
    with mp.Pool(processes=n_jobs) as pool:
        waiters = [pool.apply_async(single_model_MHT_experiment,
                                    (model_class(dim, density), n_samples,
                                     num_repl, metrics,
                                     solver, procedures, dist), distkwargs)
                   for _ in range(num_iter)]
        results = np.stack([waiter.get() for waiter in waiters])

    end = perf_counter()
    if verbose:
        print((f'Family-wise MHT experiment'
               f'with {np.around(density, 3)}'
               f'completed in time: {np.around(end - start, 3)}s'))
    return results


def single_model_GL_experiment(model, n_samples, num_iter,
                               solver, dist=multivariate_normal, **distkwargs):
    experiment_data = np.zeros((num_iter, 4, 1))

    for i in range(num_iter):
        samples = model.sample(n_samples, dist=dist, **distkwargs)
        try:
            solver.fit(samples)
        except:
            continue

        pr = solver.get_precision()
        np.fill_diagonal(pr, 0)
        pred_graph = from_numpy_array((pr != 0.).astype(int))

        eval_metrics = model.confusion(pred_graph)
        eval_metrics = (eval_metrics[1], eval_metrics[3],
                        eval_metrics[2], eval_metrics[0])

        experiment_data[i, :, 0] += eval_metrics

    return experiment_data


def familywise_GL_experiments(model_class, solver,
                              dim, density,
                              n_samples, num_iter,
                              num_repl, n_jobs=None,
                              verbose=False,
                              dist=multivariate_normal, **distkwargs):
    start = perf_counter()
    with mp.Pool(processes=n_jobs) as pool:
        waiters = [pool.apply_async(single_model_GL_experiment,
                                    (model_class(dim, density),
                                     n_samples,
                                     num_repl, solver, dist),
                                    distkwargs) for _ in range(num_iter)]
        results = np.stack([waiter.get() for waiter in waiters])

    end = perf_counter()
    if verbose:
        print((
            f'Family-wise GL experiment '
            f'with {np.around(density, 3)} '
            f'completed in time: {np.around(end - start, 3)}s'))
    return results, end - start
