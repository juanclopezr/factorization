import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import shutil
import time
import logging
import pymc3 as pm
import theano
import scipy as sp

theano.config.compute_test_value = 'ignore'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def models(dim, prior, std=0.01, alpha=2, bounds=(-10,10)):
        
    data = prior.copy()
    n, m = data.shape

    # Perform mean value imputation
    nan_mask = np.isnan(data)
    data[nan_mask] = data[~nan_mask].mean()

    # Low precision reflects uncertainty; prevents overfitting.
    # Set to the mean variance across users and items.
    alpha_u = 1 / data.var(axis=1).mean()
    alpha_v = 1 / data.var(axis=0).mean()

    # Specify the model.
    logging.info('building the PMF model')
    with pm.Model() as pmf:
        U = pm.MvNormal(
             'U', mu=0, tau=alpha_u * np.eye(dim),
            shape=(n, dim), testval=np.random.randn(n, dim) * std)
        V = pm.MvNormal(
             'V', mu=0, tau=alpha_v * np.eye(dim),
            shape=(m, dim), testval=np.random.randn(m, dim) * std)
        R = pm.Normal(
            'R', mu=theano.tensor.dot(U, V.T), tau=alpha * np.ones((n, m)),
            observed=data)
    return pmf

try:
    import ujson as json
except ImportError:
    import json

    
def save_np_vars(vars, savedir):
    """Save a dictionary of numpy variables to `savedir`. We assume
    the directory does not exist; an OSError will be raised if it does.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    shapes = {}
    for varname in vars:
        data = vars[varname]
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)

def load_np_vars(savedir):
    """Load numpy variables saved with `save_np_vars`."""
    shape_file = os.path.join(savedir, 'shapes.json')
    with open(shape_file, 'r') as sfh:
        shapes = json.load(sfh)

    vars = {}
    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)
        
    return vars

def load_train_test(name):
    """Load the train/test sets."""
    savedir = os.path.join('./jester_dataset_1_1/', name)
    vars = load_np_vars(savedir)
    return vars['train'], vars['test']

def rmse(test_data, predicted):
    """Calculate root mean squared error.
    Ignoring missing values in the test data.
    """
    I = ~np.isnan(test_data)   # indicator for missing values
    N = I.sum()                # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = sqerror[I].sum() / N                 # mean squared error
    return np.sqrt(mse)  
    
def predict(U, V):
    bounds = (-10,10)
    R = np.dot(U, V.T)
    n, m = R.shape
    sample_R = np.array([[np.random.normal(R[i,j], 0.01) for j in range(m)] for i in range(n)])
    low, high = bounds
    sample_R[sample_R < low] = low
    sample_R[sample_R > high] = high
    return sample_R

def eval_map(mape, train, test):
    U = mape['U']
    V = mape['V']
    
    predictions = predict(U, V)
    train_rmse = rmse(train, predictions)
    test_rmse = rmse(test, predictions)
    overfit = test_rmse-train_rmse
    
    print('PMF MAP training RMSE: %.5f' % train_rmse)
    print('PMF MAP testing RMSE:  %.5f' % test_rmse)
    print('Train/test difference: %.5f' % overfit)
    
    return test_rmse

def running_rmse(trace, test_data, train_data, burn_in = 0, plot=True):
    """Calculate RMSE for each step of the trace to monitor convergence.
    """
    burn_in = burn_in if len(trace) >= burn_in else 0
    results = {'per-step-train': [], 'running-train': [],
               'per-step-test': [], 'running-test': []}
    R = np.zeros(test_data.shape)
    for cnt, sample in enumerate(trace[burn_in:]):
        sample_R = predict(sample['U'], sample['V'])
        R += sample_R
        running_R = R / (cnt + 1)
        results['per-step-train'].append(rmse(train_data, sample_R))
        results['running-train'].append(rmse(train_data, running_R))
        results['per-step-test'].append(rmse(test_data, sample_R))
        results['running-test'].append(rmse(test_data, running_R))

    results = pd.DataFrame(results)

    if plot:
        results.plot(
            kind='line', grid=False, figsize=(15, 7),
            title='Per-step and Running RMSE From Posterior Predictive')

    # Return the final predictions, and the RMSE calculations
    return running_R, results

def norms(trace, monitor=('U', 'V'), ord='fro'):
    """Return norms of latent variables at each step in the
    sample trace. These can be used to monitor convergence
    of the sampler.
    """
    monitor = ('U', 'V')
    norms = {var: [] for var in monitor}
    for sample in trace:
        for var in monitor:
            norms[var].append(np.linalg.norm(sample[var], ord))
    return norms

def traceplot(trace):
    """Plot Frobenius norms of U and V as a function of sample #."""
    trace_norms = norms(trace)
    u_series = pd.Series(trace_norms['U'])
    v_series = pd.Series(trace_norms['V'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    u_series.plot(kind='line', ax=ax1, grid=False,
                  title="$\|U\|_{Fro}^2$ at Each Sample")
    v_series.plot(kind='line', ax=ax2, grid=False,
                  title="$\|V\|_{Fro}^2$ at Each Sample")
    ax1.set_xlabel("Sample Number")
    ax2.set_xlabel("Sample Number")
