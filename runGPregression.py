## Import basic packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import time
from multiprocessing import Pool
import argparse 
import pathlib

## My functions
from utils_functions import *
from CTF import * 
from graphs import * 

"""
Parameters
----------
initial_num_obs_samples : int
    Initial number of observational samples
type_cost : int 
    Type of cost per node. Fix_equal = 1, Fix_different = 2, Fix_different_variable = 3, Fix_equal_variable = 4
num_interventions : int   
    Size of the initial interventional dataset. Can be <=20
num_additional_observations: int
    Number of additional observations collected for every decision to observe
num_trials: int
    Number of BO trials
name_index : int
    Index of interventional dataset used. 
"""


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--n_obs', default = 100, type = int, help = 'num observations')
parser.add_argument('--n_int', default = 5, type = int, help = 'num interventions')
parser.add_argument('--type_cost', default = 1, type = int, help = 'intervention costs')
parser.add_argument('--name_index', default = 1, type = int, help = 'name_index')
parser.add_argument('--causal_prior', default = False,  type = bool, help = 'Do not specify when want to set to False')
parser.add_argument('--experiment', default = 'ToyGraph', type = str, help = 'experiment')
args = parser.parse_args()


## Set the seed
seed = 9
np.random.seed(seed=int(seed))

## Set the parameters
n_obs = args.n_obs
n_int = args.n_int
type_cost = args.type_cost
name_index = args.name_index
causal_prior = args.causal_prior
experiment = args.experiment

print('n_int', n_int)
print('causal_prior', causal_prior)
print('experiment', experiment)


## Import observational data
observational_data = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')[:n_obs]
full_observational_data = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')
## Import interventional data
interventional_data = np.load('./Data/' + str(args.experiment) + '/' + 'interventional_data.npy', allow_pickle=True)


if experiment == 'ToyGraph':
    graph = ToyGraph(observational_data)

if experiment == 'ConfoundedToyGraph':
    graph = ConfoundedToyGraph(observational_data)

if experiment == 'CompleteGraph':
    graph = CompleteGraph(observational_data)

if experiment == 'PSAGraph':
    graph = PSAGraph(observational_data)

## Set folder where to save objects
folder = set_saving_folder_CTF(args)
pathlib.Path("./Data/" + folder).mkdir(parents=True, exist_ok=True)

## Givent the data fit all models used for do calculus
functions = graph.fit_all_models(algorithm = 'reg')

## Define optimisation sets and the set of manipulative variables
## Get test inputs -- inputs for which we want to compute the functions
ES, _, _ = graph.get_sets_CTF()
_, _, index_BF = graph.get_info_BF()
test_inputs_list = graph.get_test_inputs_list(size = 300)

## Get the initial optimal solution and the interventional data corresponding to a random permutation of the intervential data with seed given by name_index
BF_data, PF_data = define_initial_data(graph, interventional_data, n_int, name_index)


def fit_single_GP(causal_prior, data, inputs_dim, mean_function_do, var_function_do):
    if causal_prior == True:
        mf = GPy.core.Mapping(inputs_dim, 1)
        mf.f = lambda x: mean_function_do(x)
        mf.update_gradients = lambda a, b: None

        causal_kernel = CausalRBF(input_dim=inputs_dim, variance_adjustment=var_function_do)
        #causal_kernel.variance.fix(1.)

        gpy_model = GPy.models.GPRegression(data[0], data[1], causal_kernel, mean_function=mf)
        gpy_model.likelihood.variance.fix(1e-5)

    else:

        rbf_kernel = RBF(inputs_dim, lengthscale=1., variance = 1.)
        gpy_model = GPy.models.GPRegression(data[0], data[1], rbf_kernel)

        gpy_model.likelihood.variance.fix(1e-5)
    return gpy_model


BF_model = []
pred_mean_list = []
pred_var_list = []

for i in range(len(ES)):
    if i == index_BF and BF_data[0] is None:
        BF_model = None
    else:
        if i == index_BF and BF_data[0] is not None:
            mean_function_do, var_function_do = get_do_fun(graph, functions, observational_data, function_name = "BF")
            model = fit_single_GP(causal_prior, BF_data, len(ES[i]), mean_function_do, var_function_do)
        else:
            mean_function_do, var_function_do = get_do_fun(graph, functions, observational_data, variables = ES[i])
            model = fit_single_GP(causal_prior, [PF_data[0][i], PF_data[1][i]], len(ES[i]), mean_function_do, var_function_do)
        
        pred_mean, pred_var = model.predict(test_inputs_list[i])
        pred_mean_list.append(pred_mean)
        pred_var_list.append(pred_var)


save_results_GPreg(folder,  args, causal_prior, pred_mean_list, pred_var_list)



print('Results saved')

print('Algorithm: GP reg')
print('causal_prior:', args.causal_prior)
print('folder:', folder)










