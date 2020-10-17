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
from sklearn.metrics import mean_squared_error
from math import sqrt

## My functions
from utils_functions import *
from CTF import * 
from graphs import * 

from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
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
parser.add_argument('--n_steps', default = 5, type = int, help = 'n_steps AL')
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
n_steps = args.n_steps

print('n_int:', n_int)
print('causal_prior:', causal_prior)
print('experiment:', experiment)


## Import observational data
observational_data = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')[:n_obs]
full_observational_data = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')
## Import interventional data
interventional_data = np.load('./Data/' + str(args.experiment) + '/' + 'interventional_data.npy', allow_pickle=True)

if experiment == 'ToyGraph':
    graph = ToyGraph(observational_data)
    mean_t_x = np.load("./Data/ToyGraph/intervention_function_x.npy")
    mean_t_z = np.load("./Data/ToyGraph/intervention_function_z.npy")
    true_functions = OrderedDict ([('X', mean_t_x), ('Z', mean_t_z)])


if experiment == 'ConfoundedToyGraph':
    graph = ConfoundedToyGraph(observational_data)
    mean_t_x = np.load("./Data/ConfoundedToyGraph/intervention_function_x.npy")
    mean_t_z = np.load("./Data/ConfoundedToyGraph/intervention_function_z.npy")
    true_functions = OrderedDict ([('X', mean_t_x), ('Z', mean_t_z)])


if experiment == 'CompleteGraph':
    graph = CompleteGraph(observational_data)
    true_functions = np.load("./Data/CompleteGraph/true_function.npy", allow_pickle=True)
    true_functions = OrderedDict ([
      ('B', true_functions[0]),
      ('D', true_functions[1]),
      ('E', true_functions[2]),
      ('BD', true_functions[3]),
      ('BE', true_functions[4]), 
      ('DE', true_functions[5])
        
    ])

if experiment == 'PSAGraph':
    graph = PSAGraph(observational_data)
    mean_t_x = np.load("./Data/PSAGraph/intervention_function_aspirin.npy")
    mean_t_z = np.load("./Data/PSAGraph/intervention_function_statin.npy")
    mean_t_xz = np.load("./Data/PSAGraph/intervention_function_aspirinstatin.npy")
    true_functions = OrderedDict ([
      ('aspirin', mean_t_x),
      ('statin', mean_t_z),
      ('aspirinstatin', mean_t_xz)])   

## Set folder where to save objects
folder = set_saving_folder_CTF(args)
pathlib.Path("./Data/" + folder).mkdir(parents=True, exist_ok=True)

## Givent the data fit all models used for do calculus
functions = graph.fit_all_models(algorithm = 'reg')
interventional_ranges = graph.get_interventional_ranges()

## Define optimisation sets and the set of manipulative variables
## Get test inputs -- inputs for which we want to compute the functions
ES, _, _ = graph.get_sets_CTF()
sem = graph.define_SEM()
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

        gpy_model = GPy.models.GPRegression(data[0], data[1], causal_kernel, mean_function=mf)
        gpy_model.likelihood.variance.fix(1e-5)

    else:

        rbf_kernel = RBF(inputs_dim, lengthscale=1., variance = 1.)
        gpy_model = GPy.models.GPRegression(data[0], data[1], rbf_kernel)

        gpy_model.likelihood.variance.fix(1e-5)
    return gpy_model


def AL_reg(emukit_model, target_function, space, initial_data, n_steps, 
            test_inputs, true_function, acquisition = 'ModelVariance'):
    rmse_list = []
    
    mean_list = []
    var_list = []
    
    inputs_new = initial_data[0]
    outputs_new = initial_data[1]
    model = copy.deepcopy(emukit_model)
    for i in range(n_steps):
        print('step:', i)

        mean, var = model.predict(test_inputs)
        index = np.where(var == np.max(var))[0][0]
        max_value = var[index]
        x_new = np.transpose(test_inputs[index][:,np.newaxis])

        y_new = target_function(x_new)
        
        inputs_new = np.append(inputs_new, x_new, axis=0)
        outputs_new = np.append(outputs_new, y_new, axis=0)


        model.set_data(inputs_new, outputs_new)
        
        mu_plot, var_plot = model.predict(test_inputs)

        mean_list.append(mu_plot)
        var_list.append(var_plot)
        rmse_list.append(sqrt(mean_squared_error(true_function, mu_plot)))
        
    return model, inputs_new, outputs_new, rmse_list, mean_list, var_list



import pickle
model_list_inputs = [None]*len(ES)
model_list_outputs = [None]*len(ES)
prediction_mean_list = [None]*len(ES)
prediction_var_list = [None]*len(ES)
rmse_list = [None]*len(ES)

for i in range(len(ES)):
    set_variable = ES[i][0]
    if i == index_BF and BF_data[0] is None:
        BF_model = None
    else:
        if i == index_BF and BF_data[0] is not None:
            ## Fit model
            mean_function_do, var_function_do = get_do_fun(graph, functions, observational_data, function_name = "BF")
            model = GPyModelWrapper(fit_single_GP(causal_prior, BF_data, len(ES[i]), mean_function_do, var_function_do))

            ## Get Intervention_function and space
            target_function, space = Intervention_function({set_variable:''}, model = sem, target_variable = 'Y',  
                                                            min_intervention=[interventional_ranges[set_variable][0]], max_intervention=[interventional_ranges[set_variable][1]])
            ## run AL loop
            new_model, _, _, rmse, mean_list, var_list = AL_reg(model, target_function, space, BF_data, n_steps, test_inputs_list[i], 
                                                                true_functions[set_variable], acquisition = 'ModelVariance')
        
        else:
            mean_function_do, var_function_do = get_do_fun(graph, functions, observational_data, variables = ES[i])
            model = GPyModelWrapper(fit_single_GP(causal_prior, [PF_data[0][i], PF_data[1][i]], len(ES[i]), mean_function_do, var_function_do))

            if len(ES[i]) == 1:
                target_function, space = Intervention_function({set_variable:''}, model = sem, target_variable = 'Y',  
                                                            min_intervention=[interventional_ranges[set_variable][0]], max_intervention=[interventional_ranges[set_variable][1]])
            else:
                target_function, space = Intervention_function({ES[i][0]:'', ES[i][1]:''}, model = sem, target_variable = 'Y',  
                                                            min_intervention=[interventional_ranges[ES[i][0]][0],interventional_ranges[ES[i][1]][0]], 
                                                            max_intervention=[interventional_ranges[ES[i][0]][1],interventional_ranges[ES[i][1]][1]])


            new_model, _, _, rmse, mean_list, var_list = AL_reg(model, target_function, space, [PF_data[0][i], PF_data[1][i]], 
                                    n_steps, test_inputs_list[i], true_functions[''.join(ES[i])], acquisition = 'ModelVariance')

        model_list_inputs[i] = new_model.model.X
        model_list_outputs[i] = new_model.model.Y
        prediction_mean_list[i] = mean_list
        prediction_var_list[i] = var_list
        rmse_list[i] = rmse



save_results_ALreg(folder,  args, causal_prior, rmse_list, model_list_inputs, model_list_outputs, prediction_mean_list, prediction_var_list)



print('Results saved')

print('Algorithm: AL GP reg')
print('causal_prior:', args.causal_prior)
print('folder:', folder)










