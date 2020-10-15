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

print('n_int', n_int)
print('causal_prior', causal_prior)
print('experiment', experiment)


## Import observational data
observational_data = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')[:n_obs]
full_observational_data = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')
## Import interventional data
interventional_data = np.load('./Data/' + str(args.experiment) + '/' + 'interventional_data.npy', allow_pickle=True)

print('size obs dataset:', observational_data.shape[0])

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



## Get the initial design (initial interval data) corresponding to a random permutation of the intervential data with seed given by name_index
BF_data, PF_data = define_initial_data(graph, interventional_data, n_int, name_index)

##get points that are potential locations for function evaluation  
test_inputs_list = graph.get_test_inputs_list(size = 300)
#np.save("./Data/" + folder + "test_inputs_list.npy", test_inputs_list)


functions = graph.fit_all_models()
dim_BF, inputs_BF, index_BF = graph.get_info_BF()
exploration_set, _, _ = graph.get_sets_CTF()
integrating_measures = graph.get_IMs(functions)
target_functions, _ = graph.get_intervention_functions()


Transferred_mean_list_total = []
Transferred_covariance_list_BF = []
Transferred_covariance_list_PF = []
max_values_list = []
point_values_list = []
function_values_list = []
BF_data_list = []
PF_data_list = []

print('Fitting all conditional probabilities')
functions = graph.fit_all_models()


for i in range(n_steps):
    print('Step:', i)

    if i == 0:
        total_samples = None
        total_samples_test_inputs = None
        d = None
    else:
        ## Get sample for new point 
        dict_interventions = initialise_dicts_CTF(exploration_set, PF_data[0], BF_data[0], index_BF)  
        total_samples = increment_full_samples_IM(index_function, integrating_measures, total_samples, dict_interventions, inputs_BF)

        total_samples_test_inputs = total_samples_test_inputs        



    (Transferred_mean_list, Transferred_covariance_list,
    total_samples, total_samples_test_inputs, d) = CTF(BF_data, PF_data, observational_data, graph, 
                                                            Causal_prior = causal_prior, functions = functions, total_samples = total_samples, 
                                                            total_samples_test_inputs = total_samples_test_inputs, d = d)

    max_values, point_values, function_values = get_max_variance_values(exploration_set, Transferred_covariance_list, test_inputs_list, integrating_measures)

    new_BF_data, new_PF_data, index_function = get_new_dataset(max_values, point_values, function_values, BF_data, PF_data, integrating_measures, target_functions)

    Transferred_mean_list_total.append(Transferred_mean_list)
    
    # Transferred_covariance_list_BF.append(np.diagonal(Transferred_covariance_list[1]))
    # Transferred_covariance_list_PF.append(np.diagonal(Transferred_covariance_list[0]))
    max_values_list.append(max_values)
    point_values_list.append(point_values)
    function_values_list.append(index_function)

    BF_data_list.append(BF_data)
    PF_data_list.append(PF_data)
    

    BF_data = new_BF_data
    PF_data = new_PF_data


save_results_AL_CTF(folder,  args, causal_prior, Transferred_mean_list_total, Transferred_covariance_list_BF, Transferred_covariance_list_PF, 
                    max_values_list, point_values_list,function_values_list, BF_data_list, PF_data_list)


print('Saved results')

print('Algorithm: AL CTF')
print('causal_prior', args.causal_prior)
print('name_index', name_index)
print('folder', folder)



