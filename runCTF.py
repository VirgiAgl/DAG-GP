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

## Get the initial optimal solution and the interventional data corresponding to a random permutation of the intervential data with seed given by name_index
BF_data, PF_data = define_initial_data(graph, interventional_data, n_int, name_index)

print('Fitting all conditional probabilities')
functions = graph.fit_all_models()

print('Learning with MCGP and Causal prior = ' + str(causal_prior))

(Transferred_mean_list, Transferred_covariance_list, _, _, _) = CTF(BF_data, PF_data, observational_data, 
                                                                graph, functions, Causal_prior = causal_prior)

save_results_CTF(folder,  args, causal_prior, Transferred_mean_list, Transferred_covariance_list, BF_data, PF_data)



print('Saved results')

print('Algorithm: CTF')
print('causal_prior', args.causal_prior)
print('name_index', name_index)
print('folder', folder)



