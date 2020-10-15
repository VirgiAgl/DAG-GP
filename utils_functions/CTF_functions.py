## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from .causal_kernels import CausalRBF


def define_initial_data(graph, interventional_data, num_interventions, name_index):
	## Unpacking the interventional data in PF and BF lists

	PF_data = [[], []]
	BF_data = [None, None]

	_, _, index_BF = graph.get_info_BF()

	exploration_set, _ , _ = graph.get_sets_CTF()

	## For all sets in ES this is getting the interventional data, shuffling them and creating a list 
	for j in range(len(exploration_set)):

		if j == index_BF and interventional_data[j] is None:
			BF_data[0] = None
			BF_data[1] = None

		elif j == index_BF and interventional_data[j] is not None:
			subset_all_data = shuffle_select_interventional_data(interventional_data[j], num_interventions, name_index)
			BF_data[0] = subset_all_data[:,:-1]
			BF_data[1] = subset_all_data[:, -1][:,np.newaxis]
		else:
			subset_all_data = shuffle_select_interventional_data(interventional_data[j], num_interventions, name_index)

			PF_data[0].append(subset_all_data[:,:-1])
			PF_data[1].append(subset_all_data[:, -1][:,np.newaxis])


	return BF_data, PF_data


def get_parameters_BF(Causal_prior, graph, observational_samples, functions):


	dim_BF, _ , _ = graph.get_info_BF()
	do_function_BF = graph.get_all_do()['BF']

	def mean_function_do(x):
		mean , _ = do_function_BF(observational_samples, functions, value = x)
		return mean

	def var_function_do(x):
		_ , var = do_function_BF(observational_samples, functions, value = x)
		return var
	
	if Causal_prior == True:
		## Specify parameters via do-calculus
		mean = mean_function_do
		kernel = CausalRBF(input_dim=dim_BF, variance_adjustment=var_function_do)
		#mean = None
		#kernel = RBF(dim_BF, lengthscale=1., variance = 1.)
	else:
		mean = None
		kernel = RBF(dim_BF, lengthscale=1., variance = 1.)
	  
	return mean, kernel


def shuffle_select_interventional_data(interventional_data_element, num_interventions, name_index):
	data = interventional_data_element.copy()
	num_variables = data[0]
	if num_variables == 1:
		data_x = np.asarray(data[(num_variables+1)])
		data_y = np.asarray(data[-1])
	else:
		data_x = np.asarray(data[(num_variables+1):(num_variables*2)][0])
		data_y = np.asarray(data[-1])


	if len(data_y.shape) == 1:
		data_y = data_y[:,np.newaxis]

	if len(data_x.shape) == 1:
		data_x = data_x[:,np.newaxis]
		
	all_data = np.concatenate((data_x, data_y), axis =1)

	## Need to reset the global seed 
	state = np.random.get_state()

	np.random.seed(name_index)
	np.random.shuffle(all_data)

	np.random.set_state(state)

	subset_all_data = all_data[:num_interventions]

	return subset_all_data
