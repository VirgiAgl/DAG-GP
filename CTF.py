import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import time 
from utils_functions import *



def CTF(BF_data, PF_data, observational_data, graph, functions, Causal_prior=False, total_samples = None, total_samples_test_inputs = None, d=None):
	
	## Define list to store info
	Transferred_mean_list = [None, None]
	Transferred_covariance_list = [None, None] 


	############################# LOOP
	start_time = time.time()

	## Get test inputs -- inputs for which we want to compute the functions
	test_inputs_list = graph.get_test_inputs_list(size = 300)
	

	## Define prior model for BF
	mean_function_BF, kernel_function_BF = get_parameters_BF(Causal_prior, graph, observational_data, functions)
	
	
	integrating_measures = graph.get_IMs(functions)
	dim_BF, inputs_BF, index_BF = graph.get_info_BF()
	exploration_set, _ , _ = graph.get_sets_CTF()
	dict_interventions = initialise_dicts_CTF(exploration_set, PF_data[0], BF_data[0], index_BF)  

	n_samples = 100
	if d is not None and total_samples_test_inputs is not None:
		print('Not None')
	
	if total_samples is None:
		total_samples = generate_full_samples_IM(integrating_measures, n_samples, dim_BF, dict_interventions, inputs_BF)
	
	if total_samples_test_inputs is None:
		dict_test_inputs = initialise_dicts_CTF(exploration_set[:-1], test_inputs_list[:-1])
		total_samples_test_inputs = generate_full_samples_IM(integrating_measures, n_samples, dim_BF, dict_test_inputs, inputs_BF)


	## Udate the base function if it is part of T 
	if any(m==1 for m in graph.get_IMs(functions)):
		Transferred_mean_list[1], Transferred_covariance_list[1] = forward_parameters(BF_inputs = test_inputs_list[-1], 
																					BF_data=BF_data, 
																					PF_data = PF_data,  
																					graph = graph,
																					functions = functions,
																					kernel_function_BF = kernel_function_BF,
																					mean_function_BF = mean_function_BF,
																					total_samples = total_samples,
																					kernel = 'causal',
																					n_samples=10)
	

	## Update all PFs - The base set is the last element of the exploration_set so we can just exclude this term 
	Transferred_mean_list[0], Transferred_covariance_list[0], d = backward_parameters(PF_inputs = test_inputs_list[:-1], 
																			 BF_data = BF_data, 
																			 PF_data = PF_data,
																			 graph = graph, 
																			 functions = functions,
																			 kernel_function_BF = kernel_function_BF,
																			 mean_function_BF = mean_function_BF,
																			 total_samples = total_samples,
																			 total_samples_test_inputs = total_samples_test_inputs,
																			 d = d,
																			 kernel = 'causal',
																			 n_samples=n_samples)
								  

	## Compute total time for training
	total_time = time.time() - start_time
	print('Total time for training:', total_time)
	
	
	return (Transferred_mean_list, Transferred_covariance_list, total_samples, total_samples_test_inputs, d)

