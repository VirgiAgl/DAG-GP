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


def get_max_variance_values(ES, cov_list_mf, test_inputs_list, measures = None):
	initial_dim = 0
	list_max = []
	
	max_values = []
	point_values = []
	function_values = []
	
	for i in range(len(measures)):
		#print('i', i)
		if measures[i] == 1:
			#print('Doing BF')
			#print('i',i)
			variance_values = np.diagonal(cov_list_mf[1])
			#print('variance_values', variance_values[:5])
			index = np.where(variance_values == np.max(variance_values))[0][0]
			max_value = variance_values[index]
			inputs_value = test_inputs_list[-1][index]
			function_number = len(ES) - 1    
   
			#to_append = [max_value, inputs_value, function_number]
			#list_max.append(to_append)  
			
			max_values.append(max_value)
			point_values.append(inputs_value)
			function_values.append(function_number)
			# print('max_value', max_value)
			# print('inputs_value', inputs_value)
			# print('function_number', function_number)
		else:
			# print('Doing PFs')
			# print('i',i)
			inputs = test_inputs_list[i]
			dim_test_inputs = inputs.shape[0]
			# print('initial_dim', initial_dim)
			# print('(dim_test_inputs+initial_dim)', (dim_test_inputs+initial_dim))
			variance_values = np.diagonal(cov_list_mf[0][initial_dim:(dim_test_inputs+initial_dim),
															initial_dim:(dim_test_inputs+initial_dim)])
			
			#print('variance_values', variance_values[:5])
			index = np.where(variance_values == np.max(variance_values))[0][0]
			max_value = variance_values[index]
			inputs_value = inputs[index]
			
			function_number = i
			#to_append = [max_value, inputs_value, function_number]
	
			# print('max_value', max_value)
			# print('inputs_value', inputs_value)
			# print('function_number', function_number)
			#list_max.append(to_append)
			
			max_values.append(max_value)
			point_values.append(inputs_value)
			function_values.append(function_number)
			
			initial_dim += dim_test_inputs 

	# print('max_values', max_values)
	# print('point_values', point_values)
	# print('function_values', function_values)

	return max_values, point_values, function_values


def get_next_point_function(max_values, point_values, function_values):
	index = np.where(max_values == np.max(max_values))[0][0]
	point = point_values[index]
	n_function = function_values[index]
	return point, n_function


def get_new_dataset(max_values, point_values, function_values, BF_data, PF_data, measures, target_functions):
	point, index_function = get_next_point_function(max_values, point_values, function_values)

	# print('point', point)
	# print('point', point.shape)
	# print('index_function', index_function)

	new_value = target_functions[index_function](np.transpose(point[:,np.newaxis]))


	# if measures[index_function] == 1 and BF_data[0] is not None:
	# 	## we observe the base function thus append it to BF data
	# 	new_inputs = np.append(BF_data[0], point[:,np.newaxis], axis=0)
	# 	new_outputs = np.append(BF_data[1], new_value, axis=0)
	# 	BF_data = [new_inputs, new_outputs]
	# else:
	# 	## Append it to PF data
	# 	new_PF_data = [[None]*len(PF_data[0]),[None]*len(PF_data[1])]
	# 	for i in range(len(PF_data[0])):
	# 		if i == index_function:
	# 			new_PF_data[0][i] = np.append(PF_data[0][index_function], 
	# 													   point[:,np.newaxis], axis=0)
	# 			new_PF_data[1][i] = np.append(PF_data[1][index_function], 
	# 													   new_value, axis=0)
	# 		else:
	# 			new_PF_data[0][i] = PF_data[0][index_function]
	# 			new_PF_data[1][i] = PF_data[1][index_function]
		
	# 	PF_data = new_PF_data

	print('### index_function', index_function)

	# print('BF_data inside', BF_data)
	# print('PF_data inside', PF_data)

	new_PF_data = [[None]*len(PF_data[0]),[None]*len(PF_data[1])]
	for i in range(len(measures)):
		#print('i', i)
		#print('i', measures[i])
		if i == index_function:
			if measures[i] == 1:
				## we observe the base function thus append it to BF data
				new_inputs = np.append(BF_data[0], np.transpose(point[:,np.newaxis]), axis=0)
				new_outputs = np.append(BF_data[1], new_value, axis=0)
				BF_data = [new_inputs, new_outputs]
			else:
				## Append it to PF data
				new_PF_data[0][i] = np.append(PF_data[0][i], np.transpose(point[:,np.newaxis]), axis=0)
				new_PF_data[1][i] = np.append(PF_data[1][i], new_value, axis=0)
				
		else:
			if measures[i] != 1:
				new_PF_data[0][i] = PF_data[0][i]
				new_PF_data[1][i] = PF_data[1][i]
				
		
	PF_data = new_PF_data

	return BF_data, PF_data, index_function
