import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from utils_functions import fit_single_GP_model

## Import Graph function
from . import graph
from .ToyGraph_DoFunctions import *
from .ToyGraph_CostFunctions import define_costs

## Import function to define IM
from utils_functions import GeneralMeasure
from utils_functions import ConditionalDistribution
from utils_functions import get_interventional_dict
from utils_functions import Intervention_function
from utils_functions import list_interventional_ranges


class ToyGraph(graph.GraphStructure):
	"""
	An instance of the class graph giving the graph structure in the toy example 
	
	Parameters
	----------
	"""
	def __init__(self, observational_samples):

		self.X = np.asarray(observational_samples['X'])[:,np.newaxis]
		self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]
		self.Z = np.asarray(observational_samples['Z'])[:,np.newaxis]

	def define_SEM(self):

		def fx(epsilon, **kwargs):
		  return epsilon[0]

		def fz(epsilon, X, **kwargs):
		  return np.exp(-X) + epsilon[1]

		def fy(epsilon, Z, **kwargs):
		  return np.cos(Z) - np.exp(-Z/20.) + epsilon[2]  

		graph = OrderedDict ([
		  ('X', fx),
		  ('Z', fz),
		  ('Y', fy),
		])

		return graph


	def get_sets_CTF(self):
		## ES is the exploration set thus includes all sets for which we want to learn the intervention function plus 
		## the base set as last position in the list 
		## I is the intervention set in the base function
		## C is the confounders set in the base function which in this case is empty
		ES = [['X'], ['Z']]
		I = [['Z']]
		C = []
		return ES, I , C
	
	def get_sets(self):
		MIS = [['X'], ['Z']]
		POMIS = [['Z']]
		manipulative_variables = ['X', 'Z']
		return MIS, POMIS, manipulative_variables


	def get_info_BF(self):
		dim = 1.
		inputs = ['Z']
		index = 1
		return dim, inputs, index


	def get_IMs(self, functions):
		## This function fits all conditional distributions needed as IMs
		## Function returning the IM for each set in ES. When the IM is an empty list the set is the BS
		gp_Z = functions['Z']

		Z_X = ConditionalDistribution(inputs=['X'], gp_function = gp_Z, inter_var = 'X')

		IM_x = [GeneralMeasure('Z', 'X', type_dist='conditional', cond_gp = Z_X)]

		return [IM_x, 1.]


	def get_intervention_functions(self):
		exploration_set, _, _ = self.get_sets_CTF()

		target_function_list = [None]*len(exploration_set)
		space_list = [None]*len(exploration_set)

		## Define intervention function
		for s in range(len(exploration_set)):
			target_function_list[s], space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
												model = self.define_SEM(), target_variable = 'Y',
												min_intervention = list_interventional_ranges(self.get_interventional_ranges(), exploration_set[s])[0],
												max_intervention = list_interventional_ranges(self.get_interventional_ranges(), exploration_set[s])[1])

		return target_function_list, space_list


	def get_test_inputs_list(self, size):
		## To be extended to have cartesian product
		test_inputs_list = []

		dict_ranges = self.get_interventional_ranges()
		ES, _ , _= self.get_sets_CTF()
		
		for j in range(len(ES)):
			variable = ES[j][0]
			min_value = dict_ranges[variable][0]
			max_value = dict_ranges[variable][1]

			inputs = np.linspace(min_value, max_value, size)[:,np.newaxis]
			test_inputs_list.append(inputs)

		return test_inputs_list


	def get_interventional_ranges(self):
		min_intervention_x = -5
		max_intervention_x = 5

		min_intervention_z = -5
		max_intervention_z = 20


		dict_ranges = OrderedDict ([
		  ('X', [min_intervention_x, max_intervention_x]),
		  ('Z', [min_intervention_z, max_intervention_z]),
		])
		return dict_ranges


	def fit_all_models(self, algorithm = None):
		## This function fits all conditional distributions needed for do calculus 
		functions = {}

		kernel = RBF(self.Z.shape[1], ARD = False, lengthscale=1., variance = 1.) 
		gp_Y = GPRegression(X = self.Z, Y = self.Y, kernel = kernel, noise_var= 1.)
		gp_Y.optimize()

		kernel = RBF(self.X.shape[1], ARD = False, lengthscale=1., variance =1.) 
		gp_Z = GPRegression(X = self.X, Y = self.Z, kernel = kernel)
		gp_Z.optimize()

		functions = OrderedDict ([
			('Y', gp_Y),
			('Z', gp_Z)
			])

		return functions


	def refit_models(self, observational_samples, model_type = 0):
		X = np.asarray(observational_samples['X'])[:,np.newaxis]
		Z = np.asarray(observational_samples['Z'])[:,np.newaxis]
		Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

		functions = {}

		num_features = Z.shape[1]
		kernel = RBF(num_features, ARD = False, lengthscale=1., variance = 1.) 
		gp_Y = GPRegression(X = Z, Y = Y, kernel = kernel, noise_var= 1.)
		
		gp_Y.optimize()

		num_features = X.shape[1]
		kernel = RBF(num_features, ARD = False, lengthscale=1., variance =1.) 
		gp_Z = GPRegression(X = X, Y = Z, kernel = kernel)
		gp_Z.optimize()
		
		functions = OrderedDict ([
			('Y', gp_Y),
			('Z', gp_Z)
			])


		return functions


	def get_cost_structure(self, type_cost):
		costs = define_costs(type_cost)
		return costs


	def get_all_do(self):
		do_dict = {}
		do_dict['compute_do_X'] = compute_do_X
		do_dict['compute_do_Z'] = compute_do_Z
		do_dict['compute_do_XZ'] = compute_do_XZ
		do_dict['BF'] = compute_do_Z
		return do_dict



