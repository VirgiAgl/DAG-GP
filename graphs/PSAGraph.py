## Import basic packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
from scipy.interpolate import interp1d
import scipy
import itertools
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from scipy.stats import truncnorm
import math
from scipy.stats import norm

from . import graph
from .PSAGraph_DoFunctions import *
from .PSAGraph_CostFunctions import define_costs

from utils_functions import GeneralMeasure
from utils_functions import ConditionalDistribution
from utils_functions import get_interventional_dict
from utils_functions import Intervention_function
from utils_functions import list_interventional_ranges
from utils_functions import combine_arrays


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


class PSAGraph(graph.GraphStructure):

	def __init__(self, observational_samples):

		self.age = np.asarray(observational_samples['age'])[:,np.newaxis]
		self.bmi = np.asarray(observational_samples['bmi'])[:,np.newaxis]
		self.A = np.asarray(observational_samples['A'])[:,np.newaxis]
		self.S = np.asarray(observational_samples['S'])[:,np.newaxis]
		self.cancer = np.asarray(observational_samples['cancer'])[:,np.newaxis]
		self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

	def define_SEM(self):

		def f_age(epsilon, **kwargs):
		  return np.random.uniform(low=55, high=75)

		def f_bmi(epsilon, age, **kwargs):
		  return np.random.normal(27.0 - 0.01*age, 0.7) 

		def f_A(epsilon, age, bmi, **kwargs):
		  return sigmoid(-8.0 + 0.10*age + 0.03*bmi)

		def f_S(epsilon, age, bmi, **kwargs):
		  return sigmoid(-13.0 + 0.10*age + 0.20*bmi)

		def f_cancer(epsilon, age, bmi, S, A, **kwargs):
		  return sigmoid(2.2 - 0.05*age + 0.01*bmi - 0.04*S + 0.02*A)

		def f_Y(epsilon, age, bmi, S, A, cancer, **kwargs):
		  return np.random.normal(6.8 + 0.04*age - 0.15*bmi - 0.60*S 
								+ 0.55*A + 1.00*cancer, 0.4)


		graph = OrderedDict ([
				('age', f_age),
				('bmi', f_bmi),
				('A', f_A),
				('S', f_S),
				('cancer', f_cancer),
				('Y', f_Y)
				])

		return graph

	def get_sets(self):
		MIS = [['A'], ['S'], ['A', 'S']]
		POMIS = [['A'], ['S'], ['A', 'S']]
		manipulative_variables = ['A', 'S']
		return MIS, POMIS, manipulative_variables


	def get_sets_CTF(self):
		## ES is the exploration set thus includes all sets for which we want to learn the intervention function plus 
		## the base set as last position in the list 
		## I is the intervention set in the base function
		## C is the confounders set in the base function which in this case is empty
		ES = [['A'], ['S'], ['A', 'S'], 
			['A', 'S', 'cancer', 'age', 'bmi']]
		I = [['A']]
		C = []
		return ES, I , C
	
	def get_cost_structure(self, type_cost):
		costs = define_costs(type_cost)
		return costs
	
	def get_info_BF(self):
		dim = 5.
		inputs = ['A', 'S', 'cancer', 'age', 'bmi']
		index = 3
		return dim, inputs, index


	def get_IMs(self, functions):
		## This function fits all conditional distributions needed as IMs

		gp_bmiage = functions['gp_bmiage']
		gp_Sagebmi = functions['gp_Sagebmi']
		gp_Aagebmi = functions['gp_Aagebmi']
		gp_cancerASagebmi = functions['gp_cancerASagebmi']

		mu, std = norm.fit(self.age)

		bmi_age = ConditionalDistribution(inputs=['age'], gp_function = gp_bmiage)
		S_agebmi = ConditionalDistribution(inputs=['age', 'bmi'], gp_function = gp_Sagebmi)
		A_agebmi = ConditionalDistribution(inputs=['age', 'bmi'], gp_function = gp_Aagebmi)
				

		cancer_ASagebmi = ConditionalDistribution(inputs=['A', 'S', 'age', 'bmi'], 
									gp_function = gp_cancerASagebmi, inter_var = 'A')
		cancer_ASagebmi2 = ConditionalDistribution(inputs=['A', 'S', 'age', 'bmi'], 
									gp_function = gp_cancerASagebmi, inter_var = 'S')
		cancer_ASagebmi3 = ConditionalDistribution(inputs=['A', 'S', 'age', 'bmi'], 
									gp_function = gp_cancerASagebmi, inter_var = ['A', 'S'])

		IM_A = [GeneralMeasure('age', 'A', mean = mu, variance= std**2), 
					  GeneralMeasure('bmi', 'A', type_dist='conditional', cond_gp = bmi_age),
					  GeneralMeasure('S', 'A', type_dist='conditional', cond_gp = S_agebmi),
					  GeneralMeasure('cancer', 'A', type_dist='conditional', cond_gp = cancer_ASagebmi)]

		IM_S = [GeneralMeasure('age', 'S', mean = mu, variance= std**2), 
					  GeneralMeasure('bmi', 'S', type_dist='conditional', cond_gp = bmi_age),
					  GeneralMeasure('A', 'S', type_dist='conditional', cond_gp = A_agebmi),
					  GeneralMeasure('cancer', 'S', type_dist='conditional', cond_gp = cancer_ASagebmi2)]


		IM_A_S = [GeneralMeasure('age', ['A', 'S'], mean = mu, variance= std**2), 
							GeneralMeasure('bmi', ['A', 'S'], type_dist='conditional', cond_gp = bmi_age),
							GeneralMeasure('cancer', ['A', 'S'], type_dist='conditional', cond_gp = cancer_ASagebmi3)]


		return [IM_A, IM_S, IM_A_S]


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
		ES, _ , _ = self.get_sets_CTF()
		
		## TO DO -- check if it expands to generic dimensions
		for j in range(len(ES)):
			if len(ES[j]) == 1:
				variable = ES[j][0]
				min_value = dict_ranges[variable][0]
				max_value = dict_ranges[variable][1]

				inputs = np.linspace(min_value, max_value, size)[:,np.newaxis]
				test_inputs_list.append(inputs)
			else:
				subset_inputs = []
				# Need to reduce the size for computational reasons
				size =int(np.sqrt(size))


				for i in range(len(ES[j])):
					variable = ES[j][i]
					min_value = dict_ranges[variable][0]
					max_value = dict_ranges[variable][1]     
					inputs = np.linspace(min_value, max_value, 20)[:, None]
					subset_inputs.append(inputs)

				inputs = combine_arrays(subset_inputs)
				test_inputs_list.append(inputs)

		return test_inputs_list


	def get_interventional_ranges(self):
		min_intervention_A = 0.
		max_intervention_A = 1.

		min_intervention_S = 0.
		max_intervention_S = 1.

		dict_ranges = OrderedDict ([
		  ('A', [min_intervention_A, max_intervention_A]),
		  ('S', [min_intervention_S, max_intervention_S]),
		  ('cancer', [0., 0.5]),
		  ('age', [55, 75]),
		  ('bmi', [24, 28]),
		])
		return dict_ranges


	def fit_all_models(self, algorithm = None):
		noise_var = 10.

		### Fit all conditional models
		kernel = RBF(self.age.shape[1], variance=1.0, ARD=True)
		gp_bmiage = GPRegression(X = self.age, Y = self.bmi, kernel = kernel, noise_var= 0.0001)
		gp_bmiage.likelihood.variance.fix(1e-2)
		gp_bmiage.optimize()

		kernel = RBF(np.hstack((self.age,self.bmi)).shape[1], variance=1.0, ARD=True)
		gp_Sagebmi = GPRegression(X = np.hstack((self.age,self.bmi)), Y = self.S, kernel = kernel, noise_var= 0.0001)
		gp_Sagebmi.likelihood.variance.fix(1e-2)
		gp_Sagebmi.optimize()

		kernel = RBF(np.hstack((self.age,self.bmi)).shape[1], variance=1.0, ARD=True)
		gp_Aagebmi = GPRegression(X = np.hstack((self.age,self.bmi)), Y = self.A, kernel = kernel, noise_var= 0.0001)
		gp_Aagebmi.likelihood.variance.fix(1e-2)
		gp_Aagebmi.optimize()

		kernel = RBF(np.hstack((self.A, self.S, self.age,self.bmi)).shape[1], variance=1.0, ARD=True)
		gp_cancerASagebmi = GPRegression(X = np.hstack((self.A, self.S, self.age,self.bmi)), Y = self.cancer, kernel = kernel, noise_var= 0.0001)
		gp_cancerASagebmi.likelihood.variance.fix(1e-2)
		gp_cancerASagebmi.optimize()

		kernel = RBF(np.hstack((self.S, self.age, self.bmi)).shape[1], variance=1.0, ARD=True)
		gp_YSagebmi= GPRegression(X = np.hstack((self.S, self.age, self.bmi)), 
								Y = self.Y, kernel = kernel, noise_var= 0.0001)
		gp_YSagebmi.likelihood.variance.fix(1e-2)
		gp_YSagebmi.optimize()  


		kernel = RBF(np.hstack((self.A, self.age, self.bmi)).shape[1], variance=1.0, ARD=True)
		gp_YAagebmi= GPRegression(X = np.hstack((self.A, self.age, self.bmi)), 
								Y = self.Y, kernel = kernel, noise_var= 0.0001)
		gp_YAagebmi.likelihood.variance.fix(1e-2)
		gp_YAagebmi.optimize()  


		kernel = RBF(np.hstack((self.A, self.S, self.age, self.bmi)).shape[1], variance=1.0, ARD=True)
		gp_YASagebmi= GPRegression(X = np.hstack((self.A, self.S, self.age, self.bmi)), 
								Y = self.Y, kernel = kernel, noise_var= 0.0001)
		gp_YASagebmi.likelihood.variance.fix(1e-2)
		gp_YASagebmi.optimize()  


		### ALL
		kernel = RBF(np.hstack((self.A, self.S, self.age,self.bmi, self.cancer)).shape[1], variance=1.0, ARD=True)
		gp_YASagebmicancer = GPRegression(X = np.hstack((self.A, self.S, self.age,self.bmi, self.cancer)), Y = self.Y, kernel = kernel, noise_var= 0.0001)
		gp_YASagebmicancer.likelihood.variance.fix(1e-2)
		gp_YASagebmicancer.optimize()


		print('I have fitted the models')

		## Aggregate all the estimated distributions
		functions = OrderedDict ([
						('gp_YASagebmicancer', gp_YASagebmicancer),
						('gp_bmiage', gp_bmiage),
						('gp_Sagebmi', gp_Sagebmi),
						('gp_Aagebmi', gp_Aagebmi),
						('gp_cancerASagebmi', gp_cancerASagebmi),
						('gp_YSagebmi', gp_YSagebmi),
						('gp_YAagebmi', gp_YAagebmi),
						('gp_YASagebmi',gp_YASagebmi)])

		return functions


	def refit_models(self, observational_samples, model_type = 0):
		A = np.asarray(observational_samples['A'])[:,np.newaxis]
		S = np.asarray(observational_samples['S'])[:,np.newaxis]
		bmi = np.asarray(observational_samples['bmi'])[:,np.newaxis]
		cancer = np.asarray(observational_samples['cancer'])[:,np.newaxis]
		age = np.asarray(observational_samples['age'])[:,np.newaxis]
		Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

		noise_var = 10.

		### Fit all conditional models
		kernel = RBF(age.shape[1], variance=1.0, ARD=True)
		gp_bmiage = GPRegression(X = age, Y = bmi, kernel = kernel, noise_var= 0.0001)
		gp_bmiage.likelihood.variance.fix(1e-2)
		gp_bmiage.optimize()

		kernel = RBF(np.hstack((age,bmi)).shape[1], variance=1.0, ARD=True)
		gp_Sagebmi = GPRegression(X = np.hstack((age,bmi)), Y = S, kernel = kernel, noise_var= 0.0001)
		gp_Sagebmi.likelihood.variance.fix(1e-2)
		gp_Sagebmi.optimize()

		kernel = RBF(np.hstack((age,bmi)).shape[1], variance=1.0, ARD=True)
		gp_Aagebmi = GPRegression(X = np.hstack((age,bmi)), Y = A, kernel = kernel, noise_var= 0.0001)
		gp_Aagebmi.likelihood.variance.fix(1e-2)
		gp_Aagebmi.optimize()

		kernel = RBF(np.hstack((A, S, age,bmi)).shape[1], variance=1.0, ARD=True)
		gp_cancerASagebmi = GPRegression(X = np.hstack((A, S,age,bmi)), Y = cancer, kernel = kernel, noise_var= 0.0001)
		gp_cancerASagebmi.likelihood.variance.fix(1e-2)
		gp_cancerASagebmi.optimize()

		kernel = RBF(np.hstack((S, age, bmi)).shape[1], variance=1.0, ARD=True)
		gp_YSagebmi= GPRegression(X = np.hstack((S, age, bmi)), 
								Y = Y, kernel = kernel, noise_var= 0.0001)
		gp_YSagebmi.likelihood.variance.fix(1e-2)
		gp_YSagebmi.optimize()  


		kernel = RBF(np.hstack((A, age, bmi)).shape[1], variance=1.0, ARD=True)
		gp_YAagebmi= GPRegression(X = np.hstack((A, age, bmi)), Y = Y, kernel = kernel, noise_var= 0.0001)
		gp_YAagebmi.likelihood.variance.fix(1e-2)
		gp_YAagebmi.optimize()  


		kernel = RBF(np.hstack((A, S, age, bmi)).shape[1], variance=1.0, ARD=True)
		gp_YASagebmi= GPRegression(X = np.hstack((A, S, age, bmi)), Y = Y, kernel = kernel, noise_var= 0.0001)
		gp_YASagebmi.likelihood.variance.fix(1e-2)
		gp_YASagebmi.optimize()  


		### ALL
		kernel = RBF(np.hstack((A, S, age,bmi, cancer)).shape[1], variance=1.0, ARD=True)
		gp_YASagebmicancer = GPRegression(X = np.hstack((A, S, age,bmi, cancer)), Y = Y, kernel = kernel, noise_var= 0.0001)
		gp_YASagebmicancer.likelihood.variance.fix(1e-2)
		gp_YASagebmicancer.optimize()


		print('I have fitted the models')

		## Aggregate all the estimated distributions
		functions = OrderedDict ([
						('gp_YASagebmicancer', gp_YASagebmicancer),
						('gp_bmiage', gp_bmiage),
						('gp_Sagebmi', gp_Sagebmi),
						('gp_Aagebmi', gp_Aagebmi),
						('gp_cancerASagebmi', gp_cancerASagebmi),
						('gp_YSagebmi', gp_YSagebmi),
						('gp_YAagebmi', gp_YAagebmi),
						('gp_YASagebmi',gp_YASagebmi)])
		return functions

	def get_all_do(self):
		do_dict = {}
		do_dict['compute_do_S'] = compute_do_S
		do_dict['compute_do_A'] = compute_do_A
		do_dict['compute_do_AS'] = compute_do_AS
		do_dict['BF'] = BF
		return do_dict



