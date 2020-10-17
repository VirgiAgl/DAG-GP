## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns

## Import emukit function
import emukit
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.acquisition import Acquisition
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core.optimization import GradientAcquisitionOptimizer

import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .cost_functions import *
from .causal_acquisition_functions import CausalExpectedImprovement
from .causal_optimizer import CausalGradientAcquisitionOptimizer


def get_new_dict_x(x_new, intervention_variables):
    x_new_dict = {}

    for i in range(len(intervention_variables)):
      x_new_dict[intervention_variables[i]] = x_new[0, i]
    return x_new_dict


def list_interventional_ranges(dict_ranges, intervention_variables):
    list_min_ranges = []
    list_max_ranges = []
    for j in range(len(intervention_variables)):
      list_min_ranges.append(dict_ranges[intervention_variables[j]][0])
      list_max_ranges.append(dict_ranges[intervention_variables[j]][1])
    return list_min_ranges, list_max_ranges


def get_interventional_dict(intervention_variables):
    interventional_dict = {}
    for i in range(len(intervention_variables)):
      interventional_dict[intervention_variables[i]] = ''
    return interventional_dict


def initialise_dicts(exploration_set, task):
    current_best_x = {}
    current_best_y = {}
    x_dict_mean = {}
    x_dict_var = {}
    dict_interventions = []


    for i in range(len(exploration_set)):
      variables = exploration_set[i]
      if len(variables) == 1:
        variables = variables[0]
      if len(variables) > 1:
        num_var = len(variables)
        string = ''
        for j in range(num_var):
          string += variables[j]
        variables = string

      ## This is creating a list of strings 
      dict_interventions.append(variables)


      current_best_x[variables] = []
      current_best_y[variables] = []

      x_dict_mean[variables] = {}
      x_dict_var[variables] = {}

      ## Assign initial values
      if task == 'min':
        current_best_y[variables].append(np.inf)
        current_best_x[variables].append(np.inf)
      else:
        current_best_y[variables].append(-np.inf)
        current_best_x[variables].append(-np.inf)
      
    return current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions


def initialise_dicts_CTF(exploration_set, PF_data_inputs, BF_data_inputs = None, index_BF = None):
    dict_interventions = {}

    for i in range(len(exploration_set)):
        variables = exploration_set[i]
        if len(variables) == 1:
            name = variables[0]
        else:
            num_var = len(variables)
            string = ''
            for j in range(num_var):
                string += variables[j]
            variables = string
            name = variables

        if  index_BF is not None and i == index_BF:
            dict_interventions[name] = BF_data_inputs
        else:
            dict_interventions[name] = PF_data_inputs[i]
      
    return dict_interventions


def add_data(original, new):
    data_x = np.append(original[0], new[0], axis=0)
    data_y = np.append(original[1], new[1], axis=0)
    return data_x, data_y


def find_current_global(current_y, dict_interventions, task):
    ##�This function finds the optimal value and variable at every iteration
    dict_values = {}
    for j in range(len(dict_interventions)):
        dict_values[dict_interventions[j]] = []

    for variable, value in current_y.items():
        if len(value) > 0:
          if task == 'min':
            dict_values[variable] = np.min(current_y[variable])
          else:
            dict_values[variable] = np.max(current_y[variable])
    if task == 'min':        
      opt_variable = min(dict_values, key=dict_values.get)
    else:
      opt_variable = max(dict_values, key=dict_values.get)
    
    opt_value = dict_values[opt_variable]
    return opt_value


def find_next_y_point_CTF(Transferred_mean, Transferred_covariance, current_global, costs, graph):
    ##This function optimises the acquisition function and return the next point together with the 
    ## corresponding y value for the acquisition function
    test_inputs_list = graph.get_test_inputs_list(size = 300)
    
    exploration_set, I , C = graph.get_sets_CTF()

    y_acquisition_list = []
    x_new_list = []
    improvement_list =[]


    start_index = 0
    for i in range(len(exploration_set[:-1])):
        n_test = test_inputs_list[i].shape[0]


        standard_deviation = np.sqrt(np.clip(np.diagonal(Transferred_covariance[0])[start_index:(n_test+start_index)], 0.0001, np.inf))
        u, pdf, cdf = get_standard_normal_pdf_cdf(current_global, Transferred_mean[0][start_index:(n_test+start_index),0], standard_deviation)

        improvement = (standard_deviation * (u * cdf + pdf))/len(exploration_set[i])
        
        index = np.where(improvement == np.max(improvement))[0][0]

        start_index += n_test

        x_new_list.append(test_inputs_list[i][index])
        improvement_list.append(improvement[index])

    ## BF computation 
    if len(C) ==0:
        standard_deviation = np.sqrt(np.clip(np.diagonal(Transferred_covariance[1]), 0.0001, np.inf))
        u, pdf, cdf = get_standard_normal_pdf_cdf(current_global, Transferred_mean[1][:,0], standard_deviation)

        improvement = (standard_deviation * (u * cdf + pdf))/len(I)
        
        index = np.where(improvement == np.max(improvement))[0][0]
        x_new_list.append(test_inputs_list[-1][index])
        improvement_list.append(improvement[index])

    return improvement_list, x_new_list    

        
def find_next_y_point(space, model, current_global_best, evaluated_set, costs_functions, graph, task = 'min'):
    ## This function optimises the acquisition function and return the next point together with the 
    ## corresponding y value for the acquisition function

    if len(evaluated_set) == 1:
        min_intervention0 = list_interventional_ranges(graph.get_interventional_ranges(), evaluated_set[0])[0]
        max_intervention0 = list_interventional_ranges(graph.get_interventional_ranges(), evaluated_set[0])[1]

        inputs = np.linspace(min_intervention0, max_intervention0, 300)
    else:
        min_intervention0 = list_interventional_ranges(graph.get_interventional_ranges(), evaluated_set[0])[0]
        max_intervention0 = list_interventional_ranges(graph.get_interventional_ranges(), evaluated_set[0])[1]

        min_intervention1 = list_interventional_ranges(graph.get_interventional_ranges(), evaluated_set[1])[0],
        max_intervention1 = list_interventional_ranges(graph.get_interventional_ranges(), evaluated_set[1])[1]

        inputs0 = np.linspace(min_intervention0, max_intervention0, 20)
        inputs1 = np.linspace(min_intervention1, max_intervention1, 20)

        inputs = combine_arrays([inputs0, inputs1])
        #print('inputs', inputs)

    mean, var = model.predict(inputs)
    standard_deviation = np.sqrt(var)

    u, pdf, cdf = get_standard_normal_pdf_cdf(current_global_best, mean, standard_deviation)
    improvement = (standard_deviation * (u * cdf + pdf))/len(evaluated_set)

    index = np.where(improvement == np.max(improvement))[0][0]

    x_new = inputs[index]
    y_acquisition = improvement[index]



    return y_acquisition, x_new    


def fit_single_GP_model(X, Y, parameter_list, ard = False):
    kernel = RBF(X.shape[1], ARD = parameter_list[2], lengthscale=parameter_list[0], variance = parameter_list[1]) 
    gp = GPRegression(X = X, Y = Y, kernel = kernel, noise_var= 1e-5)
    gp.optimize()
    return gp

def get_standard_normal_pdf_cdf(x: np.array, mean: np.array, standard_deviation: np.array):
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf

def clip_negative_values(variance):
    ## If values of variance are negative cause of the MC approximation this function clips them
    
    modified_variance = copy.deepcopy(variance)
    for i in range(variance.shape[0]):
        if i == 0:
            if variance[i] < 0:
                if variance[i+1]<0:
                    modified_variance[i] = variance[i+2]
                else:
                    modified_variance[i] = variance[i+1]
                        
        else:
            if variance[i] < 0:
                if variance[i-1] < 0:
                    if variance[i-2] < 0:
                        if variance[i-3] < 0:
                            if variance[i-4] < 0:
                                modified_variance[i] = 0.
                            else:
                                modified_variance[i] = variance[i-4]
                        else:
                            modified_variance[i] = variance[i-3]
                    else:
                        modified_variance[i] = variance[i-2]
                else:
                    modified_variance[i] = variance[i-1]
    
    return modified_variance

def combine_arrays(inputs):
    if len(inputs) == 1:
        value = inputs[0]
    else:
        inputs = tuple(inputs)
        value = np.dstack(np.meshgrid(*inputs)).ravel('F').reshape(len(inputs),-1).T
    return value

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


