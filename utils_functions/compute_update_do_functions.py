## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns


def get_do_function_name(intervention_variables):
    string = ''
    for i in range(len(intervention_variables)):
        string += str(intervention_variables[i]) 
    total_string = 'compute_do_' + string
    return total_string


def update_all_do_functions(graph, exploration_set, functions, dict_interventions, observational_samples, x_dict_mean, x_dict_var):
    ## This function is computing the DO mean and variance for each set in exploration set 
    mean_functions_list = []
    var_functions_list = []

    for j in range(len(exploration_set)):
        mean_functions_list.append(update_mean_fun(graph, functions, dict_interventions[j], observational_samples, x_dict_mean))
        var_functions_list.append(update_var_fun(graph, functions, dict_interventions[j], observational_samples, x_dict_var))
    return mean_functions_list, var_functions_list


def get_do_fun(graph, functions, observational_samples, variables = None, function_name = None):

    def compute_values(x, compute_do):
        mean_do = np.zeros((x.shape[0], 1))
        var_do = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            mean_do[i], var_do[i] = compute_do(observational_samples, functions, value = x[i][:,np.newaxis])
        return mean_do, var_do


    do_functions = graph.get_all_do()
    
    if function_name is None:
        function_name = get_do_function_name(variables)


    def mean_function_do(x):
        mean_do, _ = compute_values(x, do_functions[function_name])
        return np.float64(mean_do)

    def var_function_do(x):
        _ , var_do = compute_values(x, do_functions[function_name])
        return np.float64(var_do)

    return mean_function_do, var_function_do

  
def update_mean_fun(graph, functions, variables, observational_samples, xi_dict_mean):

    def compute_mean(num_interventions, x, xi_dict_mean, compute_do):
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_mean:
                mean_do[i] = xi_dict_mean[xi_str]
            else:
                mean_do[i], _ = compute_do(observational_samples, functions, value = x[i])
                xi_dict_mean[xi_str] = mean_do[i]

        return mean_do


    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def mean_function_do(x):
        num_interventions = x.shape[0]
        mean_do = compute_mean(num_interventions, x, xi_dict_mean[variables], do_functions[function_name])
        return np.float64(mean_do)

    return mean_function_do


def update_var_fun(graph, functions, variables, observational_samples, xi_dict_var):

    def compute_var(num_interventions, x, xi_dict_var, compute_do):
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_var:
                var_do[i] = xi_dict_var[xi_str]
            else:
                _, var_do[i] = compute_do(observational_samples, functions, value = x[i])
                xi_dict_var[xi_str] = var_do[i]

        return var_do

    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def var_function_do(x):
        num_interventions = x.shape[0]    
        var_do = compute_var(num_interventions, x, xi_dict_var[variables], do_functions[function_name])
        return np.float64(var_do)

    return var_function_do  