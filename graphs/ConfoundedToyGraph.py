import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from scipy.stats import norm
from collections import OrderedDict

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from utils_functions import fit_single_GP_model

## Import Graph function
from . import graph
from .ConfoundedToyGraph_DoFunctions import *

## Import function to define IM
from utils_functions import GeneralMeasure
from utils_functions import ConditionalDistribution
from utils_functions import get_interventional_dict
from utils_functions import Intervention_function
from utils_functions import list_interventional_ranges

from utils_functions import combine_arrays


class ConfoundedToyGraph(graph.GraphStructure):
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

        def fU1(epsilon, **kwargs):
            return epsilon[0]

        def fx(epsilon, U1, **kwargs):
            return epsilon[1] + U1

        def fz(epsilon, X, **kwargs):
            return np.exp(-X) + epsilon[2]

        def fy(epsilon, Z, U1, **kwargs):
            return np.cos(Z) - np.exp(-Z/20.) + epsilon[3] + U1

        graph = OrderedDict ([
          ('U1', fU1),
          ('X', fx),
          ('Z', fz),
          ('Y', fy),
        ])

        return graph


    def get_sets_CTF(self):
        ## ES is the exploration set thus includes all sets for which we want to learn the intervention function plus 
        ## the base set as last position in the list 
        ## I is the intervention set in the base function
        ## C is the confounders set in the base function
        ES = [['X'], ['Z'], ['Z', 'X']]
        I = [['Z']]
        C = [['X']]
        return ES, I , C


    def get_info_BF(self):
        dim = 2.
        inputs = ['Z', 'X']
        index = 2.
        return dim, inputs, index


    def get_IMs(self, functions):
        ## This function fits all conditional distributions needed as IMs
        ## Function returning the IM for each set in ES. When the IM is an empty list the set is the BS
        # num_features = self.X.shape[1]
        # kernel = RBF(num_features, ARD = False, lengthscale=1., variance =1.) 
        # gp_Z = GPRegression(X = self.X, Y = self.Z, kernel = kernel)
        # gp_Z.optimize()

        gp_Z = functions['gp_Z']

        mu, std = norm.fit(self.X)

        Z_X = ConditionalDistribution(inputs=['X'], gp_function = gp_Z, inter_var = 'X')

        IM_x = [GeneralMeasure('X', 'X', mean = mu, variance= std**2), 
                GeneralMeasure('Z', 'X', type_dist='conditional', cond_gp = Z_X)]

        IM_z = [GeneralMeasure('X', 'Z', mean = mu, variance= std**2)]

        return [IM_x, IM_z]
    

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

        kernel = RBF(self.X.shape[1], ARD = False, lengthscale=1., variance =1.) 
        gp_Z = GPRegression(X = self.X, Y = self.Z, kernel = kernel)
        gp_Z.optimize()

        kernel = RBF(np.hstack((self.Z,self.X)).shape[1], ARD = False, lengthscale=1., variance =1.) 
        gp_YZX = GPRegression(X = np.hstack((self.Z,self.X)), Y = self.Y, kernel = kernel)
        gp_YZX.optimize()

        functions = OrderedDict ([
            ('gp_Z', gp_Z),
            ('gp_YZX', gp_YZX)
            ])

        return functions


    def get_all_do(self):
        do_dict = {}
        do_dict['compute_do_X'] = compute_do_X
        do_dict['compute_do_Z'] = compute_do_Z
        do_dict['BF'] = BF
        return do_dict



