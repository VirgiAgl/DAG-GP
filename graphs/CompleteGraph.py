import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
import math
from scipy.stats import norm
from collections import OrderedDict

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from utils_functions import fit_single_GP_model

from . import graph
from .CompleteGraph_DoFunctions import *
from .CompleteGraph_CostFunctions import define_costs

## Import function to define IM
from utils_functions import GeneralMeasure
from utils_functions import ConditionalDistribution
from utils_functions import combine_arrays
from utils_functions import get_interventional_dict
from utils_functions import Intervention_function
from utils_functions import list_interventional_ranges


class CompleteGraph(graph.GraphStructure):
    """
    An instance of the class graph giving the graph structure in the synthetic example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples):

        self.A = np.asarray(observational_samples['A'])[:,np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:,np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:,np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:,np.newaxis]
        self.E = np.asarray(observational_samples['E'])[:,np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

    def define_SEM(self):

        def fU1(epsilon, **kwargs):
          return epsilon[0]

        def fU2(epsilon, **kwargs):
          return epsilon[1]

        def fF(epsilon, **kwargs):
          return epsilon[8]

        def fA(epsilon, U1, F, **kwargs):
          return F**2 + U1 + epsilon[2]

        def fB(epsilon, U2, **kwargs):
          return U2 + epsilon[3]

        def fC(epsilon, B, **kwargs):
          return np.exp(-B) + epsilon[4]

        def fD(epsilon, C, **kwargs):
          return np.exp(-C)/10. + epsilon[5]

        def fE(epsilon, A, C, **kwargs):
          return np.cos(A) + C/10. + epsilon[6]

        def fY(epsilon, D, E, U1, U2, **kwargs):
          return np.cos(D) - D/5. + np.sin(E) - E/4. + U1 + np.exp(-U2) + epsilon[7]

        graph = OrderedDict ([
              ('U1', fU1),
              ('U2', fU2),
              ('F', fF),
              ('A', fA),
              ('B', fB),
              ('C', fC),
              ('D', fD),
              ('E', fE),
              ('Y', fY),
            ])
        return graph
    
    def get_sets(self):
        MIS = [['B'], ['D'], ['E'], ['B', 'D'], ['B', 'E'], ['D', 'E']]
        POMIS = [['B'], ['D'], ['E'], ['B', 'D'], ['D', 'E']]
        manipulative_variables = ['B', 'D', 'E']
        return MIS, POMIS, manipulative_variables

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs

    def get_sets_CTF(self):
        ## ES is the exploration set thus includes all sets for which we want to learn the intervention function plus 
        ## the base set as last position in the list 
        ## I is the intervention set in the base function
        ## C is the confounders set in the base function
        ES = [['B'], ['D'], ['E'], ['B', 'D'], ['B','E'], ['D','E'], ['D', 'E', 'A', 'B']]
        #ES = [['B'], ['D'], ['E'], ['D', 'E', 'A', 'B']]
        I = [['D', 'E']]
        C = [['A', 'B']]
        return ES, I , C


    def get_info_BF(self):
        dim = 4.
        inputs = ['D', 'E', 'A', 'B']
        index = 6.
        return dim, inputs, index


    def get_IMs(self, functions):
        ## This function fits all conditional distributions needed as IMs
        ## Function returning the IM for each set in ES. When the IM is an empty list the set is the BS
        muA, stdA = norm.fit(self.A)
        muB, stdB = norm.fit(self.B)

        gp_EAB = functions['gp_EAB']
        gp_DABE = functions['gp_DABE']
        gp_DB = functions['gp_DB']

        E_AB = ConditionalDistribution(inputs=['A', 'B'], gp_function = gp_EAB, inter_var = 'B')
        E_AB2 = ConditionalDistribution(inputs=['A', 'B'], gp_function = gp_EAB)
        
        D_ABE = ConditionalDistribution(inputs=['A', 'B', 'E'], gp_function = gp_DABE, inter_var = 'B')
        D_B = ConditionalDistribution(inputs=['B'], gp_function = gp_DB)
        D_B2 = ConditionalDistribution(inputs=['B'], gp_function = gp_DB, inter_var = 'B')


        IM_b = [GeneralMeasure('A', ['B'], mean = muA, variance= stdA**2), 
                GeneralMeasure('B', ['B'], mean = muB, variance= stdB**2),
                GeneralMeasure('E', ['B'], type_dist='conditional', cond_gp = E_AB),
                GeneralMeasure('D', ['B'], type_dist='conditional', cond_gp = D_ABE)]

        IM_d = [GeneralMeasure('A', ['D'], mean = muA, variance= stdA**2), 
                GeneralMeasure('B', ['D'], mean = muB, variance= stdB**2),
                GeneralMeasure('E', ['D'], type_dist='conditional', cond_gp = E_AB2)]
        
        IM_e = [GeneralMeasure('A', ['E'], mean = muA, variance= stdA**2), 
                GeneralMeasure('B', ['E'], mean = muB, variance= stdB**2),
                GeneralMeasure('D', ['E'], type_dist='conditional', cond_gp = D_B)]

        IM_bd = [GeneralMeasure('A', ['B', 'D'], mean = muA, variance= stdA**2), 
                GeneralMeasure('B', ['B', 'D'], mean = muB, variance= stdB**2),
                GeneralMeasure('E', ['B', 'D'], type_dist='conditional', cond_gp = E_AB)]

        IM_be = [GeneralMeasure('A',  ['B', 'E'], mean = muA, variance= stdA**2), 
                GeneralMeasure('B', ['B', 'E'], mean = muB, variance= stdB**2),
                GeneralMeasure('D', ['B', 'E'], type_dist='conditional', cond_gp = D_B2)]
        
        IM_de = [GeneralMeasure('A', ['D', 'E'], mean = muA, variance= stdA**2), 
                GeneralMeasure('B', ['D', 'E'], mean = muB, variance= stdB**2)]


        return [IM_b, IM_d, IM_e, IM_bd, IM_be, IM_de]
        #return [IM_b, IM_d, IM_e]


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
                size_es = math.pow(size, 1./len(ES[j]))
                size_es = 20

                for i in range(len(ES[j])):
                    variable = ES[j][i]
                    min_value = dict_ranges[variable][0]
                    max_value = dict_ranges[variable][1]     
                    inputs = np.linspace(min_value, max_value, size_es)[:, None]
                    subset_inputs.append(inputs)

                inputs = combine_arrays(subset_inputs)
                test_inputs_list.append(inputs)

        return test_inputs_list

    def get_interventional_ranges(self):
        min_intervention_a = -3
        max_intervention_a = 6

        min_intervention_b = -3
        max_intervention_b = 4

        min_intervention_c = -3
        max_intervention_c = 10

        min_intervention_d = -3
        max_intervention_d = 3

        min_intervention_e = -3
        max_intervention_e = 3

        #########
        # min_intervention_e = -6
        # max_intervention_e = 3

        # min_intervention_b = -5
        # max_intervention_b = 4

        # min_intervention_d = -5
        # max_intervention_d = 5



        dict_ranges = OrderedDict ([
          ('A', [min_intervention_a, max_intervention_a]),
          ('B', [min_intervention_b, max_intervention_b]),
          ('C', [min_intervention_c, max_intervention_c]),
          ('D', [min_intervention_d, max_intervention_d]),
          ('E', [min_intervention_e, max_intervention_e])
        ])
        return dict_ranges


    def fit_all_models(self, algorithm = None):
        functions = {}

        if algorithm is None:
            ## We need this for multi task 
            inputs_list = [np.hstack((self.D,self.E,self.A,self.B)), np.hstack((self.A,self.B)), np.hstack((self.A,self.B,self.E)), self.B]
            output_list = [self.Y, self.E, self.D, self.D]
            name_list = ['gp_YDEAB', 'gp_EAB', 'gp_DABE', 'gp_DB']
            parameter_list = [[1.,1., False], [1.,1., False], [1.,1., False], [1.,1., False]]
        else:   
            ## We need this for single task 
            inputs_list = [self.B, np.hstack((self.D,self.C)), np.hstack((self.B,self.C)), np.hstack((self.A,self.C,self.E)), np.hstack((self.B,self.C,self.D)), 
                        np.hstack((self.D,self.E,self.C,self.A)),np.hstack((self.B,self.E,self.C,self.A)), np.hstack((self.A,self.B,self.C,self.D,self.E))]
            output_list = [self.C, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y]
            name_list = ['gp_C', 'gp_D_C', 'gp_B_C', 'gp_A_C_E', 'gp_B_C_D', 'gp_D_E_C_A', 'gp_B_E_C_A', 'gp_A_B_C_D_E']
            parameter_list = [[1.,1., False], [1.,1.,False], [1.,1., True], [1.,1., False], 
                                [1.,1., False], [1.,1., False], [1.,1., False], [1.,1., False]]


        ## Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_single_GP_model(X, Y, parameter_list[i])


        return functions

    def refit_models(self, observational_samples, model_type = 0):
        A = np.asarray(observational_samples['A'])[:,np.newaxis]
        B = np.asarray(observational_samples['B'])[:,np.newaxis]
        C = np.asarray(observational_samples['C'])[:,np.newaxis]
        D = np.asarray(observational_samples['D'])[:,np.newaxis]
        E = np.asarray(observational_samples['E'])[:,np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

        functions = {}
        if model_type==0:
            inputs_list = [B, np.hstack((A,C,E)), np.hstack((D,C)), np.hstack((B,C)), np.hstack((B,C,D)), 
                        np.hstack((D,E,C,A)),np.hstack((B,E,C,A))]
            output_list = [C, Y, Y, Y, Y, Y, Y, Y]
            name_list = ['gp_C', 'gp_A_C_E', 'gp_D_C', 'gp_B_C', 'gp_B_C_D', 'gp_D_E_C_A', 'gp_B_E_C_A']
            parameter_list = [[1.,1.,10., False], [1.,1.,10., False], [1.,1.,1., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False]]
        else:
            inputs_list = [np.hstack((D,E,A,B)), np.hstack((A,B)), np.hstack((A,B,E)), B]
            output_list = [Y, E, D, D]
            name_list = ['gp_YDEAB', 'gp_EAB', 'gp_DABE', 'gp_DB']
            parameter_list = [[1.,1., False], [1.,1., False], [1.,1., False], [1.,1., False]] 

        ## Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_single_GP_model(X, Y, parameter_list[i])
  
        return functions

    def get_all_do(self):
        do_dict = {}
        do_dict['compute_do_BDEF'] = compute_do_BDEF
        do_dict['compute_do_BDE'] = compute_do_BDE
        do_dict['compute_do_BD'] = compute_do_BD
        do_dict['compute_do_BE'] = compute_do_BE
        do_dict['compute_do_DE'] = compute_do_DE
        do_dict['compute_do_B'] = compute_do_B
        do_dict['compute_do_D'] = compute_do_D
        do_dict['compute_do_E'] = compute_do_E
        do_dict['BF'] = BF
        return do_dict



