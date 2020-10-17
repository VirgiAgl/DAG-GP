import numpy as np

from .c_function import *
from .d_function import *
from .sample_IM import *
from .utils import *
from .compute_mean_vectors import *

from .covariance_test_inputs import *
from .covariance_test_inputs_data import *
from .covariance_data import *
import matplotlib.pyplot as plt


def backward_parameters(PF_inputs, BF_data, PF_data, graph, functions,
                        kernel_function_BF, 
                        mean_function_BF = None, 
                        total_samples = None,
                        total_samples_test_inputs = None, 
                        d = None, 
                        kernel = 'causal', 
                        n_samples = 10):
    '''
    This function is updating the parameters of the peripheral function 
    given the distribution on the base function.

    Args:
        :params PF_inputs: test inputs for PFs 
        :params BF_data: data for the base function 
        :params PF_data: data for the PFs
        :params integrating_measures: list of measures for all the functions in T 
        :params kernel_function_BF: K_f
        :params mean_function_BF: m_f
        :params kernel: type of kernel for K_f
        :params n_samples: n samples for MC approximation
        :params dim_BF: dimensionality of BF

        :return: posterior mean and covariance for the PFs computed at the PF_inputs
    '''
    ## Raise an error if PF_data is all None and BF_data[0] is None
    if (all(d is None for d in PF_data[1]) == True and BF_data[0] is None):
        raise ValueError("No data provided")
    
    integrating_measures = graph.get_IMs(functions)

    prior_mean_BF, prior_mean_PF, prior_mean_PF_inputs = get_prior_mean_vectors(PF_inputs, 
                                                                                BF_data, PF_data,
                                                                                total_samples, 
                                                                                integrating_measures,
                                                                                mean_function_BF, 
                                                                                total_samples_test_inputs)

    ## Aggregate mean values and output vectors         
    prior_mean_vector, outputs = aggregate_mean_ouput_vectors(BF_data, PF_data, prior_mean_BF, prior_mean_PF)

    if d is None:
        d = covariance_test_inputs(PF_inputs, kernel_function_BF, samples = total_samples_test_inputs, n_samples = n_samples)

    qKq = covariance_test_inputs_data(PF_inputs, BF_data, kernel_function_BF, integrating_measures, total_samples, 
                                      total_samples_test_inputs)

    sigma = covariance_data(BF_data, PF_data, kernel_function_BF, integrating_measures, total_samples)
            
    ## Compute posterior parameters for the PFs
    mean = prior_mean_PF_inputs + np.matmul(np.matmul(qKq, np.linalg.inv(sigma)), outputs-prior_mean_vector)
    covariance = d - np.matmul(np.matmul(qKq, np.linalg.inv(sigma)), np.transpose(qKq))

    return mean, covariance, d

