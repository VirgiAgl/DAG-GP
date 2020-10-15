import numpy as np

from .c_function import *
from .d_function import *

from .sample_IM import *
from .compute_mean_vectors import *
from .utils import *


from .covariance_test_inputs_data import *
from .covariance_data import *



def forward_parameters(BF_inputs, BF_data, PF_data, graph, functions,
                       kernel_function_BF, 
                       mean_function_BF = None, 
                       total_samples = None, kernel = 'causal', 
                       n_samples=10): 
    '''
    This function is updating the parameters of the base function given the function obs (if any)
    and the integral obs - that is the observation of the PFs with respect to different IMs.
    
    Args:
        :params BF_inputs: test inputs for the base function    
        :params BF_data: observed data for the base function
        :params PF_data: list of lists of data for the functions in T excluding the BF
        :params integrating_measures: list of integrating measures from which we can sample - if 1. it denotes the base function 
        :params kernel_function_BF: K_f
        :params mean_function_BF: m_f
        :params kernel: type of kernel for K_f
        :params n_samples: n samples for MC approximation
        :params dim_BF: dimensionality of BF
        
        :return: posterior mean and covariance for the base function computed at BF_inputs
    '''
    ## Raise an error if PF_data is all None and BF_data[0] is None
    if (all(d is None for d in PF_data[1]) == True and BF_data[0] is None):
        raise ValueError("No data provided")
    

    integrating_measures = graph.get_IMs(functions)

    ## Compute prior mean
    prior_mean_BF, prior_mean_PF, prior_mean_BF_inputs = get_prior_mean_vectors(BF_inputs, 
                                                                                BF_data, PF_data, 
                                                                                total_samples, 
                                                                                integrating_measures,
                                                                                mean_function_BF)
    
    print('I have computed the mean')
    ## Aggregate mean values and output vectors         
    prior_mean_vector, outputs = aggregate_mean_ouput_vectors(BF_data, PF_data, prior_mean_BF, prior_mean_PF)
    print('Done aggregating')
        
    ## Construct covariance terms
    sigma = covariance_data(BF_data, PF_data, kernel_function_BF, integrating_measures, total_samples)
    print('Done sigma')
        
    qKq = covariance_test_inputs_data(BF_inputs, BF_data, kernel_function_BF, integrating_measures, total_samples)
    print('Done qKq')

    ## Compute posterior parameters
    mean = prior_mean_BF_inputs + np.matmul(np.matmul(qKq, np.linalg.inv(sigma)), outputs-prior_mean_vector)
    print('Done mean')

    covariance = (kernel_function_BF.K(BF_inputs, BF_inputs) - np.matmul(np.matmul(qKq, np.linalg.inv(sigma)), np.transpose(qKq)))
    print('Done covariance')
        
    return mean, covariance