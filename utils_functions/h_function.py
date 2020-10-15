import numpy as np
from .c_function import *
from .d_function import *

def h_function(PF_inputs, BF_data_inputs, PF_data_inputs, PF_inputs2, integrating_measure, BM, kernel, n_samples):

    c1_value = c_function(PF_inputs, BF_data_inputs, integrating_measure, BM, kernel, n_samples)
    c2_value = c_function(PF_inputs2, BF_data_inputs, integrating_measure, BM, kernel, n_samples)
    
    d1_value = d_function(PF_inputs, PF_data_inputs, integrating_measure, BM, kernel, n_samples)
    d2_value = d_function(PF_inputs2, PF_data_inputs, integrating_measure, BM, kernel, n_samples)
    
    h1_value = np.concatenate((np.transpose(c1_value), d1_value), axis = 1)
    h2_value = np.concatenate((np.transpose(c2_value), d2_value), axis = 1)
    
    
    return h1_value,h2_value