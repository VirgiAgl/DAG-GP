import numpy as np
from .c_function import *

def e_function(BF_inputs, BF_data_inputs, PF_data_inputs, BF_inputs2, integrating_measure, BM, kernel, n_samples):

    k1 = BM.model.kern.K(BF_inputs, BF_data_inputs)
    k2 = BM.model.kern.K(BF_inputs2, BF_data_inputs)
    
    c1_value = c_function(PF_data_inputs, BF_inputs, integrating_measure, BM, kernel, n_samples)
    c2_value = c_function(PF_data_inputs, BF_inputs2, integrating_measure, BM, kernel, n_samples)
    
    e_value1 = np.concatenate((k1, c1_value), axis =1)
    e_value2 = np.concatenate((k2, c2_value), axis =1)

    return e_value1,e_value2