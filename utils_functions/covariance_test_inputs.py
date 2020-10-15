import numpy as np
from .d_function import *

def covariance_test_inputs(test_inputs, kernel_function_BF, samples = None, 
                           n_samples = None, kernel = 'causal'):
        
    if isinstance(test_inputs, list):
        if any(d is None for d in test_inputs) == True:
            raise ValueError("Need to provide test inputs for each function")
    
        dim = 0
        
        for i in range(len(test_inputs)):
            dim += test_inputs[i].shape[0]

        #print('dim', dim)
        d = np.zeros((dim, dim))
        
        n_row = 0
        for i in range(len(test_inputs)):
            n_col = 0
            for j in range(len(test_inputs)):
                matrix = d_function(kernel_function_BF = kernel_function_BF, kernel = 'causal', 
                           samples1=samples[i], samples2= samples[j], n_samples=n_samples)

                d[n_row:(matrix.shape[0] + n_row), n_col: (matrix.shape[1]+n_col)] = matrix
            
                n_col += matrix.shape[1]
            n_row += matrix.shape[0]
            
    else:
        d = kernel_function_BF.K(test_inputs)
    return d