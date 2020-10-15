import numpy as np
from .c_function import *
from .d_function import *


def covariance_test_inputs_data(test_inputs, BF_data, kernel_function_BF, integrating_measures, 
                                total_samples, total_samples_test_inputs = None):
    
    ## Remove null values from total_samples and filter the integrating_measures and the PF data
    filtered_samples = [x for x in total_samples if x is not None]
    filtered_integrating_measures = [integrating_measures[i] for i in range(len(integrating_measures))
                                     if total_samples[i] is not None]
    
    if isinstance(test_inputs, list):
        ## We are doing backward propagation
        if any(d is None for d in test_inputs) == True:
            raise ValueError("Need to provide test inputs for each function")
    
        total_matrix = []
        for i in range(len(test_inputs)):
            vector = []
            for j in range(len(integrating_measures)):
                if (integrating_measures[i] !=1. and  integrating_measures[j]!=1.):
                    matrix = d_function(kernel_function_BF = kernel_function_BF, kernel = 'causal',
                                    samples1 = total_samples_test_inputs[i], samples2 = filtered_samples[j],
                                    n_samples = filtered_samples[0].shape[0])
                    vector.append(matrix)
                elif (integrating_measures[i] !=1. and  integrating_measures[j]==1.):
                    matrix = np.transpose(c_function(kernel_function_BF = kernel_function_BF, 
                                        BF_inputs = BF_data[0], samples = total_samples_test_inputs[i],
                                        n_samples = total_samples[0].shape[0]))

                    vector.append(matrix)

            total_matrix.append(np.hstack(vector))
        total_matrix = np.vstack(total_matrix)
 
  
            
    else:
        ## We are doing forward propagation
        total_matrix = []
        for i in range(len(integrating_measures)):
            if (integrating_measures[i] !=1.):
                matrix = c_function(kernel_function_BF = kernel_function_BF, 
                                        BF_inputs = test_inputs, samples = filtered_samples[i],
                                        n_samples = filtered_samples[0].shape[0])  
                total_matrix.append(matrix)
            else:
                matrix = kernel_function_BF.K(test_inputs, BF_data[0])
                total_matrix.append(matrix)
            
        total_matrix = np.hstack(total_matrix)
    return total_matrix