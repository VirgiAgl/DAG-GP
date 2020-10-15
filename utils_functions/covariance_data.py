import numpy as np
from .c_function import *
from .d_function import *

def covariance_data(BF_data, PF_data, kernel_function_BF, integrating_measures, total_samples):
    
    ## Raise an error if PF_data is all None and BF_data[0] is None
    if (all(d is None for d in PF_data[1]) == True and BF_data[0] is None):
        raise ValueError("No data provided")
    
    if any(d is None for d in PF_data[1]) == True:
        raise ValueError("Some PF_data is None, check computation")
            
    ## Remove null values from total_samples and filter the integrating_measures and the PF data
    filtered_samples = [total_samples[i] for i in range(len(PF_data[0])) if PF_data[0][i] is not None]
    filtered_integrating_measures = [integrating_measures[i] for i in range(len(PF_data[0]))
                                     if PF_data[0][i] is not None]
    

    PF_data[0] = [PF_data[0][i] for i in range(len(PF_data[0])) if total_samples[i] is not None]   
    PF_data[1] = [PF_data[1][i] for i in range(len(PF_data[1])) if total_samples[i] is not None]   
    
    if all(m!=1. for m in integrating_measures) and all(d is None for d in PF_data[1]) == False: 
        ## In this case BF_data[0] is None
        ## If f does not belong to T, this is computing the covariance with d func
        ## Get dim 1 and dim 2 and compute double integralss
        dim = np.sum([samples.shape[1] for samples in filtered_samples])

        qKq = np.zeros((dim, dim))
        
        n_row = 0
        for i in range(len(filtered_integrating_measures)):
            n_col = 0
            for j in range(len(filtered_integrating_measures)):
                matrix = d_function(kernel_function_BF = kernel_function_BF, kernel = 'causal',
                                    samples1 = filtered_samples[i], samples2 = filtered_samples[j],
                                    n_samples = filtered_samples[0].shape[0])

                qKq[n_row:(matrix.shape[0] + n_row), n_col: (matrix.shape[1]+n_col)] = matrix
                n_col += matrix.shape[1] ##n_X
            n_row += matrix.shape[0]

    elif all(m!=1. for m in filtered_integrating_measures) and all(d is None for d in PF_data[1]) == True:
        raise ValueError("No data provided for PF")
    else:
        ## If f belongs to T 
        ## Some terms are d, some c, some K
        indicator = 0 if BF_data[0] is None else 1
        
        dim = 0
        for i in range(len(integrating_measures)):
            if integrating_measures[i] != 1.:
                dim += PF_data[0][i].shape[0]
            elif indicator==1.:
                dim += BF_data[0].shape[0]

        qKq = np.zeros((dim, dim))

        n_row = 0
        for i in range(len(integrating_measures)):
            n_col = 0
            for j in range(len(integrating_measures)):
                if integrating_measures[i] !=1. and  integrating_measures[j]!=1.:
                    matrix = d_function(kernel_function_BF = kernel_function_BF, 
                                        samples1 = filtered_samples[i], samples2 = filtered_samples[j],
                                        n_samples = filtered_samples[0].shape[0])
                    
                elif (integrating_measures[i] !=1. and integrating_measures[j]==1. and indicator==1):
                    matrix = np.transpose(c_function(kernel_function_BF = kernel_function_BF, 
                                        BF_inputs = BF_data[0], samples = filtered_samples[i],
                                        n_samples = filtered_samples[0].shape[0]))
                    
                elif (integrating_measures[i] ==1. and integrating_measures[j]!=1. and indicator==1):
                    matrix = c_function(kernel_function_BF = kernel_function_BF, 
                                        BF_inputs = BF_data[0], samples = filtered_samples[j],
                                        n_samples = filtered_samples[0].shape[0])
                    
                elif (integrating_measures[i] ==1. and integrating_measures[j]==1. and indicator==1):
                    matrix = kernel_function_BF.K(BF_data[0])
                    
                qKq[n_row:(matrix.shape[0] + n_row), n_col: (matrix.shape[1]+n_col)] = matrix
                n_col += matrix.shape[1]
            
            n_row += matrix.shape[0]
            
    return qKq