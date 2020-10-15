import numpy as np
import time

def c_function(kernel_function_BF, kernel = 'causal', 
               PF_inputs = None, BF_inputs = None, 
               integrating_measure = None, n_samples = None, 
               samples = None):
    '''
    This function is computing 
    
        int K_f(b, b')p(b'|X = x_I) d_b
    
    in closed from when K_f and the conditional distribution is Gaussians. 
    Otheriwise it is computed via MC integration using the provided samples.
        
    ###
    Args:
    :params kernel_function_BF: function for K_f
    :params kernel: str, kernel type for K_f
    :params PF_inputs: inputs for peripheral function
    :params BF_inputs: inputs for base function
    :params integrating_measure: p  
    :params n_samples: number of samples for MC approximation   
    :params samples: samples for p(b|X = x_I)
    
    :return: value of the partial integral (KME)
    
    '''
    np.random.seed(1)
    if kernel == 'rbf':
        ## Closed form computation when kernel for the base function is RBF
        s_square = kernel_function_BF.variance[0]
        l_square = (kernel_function_BF.lengthscale[0])**2

        b, B = integrating_measure.predict(PF_inputs)

        b_vector = np.repeat(np.transpose(b), BF_inputs.shape[0], axis = 0)
        l_square_B_vector = np.repeat(np.transpose(l_square+B), BF_inputs.shape[0], axis = 0)
        B_vector = np.repeat(np.transpose(B), BF_inputs.shape[0], axis = 0)

        g1_term = s_square*1./np.sqrt((1./l_square)*B_vector + 1.)

        exponential_term = np.exp(-0.5*(BF_inputs-b_vector)*(1./l_square_B_vector)*(BF_inputs-b_vector))

        c = g1_term*exponential_term
    
    else:
        ## MonteCarlo approximation when kernel for the base function is not RBF or IM is not Gaussian
        c = np.transpose(np.stack([np.mean(kernel_function_BF.K(BF_inputs, 
                            samples[:, s, :]), axis =1) for s in range(samples.shape[1])]))
    return c