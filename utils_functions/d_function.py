import numpy as np
import time

def d_function(kernel_function_BF, kernel = 'causal', 
               PF_inputs = None, PF_inputs2 = None, 
               integrating_measure = None, 
               samples1 = None, samples2 = None, n_samples = None):
    '''
    This function is computing 
    
        int int K_f(b, b')p(b|X = x_I)p(b'|X = x_I') d_b db'
    
    in closed from when K_f and the conditional distributions are Gaussians. 
    Otheriwise it is computed via MC integration using the provided samples.
        
    Args:
    :params kernel_function_BF: function for K_f
    :params kernel: str, kernel type for K_f 
    :params PF_inputs: inputs for peripheral function
    :params PF_inputs2: other inputs for peripheral function
    :params integrating_measure: p
    :params n_samples: number of samples for MC approximation  
    :params samples1: samples for p(b|X = x_I)
    :params samples2: samples for p(b'|X = x_I')
    
    :return: value of the doubly integrated kernel

    '''
    np.random.seed(1)
    if samples2 is None:
        samples2 = samples1
    if kernel == 'rbf':
        ## TO DO - ensure this is ok with different inputs dimensionality
        ## Closed form computation when kernel for the base function is RBF
        input1_transformed = np.reshape(np.repeat(PF_inputs, PF_inputs2.shape[0], axis =1),
                                        ((PF_inputs.shape[0]*PF_inputs2.shape[0]),1))
        input2_transformed = np.reshape(np.repeat(PF_inputs2, PF_inputs.shape[0], axis =1),
                                        ((PF_inputs.shape[0]*PF_inputs2.shape[0]),1), order='F')

        shape1 = PF_inputs.shape[0]
        shape2 = PF_inputs2.shape[0]

        s_square = kernel_function_BF.variance[0]
        l_square = (kernel_function_BF.lengthscale[0])**2

        b, B = integrating_measure.predict(input1_transformed)

        b_prime, B_prime = integrating_measure.predict(input2_transformed)

        b_b_prime = b - b_prime
        l_square_B_B_prime = B + B_prime + l_square
        l_square_inverse_B_B_prime = (1./l_square)*B + (1./l_square)*B_prime + 1.

        g2_term = (s_square*(1./(np.sqrt(l_square_inverse_B_B_prime)))).reshape(shape1, shape2)

        exponential_term = np.exp(-0.5*b_b_prime*(1./(l_square_B_B_prime))*b_b_prime).reshape(shape1, shape2)

        d = g2_term*exponential_term     
    else:
        ## MC approximation when kernel for the base function is not RBF or IM is not Gaussian
        samples_kernel = np.stack([kernel_function_BF.K(samples1[s, :, :], 
                                                   samples2[s, :, :]) for s in range(int(n_samples))])
        d = np.mean(samples_kernel, axis =0)
    return d  