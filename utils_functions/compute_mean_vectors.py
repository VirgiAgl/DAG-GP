import numpy as np


def aggregate_mean_ouput_vectors(BF_data, PF_data, prior_mean_BF, prior_mean_PF):
    if BF_data[0] is None:
        outputs = np.vstack([x for x in PF_data[1] if x is not None])
        prior_mean_vector = prior_mean_PF
    elif all(d is None for d in PF_data[1]) == True:
        outputs = BF_data[1]
        prior_mean_vector = prior_mean_BF
    else:
        PF_data_outputs = np.vstack([x for x in PF_data[1] if x is not None])
        outputs = np.concatenate((PF_data_outputs, BF_data[1]), axis=0)
        prior_mean_vector = np.concatenate((prior_mean_PF, prior_mean_BF), axis=0)
    return prior_mean_vector, outputs


def get_prior_mean_vectors(test_inputs, BF_data, PF_data, samples, integrating_measures, 
                           mean_function_BF, samples_test_inputs = None):
    """
    Compute prior mean at the data point and at the provided test inputs
    
    Args:
	    :params test_inputs: list of inputs (for PF) or array (for BF) at which to compute the mean functions
	    :params BF_data: observed data for the base function
	    :params PF_data: observed data for the peripheral functions
	    :params samples: samples from the integrating measures for each function in T 
	    :params integrating_measures: list of integrating measures for each function in T 
	    :params mean_function_BF: mean function for the base function
	    :params samples_test_inputs: test inputs at which to compute the mean function
	    
	    :return: the mean function at BF_data, at PF_data and at the test inputs

    """
    ## Compute mean in BF data 
    if BF_data[0] is not None:
        prior_mean_BF = compute_mean_BF(BF_data[0], mean_function_BF)
    else:
        prior_mean_BF = None
    
    ## Compute mean in PF data 
    if all(d is None for d in PF_data[0]) == False:
        prior_mean_PF = compute_mean_PF(samples, integrating_measures, mean_function_BF, PF_data[0])
    else:
        prior_mean_PF = None
         
    ## Compute mean in test inputs 
    if isinstance(test_inputs, list):
        ## We have a list of test inputs corresponding to the different functions in T
        prior_mean_test_inputs = compute_mean_PF(samples_test_inputs, integrating_measures, 
                                                      mean_function_BF, test_inputs)
    else:
        ## Only oone array of inputs corresponding to the base function
        prior_mean_test_inputs = compute_mean_BF(test_inputs, mean_function_BF)
            
    return prior_mean_BF, prior_mean_PF, prior_mean_test_inputs



def compute_mean_BF(inputs, mean_function_BF):
    """
    Compute prior mean function for the base function that is m_f(inputs)

    Args:
        :params inputs: inputs in which we want to compute the prior mean
        :params mean_function_BF: mean function for the base function

        :return: mean vector, shape (inputs.shape[0], 1)
    """
    if mean_function_BF is not None:
        prior_mean_BF = mean_function_BF(inputs)
        if prior_mean_BF.shape == ():
            prior_mean_BF = np.array(prior_mean_BF)[np.newaxis,np.newaxis]
    else:
        prior_mean_BF = np.repeat(0., inputs.shape[0])[:,np.newaxis]
    return prior_mean_BF


def compute_mean_PF(total_samples, integrating_measure, mean_function_BF, data_inputs):
    """
    Compute prior mean function for the functions in T as 
    
    int m_f(b) p(b'|X = x_I) d_b
    
    Args:
        :params total_samples: list of samples from p(b'|X = x_I) for the IM of each function in T
        :params integrating_measure: list of measures for function in T 
        :params data_inputs: inputs at which to compute the mean function for each function in T 
        :params mean_function_BF: m_f
        
        :return: list of mean vectors, len(integrating_measure) 
                with shape of each element i in the list given by (data_inputs[i].shape[0],1)
    """
    prior_mean_PF_list = []
    for i in range(len(integrating_measure)):
        if integrating_measure[i] != 1.:
            if data_inputs[i] is not None: ## Dont compute prior mean is no inputs are provided
                if mean_function_BF is not None:
                    samples = total_samples[i]

                    prior_mean_PF = np.stack([np.mean(mean_function_BF(samples[:,s,:]), axis = 0) 
                                          for s in range(samples.shape[1])])
                    prior_mean_PF_list.append(prior_mean_PF)
                else:
                    prior_mean_PF = np.repeat(0., data_inputs[i].shape[0])[:,np.newaxis] 
                    prior_mean_PF_list.append(prior_mean_PF)
                
    prior_mean = np.vstack(prior_mean_PF_list)
    return prior_mean  

    