import numpy as np

def increment_full_samples_IM(index_function, integrating_measures, total_samples, dict_interventions, inputs_BF):
    """
    Generate samples for each probability in the list of integrating measures
    
    Args:
        :params integrating_measures: list of distributions
        :params n_samples: n samples to draw for each integrating_measures
        
        :return: list of samples from integrating measure for each function in T
    """
    n_samples = total_samples[0].shape[0]
    new_sample = []

    if integrating_measures[index_function] == 1.:
        new_sample = None
    else:
        new_sample = samples_integrating_measures(integrating_measures[index_function], 
                                                       n_samples, dict_interventions, inputs_BF)

    
    
    total_samples[index_function] = new_sample
    return total_samples


def generate_full_samples_IM(integrating_measures, n_samples, dim_BF, dict_interventions, inputs_BF):
    """
    Generate samples for each probability in the list of integrating measures
    
    Args:
        :params integrating_measures: list of distributions
        :params n_samples: n samples to draw for each integrating_measures
        
        :return: list of samples from integrating measure for each function in T
    """
    total_samples = []
    for i in range(len(integrating_measures)):
        # print('##### I am doint the integrating_measures:', i)
        if integrating_measures[i] == 1.:
            total_samples.append(None)
        else:
            #print('i', i)
            total_samples.append(samples_integrating_measures(integrating_measures[i], 
                                 n_samples, dict_interventions, inputs_BF))
    return total_samples


def samples_integrating_measures(integrating_measure, n_samples, dict_interventions, inputs_BF):
    """
    Sample from the joint integrating_measure.   
    

    Args:
        :params integrating_measure: distribution from which we want to sample. It is a list when 
                                     the integrating_measure includes different distribution and 
                                     we want to sample from the joint
        :params n_samples: n of samples to draw
        
        :return: array of samples, (n_samples, dim PF_data, dim base set)
    """
    np.random.seed(1)
    samples_dict = {}
    

    for j in range(len(integrating_measure)):
        samples_dict[integrating_measure[j].variable] = integrating_measure[j].get_samples(n_samples = n_samples, 
                                                                                            inter_values = dict_interventions,
                                                                                            sampled_data = samples_dict)       

    intervention_set = integrating_measure[0].intervention_set
    n = dict_interventions[''.join(intervention_set)].shape[0]

    ## When len(integrating_measure) > 1 this is creating an array of draws from the joint
    total_samples = concatenate_samples(n_samples, dict_interventions, samples_dict, inputs_BF, n, intervention_set)

    return total_samples



def concatenate_samples(n_samples, dict_interventions, samples_dict, inputs_BF, n, intervention_set):
    """
    Aggregate the samples in a list to create an array of samples from the joint distribution. 
    
    Args:
        :params samples: list of samples
        
    """
    
    new_sample = [None]*len(inputs_BF)
    for i in range(len(inputs_BF)):
        variable = inputs_BF[i]

        if variable in samples_dict.keys():
            ## If the variable has been sampled get the values from the samples and put it in shape
            single_sample = samples_dict[variable]
            if single_sample.shape[1] == n:
                new_sample[i] = single_sample[:,:,np.newaxis]
            else:
                new_sample[i] = np.tile(single_sample[:,np.newaxis,:], (1, n, 1))
        else:
            ## If the variable has NOT been sampled get the values from the intervention values
            data_int = dict_interventions[''.join(intervention_set)]
            col_to_get = intervention_set.index(inputs_BF[i])
            new_sample[i] = np.tile(np.expand_dims(data_int[:,col_to_get][:,np.newaxis], axis = 0), (n_samples,1,1))


    final_sample = np.dstack(new_sample)

    return final_sample

