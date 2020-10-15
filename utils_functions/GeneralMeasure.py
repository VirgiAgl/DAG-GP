import numpy as np



class GeneralMeasure():
    """
    This is a general measure from which we can sample 

    """

    def __init__(self, variable, intervention_set, type_dist:str ='marginal', 
                       mean: np.ndarray = None, variance: float = None, 
                       cond_gp = None):

        """
        :param mean: the mean of the Gaussian, shape (num_dimensions, )
        :param variance: the scalar variance of the isotropic covariance matrix of the Gaussian.
        """
        self.type_dist = type_dist
        self.mean = mean
        self.variance = variance
        self.cond_gp = cond_gp
        self.variable = variable
        self.intervention_set = intervention_set


    def get_interv_var(self):
        if self.type_dist == 'marginal':
            value = None
        else:
            value = self.cond_gp.inter_var
        return value


    def get_samples(self, n_samples: int, 
                    inter_values = None, 
                    sampled_data = None) -> np.ndarray:

        
        """
        Samples from the measure.

        Args:
        
        num_samples: number of samples
        data: when dist to sample is a conditional we use a GP and compute parameters in data

        :return: samples, shape (num_samples, num_dimensions)
        """
        np.random.seed(1)
        if self.type_dist is 'marginal':
            samples = (np.transpose(self.mean) + np.transpose(np.sqrt(self.variance)) 
                   * np.random.randn(n_samples, 1))
        else:
            ## TO DO -- this needs to be changed to exploit full variability in the distribution
            ## Problematic cause of the high variance
            samples = self.cond_gp.predict(n_samples = n_samples, 
                                           inter_values = inter_values, 
                                           sampled_data = sampled_data,
                                           intervention_set = self.intervention_set)
        return samples


class ConditionalDistribution():
    
    def __init__(self, inputs, gp_function, inter_var = None):
        self.inputs = inputs
        self.gp_function = gp_function
        self.inter_var = inter_var
    
    
    def predict(self, n_samples, inter_values, sampled_data = None, intervention_set = None):

        ## Dimension of the conditinal distribution we want to sample from
        n_dim = len(self.inputs)

        ## Get number of interventional values used as inputs of the conditional distribution
        n_inter_values = inter_values[ ''.join(intervention_set)].shape[0]

        ## Create matrix of inputs for the conditional distribution
        inpus_matrix = np.ones((n_samples, n_inter_values, n_dim))


        ## Loop over the inputs and collect the relevant input values from the sampled value or the passed interventional data
        for i in range(len(self.inputs)):
            ## If the inputs of the conditional distribution are intervened we get the intervention values 
            if self.inputs[i] in intervention_set:
                intervened_values = inter_values[''.join(intervention_set)]

                col_to_get = intervention_set.index(self.inputs[i])
                inpus_matrix[:,:,i] = np.tile(np.transpose(intervened_values[:,col_to_get][np.newaxis]), (n_samples, 1, 1))[:,:,0]
            else:
                ## If the inputs are not intervened we get the values from the samples 
                if sampled_data[self.inputs[i]].shape[1] == n_inter_values:
                    ## This sample is coming from a conditional distribution thus it has n_inter_values 
                    ## has second dimension. We dont need to expand the dimension
                    inpus_matrix[:,:,i] = sampled_data[self.inputs[i]]
                else:
                    ## This sample is coming from a marginal we thus want to expand the second dimension
                    ## to make it equal n_inter_values
                    inpus_matrix[:,:,i] = np.tile(sampled_data[self.inputs[i]][:,np.newaxis,:], (1,n_inter_values,1))[:,:,0]

        ## For each sample of the full input vector we want to predict the variable
        output_values = []

        for i in range(n_inter_values):
            mean, var = self.gp_function.predict(inpus_matrix[:,i,:])
            np.random.seed(1)
            values = np.random.normal(mean, np.sqrt(var))  


            output_values.append(values)

        pred_values = np.hstack(output_values)
    
        return pred_values

