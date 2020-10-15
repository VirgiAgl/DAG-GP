import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict


def compute_do_X(observational_samples, functions, value):
    gp_Z = functions['gp_Z']
    gp_YZX = functions['gp_YZX']

    X = np.asarray(observational_samples['X'])[:,np.newaxis]
    if len(value.shape) == 1:
        value = value[:,np.newaxis]
    
    mean_do = np.zeros((value.shape[0],1))
    var_do = np.zeros((value.shape[0],1))
    for i in range(value.shape[0]):    
        
        if len(value[i].shape) == 1:
            value_i = value[i][:,np.newaxis]
        else:
            value_i = value[i]

        z_mean, z_var = gp_Z.predict(value_i)

        n_samples = 100

        np.random.seed(1)
        #samples_z = np.tile(z_mean[:,0], (n_samples,1)).reshape(n_samples*value_i.shape[0],1)
        samples_z = np.random.normal(z_mean, np.sqrt(z_var), (n_samples, z_mean.shape[0],z_mean.shape[1])).reshape(n_samples*value_i.shape[0],1)   

        
        xprimevalues = np.tile(X, (samples_z.shape[0],1))
        zvalues = np.tile(samples_z, (X.shape[0],1))
        
        
        intervened_inputs = np.hstack((zvalues,xprimevalues))
        
           
        first_mc = np.mean((gp_YZX.predict(intervened_inputs)[0]).reshape(X.shape[0],samples_z.shape[0]),axis =0)
        second_mc = np.mean(first_mc.reshape(n_samples, value_i.shape[0]), axis =0)

        first_mc_var = np.mean((gp_YZX.predict(intervened_inputs)[1]).reshape(X.shape[0], samples_z.shape[0]),axis =0)
        second_mc_var = np.mean(first_mc_var.reshape(n_samples, value_i.shape[0]), axis =0)

        mean_do[i] = second_mc

        var_do[i] = second_mc_var

    return mean_do, var_do


def compute_do_Z(observational_samples, functions, value):
    
    gp_YZX = functions['gp_YZX']

    X = np.asarray(observational_samples['X'])[:,np.newaxis]
    
    Zvalues = np.repeat(value, X.shape[0])[:,np.newaxis]
    Xvalues = np.tile(X, (value.shape[0],1))

    intervened_inputs = np.hstack((Zvalues,Xvalues))
    
    mean_do = np.mean((gp_YZX.predict(intervened_inputs)[0]).reshape(value.shape[0],X.shape[0]), axis =1)
    var_do = np.mean((gp_YZX.predict(intervened_inputs)[1]).reshape(value.shape[0],X.shape[0]), axis =1)

    if len(mean_do.shape) ==1:
        mean_do = mean_do[:,np.newaxis]

    if len(var_do.shape) ==1:
        var_do = var_do[:,np.newaxis]


    return mean_do, var_do


def BF(observational_samples, functions, value):
    
    gp_YZX = functions['gp_YZX']

    mean_do, var_do = gp_YZX.predict(value)

    return mean_do, var_do


