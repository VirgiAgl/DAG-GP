import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict


def compute_do_X(observational_samples, functions, value):
    gp_Y = functions['Y']
    gp_Z = functions['Z']

    if len(value.shape)==1:
        value = value[:,np.newaxis]

    z_mean, z_var = gp_Z.predict(value)
    
    np.random.seed(1)
    z = np.random.normal(z_mean, np.sqrt(z_var), (1000, z_mean.shape[0],z_mean.shape[1]))    
    #z = np.tile(z_mean[np.newaxis], (100, 1,1))
    #print('z shape',z.shape)
    
    mean_do = np.stack([np.mean(gp_Y.predict(z[:,s,:])[0]) for s in range(z.shape[1])])
    var_do = np.stack([np.mean(gp_Y.predict(z[:,s,:])[1]) for s in range(z.shape[1])])

    #mean_do, var_do = gp_Z.predict(value)

    return mean_do[:,np.newaxis], var_do[:,np.newaxis]


def compute_do_Z(observational_samples, functions, value):
    
    gp_Y = functions['Y']
    
    if len(value.shape)==1:
        value = value[:,np.newaxis]

    mean_do = gp_Y.predict(value)[0]
    
    var_do = gp_Y.predict(value)[1]

    return mean_do, var_do
   

def compute_do_XZ(observational_samples, functions, value):
    
    gp_X_Z = functions['gp_X_Z']
    
    X = np.asarray(observational_samples['X'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], X.shape[0])[:,np.newaxis], np.repeat(value[1], X.shape[0])[:,np.newaxis]))
    mean_do = np.mean(gp_X_Z.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_X_Z.predict(intervened_inputs)[1])


    return mean_do, var_do

