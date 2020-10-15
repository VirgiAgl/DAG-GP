import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
from scipy.interpolate import interp1d
import scipy
import itertools

from numpy.random import randn
import copy
import seaborn as sns


def BF(observational_samples, functions, value):
    gp_YASagebmicancer = functions['gp_YASagebmicancer']
    mean_do, var_do = gp_YASagebmicancer.predict(value)
    return mean_do, var_do


def compute_do_S(observational_samples, functions, value):
    gp_S_A_B = functions['gp_YSagebmi']

    age = np.asarray(observational_samples['age'])[:,np.newaxis]
    bmi = np.asarray(observational_samples['bmi'])[:,np.newaxis]

    intervened_inputs = np.hstack((np.repeat(value, age.shape[0])[:,np.newaxis], age, bmi))

    mean_do = np.mean(gp_S_A_B.predict(intervened_inputs)[0])

    var_do = np.mean(gp_S_A_B.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_A(observational_samples, functions, value):

    gp_A_B_As = functions['gp_YAagebmi']

    age = np.asarray(observational_samples['age'])[:,np.newaxis]
    bmi = np.asarray(observational_samples['bmi'])[:,np.newaxis]

    intervened_inputs = np.hstack((np.repeat(value, bmi.shape[0])[:,np.newaxis], age,bmi))
    mean_do = np.mean(gp_A_B_As.predict(intervened_inputs)[0])

    var_do = np.mean(gp_A_B_As.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_AS(observational_samples, functions, value):

    gp_As_S_B_A = functions['gp_YASagebmi']

    age = np.asarray(observational_samples['age'])[:,np.newaxis]
    bmi = np.asarray(observational_samples['bmi'])[:,np.newaxis]

    intervened_inputs = np.hstack((np.repeat(value[0], age.shape[0])[:,np.newaxis], 
                                    np.repeat(value[1], age.shape[0])[:,np.newaxis], age,bmi))
    mean_do = np.mean(gp_As_S_B_A.predict(intervened_inputs)[0])

    var_do = np.mean(gp_As_S_B_A.predict(intervened_inputs)[1])

    return mean_do, var_do



