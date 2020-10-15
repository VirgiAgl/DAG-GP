## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns


def set_saving_folder(args):
  if args.type_cost == 1:
    folder = str(args.experiment) + '/Fix_equal/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  if args.type_cost == 2:
    folder = str(args.experiment) + '/Fix_different/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  if args.type_cost == 3:
    folder = str(args.experiment) + '/Fix_different_variable/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  if args.type_cost == 4:
    folder = str(args.experiment) + '/Fix_equal_variable/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  return folder


def set_saving_folder_CTF(args):
    folder = str(args.experiment) + '/' + str(args.n_obs) + '/' + str(args.n_int) + '/' 
    return folder



def save_results(folder,  args, current_cost, current_best_x, current_best_y, global_opt, observed, total_time):
    np.save("./Data/" + folder + "cost_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy", current_cost)
    np.save("./Data/" + folder + "best_x_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy",current_best_x)
    np.save("./Data/" + folder + "best_y_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy", current_best_y)
    np.save("./Data/" + folder + "total_time_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy",total_time)
    np.save("./Data/" + folder + "observed_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy", observed)
    np.save("./Data/" + folder + "global_opt_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy",global_opt)


def save_results_CBO_CTF(folder,  args, current_cost, current_best_x, current_best_y, global_opt, observed, total_time, model_type):
    np.save("./Data/" + folder + "cost_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(model_type) + str(args.name_index) + ".npy", current_cost)
    np.save("./Data/" + folder + "best_x_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(model_type) + str(args.name_index) + ".npy",current_best_x)
    np.save("./Data/" + folder + "best_y_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(model_type) + str(args.name_index) + ".npy", current_best_y)
    np.save("./Data/" + folder + "total_time_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(model_type) + str(args.name_index) + ".npy",total_time)
    np.save("./Data/" + folder + "observed_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(model_type) + str(args.name_index) + ".npy", observed)
    np.save("./Data/" + folder + "global_opt_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(model_type) + str(args.name_index) + ".npy",global_opt)




def save_results_BO(folder,  args, current_cost, current_best_x, current_best_y, total_time, Causal_prior):
    np.save("./Data/" + folder + "cost_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy", current_cost)
    np.save("./Data/" + folder + "best_x_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy",current_best_x)
    np.save("./Data/" + folder + "best_y_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy", current_best_y)
    np.save("./Data/" + folder + "total_time_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy",total_time)


def save_results_CTF(folder,  args, Causal_prior, Transferred_mean_list, Transferred_covariance_list, BF_data, PF_data):
    np.save("./Data/" + folder + "Transferred_mean_list_" + str(Causal_prior) + str(args.name_index) + ".npy", Transferred_mean_list)
    
    if BF_data[0] is not None:
        np.save("./Data/" + folder + "Transferred_covariance_BF_" + str(Causal_prior) + str(args.name_index) + ".npy", 
                                                                                    np.diagonal(Transferred_covariance_list[1]))

    
    np.save("./Data/" + folder + "Transferred_covariance_PF_" + str(Causal_prior) + str(args.name_index) + ".npy", 
                                                                                    np.diagonal(Transferred_covariance_list[0]))

    np.save("./Data/" + folder + "BF_data_" + str(args.name_index) + ".npy", BF_data)
    np.save("./Data/" + folder + "PF_data_outputs_" + str(args.name_index) + ".npy", PF_data[1])
    np.savez("./Data/" + folder + "PF_data_inputs_" + str(args.name_index), *PF_data[0])


def save_results_GPreg(folder,  args, Causal_prior, pred_mean_list, pred_var_list):
    np.save("./Data/" + folder + "GP_reg_mean_list_" + str(Causal_prior) + str(args.name_index) + ".npy", pred_mean_list)
    np.save("./Data/" + folder + "GP_reg_covariance_list_" + str(Causal_prior) + str(args.name_index) + ".npy",pred_var_list)


def save_results_ALreg(folder,  args, Causal_prior, rmse, model_list_inputs, model_list_outputs,prediction_mean_list, prediction_var_list):
    np.save("./Data/" + folder + "AL_GP_reg_rmse_" + str(Causal_prior) + str(args.name_index) + ".npy", rmse)

    np.save("./Data/" + folder + "AL_GP_reg_models_inputs_" + str(Causal_prior) + str(args.name_index) + ".npy", model_list_inputs)
    np.save("./Data/" + folder + "AL_GP_reg_models_outputs_" + str(Causal_prior) + str(args.name_index) + ".npy", model_list_outputs)

    np.save("./Data/" + folder + "AL_GP_reg_prediction_mean_list_" + str(Causal_prior) + str(args.name_index) + ".npy", prediction_mean_list)
    np.save("./Data/" + folder + "AL_GP_reg_prediction_var_list_" + str(Causal_prior) + str(args.name_index) + ".npy", prediction_var_list)


def save_results_AL_CTF(folder,  args, Causal_prior, Transferred_mean_list_total, Transferred_covariance_list_BF, Transferred_covariance_list_PF, 
                        max_values_list, point_values_list,function_values_list, BF_data_list, PF_data_list):
    
    np.save("./Data/" + folder + "AL_CTF_Transferred_mean_list_" + str(Causal_prior) + str(args.name_index) + ".npy", Transferred_mean_list_total)
    np.save("./Data/" + folder + "AL_CTF_Transferred_covariance_BF_" + str(Causal_prior) + str(args.name_index) + ".npy", Transferred_covariance_list_BF)
    np.save("./Data/" + folder + "AL_CTF_Transferred_covariance_PF_" + str(Causal_prior) + str(args.name_index) + ".npy", Transferred_covariance_list_PF)


    np.save("./Data/" + folder + "AL_CTF_max_values_list_" + str(Causal_prior) + str(args.name_index) + ".npy", max_values_list)
    np.save("./Data/" + folder + "AL_CTF_point_values_list_" + str(Causal_prior) + str(args.name_index) + ".npy", point_values_list)
    np.save("./Data/" + folder + "AL_CTF_function_values_list_" + str(Causal_prior) + str(args.name_index) + ".npy", function_values_list)

    np.save("./Data/" + folder + "AL_CTF_BF_data_list_" + str(Causal_prior) + str(args.name_index) + ".npy", BF_data_list)
    np.save("./Data/" + folder + "AL_CTF_PF_data_list_" + str(Causal_prior) + str(args.name_index) + ".npy", PF_data_list)



