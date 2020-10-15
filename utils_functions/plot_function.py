import numpy as np
import matplotlib.pyplot as plt

def plot_current_model(variable, observational_samples, true_functions, Transferred_mean, additional_means, Transferred_covariance, input_dict,
                       ranges_dict, intervention_data_dic, label_true, label_y, label_x, label_our, label_std, 
                       additional_labels, title, lim_y, transfer = True):
    col_std = 'green'
    col_other_mean = 'C0'
    color_data_BF = 'red' 
    color_our_mean = 'green'
    color_data_PF = 'magenta' 
    color_true_fun = 'black'
    size_label_ticks = 25
    size_title = 25
    size_legend = 15
    linewidth = 3
    low_ylim = lim_y[0]
    up_ylim = lim_y[1]
    col_obs = 'black'
    
    list_variables = list(intervention_data_dic.keys())
    if variable in list_variables: list_variables.remove(variable)
    
    if Transferred_covariance.shape[1]>1:
        Transferred_variance = clip_negative_values(np.diagonal(Transferred_covariance)[:,np.newaxis])
    else:
        Transferred_variance = Transferred_covariance

    inputs = input_dict[variable]
    
    ## This is plotting the data for the other function
    if transfer == True:
        variable_to_plot = list_variables[0]
        for i in range(len(intervention_data_dic[variable_to_plot])):
            if i ==0:
                if intervention_data_dic[variable_to_plot][i][1] is not None:
                    y = (intervention_data_dic[variable_to_plot][i][1]).shape[0]
                    plt.scatter(intervention_data_dic[variable_to_plot][i][0], np.repeat(0,y)[:,np.newaxis], 
                                color = color_data_PF, 
                                label = r"$"+ 'D^I_'+variable_to_plot+"$", zorder=10)
                else:
                    plt.scatter(intervention_data_dic[variable_to_plot][i][0], intervention_data_dic['X'][i][1], 
                                color = color_data_PF, 
                                label = r"$"+ 'D^I_'+variable_to_plot+"$", zorder=10)  
            else:
                y = (intervention_data_dic[variable_to_plot][i][1]).shape[0]
                plt.scatter(intervention_data_dic[variable_to_plot][i][0], np.repeat(low_ylim+0.2,y)[:,np.newaxis], 
                            color = color_data_PF, zorder=10)

    ## This is plotting the data for the function
    for j in range(len(intervention_data_dic[variable])):
        if j ==0:
            plt.scatter(intervention_data_dic[variable][j][0], intervention_data_dic[variable][j][1], color = color_data_BF, 
                label = r"$"+ 'D^I_'+variable+"$", zorder=10)
        else:
             plt.scatter(intervention_data_dic[variable][j][0], intervention_data_dic[variable][j][1], 
                         color = color_data_BF
                         ,zorder=10)           
    
    plt.scatter(observational_samples[variable], np.repeat(low_ylim+0.2, observational_samples[variable].shape[0]), 
               marker = 'x', label='$D^O$', color =col_obs, alpha = 0.5)
    
    plt.plot(inputs, true_functions[variable], color_true_fun, label=label_true, zorder=2, linewidth=linewidth)
    plt.plot(inputs, Transferred_mean, color_our_mean, label=label_our, zorder=3, linewidth=linewidth)
    
    ## Additional line we want to display
    if all(d is None for d in additional_means) == False:
        for i in range(len(additional_means)):
            values_to_plot = additional_means[i]
            plt.plot(inputs, values_to_plot, label=additional_labels, 
                     zorder=1, linewidth=linewidth)
   
    ## Uncertainty
    plt.fill_between(inputs[:, 0],
                     Transferred_mean[:, 0] + 2* np.sqrt(Transferred_variance)[:, 0],
                     Transferred_mean[:, 0] - 2* np.sqrt(Transferred_variance)[:, 0], color=col_std, alpha=0.3,
                    label=label_std)

    plt.legend(loc=2, prop={'size': size_legend}, ncol=4)
    plt.xlabel(label_x, fontsize = size_label_ticks)
    plt.ylabel(label_y, fontsize = size_label_ticks)
    plt.xticks(size=size_label_ticks)
    plt.yticks(size=size_label_ticks)
    plt.grid(False)
    plt.xlim(ranges_dict[variable][0]-0.1, ranges_dict[variable][1]+0.1)
    plt.ylim(low_ylim, up_ylim)
    plt.title(title, size =size_title)



def plot_2D(inputs, Y):
    ## Plot a 2D function
    num_points = int(np.sqrt(Y.shape[0]))
    
    X_plot, Z_plot = np.meshgrid(*inputs)
    Y_plot = Y.reshape(num_points,num_points)

    # Plot the surface.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X_plot, Z_plot, Y_plot, cmap='Reds', linewidth=0, antialiased=False)
    