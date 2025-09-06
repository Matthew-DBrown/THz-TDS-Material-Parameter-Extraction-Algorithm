# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:54:16 2024

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import scipy as sc
from scipy import signal
import pandas as pd
import spectra as sp

plt.rcParams["font.family"] = "Times New Roman"

tableau_colors = ['b', 'r', 'g', '#FFA500', '#800080', '#008080']
norm = 1e12
colors_13 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a"
]


sub_colors1 = ['b', 'r', 'g', 'k', 'm']
sub_colors2 = ['b', 'm', 'r', 'k', 'g', 'orange', 'fuchsia', 'steelblue']
sub_colors3 = ['darkblue', 'b', 'limegreen', 'darkgreen', 'indigo', 'm', 'orangered', 'red']
blues = ['midnightblue', 'navy', 'darkblue', 'blue', 'mediumblue',]
darks = ['maroon', 'darkgreen', 'darkcyan', 'darkblue', 'darkmagenta']

class File:
    
    def __init__(self, name, column_data, parent_path):
        self.name = name
        self.column_data = column_data
        self.parent_path = parent_path
        self.label_name = self.parent_path.rsplit('\\', 2)[1].upper()#, self.parent_path.rsplit('\\', 2)[1].rsplit('_')[1].upper()
        
def extract_columns_files(parent_path):
    files = {}
    for file in os.listdir(parent_path):
        if file.endswith(".csv"):
            file_path = os.path.join(parent_path, file)
            df = pd.read_csv(file_path)
            column_data = {}
            for column in df.columns:
                column_data["".join(column.split())] = df[column].to_numpy()
            files[file] = File(file, column_data, parent_path)
        
    print("DONE")
    return files

def smooth_data(array, window_length, mode, polynomial_degree):
    '''
    Uses the Savitzky-Golay Filtering technique. This preserves the overall
    trend without fitting a single polynomial over the entire range which may
    not be representative of the data.

    Parameters
    ----------
    array : Signal e.g. kappa, re_ind
        Numpy array.

    Returns
    -------
    Array of smoothed data.
    '''
    smooth_dat = signal.savgol_filter(array, window_length, polynomial_degree, mode=mode)
    return smooth_dat
    

def sigma_region(stack):
    one_sigma = np.std(stack, axis=0)
    return one_sigma

def plot_ref_ind(files):
    i = 0
    plt.figure(figsize=(11,9))
    for file, file_obj in files.items():
        plt.plot(file_obj.column_data['freq']/norm, file_obj.column_data['ref_ind'], linewidth=2, color=tableau_colors[i], label=f'File: {file}')
        i += 1
        
    
    plt.xlim(np.min(file_obj.column_data['freq']/norm), np.max(file_obj.column_data['freq']/norm))
    plt.ylabel("Refractive Index", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.tick_params('both', labelsize=14)
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)
    
def plot_avg_ref_ind(files, xlims=None, title=None):
    n_data_tot_list = []
    for file, file_obj in files.items():
        f_data = file_obj.column_data['freq']/norm
        n_data = files[file].column_data['ref_ind']
        n_data_tot_list.append(n_data)
    
    n_data_stack = np.stack(n_data_tot_list, axis=0)
    one_sigma = sigma_region(n_data_stack)
    

    n_max = np.max(n_data_stack, axis=0)  
    n_min = np.min(n_data_stack, axis=0)  
    
    
    plt.figure(figsize=(9,7))
    
    avg = np.average(n_data_stack, axis=0)
    smoothed_avg = smooth_data(avg, 4, 'nearest', 1)
    
    
    plt.fill_between(f_data, avg-one_sigma, avg+one_sigma, color='orange', alpha=0.3, label="Standard Deviation")
    #plt.plot(f_data, avg-one_sigma, linestyle='--', color='b')
    #plt.plot(f_data, avg+one_sigma, linestyle='--', color='b')
    #plt.scatter(f_data, avg, alpha=0.5, label=r'Average $n$')
    plt.plot(f_data, smoothed_avg, linestyle='-', linewidth=2, color='r', label='Smoothed Average')
    plt.xlim(np.min(f_data), np.max(f_data))
    plt.ylabel("Refractive Index", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    #plt.plot(f_data, np.average(n_data_stack, axis=0), color='r', linewidth=2, label='Average')
    plt.legend(loc='lower left', fontsize=14)
    plt.tick_params('both', labelsize=14)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)   
    #plt.ylim(1.5, 2)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if title is not None:
        plt.title(f"Sample: {title}", fontsize=14)
    else:
        plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    
    return n_data_stack

def plot_k(files):
    i = 0
    plt.figure(figsize=(11,9))
    for file, file_obj in files.items():
        plt.plot(file_obj.column_data['freq']/norm, file_obj.column_data['kappa'], color=tableau_colors[i], label=f'File: {file}')
        i += 1
        
    plt.xlim(np.min(file_obj.column_data['freq']/norm), np.max(file_obj.column_data['freq']/norm))
    plt.tick_params('both', labelsize=14)
    plt.ylabel(r"Extinction Coefficient, $\kappa$", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)

def plot_avg_k(files, title=None):
    k_data_tot_list = []
    for file, file_obj in files.items():
        f_data = file_obj.column_data['freq']/norm
        k_data = files[file].column_data['kappa']
        k_data_tot_list.append(k_data)
    
    k_data_stack = np.stack(k_data_tot_list, axis=0)
    n_max = np.max(k_data_stack, axis=0)  
    n_min = np.min(k_data_stack, axis=0)  
    
    plt.figure(figsize=(9,7))
    #plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    plt.fill_between(f_data, n_min, n_max, color='dodgerblue', alpha=0.4, label="Region of variation")
    plt.xlim(np.min(f_data), np.max(f_data))
    plt.ylabel(r"Extinction Coefficient, $\kappa$", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.plot(f_data, np.average(k_data_stack, axis=0), color='b', linewidth=2, label='Average')
    plt.legend(loc='upper left', fontsize=14)
    plt.tick_params('both', labelsize=14)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)  
    if title is not None:
        plt.title(f"Sample: {title}", fontsize=14)
    else:
        plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    
    
    return k_data_stack

def plot_loss_tangent(files, title=None):
    loss_tan_data_tot_list = []
    for file, file_obj in files.items():
        f_data = file_obj.column_data['freq']/norm
        epsilon_r = files[file].column_data['epsilon_r']
        epsilon_i = files[file].column_data['epsilon_i']
        loss_tan_data_tot_list.append(epsilon_i/epsilon_r)
    
    loss_tan_data_stack = np.stack(loss_tan_data_tot_list, axis=0)
    loss_tan_max = np.max(loss_tan_data_stack, axis=0)  
    loss_tan_min = np.min(loss_tan_data_stack, axis=0)  
    
    plt.figure(figsize=(9,7))
    #plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    plt.fill_between(f_data, loss_tan_min, loss_tan_max, color='pink', alpha=0.4, label="Region of variation")
    plt.xlim(np.min(f_data), np.max(f_data))
    plt.ylabel(r"$\tan(\delta)$", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.plot(f_data, np.average(loss_tan_data_stack, axis=0), color='m', linewidth=2, label='Average')
    plt.legend(loc='upper left', fontsize=14)
    plt.tick_params('both', labelsize=14)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)  
    
    if title is not None:
        plt.title(f"Sample: {title}", fontsize=14)
    else:
        plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    
    
    return loss_tan_data_stack

def plot_epsilon_r(files, title=None):
    epsilonR_data_tot_list = []
    for file, file_obj in files.items():
        f_data = file_obj.column_data['freq']/norm
        epsilonR_data = files[file].column_data['epsilon_r']
        epsilonR_data_tot_list.append(epsilonR_data)
    
    epsilonR_data_stack = np.stack(epsilonR_data_tot_list, axis=0)
    epsilonR_max = np.max(epsilonR_data_stack, axis=0)  
    epsilonR_min = np.min(epsilonR_data_stack, axis=0)  
    
    plt.figure(figsize=(9,7))
    #plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    plt.fill_between(f_data, epsilonR_min, epsilonR_max, color='gray', alpha=0.4, label="Region of variation")
    plt.xlim(np.min(f_data), np.max(f_data))
    plt.ylabel(r"$\epsilon_r$", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.plot(f_data, np.average(epsilonR_data_stack, axis=0), color='k', linewidth=2, label='Average')
    plt.legend(loc='lower left', fontsize=14)
    plt.tick_params('both', labelsize=14)
    plt.grid(True, color='k', linestyle='--', alpha=0.4)  
    if title is not None:
        plt.title(f"Sample: {title}", fontsize=14)
    else:
        plt.title(f"Sample: {file_obj.name.rsplit('_6')[0]}", fontsize=14)
    
    
    return epsilonR_data_stack


def plot_n_summary(file_paths, xlims=None, label_cond='side', color_set=colors_13, radix_inset=False, title=None):
    fig, ax = plt.subplots(figsize=(10,9))
    if radix_inset:
        zoomed_ax = inset_axes(ax, width="50%", height="30%", loc="lower right", bbox_to_anchor=(0.00, 0.04, 1,0.8),bbox_transform=ax.transAxes)
        zoomed_ax.tick_params('both', labelsize=14)
    #plt.figure(figsize=(12, 8))
    # Adjust the plot area to make it smaller, leaving space on the right
    plt.subplots_adjust(left=0.1, right=0.78, top=0.93, bottom=0.07)
    i=0
    n_all = []
    
    for path in file_paths:
        sample_files  = extract_columns_files(path)
    
        n_data_tot_list = []
        for file, file_obj in sample_files.items():
            f_data = file_obj.column_data['freq']/norm
            max_idx = sp.find_index(f_data, xlims[1])
            min_idx = sp.find_index(f_data, xlims[0])
            f_data = f_data[min_idx:max_idx]
            n_data = (sample_files[file].column_data['ref_ind'])[min_idx:max_idx]
            n_data_tot_list.append(n_data)
            #print(min_idx, " ", max_idx)
        n_data_stack = np.stack(n_data_tot_list, axis=0)
        one_sigma = sigma_region(n_data_stack)
    
        n_max = np.max(n_data_stack, axis=0)  
        n_min = np.min(n_data_stack, axis=0)  
    
        avg = np.average(n_data_stack, axis=0)
        smoothed_avg = smooth_data(avg, 4, 'nearest', 1)
    
        ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
        #plt.plot(f_data, avg-one_sigma, linestyle='--', color='b')
        #plt.plot(f_data, avg+one_sigma, linestyle='--', color='b')
        #plt.scatter(f_data, avg, alpha=0.5, label=r'Average $n$')

        temp_string = file_obj.label_name.removeprefix('ROGERS_')
        temp_string = temp_string.replace('RT_DURIOD_', '')
        temp_string = temp_string.replace('RADIX_', '')
        temp_string = temp_string.replace('RT_DUROID_', '')
        if "4.6" and "film" in path:
            temp_string = temp_string + "_FILM"
        elif "4.6" and "plate" in path:
            temp_string = temp_string + "_PLATE"
            
        ax.plot(f_data, smoothed_avg, linestyle='-', linewidth=2, color=color_set[i], label=f"{temp_string}")
        ax.set_xlim(np.min(f_data), np.max(f_data))
        ax.set_ylabel("Refractive Index", fontsize=14)
        ax.set_xlabel("Frequency (THz)", fontsize=14)
        #plt.plot(f_data, np.average(n_data_stack, axis=0), color='r', linewidth=2, label='Average')
        
        if "2.8" in path:
            # Plot the same lines on the zoomed-in axis, but with different limits
            zoomed_ax.plot(f_data, smoothed_avg, linestyle='-', linewidth=1, color=color_set[i])
            zoomed_ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
        
        if label_cond=='side':
            
            x_pos = f_data[-1] 
            y_pos = avg[-1]   
            
            
            ax.annotate(
                f"{temp_string}", 
                xy=(x_pos, y_pos),             
                xytext=(5, 0),                 
                textcoords="offset points",    
                ha="left",                     
                va="center",                   
                color=color_set[i],            
                fontsize=14,
                fontweight='bold'
                )   
        
        elif label_cond=='ontop':
            
            x_pos = f_data[50]#f_data[len(f_data) // 2]    # Middle of x values for central placement
            y_pos = avg[0] #avg[len(avg) // 2]    # Corresponding y value


            plt.text(
                x_pos, y_pos,               
                f"{temp_string}",                  
                color=color_set[i],         
                fontsize=12,                
                fontweight='bold',          
                ha='center',                
                va='center',                
                backgroundcolor="white",    
                alpha=0.8                   
                )
                
        elif label_cond=='legend':
            ax.legend(loc='lower left', fontsize=14)
        
        elif label_cond=='legend-outside':
            ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), borderaxespad=0.)
        
        ax.tick_params('both', labelsize=14)
        ax.grid(True, color='k', linestyle='--', alpha=0.4)   
        #plt.ylim(1.5, 2)
        i += 1
        n_all.append(n_data_stack)
    #print(np.stack(n_all, axis=0))
    #print(np.stack(n_all, axis=0))
    
    #plt.legend(loc='lower left', fontsize=14)
    y_lim_max = np.max(np.stack(n_all, axis=0))    
    ax.set_ylim(1, y_lim_max*1.05)
    
    if xlims is not None:
        ax.set_xlim(np.min(f_data), np.max(f_data))
        # Set the zoomed-in view limits
        zoomed_ax.set_xlim(np.min(f_data), np.max(f_data))
    if title is not None:
        ax.title(f"{title}", fontsize=14)

    return sample_files
    

###############################################################################
###############################################################################

def plot_epsilonR_summary(file_paths, xlims=None, label_cond='side', color_set=colors_13, radix_inset=False, title=None):
    fig, ax = plt.subplots(figsize=(10,11))
    if radix_inset:
        zoomed_ax = inset_axes(ax, width="50%", height="30%", loc="lower right", bbox_to_anchor=(0.00, 0.04, 1,0.67),bbox_transform=ax.transAxes)
        zoomed_ax.tick_params('both', labelsize=16)
    
    plt.subplots_adjust(left=0.1, right=0.78, top=0.93, bottom=0.07)
    i=0
    epsilonR_all = []
    added_labels = set()
    for path in file_paths:
        print(f"#########################################\nProcessing {path}")
        sample_files  = extract_columns_files(path)
    
        epsilonR_data_tot_list = []
        for file, file_obj in sample_files.items():
            f_data = file_obj.column_data['freq']/norm
            max_idx = sp.find_index(f_data, xlims[1])
            min_idx = sp.find_index(f_data, xlims[0])
            f_data = f_data[min_idx:max_idx]
            epsilon_r = sample_files[file].column_data['epsilon_r'][min_idx:max_idx]
            epsilon_i = sample_files[file].column_data['epsilon_i'][min_idx:max_idx]
            epsilonR_data_tot_list.append(epsilon_r)
            
        epsilonR_data_stack = np.stack(epsilonR_data_tot_list, axis=0)
        one_sigma = sigma_region(epsilonR_data_stack)
    
        epsilonR_max = np.max(epsilonR_data_stack, axis=0)  
        epsilonR_min = np.min(epsilonR_data_stack, axis=0)  
    
        avg = np.average(epsilonR_data_stack, axis=0)
        smoothed_avg = smooth_data(avg, 4, 'nearest', 1)
    
        ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
       

        temp_string = file_obj.label_name.removeprefix('ROGERS_')
        temp_string = temp_string.replace('RT_DURIOD_', '')
        temp_string = temp_string.replace('RADIX_', '')
        temp_string = temp_string.replace('RT_DUROID_', '')
        if "4.6" and "film" in path:
            temp_string = temp_string + "_FILM"
        elif "4.6" and "plate" in path:
            temp_string = temp_string + "_PLATE"
            
        label = f"{temp_string}"
        if label not in added_labels:
            print(f"Plotted {temp_string}")
            ax.plot(f_data, smoothed_avg, linestyle='-', linewidth=2, color=color_set[i], label=label)
            added_labels.add(label)  # Add label to the set
        else:
            print(f"Plotted {temp_string}")
            ax.plot(f_data, smoothed_avg, linestyle='--', linewidth=2, color=color_set[i])    
        
        
        ax.set_xlim(np.min(f_data), np.max(f_data))
        ax.set_ylabel(rf"$\epsilon_r$", fontsize=24)
        ax.set_xlabel("Frequency (THz)", fontsize=24)
        #plt.plot(f_data, np.average(n_data_stack, axis=0), color='r', linewidth=2, label='Average')
        
        if "2.8" in path:
            # Plot the same lines on the zoomed-in axis, but with different limits
            if radix_inset:
                if i > 13:
                    linestyle_cond = '--'
                    if np.max(smoothed_avg) < 2.4:
                        print(f"Skipping {path}: max(smoothed_avg) < 2.7")
                        continue
                    zoomed_ax.plot(f_data, smoothed_avg, linestyle=linestyle_cond, linewidth=1, color=color_set[i])
                    zoomed_ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
                else:
                    if np.max(smoothed_avg) < 2.7:
                        print(f"Skipping {path}: max(smoothed_avg) < 2.7")
                        continue
                    zoomed_ax.plot(f_data, smoothed_avg, linestyle='-', linewidth=1, color=color_set[i])
                    zoomed_ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
                
        if label_cond=='side':
            
            x_pos = f_data[-1]   
            y_pos = avg[-1]   
            
            
            ax.annotate(
                f"{temp_string}", 
                xy=(x_pos, y_pos),             
                xytext=(5, 0),                 
                textcoords="offset points",    
                ha="left",                     
                va="center",                   
                color=color_set[i],            
                fontsize=21,
                fontweight='bold'
                )   
        
        elif label_cond=='ontop':
            
            x_pos = f_data[50]#f_data[len(f_data) // 2]   
            y_pos = avg[0] #avg[len(avg) // 2]    


            plt.text(
                x_pos, y_pos,               
                f"{temp_string}",           
                color=color_set[i],         
                fontsize=21,                
                fontweight='bold',          
                ha='center',                
                va='center',                
                backgroundcolor="white",    
                alpha=0.8                   
                )
                
        elif label_cond=='legend':
            ax.legend(loc='lower left', fontsize=21)
        
        elif label_cond=='legend-outside':
            ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=21, borderaxespad=0.)
        
        ax.tick_params('both', labelsize=21)
        ax.grid(True, color='k', linestyle='--', alpha=0.4)   
        
        i += 1
        epsilonR_all.append(epsilonR_data_stack)
        print(f"Processed {temp_string} | Color = {(colors_13*2)[i]}")
        
    #y_lim_max = np.max(np.stack(epsilonR_all, axis=0))
    #print(y_lim_max)
    #ax.set_ylim(1, y_lim_max*1.05)
    ax.set_ylim(1, 5.2)
    if xlims is not None:
        ax.set_xlim(np.min(f_data), np.max(f_data))
        if radix_inset:
            zoomed_ax.set_xlim(np.min(f_data), np.max(f_data))
    if title is not None:
        ax.title(f"{title}", fontsize=21)

    return sample_files

###############################################################################
###############################################################################

def plot_lossTan_summary(file_paths, xlims=None, label_cond='side', color_set=colors_13, radix_inset=False, title=None):
    fig, ax = plt.subplots(figsize=(10,9))
    if radix_inset:
        zoomed_ax = inset_axes(ax, width="50%", height="30%", loc="lower right", bbox_to_anchor=(0.00, 0.04, 1,0.8),bbox_transform=ax.transAxes)
        zoomed_ax.tick_params('both', labelsize=14)
    #plt.figure(figsize=(12, 8))
    
    plt.subplots_adjust(left=0.1, right=0.78, top=0.93, bottom=0.07)
    i=0
    lossTan_all = []
    for path in file_paths:
        print(f"#########################################\nProcessing {path}")
        sample_files  = extract_columns_files(path)
    
        lossTan_data_tot_list = []
        for file, file_obj in sample_files.items():
            f_data = file_obj.column_data['freq']/norm
            max_idx = sp.find_index(f_data, xlims[1])
            min_idx = sp.find_index(f_data, xlims[0])
            f_data = f_data[min_idx:max_idx]
            epsilon_r = (sample_files[file].column_data['epsilon_r'])[min_idx:max_idx]
            epsilon_i = (sample_files[file].column_data['epsilon_i'])[min_idx:max_idx]
            lossTan_data_tot_list.append(epsilon_i/epsilon_r)
            #print(min_idx, " ", max_idx)
        lossTan_data_stack = np.stack(lossTan_data_tot_list, axis=0)
        one_sigma = sigma_region(lossTan_data_stack)
    
        lossTan_max = np.max(lossTan_data_stack, axis=0)  
        lossTan_min = np.min(lossTan_data_stack, axis=0)  
    
        avg = np.average(lossTan_data_stack, axis=0)
        smoothed_avg = smooth_data(avg, 4, 'nearest', 1)
    
        ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
        #plt.plot(f_data, avg-one_sigma, linestyle='--', color='b')
        #plt.plot(f_data, avg+one_sigma, linestyle='--', color='b')
        #plt.scatter(f_data, avg, alpha=0.5, label=r'Average $n$')

        temp_string = file_obj.label_name.removeprefix('ROGERS_')
        temp_string = temp_string.replace('RT_DURIOD_', '')
        temp_string = temp_string.replace('RADIX_', '')
        if "4.6" and "film" in path:
            temp_string = temp_string + "_FILM"
        elif "4.6" and "plate" in path:
            temp_string = temp_string + "_PLATE"
            
        ax.plot(f_data, smoothed_avg, linestyle='-', linewidth=2, color=color_set[i], label=f"{temp_string}")
        ax.set_xlim(np.min(f_data), np.max(f_data))
        ax.set_ylabel(r"Loss Tangent, $\tan(\delta)$", fontsize=14)
        ax.set_xlabel("Frequency (THz)", fontsize=14)
        #plt.plot(f_data, np.average(n_data_stack, axis=0), color='r', linewidth=2, label='Average')
        
        if "2.8" in path:
            if radix_inset:
                # Plot the same lines on the zoomed-in axis, but with different limits
                zoomed_ax.plot(f_data, smoothed_avg, linestyle='-', linewidth=1, color=color_set[i])
                zoomed_ax.fill_between(f_data, avg-one_sigma, avg+one_sigma, color=color_set[i], alpha=0.3)
        
        if label_cond=='side':
            
            x_pos = f_data[-1]
            y_pos = avg[-1]   
            
            
            ax.annotate(
                f"{temp_string}", 
                xy=(x_pos, y_pos),             
                xytext=(5, 0),                 
                textcoords="offset points",    
                ha="left",                     
                va="center",                   
                color=color_set[i],            
                fontsize=14,
                fontweight='bold'
                )   
        
        elif label_cond=='ontop':
            
            x_pos = f_data[50]#f_data[len(f_data) // 2]   
            y_pos = avg[0] #avg[len(avg) // 2]    

            
            plt.text(
                x_pos, y_pos,               
                f"{temp_string}",           
                color=color_set[i],         
                fontsize=12,                
                fontweight='bold',          
                ha='center',                
                va='center',                
                backgroundcolor="white",    
                alpha=0.8                   
                )
                
        elif label_cond=='legend':
            ax.legend(loc='lower left', fontsize=14)
        
        elif label_cond=='legend-outside':
            ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), borderaxespad=0.)
        
        ax.tick_params('both', labelsize=14)
        ax.grid(True, color='k', linestyle='--', alpha=0.4)   
        #plt.ylim(1.5, 2)
        i += 1
        lossTan_all.append(lossTan_data_stack)
    #print(np.stack(n_all, axis=0))
    #print(np.stack(n_all, axis=0))
    
    #plt.legend(loc='lower left', fontsize=14)

    y_lim_max = np.max(np.stack(lossTan_all, axis=0))    
    ax.set_ylim(0, y_lim_max*1.05)
    if xlims is not None:
        ax.set_xlim(np.min(f_data), np.max(f_data))
        # Set the zoomed-in view limits
        if radix_inset:
            zoomed_ax.set_xlim(np.min(f_data), np.max(f_data))
    if title is not None:
        ax.title(f"{title}", fontsize=14)

    return sample_files


if __name__ == "__main__":
    tmm3_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\rogers_tmm3\rogerstmm3_lytera_focused"
    rogers_5880LZ_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\rogers_rt_duriod_5880LZ\duriod5880LZ_lytera_focused"
    rogers_6002_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\rogers_rt_duriod_6002\duriod6002_lytera_focused"
    rogers_5880RIC6_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\rogers_rt_duriod_5880RIC6\5880RIC6_lytera_focused"
    rogers_RO3003_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\rogers_RO3003\lytera_focused_RO3003"
    radix_46_film_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_4.6\lytera_focused_radix4.6_film"
    radix_46_plate_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_4.6\lytera_focused_radix4.6_plate"
    radix_28HT_zp15_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_2.8HT_Zprint_(15)\lytera_focused_2.8HT_zprint(15)"
    radix_28HT_zp14_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_2.8HT_Zprint_(14)\lytera_focused_2.8HT_Zprint(14)"
    radix_28HT_film_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_2.8HT\lytera_focused_radix2.8HT_film"
    radix_28HT_plate_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_2.8HT\lytera_focused_radix2.8HT_plate"
    radix_28_7_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_2.8(7)\lytera_radix2.8(7)_focused"
    radix_28_6_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_2.8(6)\lytera_radix_2.8(6)_foc"
    
    tmm3_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\rogers_tmm3\lytera_rogerstmm3_collimated"
    rogers_5880LZ_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\rogers_rt_duroid_5880LZ\lytera_duroid5880LZ_collimated"
    rogers_6002_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\rogers_rt_duriod_6002\lytera_duroid6002_collimated"
    rogers_5880RIC6_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\rogers_rt_duriod_5880RIC6\lytera_5880RIC6_collimated"
    rogers_RO3003_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\rogers_RO3003\lytera_collimated_RO3003"
    radix_46_film_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_4.6\lytera_collimated_radix4.6_film"
    radix_46_plate_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_4.6\lytera_collimated_radix4.6_plate"
    radix_28HT_zp15_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_2.8HT_Zprint_(15)\lytera_collimated_2.8HT_zprint(15)"
    radix_28HT_zp14_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_2.8HT_Zprint_(14)\lytera_collimated_2.8HT_Zprint(14)"
    radix_28HT_film_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_2.8HT\lytera_collimated_radix2.8HT_film"
    radix_28HT_plate_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_2.8HT\lytera_collimated_radix2.8HT_plate"
    radix_28_7_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_2.8(7)\lytera_radix2.8(7)_collimated"
    radix_28_6_path_col = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP_241120\y4_project\radix_2.8(6)\lytera_radix_2.8(6)_col"
    
    

    paths_list = [tmm3_path, rogers_5880LZ_path, rogers_6002_path,
              rogers_5880RIC6_path, rogers_RO3003_path, radix_46_film_path, 
              radix_46_plate_path, radix_28HT_zp15_path, radix_28HT_zp14_path,
              radix_28HT_film_path, radix_28HT_plate_path, radix_28_6_path,
              radix_28_7_path]
    
    paths_list_NO46 = [tmm3_path, rogers_5880LZ_path, rogers_6002_path,
              rogers_5880RIC6_path, rogers_RO3003_path, radix_28HT_film_path, radix_28HT_plate_path, radix_28_6_path,
              radix_28_7_path]

    paths_list_set1 = [tmm3_path, rogers_5880LZ_path, rogers_6002_path,
              rogers_5880RIC6_path, rogers_RO3003_path]

    paths_list_set2 = [radix_46_film_path, 
                       radix_46_plate_path, radix_28HT_zp15_path, radix_28HT_zp14_path,
                       radix_28HT_film_path, radix_28HT_plate_path, radix_28_6_path,
                       radix_28_7_path]
    
    paths_and_col = [tmm3_path, rogers_5880LZ_path, rogers_6002_path,
              rogers_5880RIC6_path, rogers_RO3003_path, radix_46_film_path, 
              radix_46_plate_path, radix_28HT_zp15_path, radix_28HT_zp14_path,
              radix_28HT_film_path, radix_28HT_plate_path, radix_28_6_path,
              radix_28_7_path, tmm3_path_col, rogers_5880LZ_path_col, 
              rogers_6002_path_col, rogers_5880RIC6_path_col, 
              rogers_RO3003_path_col, radix_46_film_path_col,
              radix_46_plate_path_col, radix_28HT_zp15_path_col,
              radix_28HT_zp14_path_col, radix_28HT_film_path_col,
              radix_28HT_plate_path_col, radix_28_6_path_col, radix_28_7_path_col]
    
    
    tmm3_files = extract_columns_files(tmm3_path)
    rogers_5880LZ_files = extract_columns_files(rogers_5880LZ_path)
    rogers_6002_files = extract_columns_files(rogers_6002_path)
    rogers_5880RIC6_files = extract_columns_files(rogers_5880RIC6_path)
    rogers_RO3003_files = extract_columns_files(rogers_RO3003_path)
    radix_46_film_files = extract_columns_files(radix_46_film_path)
    radix_46_plate_files = extract_columns_files(radix_46_plate_path)
    radix_28HT_zp15_files = extract_columns_files(radix_28HT_zp15_path) 
    radix_28HT_zp14_files = extract_columns_files(radix_28HT_zp14_path)
    radix_28HT_film_files = extract_columns_files(radix_28HT_film_path)
    radix_28HT_plate_files = extract_columns_files(radix_28HT_plate_path) 
    radix_28_7_files = extract_columns_files(radix_28_7_path)
    radix_28_6_files = extract_columns_files(radix_28_6_path)
  
    data = plot_epsilonR_summary(paths_and_col, xlims=[0, 1.4], label_cond='legend-outside', color_set=colors_13*2, radix_inset=True, title=None)
    '''
    plot_avg_k(radix_28_6_files, title='Radix 2.8(6)')
    plot_ref_ind(radix_28_6_files)
    plot_k(radix_28_6_files)
    plot_loss_tangent(radix_28_6_files, title='Radix 2.8(6)')
    plot_epsilon_r(radix_28_6_files, title='Radix 2.8(6)')
    '''