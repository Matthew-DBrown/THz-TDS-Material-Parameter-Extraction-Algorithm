# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:27:00 2025

@author: matth
"""

import re
import openpyxl
import spectra as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy import signal
from scipy import constants
import lyteraProcessing3 as lp3
from mpi4py import MPI

import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


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

def linear_fit(x,m,c):
    return m*x + c


def calc_epsilonR(n_data, k_data):
    return np.array(n_data)**2 - np.array(k_data)**2

def calc_lossTan(n_data, k_data):
    epsilon_real = np.array(n_data)**2 - np.array(k_data)**2
    epsilon_imag = 2*np.array(n_data)*np.array(k_data)
    return epsilon_imag/epsilon_real


class SampleRefPair:
    
    def __init__(self, sample_signal, reference_signal, measured_thickness, unwrap_range_selec=False, plot=True, preset=None):
        # sample_signal and reference_signal are objects. Derived from MultiSignals
        self.sample_signal = sample_signal
        self.reference_signal = reference_signal
        self.super_name = sample_signal.signal_file.rsplit('\\', 2)[len(self.sample_signal.signal_file.rsplit('\\', 2))-1].upper().replace("FOC", " ").replace("COL", " ").replace(".TXT", " ").replace("_", " ").replace("ROGERS", " ")
        self.measured_thickness = measured_thickness
        self.fft_freqs, self.fft_sample = sample_signal.calc_complex_fft()[0], sample_signal.calc_complex_fft()[1]
        self.fft_reference = reference_signal.calc_complex_fft()[1]
        self.time_data = self.sample_signal.x_data * 1E-12 # Picosecond normalisation
       
        self.measured_H = self.fft_sample / self.fft_reference # The measured transfer function
        self.measured_phase = np.angle(self.measured_H) # This must be unwrapped
        
        # Unwrapping process (Manual. Either unwrap is fine):-
        self.measured_phase_peaks, _ = find_peaks(self.measured_phase, height=2, distance=1) # Finding peaks in the measured phase signal
        self.measured_unwrapped_phase = np.zeros_like(self.measured_phase) # Constructing empty array to fill with unwrapped data
        
        M = 0 # peak counter set to zero (initialised)
        for i in range(len(self.measured_phase)):
            if i in self.measured_phase_peaks:
                M += 1
            self.measured_unwrapped_phase[i] = self.measured_phase[i] - (M*2*np.pi)
       
        if preset is None:
            self.f_min_param = float(input("Input Minimum Frequency for Analysis (THz): ")) * 1E12 
            self.f_max_param = float(input("Input Maximum Frequency for Analysis (THz): ")) * 1E12
            self.number_fp_pulses = int(input("Number of FP pulses expected: "))
            self.x_min_index = sp.find_index(self.fft_freqs, self.f_min_param)
            self.x_max_index = sp.find_index(self.fft_freqs, self.f_max_param)
            if unwrap_range_selec:
                self.unwrap_fit_min = float(input("Input Minimum Frequency for Phase Extrapolation (THz): ")) * 1E12
                self.unwrap_fit_max = float(input("Input Maximum Frequency for Phase Extrapolation (THz): ")) * 1E12
                self.unwrap_min_idx = sp.find_index(self.fft_freqs, self.unwrap_fit_min)
                self.unwrap_max_idx = sp.find_index(self.fft_freqs, self.unwrap_fit_max)
                
        elif preset is not None:
            self.f_min_param = preset[0] * 1E12
            self.f_max_param =  preset[1] * 1E12
            self.number_fp_pulses = preset[2]
            self.x_min_index = sp.find_index(self.fft_freqs, self.f_min_param)
            self.x_max_index = sp.find_index(self.fft_freqs, self.f_max_param)
            self.unwrap_fit_min = preset[3] * 1E12
            self.unwrap_fit_max = preset[4] * 1E12
            self.unwrap_min_idx = sp.find_index(self.fft_freqs, self.unwrap_fit_min)
            self.unwrap_max_idx = sp.find_index(self.fft_freqs, self.unwrap_fit_max)
        
        self.freqs_interest = self.fft_freqs[self.x_min_index:self.x_max_index+1] # We add 1 such that the final index is inclusive
        
        # Extrapolation of unwrapped phase to zero intercept:-
        
        self.backup_unwrap = np.unwrap(self.measured_phase)
        
        self.extrap_frequencies = self.fft_freqs[self.unwrap_min_idx:self.unwrap_max_idx] # We do not add 1 here as the range is not particularly important as we only use it once for linear fitting
        self.popt, self.pcov = curve_fit(linear_fit, self.extrap_frequencies, self.measured_unwrapped_phase[self.unwrap_min_idx:self.unwrap_max_idx]) # popt[0] = m, popt[1] = c. Need to remove the offset in the unwrapped phase so only concerned with popt[1]
        self.perr = np.sqrt(np.diag(self.pcov))
        
        self.popt_backup, self.pcov_backup = curve_fit(linear_fit, self.extrap_frequencies, self.backup_unwrap[self.unwrap_min_idx:self.unwrap_max_idx]) # popt[0] = m, popt[1] = c. Need to remove the offset in the unwrapped phase so only concerned with popt[1]
        self.perr_backup = np.sqrt(np.diag(self.pcov_backup))
        
        
        
        if self.popt[1] <= 0:
            self.zero_unwrapped_phase = self.measured_unwrapped_phase + np.abs(self.popt[1])
        elif self.popt[1] > 0:
            self.zero_unwrapped_phase = self.measured_unwrapped_phase - np.abs(self.popt[1])
        
        if self.popt_backup[1] <= 0:
            self.zero_unwrapped_phase_backup = self.backup_unwrap + np.abs(self.popt_backup[1])
        elif self.popt_backup[1] > 0:
            self.zero_unwrapped_phase_backup = self.backup_unwrap - np.abs(self.popt_backup[1])
        
        
        # Spectrogram
        
        self.fs = 1 / ((np.round(self.sample_signal.x_data[1], 3) - np.round(
        self.sample_signal.x_data[0], 3))*1e-12)
        x=1000*self.sample_signal.y_data
        Pxx, freqs, bins, _ = plt.specgram(x, NFFT=64, Fs=self.fs, mode='psd', noverlap=32, cmap='hot', vmin=-80, vmax=-60)
        #plt.close()
        
        if plot:
            self.fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 2], hspace=0.4)
            
            # Create the first three subplots with a shared x-axis
            
            self.ax1 = self.fig.add_subplot(gs[0, 0])
            self.ax2 = self.fig.add_subplot(gs[1, 0], sharex=self.ax1)
            self.ax3 = self.fig.add_subplot(gs[2, 0], sharex=self.ax1)
            
            self.ax4 = self.fig.add_subplot(gs[0, 1])
            self.ax5 = self.fig.add_subplot(gs[1, 1])
            self.ax6 = self.fig.add_subplot(gs[2, 1])
            
            self.ax1.set_title(f"Sample: {self.sample_signal.name.rsplit('.txt')[0]}", fontsize=14)
            self.ax1.plot(self.fft_freqs, self.measured_phase, color='b', label=r'Measured $\angle$H')
            self.ax1.set_ylabel("Phase (rad)", fontsize=14)
            self.ax1.yaxis.set_label_coords(-0.1, 0.5)
            self.ax1.tick_params(axis='both', labelsize=11)
            
            self.ax2.plot(self.fft_freqs, self.measured_unwrapped_phase, color='b', label='Manual unwrap')
            self.ax2.plot(self.fft_freqs, self.popt[0]*self.fft_freqs, color='darkmagenta', linestyle='--', label=r'Interpolated fit $\angle$H = '+f'({self.popt[0]:.2e}'+r'$\pm$'+f'{self.perr[0]:.2e})f')
            self.ax2.plot(self.fft_freqs, self.popt_backup[0]*self.fft_freqs, color='red', linestyle='--', label=r'Interpolated fit Numpy $\angle$H = '+f'({self.popt_backup[0]:.2e}'+r'$\pm$'+f'{self.perr_backup[0]:.2e})f')
            self.ax2.plot(self.fft_freqs, self.backup_unwrap, color='r', label='Numpy unwrap')
            self.ax2.set_ylabel("Phase (rad)", fontsize=14)
            self.ax2.set_xlim(np.min(self.fft_freqs), np.max(self.fft_freqs))
            self.ax2.yaxis.set_label_coords(-0.1, 0.5)
            self.ax2.tick_params(axis='both', labelsize=11)
        
            self.ax3.plot(self.fft_freqs, 20*np.log10(np.abs(self.fft_reference)), 'r', label='Reference FFT')
            self.ax3.plot(self.fft_freqs, 20*np.log10(np.abs(self.fft_sample)), 'b', label='Sample FFT')
            self.ax3.hlines(0, np.min(self.fft_freqs), np.max(self.fft_freqs), 'k')
            self.ax3.set_ylabel("Magnitude (dB)", fontsize=14)
            self.ax3.set_xlabel("Frequency (Hz)", fontsize=14)
            self.ax3.yaxis.set_label_coords(-0.1, 0.5)
            self.ax3.tick_params(axis='both', labelsize=11)
            
            self.ax1.axvspan(np.min(self.freqs_interest), np.max(self.freqs_interest), color='lightblue', label='Parameter Extraction Region', alpha=0.8)
            self.ax1.axvspan(self.fft_freqs[self.unwrap_min_idx], self.fft_freqs[self.unwrap_max_idx], color='green', label='Phase Extrapolation Region', alpha=0.8)
            self.ax2.axvspan(np.min(self.freqs_interest), np.max(self.freqs_interest), color='lightblue', alpha=0.8)
            self.ax3.axvspan(np.min(self.freqs_interest), np.max(self.freqs_interest), color='lightblue', alpha=0.8)
            
            self.ax1.legend(loc='lower left', fontsize=14)
            self.ax2.legend(loc='lower left', fontsize=14)
            self.ax3.legend(loc='lower left', fontsize=14)
        
            self.ax1.grid(True, linestyle='--', alpha=0.6)
            self.ax2.grid(True, linestyle='--', alpha=0.6)
            self.ax3.grid(True, linestyle='--', alpha=0.6)
            self.ax4.grid(True, linestyle='--', alpha=0.6)
            self.ax6.grid(True, linestyle='--', alpha=0.6)
            
            self.ax4.plot(self.time_data, self.sample_signal.y_data, color='b', label='Sample')
            self.ax4.plot(self.time_data, self.reference_signal.y_data, color='r', label='Reference')
            self.ax4.set_ylabel("Amplitude (a.u.)", fontsize=14)
            self.ax4.set_xlabel("Time (s)", fontsize=14)
            self.ax4.set_xlim(np.min(self.time_data), np.max(self.time_data))
            self.ax4.tick_params(axis='both', labelsize=11)
            self.ax4.legend(loc='upper right', fontsize=14)
            
            self.mesh = self.ax5.pcolormesh(bins, freqs, 20*np.log10(Pxx), shading='auto', cmap='viridis')
            self.ax5.set_title("Spectrogram of Sample")
            self.ax5.set_ylabel("Frequency (Hz)", fontsize=14)
            self.ax5.set_xlabel("Time (s)", fontsize=14)
            self.fig.colorbar(self.mesh, ax=self.ax5, orientation='horizontal', pad=0.2, fraction=0.05, label="dB")
            self.ax5.tick_params(axis='both', labelsize=11)
            #plt.close('all')
        
        

def init_params_guess(Pair, f, L):
    i_map = sp.find_index(Pair.fft_freqs, f)
    ns_guess = 1 - ((constants.c/(2*np.pi*f*L))*Pair.zero_unwrapped_phase[i_map])
    co = constants.c / (2*np.pi*f*L)
    ln_arg1 = 4*ns_guess / ((ns_guess+1)**2)
    ks_guess = co*(np.log(ln_arg1) - np.log(np.abs(Pair.measured_H[i_map])))
    return ns_guess, ks_guess

def sigma_sum_FP(Pair, ns, ks, l, f):
        ns_tilde = ns - (1j*ks)
        total = 0
        equation = (((ns_tilde - 1)/(ns_tilde + 1))**2) * (
            np.exp(-2*1j*ns_tilde*2*np.pi*l*f/constants.c))
        for i in range(0, Pair.number_fp_pulses+1):
            total += equation**i
        return total


def error_2(Pair, f, ns, ks, L):
    # This outputs the error function that is to be minimized via Nelder-Mead
    i_map = sp.find_index(Pair.fft_freqs, f)
    coeff = (4*(ns-1j*ks))/((ns-1j*ks+1)**2)
    exponent = -1j*(ns - 1j*ks -1)*2*np.pi*f*L/constants.c
    exponent_full = -1j*(ns - 1j*ks -1)*2*np.pi*Pair.fft_freqs*L/constants.c
    
    exponent2_full = -1j * 2*(ns - 1j*ks)*2*np.pi*Pair.fft_freqs*L/constants.c

    
    
    fp_val = (1 - ((ns - 1j*ks - 1)/(ns - 1j*ks + 1))**2 * np.exp(exponent2_full))**(-1)
    
    model = coeff * np.exp(exponent) * fp_val[i_map] # single value
    full_model = coeff * np.exp(exponent_full) * fp_val # array across fft_freqs
    
    M_omega = np.abs(model) - np.abs(Pair.measured_H)[i_map]
    
    # Conducting manual phase unwrapping on the phase model
    phase_peaks, _ = find_peaks(np.angle(full_model), height=2, distance=1)
    p = 0
    unwrapped_phase = np.zeros_like(np.angle(full_model))
    for n in range(len(np.angle(full_model))):
        if n in phase_peaks:
            p += 1
        unwrapped_phase[n]  = np.angle(full_model)[n] - (p*2*np.pi)

    popt, pcov = curve_fit(linear_fit, Pair.extrap_frequencies, unwrapped_phase[Pair.unwrap_min_idx:Pair.unwrap_max_idx])
    
    if popt[1]<=0:
        interpolated_model_unwrapped_phase = unwrapped_phase + np.abs(popt[1])

    elif popt[1]>0:
        interpolated_model_unwrapped_phase = unwrapped_phase - np.abs(popt[1])

    
    A_omega = interpolated_model_unwrapped_phase[i_map] - Pair.zero_unwrapped_phase_backup[i_map]
    
    
    return np.abs(M_omega) + np.abs(A_omega)


def Extract_NelderMead(Pair, plus_l, minus_l, resolution, parent_path_to_save, measurement_name):
    
    data_for_thicknesses = {}
    TV = {}
    
    thicknesses = np.linspace(Pair.measured_thickness - minus_l, Pair.measured_thickness + plus_l, resolution, endpoint=True)
    total_calculations_number = len(thicknesses)*len(Pair.freqs_interest)
    for L in thicknesses:
        thickness_iteration_number = sp.find_index(thicknesses, L)
        
        n_sample_real = []
        k_sample_imag = []
        f_num = thickness_iteration_number * len(Pair.freqs_interest)
        for frequency in Pair.freqs_interest:
            total_progress = f_num / total_calculations_number 
            ns_guess, ks_guess = init_params_guess(Pair, frequency, L)
            init_guess = [ns_guess, ks_guess]
            temp_fn = lambda t: error_2(Pair, frequency, t[0], t[1], L)
            minimized_params = minimize(temp_fn, init_guess, method='Nelder-Mead')
            print(f"[{Pair.super_name}] Progress (COARSE SWEEP): {(total_progress * 100):.2f}% ---------------------------------------------------------")
            print(f"Frequency point: {frequency/1e12} THz\nGuessed n: {ns_guess} | k = {ks_guess}")
            print(f"Minimised: n = {minimized_params.x[0]} | k = {minimized_params.x[1]}\n")
            

            n_value = minimized_params.x[0]
            k_value = minimized_params.x[1]
            
            n_sample_real.append(n_value)
            k_sample_imag.append(k_value)
            
            f_num += 1
            
        temp_stack = np.stack((n_sample_real, k_sample_imag), axis=0)

        data_for_thicknesses[L] = temp_stack
    
    
    for thickness, data_nk in data_for_thicknesses.items():
        TV[thickness] = np.sum(np.abs(np.diff(data_nk[0])) + np.abs(np.diff(data_nk[1])))
    
    thicknesses_array = []
    tv_dat = []
    for thickness, tv in TV.items():
        thicknesses_array.append(thickness)
        tv_dat.append(tv)
        
    coarse_min_thickness = np.min(thicknesses_array)
    fine_range = np.linspace(coarse_min_thickness*0.995, coarse_min_thickness*1.005, 10, endpoint=True)
    total_fine_calculations_number = len(fine_range)*len(Pair.freqs_interest)    
    
    print("Progress (COARSE SWEEP) 100%\nFINISHED\n")
    print(f"TV minimum: {np.min(thicknesses_array)}\n")
    print(f"CONTINUING WITH FINE SWEEP BETWEEN: {coarse_min_thickness*0.98} and {coarse_min_thickness*1.02}")
    
    #--- FINE SWEEP ---#
       
    data_for_fine_thicknesses = {}
    TV_fine = {}
    for L in fine_range:
        fine_thickness_iteration_number = sp.find_index(fine_range, L)
        #thickness_progress = (thickness_iteration_number / len(thicknesses)) * 100
        n_sample_real_fine = []
        k_sample_imag_fine = []
        f_num_fine = fine_thickness_iteration_number * len(Pair.freqs_interest)
        for frequency in Pair.freqs_interest:
            total_progress_fine = f_num_fine / total_fine_calculations_number
            ns_guess, ks_guess = init_params_guess(Pair, frequency, L)
            init_guess = [ns_guess, ks_guess]
            temp_fn = lambda t: error_2(Pair, frequency, t[0], t[1], L)
            minimized_params = minimize(temp_fn, init_guess, method='Nelder-Mead')
            print(f"[{Pair.super_name}] Progress (FINE SWEEP): {(total_progress_fine * 100):.2f}% ---------------------------------------------------------")
            print(f"Frequency point: {frequency/1e12} THz\nGuessed n: {ns_guess} | k = {ks_guess}")
            print(f"Minimised: n = {minimized_params.x[0]} | k = {minimized_params.x[1]}\n")
            

            n_value_fine = minimized_params.x[0]
            k_value_fine = minimized_params.x[1]
            
            n_sample_real_fine.append(n_value_fine)
            k_sample_imag_fine.append(k_value_fine)    
            
            f_num_fine += 1
            
        temp_stack_fine = np.stack((n_sample_real_fine, k_sample_imag_fine), axis=0)

        data_for_fine_thicknesses[L] = temp_stack_fine
    
    for thickness, data_nk in data_for_fine_thicknesses.items():
        TV_fine[thickness] = np.sum(np.abs(np.diff(data_nk[0])) + np.abs(np.diff(data_nk[1])))
    
    fine_thicknesses_array = []
    fine_tv_dat = []
    for thickness, tv in TV_fine.items():
        fine_thicknesses_array.append(thickness)
        fine_tv_dat.append(tv)
    
    
    epsilonR = calc_epsilonR(data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][0], data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][1])
    lossTan = calc_lossTan(data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][0], data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][1])
    
    
    # Saving all fine thicknesses
    df = pd.DataFrame()
    save_path = parent_path_to_save+measurement_name+"_(all_thicknesses).xlsx"
    df.to_excel(save_path)
    writer = pd.ExcelWriter(save_path, engine='openpyxl')
    for thickness, data in data_for_fine_thicknesses.items():
        df_for_thickness = pd.DataFrame(data={'frequencies':Pair.freqs_interest, 'n':data[0], 'k':data[1]})
        df_for_thickness.to_excel(writer, sheet_name=f'{thickness}', header=True, index=False)
    writer.close()
    print("\nSAVED ALL FINE THICKNESS DATA\n")
    # Saving optimal file
    df = pd.DataFrame()
    save_path_optimal = parent_path_to_save+measurement_name+f"OPTIMAL_T={fine_thicknesses_array[np.argmin(fine_tv_dat)]}.xlsx"
    df.to_excel(save_path_optimal)
    writer = pd.ExcelWriter(save_path_optimal, engine='openpyxl')
    optimal_n = data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][0]
    optimal_k = data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][1]
    df_optimal = pd.DataFrame(data={'frequency': Pair.freqs_interest, 'n': optimal_n, 'k': optimal_k, 'epsilon_r': epsilonR, 'loss_tan': lossTan})
    df_optimal.to_excel(writer, sheet_name=f'{fine_thicknesses_array[np.argmin(fine_tv_dat)]}', header=True, index=False)
    writer.close()
    print("\nSAVED OPTIMAL THICKNESS DATA\n")
    
    
    Pair.ax6.plot(Pair.freqs_interest, data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][0], marker='o', markersize=1, color='black', label=rf'$n_s$\nThickness: {(fine_thicknesses_array[np.argmin(fine_tv_dat)]):.7f}')
    Pair.ax6.set_ylabel("Refractive Index", color="blue")
    Pair.ax6.tick_params(axis='y')
    Pair.ax6.set_ylim(1, np.max(n_sample_real_fine)*1.05)
    
    ax6_right = Pair.ax6.twinx()
    ax6_right.plot(Pair.freqs_interest, data_for_fine_thicknesses[fine_thicknesses_array[np.argmin(fine_tv_dat)]][1], color="red", label=r"$\kappa$\nThickness: {(fine_thicknesses_array[np.argmin(fine_tv_dat)]):.4f}")
    ax6_right.set_ylabel("Extinction Coefficient")
    ax6_right.set_ylim(0, np.max(k_sample_imag_fine)*1.05)
    ax6_right.legend(loc="upper right", fontsize=14)
    Pair.ax6.legend(loc='lower right', fontsize=14)
    
    
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax7.set_ylim(1, np.max(epsilonR)*1.2)
    ax7.plot(Pair.freqs_interest/1e12, epsilonR, linestyle='--', color='firebrick', label=f'Algorithm Output\nThickness: {(fine_thicknesses_array[np.argmin(fine_tv_dat)]):.4f}')
    ax7.plot(Pair.freqs_interest/1e12, lp3.smooth_data(epsilonR, 3, 'nearest', 2), linestyle='-', color='darkred', label='Savitzky-Golay Filtering')
    ax7.set_ylabel(r"$\epsilon_r$", fontsize=14)
    ax7.set_xlabel("Frequency (THz)", fontsize=14)
    ax7.tick_params('both', labelsize=14)
    ax7.legend(fontsize=14, title=Pair.super_name)
    ax7.grid(True, linestyle='--', alpha=0.4)
    
    ax8.plot(Pair.freqs_interest/1e12, lossTan, linestyle='--', color='lightblue', label=f'Algorithm Output\nThickness: {(fine_thicknesses_array[np.argmin(fine_tv_dat)]):.4f}')
    ax8.plot(Pair.freqs_interest/1e12, lp3.smooth_data(lossTan, 3, 'nearest', 1), linestyle='-', color='blue', label='Savitzky-Golay Filtering')
    ax8.set_ylim(0, np.max(lossTan)*1.2)
    ax8.set_ylabel(r"$tan(\delta)$", fontsize=14)
    ax8.set_xlabel("Frequency (THz)", fontsize=14)
    ax8.tick_params('both', labelsize=14)
    ax8.legend(fontsize=14)
    ax8.grid(True, linestyle='--', alpha=0.4)
    
    fig3, ax9 = plt.subplots()
    ax9.plot(fine_thicknesses_array, fine_tv_dat, marker='o', color='r', label='TV (FINE)')
    ax9.legend('lower left', fontsize=14)
    ax9.set_ylabel("Total Variation", fontsize=14)
    ax9.set_xlabel("Thickness (m)", fontsize=14)
    ax9.grid(True, linestyle='--', alpha=0.4)

    
    
    return data_for_thicknesses, TV, data_for_fine_thicknesses, TV_fine

def Extract_NelderMead_MPI(Pair, plus_l, minus_l, resolution, parent_path_to_save, measurement_name):
    plt.close()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Starting script in process {rank}/{size}")
    sys.stdout.flush()
    if rank == 0:
        print(f"Starting Extract_NelderMead with {size} MPI processes...\n")
        sys.stdout.flush()
    
    data_for_thicknesses = {}
    TV = {}

    # Split the thicknesses across MPI ranks
    thicknesses = np.linspace(Pair.measured_thickness - minus_l, Pair.measured_thickness + plus_l, resolution, endpoint=True)
    local_thicknesses = np.array_split(thicknesses, size)[rank]  # Each rank gets a subset

    print(f"Rank {rank}: Processing {len(local_thicknesses)} thickness values...")
    sys.stdout.flush()

    local_data = {}

    # --- COARSE SWEEP ---
    for i, L in enumerate(local_thicknesses):
        progress = (i + 1) / len(local_thicknesses) * 100
        print(f"Rank {rank}: Coarse sweep progress {progress:.2f}% (Thickness {L:.6f})")
        sys.stdout.flush()

        n_sample_real = []
        k_sample_imag = []
        for frequency in Pair.freqs_interest:
            ns_guess, ks_guess = init_params_guess(Pair, frequency, L)
            init_guess = [ns_guess, ks_guess]
            temp_fn = lambda t: error_2(Pair, frequency, t[0], t[1], L)
            minimized_params = minimize(temp_fn, init_guess, method='Nelder-Mead')

            n_sample_real.append(minimized_params.x[0])
            k_sample_imag.append(minimized_params.x[1])

        temp_stack = np.stack((n_sample_real, k_sample_imag), axis=0)
        local_data[L] = temp_stack

    print(f"Rank {rank}: Finished coarse sweep.")
    sys.stdout.flush()

    # Gather results at rank 0
    all_data = comm.gather(local_data, root=0)

    if rank == 0:
        print("Rank 0: Gathering coarse sweep results from all processes...")
        sys.stdout.flush()
        
        for process_data in all_data:
            data_for_thicknesses.update(process_data)

        for thickness, data_nk in data_for_thicknesses.items():
            TV[thickness] = np.sum(np.abs(np.diff(data_nk[0])) + np.abs(np.diff(data_nk[1])))

        # Get optimal thickness
        coarse_min_thickness = min(TV, key=TV.get)
        print(f"Rank 0: Coarse sweep complete. Optimal thickness = {coarse_min_thickness:.6f}")
        sys.stdout.flush()

        #fine_range = np.linspace(coarse_min_thickness * 0.98, coarse_min_thickness * 1.02, 70, endpoint=True)
        fine_range = np.linspace(Pair.measured_thickness-100E-6, Pair.measured_thickness+100E-6, 100, endpoint=True)
        
    else:
        fine_range = None

    # Broadcast fine_range
    fine_range = comm.bcast(fine_range, root=0)

    print(f"Rank {rank}: Received fine range for processing.")
    sys.stdout.flush()

    # --- FINE SWEEP ---
    local_fine_thicknesses = np.array_split(fine_range, size)[rank]
    local_fine_data = {}

    for i, L in enumerate(local_fine_thicknesses):
        progress = (i + 1) / len(local_fine_thicknesses) * 100
        print(f"Rank {rank}: Fine sweep progress [{Pair.super_name}] {progress:.2f}% (Thickness {L:.6f})")
        sys.stdout.flush()

        n_sample_real_fine = []
        k_sample_imag_fine = []
        for frequency in Pair.freqs_interest:
            ns_guess, ks_guess = init_params_guess(Pair, frequency, L)
            init_guess = [ns_guess, ks_guess]
            temp_fn = lambda t: error_2(Pair, frequency, t[0], t[1], L)
            minimized_params = minimize(temp_fn, init_guess, method='Nelder-Mead')

            n_sample_real_fine.append(minimized_params.x[0])
            k_sample_imag_fine.append(minimized_params.x[1])

        temp_stack_fine = np.stack((n_sample_real_fine, k_sample_imag_fine), axis=0)
        local_fine_data[L] = temp_stack_fine

    print(f"Rank {rank}: Finished fine sweep.")
    sys.stdout.flush()

    # Gather fine sweep results
    all_fine_data = comm.gather(local_fine_data, root=0)

    if rank == 0:
        print("Rank 0: Gathering fine sweep results from all processes...")
        sys.stdout.flush()
    
        data_for_fine_thicknesses = {}
        TV_fine = {}

        for process_fine_data in all_fine_data:
            data_for_fine_thicknesses.update(process_fine_data)

        for thickness, data_nk in data_for_fine_thicknesses.items():
            TV_fine[thickness] = np.sum(np.abs(np.diff(data_nk[0])) + np.abs(np.diff(data_nk[1])))

        optimal_thickness = min(TV_fine, key=TV_fine.get)
        print(f"Rank 0: Fine sweep complete. Optimal thickness = {optimal_thickness:.6f}")
        sys.stdout.flush()

        optimal_n = data_for_fine_thicknesses[optimal_thickness][0]
        optimal_k = data_for_fine_thicknesses[optimal_thickness][1]

        epsilonR = calc_epsilonR(optimal_n, optimal_k)
        lossTan = calc_lossTan(optimal_n, optimal_k)
        
        # Save all fine thickness results in an Excel file
        fine_results_path = f"{parent_path_to_save}/{measurement_name}_fine_results.xlsx"
        with pd.ExcelWriter(fine_results_path) as writer:
            for thickness, data_nk in data_for_fine_thicknesses.items():
                df = pd.DataFrame({"n": data_nk[0], "k": data_nk[1]}, index=Pair.freqs_interest)
                df.to_excel(writer, sheet_name=f"Thickness_{thickness:.6f}")
        
        # Save optimal thickness in a separate Excel file
        optimal_results_path = f"{parent_path_to_save}/{measurement_name}_optimal_T={optimal_thickness}.xlsx"
        df_optimal = pd.DataFrame({"n": data_for_fine_thicknesses[optimal_thickness][0],
                                   "k": data_for_fine_thicknesses[optimal_thickness][1],
                                   "epsilon_r": epsilonR,
                                   "loss_tangent": lossTan,
                                   "epsilon_i": 2*data_for_fine_thicknesses[optimal_thickness][0]*data_for_fine_thicknesses[optimal_thickness][1],
                                   "conductivity [(Ohm m)^-1]": 2*data_for_fine_thicknesses[optimal_thickness][0]*data_for_fine_thicknesses[optimal_thickness][1]*2*np.pi*Pair.freqs_interest*constants.epsilon_0,
                                   "resistivity [Ohm cm]": (2*data_for_fine_thicknesses[optimal_thickness][0]*data_for_fine_thicknesses[optimal_thickness][1]*2*np.pi*Pair.freqs_interest*constants.epsilon_0)**-1 * 1E-2}, index=Pair.freqs_interest)
        df_optimal.to_excel(optimal_results_path, sheet_name="Optimal")

        print(f"Rank 0: Results saved in \n{fine_results_path} \nand \n{optimal_results_path}")
        sys.stdout.flush()

        print("Rank 0: Starting plotting...")
        sys.stdout.flush()
    
        # --- PLOTTING ---
        Pair.ax6.plot(Pair.freqs_interest, optimal_n, marker='o', markersize=1, color='black', label=f'$n_s$ Optimal\nT: {optimal_thickness:.7f}')
        Pair.ax6.set_ylabel("Refractive Index", color="blue")
        Pair.ax6.set_ylim(1, np.max(optimal_n) * 1.05)

        ax6_right = Pair.ax6.twinx()
        ax6_right.plot(Pair.freqs_interest, optimal_k, color="red", label=r"$\kappa$")
        ax6_right.set_ylabel("Extinction Coefficient")
        ax6_right.set_ylim(0, np.max(optimal_k) * 1.05)
        ax6_right.legend(loc="upper right", fontsize=14)
        Pair.ax6.legend(loc='lower right', fontsize=14)

        ## Figure 2: Epsilon and Loss Tangent
        fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax7.set_ylim(1, np.max(epsilonR)*1.2)
        ax7.plot(Pair.freqs_interest/1e12, epsilonR, linestyle='--', color='firebrick', label=f'Epsilon\nT: {optimal_thickness:.4f}')
        ax7.plot(Pair.freqs_interest/1e12, smooth_data(epsilonR, 3, 'nearest', 2), linestyle='-', color='darkred', label='Savitzky-Golay Filtering')
        ax7.set_ylabel(r"$\epsilon_r$", fontsize=14)
        ax7.set_xlabel("Frequency (THz)", fontsize=14)
        ax7.tick_params('both', labelsize=14)
        ax7.grid(True, linestyle='--', alpha=0.4)
        ax7.legend(fontsize=14, title=Pair.super_name)
        
        
        ax8.set_ylim(0, np.max(lossTan)*1.2)
        ax8.plot(Pair.freqs_interest/1e12, lossTan, linestyle='--', color='lightblue', label=f'Loss Tangent\nT: {optimal_thickness:.4f}')
        ax8.plot(Pair.freqs_interest/1e12, smooth_data(lossTan, 3, 'nearest', 1), linestyle='-', color='blue', label='Savitzky-Golay Filtering')
        ax8.set_ylabel(r"$tan(\delta)$", fontsize=14)
        ax8.set_xlabel("Frequency (THz)", fontsize=14)
        ax8.tick_params('both', labelsize=14)
        ax8.legend(fontsize=14)
        ax8.grid(True, linestyle='--', alpha=0.4)

        ## Figure 3: TV Plot
        fig3, ax9 = plt.subplots()
        ax9.plot(list(TV_fine.keys()), list(TV_fine.values()), marker='o', color='r', label='TV (Fine)')
        ax9.set_ylabel("Total Variation", fontsize=14)
        ax9.set_xlabel("Thickness (m)", fontsize=14)
        ax9.grid(True, linestyle='--', alpha=0.4)
        ax9.legend()

        plt.show()

        print("Rank 0: Plotting complete.")
        sys.stdout.flush()
        plt.close('all')
    #return data_for_thicknesses, TV, data_for_fine_thicknesses, TV_fine



if __name__ == "__main__":
    
    # Example use
    
    post_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\silicon_samples\shinetsu_wafer\post_irradiation"
    thickness = 3125E-6
    radix_46_path = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\TERA_BACKUP\y4_project\radix_4.6\collimated\241101\film_side"
    reference_obj = sp.Signal(os.path.join(radix_46_path, 'reference.txt'), name='Radix 4.6 Ref')
    dump_test = r"C:\Users\matth\Desktop"

    for file in os.listdir(radix_46_path):
        if "reference" not in file.lower() and file.endswith("txt"):
            signal_obj = sp.Signal(os.path.join(radix_46_path, file), name=f'{file}')
            Pair = SampleRefPair(signal_obj, reference_obj, thickness, unwrap_range_selec=True, plot=True, preset=[0.2, 1.7, 3, 0.3, 0.8])
            # Preset format above: [(minimum extraction frequency), (maximum extraction frequency), (number_fp), (minimum unwrap frequency range). (maximum unwrap frequency range)]
            #Extract_NelderMead_MPI(Pair, 10E-6, 10E-6, 1, dump_test, f'{file}')
            
            # ** NOTE THAT THE GEOMETRIC SUM IS USED INSTEAD AS SEEN IN NAFTALY FOR FP TREATMENT ##
            
    
    
