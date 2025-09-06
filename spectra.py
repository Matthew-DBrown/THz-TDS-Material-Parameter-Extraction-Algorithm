# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:20:42 2024

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import scipy as sc
import os
from scipy import constants
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

## NOTE: The original idea behind this file was to have the nelder-mead minimisation in here. However, this is now a utilities file that is used in the NMAlgorithm file.
## AS A RESULT, THERE ARE A LOT OF EMPTY FUNCTIONS - IGNORE
n_air = 1
k_air = 0 # For now. We will need to establish a frequency dependence here later


def Calculate_T_theory():
    pass


def zero(ft_data):
    pass # Function to zero data? May be best to have this inside the class

def calculate_dynamic_range(spectrum):
    pass # Makes a plot like in Miguel paper. Class-Method / Function?

def calc_delta_phi(arg_H_measured, H_theory):
    return arg(H_theory) - arg_H_measured # Need to make a function that calculates unwrapped phase of theory
    
def arg(function):
    pass

def calc_delta_rho(H_theory, H_measured):
    return np.ln(np.abs(H_theory)) - np.ln(np.abs(H_measured))

def calc_delta_nk(H_measured, arg_H_measured, H_theory):
    return calc_delta_phi(arg_H_measured, H_theory)**2 + calc_delta_rho(H_theory, H_measured)**2


def linear_function(x, m, c):
    return m*x + c

def pad_to_power_of_2(signal, min_length=1024):
    """
    Pads a signal array with zeros to make its length a power of 2,
    with a minimum resulting size of min_length.
    
    Parameters:
        signal (array-like): The input time-domain signal.
        min_length (int): The minimum resulting size. Default is 1024.
        
    Returns:
        padded_signal (numpy.ndarray): The zero-padded signal.
    """
    current_length = len(signal)
    target_length = max(2**int(np.ceil(np.log2(current_length))), min_length)
    padding_length = target_length - current_length
    padded_signal = np.pad(signal, (0, padding_length), mode='constant')
    return padded_signal

def find_index(array, target):
    '''
    Parameters
    ----------
    array : Numpy array
        
    target : Float

    Returns
    -------
    Index in array of a value nearest target value.

    '''
    
    diff = np.abs(array - target)
    return np.argmin(diff)

def interpolate(unwrapped_data, frequency_array, x_min, x_max, plot=True):
    '''
    Parameters
    ----------
    unwrapped_data : Numpy array
        The data points that now form a linear function for a limited frequency range (noise).
    frequency_array : Numpy array
        The frequency range associated with the unwrapped data.
    x_min:
        
    x_max:
        
    Returns
    -------
    Fit. Plot (optional)
    '''
    f_min_index = find_index(frequency_array, x_min)
    f_max_index = find_index(frequency_array, x_max)
    popt, pcov = curve_fit(linear_function, frequency_array[f_min_index:f_max_index], unwrapped_data[f_min_index:f_max_index])
    return popt, pcov



class Signal:
    
    def __init__(self, signal_file, name=None):
        '''
        Parameters
        ----------
        signal_file : RAW String
            Takes the directory of the file, with '\' as '/'.

        '''
        self.signal_file = signal_file
        self.name = name
        self.extension = self.signal_file.rsplit('.', 1)[-1] # Extracts the filetype.
        if self.extension == 'csv':
            self.df = pd.DataFrame(data=pd.read_csv(self.signal_file).values)
            self.x_data = self.df.iloc[:, 0]
            self.y_data = self.df.iloc[:, 1]
            self.data_size = len(self.y_data)
            # self.odu_data = self.df.iloc[:,2] # ODU's equivalent time measurement
        elif self.extension == 'txt':
            self.df = pd.read_csv(self.signal_file, delim_whitespace=True, header=None)
            self.x_data = self.df.iloc[:, 0]
            self.y_data = self.df.iloc[:, 1]
            self.data_size = len(self.y_data)
            # self.odu_data = self.df.iloc[:,2]
        elif self.extension == 'xlsx':
            pass
        else:
            print("Invalid Filetype!")
        
        #plt.plot(self.x_data, self.y_data, linestyle='-', marker='o', color='m', markersize=2, linewidth=1)
        #plt.xlabel("Time")
        #plt.ylabel("_____")

    def perform_fft(self, logscale=False):
        # Assumes equal interval rate
        # Something about FFT not that good for number of data =! 2^n
        self.x_rounded = np.round(self.x_data, 3)
        self.time_step = self.x_rounded[1]-self.x_rounded[0]
        self.fft_signal = np.fft.fft(self.y_data)
        n = len(self.x_data)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        # self.fft_signal_pos = self.fft_signal[:self.data_size//2]
        # self.fft_freq_array_pos = self.fft_freq_array[:self.data_size//2]
        
        plt.plot(self.fast_freqs[:self.data_size//2], np.abs(self.fft_signal[:self.data_size//2]), linestyle='-', marker='o', color='b')
        if logscale:
            plt.yscale('log')
        plt.title(self.signal_file)
        plt.ylabel("Arbitrary Units", fontsize=14)
        plt.xlabel("Frequency (THz)", fontsize=14)
        plt.grid(True, linestyle='--', color='k', alpha=0.4)
    
    def period_lombscargle(self, f_min, f_max, n, normalize):
        # Apparently Lombscargle uses angular frequencies!
        
        frequency_array = np.linspace(f_min, f_max, int(n), endpoint=True)/(2*np.pi)
        pgram = sc.signal.lombscargle(self.x_data, self.y_data, frequency_array, normalize=normalize)
        plt.title(f"Periodogram for: {self.signal_file}")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (?)")
        plt.plot(frequency_array, pgram)
        plt.show()
    
    def calc_lombscargle(self, f_min, f_max, n, normalize):
        # Does not plot --> faster to collect data
        frequency_array = np.linspace(f_min, f_max, int(n), endpoint=True)/(2*np.pi)
        pgram = sc.signal.lombscargle(self.x_data, self.y_data, frequency_array, normalize=normalize)
        return frequency_array, pgram
    
    def calc_fft(self, logscale=False):
        self.x_rounded = np.round(self.x_data, 3)
        self.time_step = self.x_rounded[1]-self.x_rounded[0]
        self.fft_signal = np.fft.fft(self.y_data)
        n = len(self.x_data)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        if logscale:
            return self.fast_freqs[:len(self.x_data)//2], np.log10(np.abs(self.fft_signal[:len(self.x_data)//2]))
        else:
            return self.fast_freqs[:len(self.x_data)//2], np.abs(self.fft_signal[:len(self.x_data)//2])
     
    def calc_complex_fft2(self, zero_pad, time_normalisation=1E-12, positive=True, logscale=False):
        '''
        PADDED!
        '''
        self.x_rounded = np.round(self.x_data, 3) * time_normalisation
        self.time_step = self.x_rounded[1] - self.x_rounded[0]
        self.padded_y = np.pad(self.y_data, (0, zero_pad - len(self.y_data)), 'constant')
        self.fft_signal = np.fft.fft(self.padded_y)
        n = len(self.padded_y)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        if positive and not logscale:
            return self.fast_freqs[:len(self.padded_y)//2], self.fft_signal[:len(self.padded_y)//2]
        elif positive and logscale:
            return self.fast_freqs[:len(self.padded_y)//2], np.log10(self.fft_signal[:len(self.padded_y)//2])
        elif not positive and not logscale:
            return self.fast_freqs, self.fft_signal
        elif not positive and logscale:
            return self.fast_freqs, np.log10(self.fft_signal) # Pretty sure this will not work anyway as some values are negative
        
    def calc_complex_fft(self, time_normalisation=1E-12, positive=True, logscale=False):
        self.x_rounded = np.round(self.x_data, 3) * time_normalisation
        self.time_step = self.x_rounded[1] - self.x_rounded[0]
        self.fft_signal = np.fft.fft(self.y_data)
        n = len(self.x_data)
        self.fast_freqs = np.fft.fftfreq(n, self.time_step)
        if positive and not logscale:
            return self.fast_freqs[:len(self.x_data)//2], self.fft_signal[:len(self.x_data)//2]
        elif positive and logscale:
            return self.fast_freqs[:len(self.x_data)//2], np.log10(self.fft_signal[:len(self.x_data)//2])
        elif not positive and not logscale:
            return self.fast_freqs, self.fft_signal
        elif not positive and logscale:
            return self.fast_freqs, np.log10(self.fft_signal) 
        
        
        
    def calculate_windowed_fft(self, t1, t2, plot=False, logscale_state=False):
        x_1_index = find_index(self.x_data, t1)
        x_2_index = find_index(self.x_data, t2)
        self.x_rounded_window = np.round(self.x_data, 3)[x_1_index:x_2_index]
        # self.window_timestep = self.x_rounded_window[1]-self.x_rounded_window[0]
        self.window_timestep = 0.156
        self.window_fft_signal = np.fft.fft(self.y_data[x_1_index:x_2_index])
        n = len(self.x_rounded_window)
        self.windowed_fast_freq = np.fft.fftfreq(n, self.window_timestep)
        if plot:
            if logscale_state:
                plt.plot(self.windowed_fast_freq[:n//2], np.abs(self.window_fft_signal[:n//2]))
                plt.yscale('log')
            else:
                plt.plot(self.windowed_fast_freq[:n//2], np.abs(self.window_fft_signal[:n//2]))
        else:
            if logscale_state==False:
                return self.windowed_fast_freq[:n//2], np.abs(self.window_fft_signal[:n//2])
            else:
                return self.windowed_fast_freq[:n//2], np.log10(np.abs(self.window_fft_signal[:n//2]))
    
    
    def re_plot(self):
        plt.plot(self.x_data, self.y_data, linestyle='-', marker='o', color='m', markersize=2, linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("_____")
        if self.name:
            plt.title(f"{self.name}")
    
    def show_data(self):
        '''
        Can be handy to see.
        '''
        return self.x_data, self.y_data #, self.odu_data
    
    def spectrogram(self):
        self.x_rounded = np.round(self.x_data, 3)
        time_step = self.x_rounded[1]-self.x_rounded[0] * 10**(-12)
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(self.x_data, self.y_data)
        ax1.set_label("Signal")
        
        Pxx, freqs, bins, im = ax2.specgram(self.y_data, NFFT=256, Fs=1/time_step)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        
        plt.show()     
        
    def measured_transfer_function(self, fft_signal, reference):
        # In the frequency domain
        self.msrd_transfer_function = fft_signal / reference.calc_complex_fft()[1]
        return self.msrd_transfer_function
        
    def phase_transfer_function(self, plot=False, name=False):
        self.phase_data = np.angle(self.msrd_transfer_function)
        frequencies = self.calc_complex_fft()[0]
        if plot:
            fig = plt.figure(figsize=(5,5))
            plt.plot(frequencies, self.phase_data/np.pi, color='b')
            plt.xlabel("Frequency (THz)", fontsize=14)
            plt.xlim(np.min(frequencies), np.max(frequencies))
            plt.ylabel(r"Angle ($\pi$ radians)", fontsize=14)
            plt.hlines([-1, 1], np.min(frequencies), np.max(frequencies), linestyle='--', color='r')
            if name:
                plt.title(f"Phase of Measured Transfer Function\n{name}", fontsize=14)
            elif not name:
                plt.title(f"Phase of Measured Transfer Function\n{self.signal_file}")
            plt.show()
       
    def count_peaks(self, data, threshold_val, dist, plot=False):
        peaks, _ = find_peaks(self.phase_data, height=threshold_val, distance=dist)
        if plot:
            freqs = self.calc_complex_fft()[0]
            plt.plot(freqs, self.phase_data, color='b', label='Phase Data')
            plt.scatter(freqs[peaks], self.phase_data[peaks], color='red', label='Identified Peaks', zorder=5)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (radians)")
            plt.title("Phase Data with Peaks")
            plt.xlim(np.min(freqs), np.max(freqs))
            plt.legend(loc='lower left', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
        return peaks, _ 
        
    def unwrap_phase(self, plot=False):
        freq_array = self.calc_complex_fft()[0]
        phase_data = np.copy(self.phase_data)

        peaks, _ = find_peaks(phase_data, height=2, distance=1)
        
        M = 0  # Counter for multiples of 2*pi to subtract
        unwrapped_phase = np.zeros_like(phase_data)  # Array for unwrapped phase data
    
        # Sweep through phase data and adjust by subtracting M*2*pi
        for i in range(len(phase_data)):

            if i in peaks:
                M += 1
        
            # Subtract M*2*pi from the current phase value
            unwrapped_phase[i] = phase_data[i] - M * 2 * np.pi

        if plot:
            plt.plot(freq_array, unwrapped_phase, color='purple', label='Unwrapped Phase')
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Unwrapped Phase (radians)")
            plt.title("Unwrapped Phase Data")
            plt.legend(loc='lower left', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()
        
        return unwrapped_phase
    
    def calc_n_sample_NEEDS_FIXING(self):
        pass
    

    
    

class MultiSignals:
    
    def __init__(self, signals_path):
        self.signals_dic = {}
        
        for file in os.listdir(signals_path):
            if file.rsplit('.', 1)[-1] =="txt":
                if file not in self.signals_dic:
                    self.signals_dic[file] = Signal(os.path.join(signals_path, file), f"{file}")
            else:
                print(f'Ignored: {file}')
        
        self.signals_dic_nref = self.signals_dic.copy() #dictionary without reference
        self.signals_dic_nref.pop('reference.txt')
        y_data_list = [np.array(signal.y_data) for signal in self.signals_dic_nref.values()]
        self.array_stack_nref = np.vstack(y_data_list)
        
        self.reference_freqs = self.signals_dic['reference.txt'].calc_fft(logscale=False)[0]
        self.reference_fft = self.signals_dic['reference.txt'].calc_fft(logscale=False)[1]
                    
                
    def plot_all(self):
        for file, signal in self.signals_dic.items():
            plt.plot(signal.x_data, signal.y_data, label=f'{file}')
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (a.u.)")
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(True, color=-'k', linestyle='--', alpha=0.5)
        plt.show()
        
    def calc_std_dev(self):
        self.std_dev_array = np.std(self.array_stack_nref, 0)
        return self.std_dev_array
    
    def calc_avg_array(self):
        self.avg_array = np.average(self.array_stack_nref, 0)
        return self.avg_array
    
    def plot_snr_average(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.signals_dic['reference.txt'].x_data, self.calc_avg_array(), color='b', linestyle='-', label="Average")
        ax1.plot(self.signals_dic['reference.txt'].x_data, 20*self.calc_std_dev(), color='g', linestyle='-', label ='Std Dev (x20)')
        ax1.set_ylabel("Amplitude (a.u.)", fontsize=14)
        ax1.set_xlabel("Time (ps)", fontsize=14)
        ax1.set_xlim(0, np.max(self.signals_dic['reference.txt'].x_data))
        ax1.grid(True, color='k', linestyle='--', alpha=0.5)
        ax2 = ax1.twinx()
        ax2.plot(self.signals_dic['reference.txt'].x_data, np.abs(self.calc_avg_array()/self.calc_std_dev()), color='r', linestyle='dotted', label='SNR')
        ax2.set_ylabel("SNR (a.u.)", fontsize=14)
        ax1.legend(loc='upper left', fontsize=14)
        ax2.legend(loc='upper right', fontsize=14)    
        fig.suptitle("Radix(6) Data")
  
    
    def plot_error_function(self, sample_measurement, sample_thickness):
        # Step 1: need the frequency domain of the sample and the reference
        # We will define the reference in the __init__ method during instant.n
        # ASSUMING ABSORPTION IN AIR IS SMALL I.E. K_air ~ 0 
        sample_fft = self.signals_dic[sample_measurement].calc_fft(logscale=False)[1]
        sample_freq = self.signals_dic[sample_measurement].calc_fft(logscale=False)[0]#
        
        T_measured = sample_fft / self.reference_fft
        n_sample_range = np.linspace(0, 10, num=len(self.reference_fft), endpoint=True)
        k_sample_range = np.linspace(0, 10, num=len(self.reference_fft), endpoint=True)
        n_sample_range_tilde = np.array([n_sample_range - (k_sample_range)*(1j)])
        
        T_theory = Calculate_T_theory(n_air, n_sample_range_tilde, n_air, self.reference_freqs, 3.01E-3)
        
        delta_rho = np.log(np.abs(T_theory)) - np.log(np.abs(T_measured))
        delta_phi = np.angle(T_theory) - np.angle(T_measured)
        
        delta_nk = delta_rho**2 + delta_phi**2
        
        return T_theory, T_measured, delta_nk, n_sample_range, k_sample_range
    
    def plot_error_function_test(self, sample_measurement, sample_thickness):
    # Step 1: need the frequency domain of the sample and the reference
    # We will define the reference in the __init__ method during instantiation.
    # ASSUMING ABSORPTION IN AIR IS SMALL I.E. K_air ~ 0
        sample_fft = self.signals_dic[sample_measurement].calc_fft(logscale=False)[1]
        sample_freq = self.signals_dic[sample_measurement].calc_fft(logscale=False)[0]
    
        T_measured = sample_fft / self.reference_fft
    
        # Create ranges for n_sample and k_sample
        n_sample_range = np.linspace(0, 10, num=len(self.reference_fft), endpoint=True)
        k_sample_range = np.linspace(0, 10, num=len(self.reference_fft), endpoint=True)
        
        # Creating a 2D meshgrid for n_sample and k_sample
        n_sample_grid, k_sample_grid = np.meshgrid(n_sample_range, k_sample_range)
        
        # Compute the complex refractive index
        n_sample_range_tilde = n_sample_grid - k_sample_grid * (1j)
        
        # Calculate theoretical transmission T_theory
        T_theory = Calculate_T_theory(n_air, n_sample_range_tilde, n_air, self.reference_freqs, 3.01E-3)
        
        # Calculate error functions
        delta_rho = np.log(np.abs(T_theory)) - np.log(np.abs(T_measured))
        delta_phi = np.angle(T_theory) - np.angle(T_measured)
    
        delta_nk = delta_rho**2 + delta_phi**2
        
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotting the surface
        ax.plot_surface(n_sample_grid, k_sample_grid, delta_nk, cmap='viridis')
        
        ax.set_xlabel('n_sample_range')
        ax.set_ylabel('k_sample_range')
        ax.set_zlabel('delta_nk')
        ax.set_title('3D Plot of delta_nk over n_sample_range and k_sample_range')
        
        plt.show()

        return delta_nk, n_sample_range, k_sample_range
    
        
if __name__ == "__main__":
    pass
    #wood2b_1_path_focused =  r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\wood2b_sample\focused\241011_2\wood2b_foc_4190_1.txt"
    #reference_focused = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\wood2b_sample\focused\241011_2\reference.txt"
    #wood2b_2_path_focused = r"C:\Users\matth\OneDrive - University of Birmingham\Y4 Project\Lab_data\wood2b_sample\focused\241011_2\wood2b_foc_4190_2.txt"
    
    #wood_obj_1 = Signal(wood2b_1_path_focused)
    #wood_obj_2 = Signal(wood2b_2_path_focused)
    #reference_obj = Signal(reference_focused)
    
    #plt.plot(wood_obj_1.x_data, wood_obj_1.y_data, linestyle='-', marker='o', color='b', markersize=2, linewidth=1.5, label='Wood2b (i) Sample (Focused)')
    #plt.plot(wood_obj_2.x_data, wood_obj_2.y_data, linestyle='-', marker='o', color='m', markersize=2, linewidth=1.5, label='Wood2b (ii) Sample (Focused)')
    #plt.plot(reference_obj.x_data, reference_obj.y_data, linestyle='--', marker='^', color='r', markersize=2, linewidth=1.5, label='Reference (Focused)')
    
    ##plt.plot(wood_obj_1.calc_fft(logscale=True)[0], wood_obj_1.calc_fft(logscale=True)[1], linestyle='-', marker='o', color='b', markersize=2, linewidth=1.5, label='Log FFT Wood2b Sample (i) (Focused)')
    ##plt.plot(wood_obj_2.calc_fft(logscale=True)[0], wood_obj_2.calc_fft(logscale=True)[1], linestyle='-', marker='*', color='m', markersize=2, linewidth=1.5, label='Log FFT Wood2b Sample (ii) (Focused)')
    ##plt.plot(reference_obj.calc_fft(logscale=True)[0], reference_obj.calc_fft(logscale=True)[1], linestyle='--', marker='^', color='r', markersize=2, linewidth=1.5, label='Log FFT Reference (Focused)')
    
    #plt.xlabel("Time (ps)", fontsize=14)
    #plt.ylabel("Amplitude (a.u.)", fontsize=14)
    #plt.grid(True, linestyle='--', color='g', linewidth=0.8, alpha=0.4)
    #plt.xlim(0, np.max(wood_obj_1.x_data))
    #plt.legend(loc='upper right', fontsize=14)