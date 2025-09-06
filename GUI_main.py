# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 22:55:07 2025

@author: matth
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spectra as sp
import numpy as np
import NMAlgorithm as ut # ut as a "utilities" file for now. Needs cleaning though
import matplotlib.pyplot as plt
import psutil
import subprocess
import os

def convert_to_none(value):
    return None if value == '' else value


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("THz-TDS Material Parameter Extraction Tool")
        self.geometry("1080x720")
        self._dflt_col_weight = 1
        self._dflt_border_width = 6
        self.max_row = 30

        # Make all rows (except last) stretchable for the plot to fill
        for i in range(self.max_row):
            weight = 1 if i < self.max_row - 1 else 0
            self.rowconfigure(i, weight=weight)
        for j in range(20):
            self.columnconfigure(j, weight=self._dflt_col_weight)

        self.create_logo()

        self.title_label = tk.Label(
            self, text="THz-TDS Material Parameter Extraction Tool", font=("Helvetica", 24, "bold"))
        self.title_label.grid(row=0, column=2, sticky="nsew")
        self.exit_button()
        self.protocol("WM_DELETE_WINDOW", self.confirm_exit)

        self.reference_file = None
        self.data_files = []

        # Reference controls
        self.add_reference_button = ttk.Button(
            self, text="Add Reference File", command=self.load_reference)
        self.add_reference_button.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.ref_listbox = tk.Listbox(self, height=1)
        self.ref_listbox.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Data files selection
        self.data_button = ttk.Button(
            self, text="Select Data Files", command=self.load_data_files)
        self.data_button.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.data_listbox = tk.Listbox(self, height=7, selectmode=tk.EXTENDED)
        self.data_listbox.grid(row=4, column=0, columnspan=2, rowspan=20, sticky="nsew")
        self.remove_files_button = ttk.Button(
            self, text="Remove selected file(s)", command=self.remove_files)
        self.remove_files_button.grid(row=27, column=0, columnspan=2, sticky="nsew")

        # Plot area: stretch from row 1 down to row 28 (leaving row 29 for bottom controls)
        self.plot_container = ttk.Frame(self)
        self.plot_container.grid(
            row=1, column=2, columnspan=15, rowspan=self.max_row+2, sticky="nsew")

        # Sub-frames inside plot_container
        self.plot_frame = ttk.Frame(self.plot_container)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.FFT_frame = ttk.Frame(self.plot_container)
        self.FFT_frame.grid(row=1, column=0, sticky="nsew")
        self.interp_frame = ttk.Frame(self.plot_container)
        self.interp_frame.grid(row=2, column=0, sticky="nsew")

        # Create canvases
        self.figure = Figure(figsize=(7, 1.9))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.FFT_figure = Figure(figsize=(7, 1.9))
        self.FFT_ax = self.FFT_figure.add_subplot(111)
        self.FFT_ax.set_xlabel("Frequency (THz)")
        self.FFT_ax.set_ylabel("Log Intensity")
        self.FFT_canvas = FigureCanvasTkAgg(self.FFT_figure, master=self.FFT_frame)
        self.FFT_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.interp_figure = Figure(figsize=(7, 1.9))
        self.interp_ax = self.interp_figure.add_subplot(111)
        self.interp_ax.set_xlabel("Frequency (THz)")
        self.interp_ax.set_ylabel("Unwrapped Phase (rad)")
        self.interp_canvas = FigureCanvasTkAgg(self.interp_figure, master=self.interp_frame)
        self.interp_canvas.get_tk_widget().pack(fill="both", expand=True)


        # Bottom buttons
        self.plot_button = ttk.Button(
            self, text="Plot Selected File", command=self.plot_selected)
        self.plot_button.grid(row=self.max_row-2, column=0, columnspan=2, sticky="nsew")
        self.clear_button = ttk.Button(
            self, text="Clear Plot Canvas", command=self.clear_canvas)
        self.clear_button.grid(row=self.max_row-1, column=0, columnspan=2, sticky="nsew")

        # Entries
        self.preProcessSec()
        self.rangeSummarySec()

    def load_reference(self):
        file = filedialog.askopenfilename(
            title="Select Reference File",
            filetypes=[("CSV and TXT files", "*.csv *.txt")]
        )
        if file:
            self.reference_file = file
            self.ref_listbox.delete(0, tk.END)
            self.ref_listbox.insert(tk.END, file)

    def load_data_files(self):
        files = filedialog.askopenfilenames(
            title="Select Data Files",
            filetypes=[("CSV and TXT files", "*.csv *.txt")]
        )
        if files:
            for f in files:
                if f not in self.data_files:
                    self.data_files.append(f)
                    self.data_listbox.insert(tk.END, f)

    def remove_files(self):
        for i in reversed(self.data_listbox.curselection()):
            self.data_listbox.delete(i)

    def create_logo(self):
        self.logo = ImageTk.PhotoImage(
            Image.open("logoV1.png").resize((250, 150)))
        logo_label = tk.Label(self, image=self.logo)
        logo_label.grid(row=0, column=0, rowspan=1, columnspan=2)

    def exit_button(self):
        exit_button = ttk.Button(self, text="QUIT", command=self.confirm_exit)
        exit_button.grid(row=self.max_row, column=0, columnspan=2, sticky="nsew")

    def confirm_exit(self):
        if messagebox.askyesno("Exit Confirmation", "Are you sure you want to quit?"):
            self.destroy()

    def plot_selected(self):
        sel = self.data_listbox.curselection()
        if sel:
            self._plot_file(self.data_files[sel[0]])
        elif self.ref_listbox.curselection():
            self._plot_file(self.reference_file)
        else:
            messagebox.showwarning("No file selected", "Please select a data file to plot.")

    def clear_canvas(self):
        self.ax.clear()
        self.FFT_ax.clear()
        self.interp_ax.clear()
        self.interp_ax.set_xlabel("Frequency (THz)")
        self.interp_ax.set_ylabel("Unwrapped Phase (rad)")
    
        self.FFT_ax.set_xlabel("Frequency (THz)")
        self.FFT_ax.set_ylabel("Log Intensity")
        self.canvas.draw()
        self.FFT_canvas.draw()
        self.interp_canvas.draw()

    def _plot_file(self, file):
        try:
            df = pd.read_csv(file, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(file)
        if df.shape[1] < 2:
            messagebox.showerror("Invalid data", "File must have at least two columns.")
            return
        x, y = df.iloc[:,0], df.iloc[:,1]
        self.ax.plot(x,y,label=file.split('/')[-1])
        self.ax.set_xlabel(df.columns[0]); self.ax.set_ylabel(df.columns[1]); self.ax.legend()
        freqs, fft_vals = sp.Signal(file).calc_complex_fft()
        self.FFT_ax.plot(freqs/1e12, 10*np.log(np.abs(fft_vals)))
        self.FFT_ax.set_xlim(0, np.max(freqs/1e12))
        
        
        self.figure.tight_layout(); self.FFT_figure.tight_layout()
        self.canvas.draw(); self.FFT_canvas.draw()

        
        

    def preProcessSec(self):
        labels = ["Average Thickness (um):","Surface Roughness (um):",
                  "Minimum Thickness (um):","Maximum Thickness (um):",
                  "Number of Thicknesses to Sweep:",
                  "Number of Observed Fabry Perot Reflections:"]
        rows = [1,3,5,7,9,11]
        
        self.average_thickness_entry = tk.Entry(self)
        self.surface_roughess_entry = tk.Entry(self)
        self.minimum_thickness_entry = tk.Entry(self)
        self.maximum_thickness_entry= tk.Entry(self)
        self.number_thicknesses_entry = tk.Entry(self)
        self.number_fabry_entry = tk.Entry(self)
        
        self.average_thickness_entry.grid(row=2, column=4, sticky='nsew')
        self.surface_roughess_entry.grid(row=4, column=4, sticky='nsew')
        self.minimum_thickness_entry.grid(row=6, column=4, sticky='nsew')
        self.maximum_thickness_entry.grid(row=8, column=4, sticky='nsew')
        self.number_thicknesses_entry.grid(row=10, column=4, sticky='nsew')
        self.number_fabry_entry.grid(row=12, column=4, sticky='nsew')
        
        
        
        for t,r in zip(labels,rows):
            tk.Label(self,text=t).grid(row=r,column=4,sticky="nsew")
        
    def rangeSummarySec(self):
        params=["Minimum Frequency (THz):","Maximum Frequency (THz):",
                "Interpolation Range [min THz, max THz]:"]
        starts=[13,15,17]
        
        self.min_freq_entry = tk.Entry(self)
        self.min_freq_entry.grid(row=14, column=4, sticky='nsew')
        
        self.max_freq_entry = tk.Entry(self)
        self.max_freq_entry.grid(row=16, column=4, sticky='nsew')
        
        self.interp_range_entry = tk.Entry(self)
        self.interp_range_entry.grid(row=18, column=4, sticky='nsew')
        
        
        for p,s in zip(params,starts):
            tk.Label(self,text=p).grid(row=s,column=4,sticky="nsew")
            
        summary_button = ttk.Button(self, text="Generate summary", command=lambda: self.generate_summary(self.interp_range_entry.get(), float(self.min_freq_entry.get()), float(self.max_freq_entry.get()), float(self.average_thickness_entry.get()), float(self.minimum_thickness_entry.get()), float(self.maximum_thickness_entry.get()), float(self.number_thicknesses_entry.get()), float(self.surface_roughess_entry.get()), float(self.number_fabry_entry.get()), self.data_listbox.get(self.data_listbox.curselection()[0]), self.ref_listbox.get(0)))
        summary_button.grid(row=20, column=4, sticky="nsew")
        
        analysis_button = tk.Button(self, text="PROCEED TO EXTRACTION\n ON SELECTED FILE", bg="pale green",  command=lambda: self.extraction_tab(self.interp_range_entry.get(), self.min_freq_entry.get(), self.max_freq_entry.get(), self.average_thickness_entry.get(), self.minimum_thickness_entry.get(), self.maximum_thickness_entry.get(), self.number_thicknesses_entry.get(), self.surface_roughess_entry.get(), self.number_fabry_entry.get(), self.data_listbox.get(self.data_listbox.curselection()[0]), self.ref_listbox.get(0)))
        analysis_button.grid(row=21, column=4, sticky="nsew")
        
        
    def generate_summary(self, interp_tuple, freq_min, freq_max,
                         avg_thickness, minimum_thickness, maximum_thickness,
                         number_thicknesses, surface_roughess, number_fp, file, reference_file):
        
        
        SampleSig = sp.Signal(rf'{file}', name=f'{file}')
        ReferenceSig = sp.Signal(rf'{reference_file}', name=f'{reference_file}')
                
        print(freq_min, freq_max, number_fp, float(interp_tuple.rsplit(',')[0]), float(interp_tuple.rsplit(',')[1]))
        
        Pair = ut.SampleRefPair(SampleSig, ReferenceSig, avg_thickness, unwrap_range_selec=True, plot=True, preset=[freq_min, freq_max, number_fp,float(interp_tuple.rsplit(',')[0]), float(interp_tuple.rsplit(',')[1])])
        
        
        pass
    
    
    def extraction_tab(self, interp_tuple, freq_min, freq_max,
                   avg_thickness, minimum_thickness, maximum_thickness,
                   number_thicknesses, surface_roughess, number_fp,
                   file, reference_file):

        # Convert inputs to None if empty
        interp_tuple       = convert_to_none(interp_tuple)
        freq_min           = convert_to_none(freq_min)
        freq_max           = convert_to_none(freq_max)
        avg_thickness      = convert_to_none(avg_thickness)
        minimum_thickness  = convert_to_none(minimum_thickness)
        maximum_thickness  = convert_to_none(maximum_thickness)
        number_thicknesses = convert_to_none(number_thicknesses)
        surface_roughess   = convert_to_none(surface_roughess)
        number_fp          = convert_to_none(number_fp)
        file               = convert_to_none(file)
        reference_file     = convert_to_none(reference_file)

        # If any required field is None, warning
        if None in [interp_tuple, freq_min, freq_max, avg_thickness,
                minimum_thickness, maximum_thickness, number_thicknesses,
                surface_roughess, number_fp, file, reference_file]:
            messagebox.showinfo("Missing Information", "Ensure all Information is Provided in the Main Page. Fill the entries on the right, and select/highlight a data file from the list. Then click proceed.")
            return  # Stop execution if validation fails

        # Convert to float 
        try:
            freq_min           = float(freq_min)
            freq_max           = float(freq_max)
            avg_thickness      = float(avg_thickness)
            minimum_thickness  = float(minimum_thickness)
            maximum_thickness  = float(maximum_thickness)
            number_thicknesses = float(number_thicknesses)
            surface_roughess   = float(surface_roughess)
            number_fp          = float(number_fp)
        except ValueError as e:
            messagebox.showinfo("Invalid Input", f"Please enter valid numbers. Error: {e}")
            return

    
        ExtractionWindow(self, interp_tuple, freq_min, freq_max,
                             avg_thickness, minimum_thickness, maximum_thickness,
                             number_thicknesses, surface_roughess, number_fp, file, reference_file)
        
    def run(self): self.mainloop()

class ExtractionWindow(tk.Toplevel):
    def __init__(self, master, interp_tuple, freq_min, freq_max,
                         avg_thickness, minimum_thickness, maximum_thickness,
                         number_thicknesses, surface_roughess, number_fp, file, reference_file):
        
        super().__init__(master)
        
        self.interp_tuple = interp_tuple
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.avg_thickness = avg_thickness
        self.minimum_thickness = minimum_thickness
        self.maximum_thickness = maximum_thickness
        self.number_thicknesses = number_thicknesses
        self.surface_roughess = surface_roughess
        self.number_fp = number_fp
        self.file = file
        self.reference_file = reference_file
        
        print(self.file)
        
        self.title("Extraction Pge")
        self.geometry("1080x720")
        self._dflt_col_weight = 1
        self._dflt_border_width = 6 
        self.max_row = 30

        for i in range(self.max_row):
            weight = 1 if i < self.max_row - 1 else 0
            self.rowconfigure(i, weight=weight)
        for j in range(20):
            self.columnconfigure(j, weight=self._dflt_col_weight)

        self.title_label = tk.Label(
            self, text="Extraction Window", font=("Helvetica", 24, "bold"))
        self.title_label.grid(row=0, column=0, sticky="nsew")
        
        # Determine core count
        self.physical_cpu = psutil.cpu_count(logical=False) 
        
        tk.Label(self, text=f"Physical Cores Available: {self.physical_cpu}", font=("Helvetica", 12)).grid(row=1, column=0)
        
        # Slider
        self.selected_cores = tk.IntVar(value=1)
        
        
        tk.Scale(
            self, from_=1, to=self.physical_cpu,
            orient=tk.HORIZONTAL,
            label="Select # Cores",
            width=25,
            variable=self.selected_cores
        ).grid(row=2, column=0, sticky='nsew')
        
        self.summary_sec()

        # Big chunky START button
        self.start_button = tk.Button(
            self, 
            text="🔎 START EXTRACTION 🔎", 
            font=("Helvetica", 18, "bold"), 
            bg="lime green", 
            fg="black", 
            height=3, width=25,
            command=self.on_start
        )
        self.start_button.grid(row=10, column=0, columnspan=5, pady=30, sticky="nsew")


    def summary_sec(self):
        """ Show a summary of extraction settings before user starts """
        summary_text = (
            f"REFERENCE FILE:\n{self.reference_file}\n\n"
            f"DATA FILE:\n{self.file}\n\n"
            f"THICKNESS SWEEP:\n"
            f"Start: {self.minimum_thickness} μm\n"
            f"Stop:  {self.maximum_thickness} μm\n"
            f"Step:  {(self.maximum_thickness - self.minimum_thickness)/self.number_thicknesses:.3f} μm\n"
            f"Total: {int(self.number_thicknesses)} thickness values\n\n"
            f"FREQUENCY RANGE:\n{self.freq_min} – {self.freq_max} THz\n"
            f"Interpolation Range: {self.interp_tuple[0]} – {self.interp_tuple[1]} THz"
        )

        self.summary_label = tk.Label(
            self, text=summary_text, justify="left", font=("Helvetica", 12), anchor="w"
        )
        self.summary_label.grid(row=3, column=0, columnspan=5, sticky="nsew", padx=10, pady=10)
    
    def on_start(self):
        
        with open("run_extraction_tmp.py", "w") as f:
            f.write(f"""
import NMAlgorithm as ut
import spectra as sp
                    
# Reconstruct signals
SampleSig = sp.Signal(r"{self.file}", name="Sample")
ReferenceSig = sp.Signal(r"{self.reference_file}", name="Reference")
                    
# Example use
Pair = ut.SampleRefPair(
    SampleSig,
    ReferenceSig,
    {self.avg_thickness},
    unwrap_range_selec=True,
    plot=True,
    preset=[{self.freq_min}, {self.freq_max}, {self.number_fp}, 
                                {self.interp_tuple.split(',')[0]}, {self.interp_tuple.split(',')[1]}])

Extract_NelderMead_MPI(Pair, self.maximum_thickness-self.avg_thickness, self.avg_thickness-self.minimum_thickness, 1, f'{os.path.dirname(self.file)}', f'{self.file}')

""")

        
        num_cores = self.selected_cores.get()
        cmd = (
        f'conda activate terahertz && '
        f'mpirun -np {num_cores} python run_extraction_tmp.py'
    )
        try:
            # Launch a new cmd window, activate env, run MPI, keep window open
            subprocess.Popen(
                ["cmd", "/k", cmd],  
                shell=True
                )
            messagebox.showinfo(
                "Extraction Started",
                f"Extraction launched in a new cmd window using {num_cores} cores "
                "inside the 'terahertz' conda environment."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Could not start extraction:\n{e}")

if __name__ == "__main__": App().run()
