# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:41:58 2022

@author: Loïc
title: detection of BBP in audios
"""

#%% Packages importations
print("\rImportation of packages...", end="\r")
import os
from datetime import datetime
import numpy as np
import pandas as pd
from librosa import load
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import find_peaks, correlate
from scipy.stats import linregress
print("Importation of packages done!")

#%% Parameters
# Paths
print("\rSetting up parameters...", end="\r")
audio_f = "/media/loic/DOLPHINFREE/Acoustique"  # Path to recordings 
csv_f = "./../CSV_data"                         # Path to data in csv
save_f = "./Results"                            # Path to results

# For functions
sr = 512000             # sample rate of the recordings
cut_low = 50000         # frequency cut in highpass
num_order = 1           # order of the highpass filter
sound_thresh = 1e-5     # arbitrary threshold used for noise detection 
click_size = 100        # mean size of a click (observations) for rolling average
max_length = 500        # maximum length to consider a sound as a click
nfft = 8192             # it is high to estimate freq precisely (we dont need good time location)
hop_length = nfft//2    # if needed, reduce hop_length to be more precise in time
flip = 60               # if there are two many BBPs selected, uses more strict criteria

# soft criteria, we will have false positives but at least we detect all BBPs
mini_space = click_size*5   # minimal space between two clicks (avoids echoes)
ICI_space = int(0.020*sr)   # maximal space between two clicks to be a BBP
nb_mini_clicks = 15         # minimal number of clicks to be a BBP
d_amp = 0.1                 # max difference of amplitude between clicks of a BBP
irregularity = 2            # limit for irregularity in a BBP (the closer to 1, the better)
mean_nrg = 0.1              # max mean energy for clicks in BBP
print("Parameters ready to use!")


#%% Data
print("\rImportation of csv data", end="\r")
from BBPUtils import get_csv, butter_pass_filter, TeagerKaiser_operator, OLDget_BBPs
data_20_21, audio_paths = get_csv(csv_f, slash="/")
print("Importation of csv data complete!")


#%%## Detection of clicks in environnement #####
print("\nMain execution: Looking for clicks in recordings.")
BBP_all = np.empty((0,3)) # [idx file, sample position, idx BBP in file]

for file in range(len(audio_paths)):
    print(f"\r\tExecution for file {file+1} on {len(audio_paths)}", end='\r')
    # load signal
    signal = load(os.path.join(audio_f, audio_paths[file][4:8], audio_paths[file]),
        sr=None)[0]
    signal_high = butter_pass_filter(signal, cut_low, sr, 
                                     num_order, mode='high')  
    tk_signal = np.abs(TeagerKaiser_operator(signal_high))

    # load or find clicks
    signal_peaks = find_peaks(tk_signal, prominence=sound_thresh, distance=mini_space, 
        width=[0,max_length])[0]
    # security
    if len(signal_peaks)==0:
        # create empty list to avoid errors
        signal_peaks = np.array([0])

    # Compute ICI
    peaks_selection = OLDget_BBPs(signal_peaks, ICI_space, nb_mini_clicks)
    # security
    if len(peaks_selection) > flip:
        peaks_selection = OLDget_BBPs(signal_peaks, int(ICI_space/2), nb_mini_clicks)

    # Apply selection criteria
    d_amp_selection = np.array([np.mean(np.abs(tk_signal[selection[1:]]-tk_signal[selection[:-1]]))<d_amp
         for selection in peaks_selection])
    regularity = np.array([np.mean(np.abs(selection[1:]-selection[:-1])[1:]/np.abs(selection[1:]-selection[:-1])[:-1])<irregularity
        for selection in peaks_selection])
    nrg = np.array([np.mean(tk_signal[selection])< mean_nrg for selection in peaks_selection])
    keeping = np.where(np.logical_and(d_amp_selection, regularity, nrg))[0]

    # just to be sure, if linreg is negative then it is prbly a false positive (a click and its echoes (diminishing in amplitude))
    linreg = np.array([linregress(tk_signal[selection],selection)[2] 
                       for selection in peaks_selection])[keeping]
    lengths = np.array([len(selection) for selection in peaks_selection])[keeping]
    keeping = keeping[np.logical_or((linreg > -0.5),(lengths > 2*nb_mini_clicks))]

    # save results
    for i in keeping:
        curr_BBP = np.append(peaks_selection[i][...,np.newaxis],
                             np.array([i]*len(peaks_selection[i]))[...,np.newaxis],
                             axis=1)
        curr_BBP = np.append(np.array([file]*len(peaks_selection[i]))[...,np.newaxis],
                             curr_BBP, axis=1)
        BBP_all = np.append(BBP_all, curr_BBP, axis=0)
        
np.save(os.path.join(save_f, datetime.now().strftime("%d-%m-%y_%Hh%M") + "_BBP_all.npy"), 
    BBP_all)