# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 12:00:00 2022

@author: Loïc
title: DECAV from Abeille, adapted and improved from a matlab script
"""# -*- coding: utf-8 -*-

#%% Packages
print("\rImportation of packages...", end="\r")
import os
import numpy as np
from librosa import load, stft, pcen, amplitude_to_db
from scipy.signal import resample
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
print("Importation of packages complete!")


#%% Parameters
# Paths
print("\rSetting up parameters...", end="\r")
audio_f = "./../Audio_data"  # Path to recordings  
csv_f = "./../CSV_data"      # Path to data in csv
save_f = "./Trajectories"	 # Path were results are stored

# Audio parameters
start = 0 		# start time for signal (in sec)
stop = 60 		# stop time for signal (in sec)
new_sr = 48000			# resampling rate
nfft = 256				# size of FFT window
noverlap = 0 			# overlapping windows
fmin = 4675				# frequency high pass selection (in Hz)

f_min = round(fmin/(new_sr/nfft))
hop_length = nfft//(noverlap+1) 

# parameters for functions 
# (dist_f = 1, dist_t = 1 --> strict criteria)
# (dist_f = 2, dist_t = 5 --> loose criteria)
nrg_rap = 6 		# energy must be higher than nrg_rap times the geom mean
dist_f = 1			# frequency distance tolerance for trace
dist_t = 1   		# time distance tolerance for trace
taille_min = 53.3 	# minimum length to keep a trace (in milliseconds)
min_acce = 0.5  	# acceleration threshold for mean acceleration
max_acce = 3    	# acceleration threshold for max acceleration
min_r_coef = 0.5    # minimal correlation coefficient to consider 2 traces as harmonics
sparsity = 0.5      # minimal data must have more than 'sparsity' true values

taille_traj_min = round(taille_min*(new_sr/hop_length)/1000)
print("Parameters ready to use!")


#%% Importation of data and functions
print("\rImportation of csv data...", end="\r")
from WhistleUtils import get_csv, plot_spectrums, get_local_maxima, get_trajectories, \
	select_trajectories, sparsity_ridoff, harmonize_trajectories
data_20_21, audio_paths = get_csv(csv_f, slash="/")
print("Importation of csv data complete!\n")


#%% Main execution
print("Spectral detector of whistles")
for file in range(len(audio_paths)):
	# import audio recording
	signal, fe = load(os.path.join(audio_f, audio_paths[file][4:8], audio_paths[file]), 
		sr=None)
	signal = signal[int(start*fe):int(stop*fe)]
	# resample
	signal_dec = resample(signal, int(((stop-start)*new_sr)))

	# extract spectral informations
	Magnitude_audio = stft(signal_dec, n_fft=nfft, hop_length=hop_length)
	spectre = np.copy(np.abs(Magnitude_audio[f_min:,:]))
	# PCEN could replace spectrogram in very noisy recordings
	#spectre_pcen = pcen(np.abs(Magnitude_audio) * (2**31), bias=10)[f_min:,:]

	# Selection algorithm
	max_loc_per_bin_check1 = get_local_maxima(spectre, spectre, nrg_rap)[1]
	trajectories = get_trajectories(max_loc_per_bin_check1, dist_f=dist_f, dist_t=dist_t)
	final_traj = select_trajectories(trajectories, taille_traj_min, min_acce, max_acce, verbose=0)
	corrected_traj = sparsity_ridoff(final_traj, error_thresh=sparsity)
	harmonized_traj = harmonize_trajectories(corrected_traj, min_r=min_r_coef, 
		min_common=taille_traj_min*2, delete=True)

	# Saving results
	values = np.unique(harmonized_traj)[1:]
	dict_traj = {}
	startstop = np.zeros((len(values), 2))
	for key, value in enumerate(values):
		dict_traj[key+1] = [np.where(harmonized_traj == value)[0].tolist(),
							np.where(harmonized_traj == value)[1].tolist()] 
		startstop[key] = np.array([min(dict_traj[key+1][1]), max(dict_traj[key+1][1])])

	f = open(os.path.join(save_f, audio_paths[file].split('/')[-1][:-4] + ".json"), "w")
	json.dump(dict_traj, f, indent=4)
	f.close()

	print(f"\r\tFile {file+1} on {len(audio_paths)}: found {len(values)} whistles", end='\r')
print("\nDetection of whistles finished!")