# -*- coding: utf-8 -*-
"""
Created on FRI Mar 04 12:00:00 2022

@author: Loïc
title: Analysis of trajectories
"""# -*- coding: utf-8 -*-

#%% Packages
print("\n")
print("\rImportation of packages...", end="\r")
import os
import numpy as np
import pandas as pd
from librosa import load, stft, amplitude_to_db
from scipy.signal import resample, decimate
import seaborn as sns
import matplotlib.pyplot as plt
import json
print("Importation of packages complete!")


#%% Parameters
print("\rSetting up parameters...", end="\r")
csv_f = "./../CSV_data"         # Path to data in csv
save_f = "./Trajectories"		# Path were trajectories are stored
results_f = "./Evaluation"		# Path were results are stored

keep_if = 1 	# minimal length of a whistle to be kept
print("Parameters ready to use!")


#%% Importation of data and functions
print("\rImportation of csv data...", end="\r")
from WhistleUtils import get_csv, get_category
data_20_21, audio_paths = get_csv(csv_f, slash="/")
print("Importation of csv data complete!\n")


#%% Get counts of whistles
print("Fetching counts of whistles")
count = np.zeros(len(audio_paths), dtype=int)
for file in range(len(audio_paths)):
	print(f"\r\tFetching trajectories in file {file+1} on {len(audio_paths)}", end='\r')

	f = open(os.path.join(save_f, audio_paths[file].split('/')[-1][:-4] + ".json"), "r")
	whistles = json.load(f)
	f.close()

	# count[file] = len(whistles.keys())
	for key in list(whistles.keys()):
		len_for_key = max(whistles[key][1])-min(whistles[key][1])
		#len_for_key = len(np.unique(whistles[key][1]))
		if (len_for_key > keep_if) :
			count[file] += len_for_key
print("\nCounts of whistles ready !")


#%% Association of each recording to its categories
files_in_folder = np.array([path[-27:] for path in audio_paths])
if input('\nSave table with categories ? [Y/n] ') == 'Y':
	count_oc = np.copy(count)
	data_to_save = pd.DataFrame(count_oc.astype(int), columns=['total_whistles_duration'])
	for cat in ['acoustic', 'fishing_net', 'behavior', 'beacon', 'date', 'number', 'net']:
		data_to_save[cat] = get_category(files_in_folder, audio_paths, data_20_21, cat)

	data_to_save['audio_names'] = [file[:-4] for file in files_in_folder]
	data_to_save.to_csv(os.path.join(results_f, "whistles_durations.csv"),
		index=False, index_label=False)