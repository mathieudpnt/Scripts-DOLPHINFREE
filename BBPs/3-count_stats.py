# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:05:33 2022

@author: loic
title: Oops i did the clicks again
"""

#%% Packages
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

#%% Parameters 
print("\rSetting up parameters...", end="\r")
csv_f = "./../CSV_data"                         # Path to data in csv
save_f = "./Results"                            # Path to results
file_BBP_all = "07-06-22_19h49_BBP_all.npy"
file_annotations = "08-06-22_15h11_annot.json"

sr = 512000
print("Parameters ready to use!")

#%% Importation of ata and function
print("\rImportation of csv data", end="\r")
from FuncUtils import get_csv, get_category
data_20_21, audio_paths = get_csv(csv_f, slash="/")
print("Importation of csv data complete!")


#%% Main
files = np.arange(len(audio_paths))
files_in_folder = np.array([audio_path[-27:-4] for audio_path in audio_paths])
BBP_all = np.load(os.path.join(save_f, file_BBP_all))

f = open(os.path.join(save_f, file_annotations),)
annotations = json.load(f)
f.close()

BBP_per_file = np.zeros(len(files))
Buzz_per_file = np.zeros(len(files))
Burst_per_file = np.zeros(len(files))
for file in files:
    if audio_paths[file] in list(annotations.keys()):
        Buzz_per_file[file] = len(np.where(np.array(annotations[audio_paths[file]]) == 0)[0])
        Burst_per_file[file] = len(np.where(np.array(annotations[audio_paths[file]]) == 1)[0])
    BBP_per_file[file] = Buzz_per_file[file] + Burst_per_file[file]

print(f"We found {int(np.sum(BBP_per_file))} BBP in the audios:\
 {int(np.sum(Buzz_per_file))} Buzzes & {int(np.sum(Burst_per_file))} Burst-pulses\n")

# get a handful of interesting features
ICI_Buzz = np.array([])
numbers_Buzz = np.array([])
duration_Buzz = np.array([])
ICI_Burst = np.array([])
numbers_Burst = np.array([])
duration_Burst = np.array([])
count=0
for file in files:
    if (audio_paths[file] in list(annotations.keys())) and (data_20_21['T'][file]==0):
        count+= 1
        use = np.where(BBP_all[:,0] == file)[0]
        detections = np.unique(BBP_all[use,-1])
        for n_detection, detection in enumerate(detections):
            BBP = BBP_all[use][np.where(BBP_all[use,-1] == detection)[0]]
            BBP_label = annotations[audio_paths[file]][n_detection]
            if BBP_label == 0:
                ICI_Buzz = np.append(ICI_Buzz, np.mean(BBP[1:,1]-BBP[:-1,1]))
                numbers_Buzz = np.append(numbers_Buzz, BBP.shape[0])
                duration_Buzz = np.append(duration_Buzz, BBP[-1,1]-BBP[0,1])
            elif BBP_label == 1:
                ICI_Burst = np.append(ICI_Burst, np.mean(BBP[1:,1]-BBP[:-1,1]))
                numbers_Burst =  np.append(numbers_Burst, BBP.shape[0])
                duration_Burst = np.append(duration_Burst, BBP[-1,1]-BBP[0,1])

print(f"Excluding control sequences, we found {len(ICI_Buzz)+len(ICI_Burst)} BBP in the audios:\
 {len(ICI_Buzz)} Buzzes & {len(ICI_Burst)} Burst-pulses")
print(f"\nMean durations are: \n\t{round(np.mean(duration_Buzz)/sr,3)} sec for Buzzes\
 \n\t{round(np.mean(duration_Burst)/sr,3)} sec for Burst-pulses")
print(f"\nMean ICI are: \n\t{round(np.mean(ICI_Buzz)/sr,4)} sec for Buzzes\
 \n\t{round(np.mean(ICI_Burst)/sr,4)} sec for Burst-pulses")
print(f"\nMean number of clicks are: \n\t{round(np.mean(numbers_Buzz),1)} for Buzzes\
 \n\t{round(np.mean(numbers_Burst),1)} for Burst-pulses")


# save categories associated to each BBP
if input('\nSave table with categories [Y/n] ? ') == 'Y':
    data_to_save = pd.DataFrame(BBP_per_file.astype(int), columns=['number_of_BBP'])
    for cat in ['acoustic', 'fishing_net', 'behavior', 'beacon', 'date', 'number', 'net']:
        data_to_save[cat] = get_category(files_in_folder, audio_paths, data_20_21, cat)

    data_to_save['audio_names'] = [file[:-10] for file in files_in_folder]
    data_to_save['Buzz'] = Buzz_per_file.astype(int)
    data_to_save['Burst-pulse'] = Burst_per_file.astype(int)
    data_to_save.to_csv(os.path.join(save_f, datetime.now().strftime("%d-%m-%y_%Hh%M")+"_number_of_BBP.csv"), 
        index=False, index_label=False)

