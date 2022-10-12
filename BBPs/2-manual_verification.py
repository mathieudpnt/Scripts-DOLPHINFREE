# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:41:58 2022

@author: Loïc
title: manual review of detected BBPs
"""

#%% Packages importations
print("\rImportation of packages...", end="\r")
import os
import json
from datetime import datetime
import numpy as np
from librosa import load, amplitude_to_db, stft, pcen
from matplotlib.widgets import Button
from matplotlib import use
use('TkAgg') 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
print("Importation of packages done!")

#%% Parameters
print("\rSetting up parameters...", end="\r")
# Paths
audio_f = "./../Audio_data"  # Path to recordings 
csv_f = "./../CSV_data"      # Path to data in csv
save_f = "./Results"         # Path to results

# For functions
sr = 512000             # sample rate of the recordings
cut_low = 50000         # frequency cut in highpass
num_order = 1           # order of the highpass filter
sound_thresh = 1e-5     # arbitrary threshold used for noise detection (lowered here)
click_size = 100        # mean size of a click (observations) for rolling average
max_length = 500        # maximum length to consider a sound as a click

mini_space = click_size*5

nfft = 512
overlap = 0.9
hop_length = int(nfft*(1-overlap))
window = int(sr/2)
print("Parameters ready to use!")


#%% Functions (button's actions)
classes = ["Buzz", "Burst-pulse", "Error", "Undefined"]
def buzz_replace(*args, **kwargs):
    global BBP_manual, main_col
    BBP_manual[main_col] = 0

def burst_replace(*args, **kwargs):
    global BBP_manual, main_col
    BBP_manual[main_col] = 1

def error_replace(*args, **kwargs):
	global BBP_manual, main_col
	BBP_manual[main_col] = 2

def display_selection(*args, **kwargs):
    global BBP_manual, main_col, text
    text.set_text(f'Current selection: {classes[BBP_manual[main_col]]}')

def exit(*args, **kwargs):
    global fig
    plt.close(fig)

def break_all(event,*args, **kwargs):
	global keep_go
	if event.key == "ctrl+q":
		keep_go = False
		plt.close(fig)


#%% Import Data and functions
print("\rImportation of csv data", end="\r")
from BBPUtils import get_csv, butter_pass_filter, TeagerKaiser_operator
data_20_21, audio_paths = get_csv(csv_f, slash="/")
BBP_all = np.load(os.path.join(save_f, "07-06-22_19h49_BBP_all.npy"))
total_BBP_number = [len(np.unique(BBP_all[np.where(BBP_all[:,0]==file)[0], -1])) 
	for file in range(len(audio_paths))]
print("Importation of csv data complete!")


#%% Main execution
print("Beginning display of BBPs...")
print(f"\tYou have {np.sum(total_BBP_number)} BBPs to annotate... Gl & Hf")
dict_annot = {}
keep_go=True

for file in range(len(audio_paths)):
	print(f"\r\tAnnotation of file {file+1} in {len(audio_paths)}", end="\r")
	use = np.where(BBP_all[:,0] == file)[0]

	idx_in_data = np.where(data_20_21['Fichier Audio'] == \
       audio_paths[file].replace('\\','/'))
	signals_data = data_20_21.iloc[idx_in_data[0][0],:]
	cat_acoustic = signals_data.iloc[3:10].astype(int).idxmax(axis=0)
	cat_behavior = signals_data.iloc[13:16].astype(int).idxmax(axis=0)

	if (len(use) > 0) and keep_go: 
		signal = load(os.path.join(audio_f, audio_paths[file][4:8], audio_paths[file]), 
					  sr=None)[0]
		signal_high = butter_pass_filter(signal, cut_low, sr, 
		                                 num_order, mode='high')  
		tk_signal = np.abs(TeagerKaiser_operator(signal_high))
		signal_peaks = find_peaks(tk_signal, prominence=sound_thresh, distance=mini_space, 
		    width=[0,max_length])[0]

		Magnitude_audio = stft(signal_high, n_fft=nfft, hop_length=hop_length)
		#spectrum = amplitude_to_db(np.abs(Magnitude_audio))
		spectrum = pcen(np.abs(Magnitude_audio) * (2**31))

		# show ICI of each BBP
		n_BBP = np.unique(BBP_all[use,-1])
		BBP_manual = [-1]*len(n_BBP)
		for main_col, i in enumerate(n_BBP):
			if keep_go:
				BBP = BBP_all[use][BBP_all[use,-1]==i,1]
				lower = max(int(BBP.min())-window, 0)
				upper = min(int(BBP.max())+window, len(signal))

				peaks = np.copy(signal_peaks)
				peaks = peaks[peaks > lower]
				peaks = peaks[peaks < upper] - lower

				ICI = peaks[1:]-peaks[:-1]
				ICI = np.append(ICI, ICI[-1])
		 
				fig, axs = plt.subplots(5,1, figsize=(32,18), gridspec_kw=
				{'height_ratios': [1,6,3,3,.5]})
				
				fig.suptitle(f"BBP {int(main_col)+1}/{len(n_BBP)} in file {file}: {audio_paths[file]}.\nSequence: {cat_acoustic}, Behaviour: {cat_behavior}\nPress 'Ctrl+q' to exit")

				axs[0].plot(tk_signal, color="black")
				axs[0].set_title("Full recording (for context)")
				for col, j in enumerate(n_BBP):
					axs[0].plot(BBP_all[use][BBP_all[use,-1]==j,1], 
						tk_signal[BBP_all[use][BBP_all[use,-1]==j,1].astype(int)],
						'.', color="C"+str(col))
				axs[0].set_ylabel("Amplitude")
				axs[0].axis(xmin=lower, xmax=upper)
				axs[0].tick_params('x', labelbottom=False, bottom=False)

				axs[1].imshow(spectrum[::-1][:,int(lower/hop_length):int(upper/hop_length)], 
				    aspect='auto', interpolation='none', cmap='jet', 
				    extent=(0,(upper-lower)/sr,0,int(sr/2)))
				axs[1].set_title("Spectrogram (dB scale)")
				axs[1].set_ylabel("Frequencies")
				axs[1].tick_params('x', labelbottom=False, bottom=False)
				
				axs[2].plot(np.linspace(0, (upper-lower)/sr, num=len(tk_signal[lower:upper])), 
					tk_signal[lower:upper], color="black")
				axs[2].set_title("Signal (filtered > 50kHz)")
				axs[2].set_ylabel("Amplitude")
				axs[2].plot(peaks/sr, tk_signal[peaks], '.', color="black")
				axs[2].plot((BBP-lower)/sr, tk_signal[BBP.astype(int)],
							'.', color="C"+str(main_col))
				axs[2].sharex(axs[1])
				axs[2].axis(ymin=-1e-4, ymax=1e-3)
				axs[2].tick_params('x', labelbottom=False, bottom=False)

				axs[3].plot(peaks/sr, np.log(ICI), '.', color="red")
				axs[3].set_title("log(ICI) = F(time)")
				axs[3].set_ylabel("log(ICI)")
				axs[3].sharex(axs[1])

				text = axs[4].text(0.5, 0.5, 
					f'Current selection: {classes[BBP_manual[int(main_col)]]}', 
					horizontalalignment='center', verticalalignment='center', 
					transform=axs[4].transAxes, fontsize=18)
				axs[4].axis('off')

				pos_buzz = plt.axes([0.275, 0, 0.15, 0.05]) #left bottom width height
				buzz = Button(pos_buzz, 'Buzz', color="yellow", hovercolor='green')
				pos_burst = plt.axes([0.425, 0, 0.15, 0.05]) 
				burst = Button(pos_burst, 'Burst-pulse', color="yellow", hovercolor='green')
				pos_error = plt.axes([0.575, 0, 0.15, 0.05]) #left bottom width height
				error = Button(pos_error, 'Error', color="yellow", hovercolor='green')
				pos_next = plt.axes([0.85, 0, 0.15, 0.05])
				nextB = Button(pos_next, 'Next', color="red", hovercolor='orange')

				buzz.on_clicked(buzz_replace)
				burst.on_clicked(burst_replace)
				error.on_clicked(error_replace)
				buzz.on_clicked(display_selection)
				burst.on_clicked(display_selection)
				error.on_clicked(display_selection)

				nextB.on_clicked(exit)
				fig.canvas.mpl_connect('key_press_event', break_all)

				plt.show(block=True)

			plt.close('all')
		del signal, signal_high, signal_peaks, tk_signal,\
		Magnitude_audio, spectrum

		dict_annot[audio_paths[file]] = BBP_manual
		# Temp save (in case of a crash)
		with open(os.path.join(save_f, "_temp_annot.json"),'w') as f:
			json.dump(dict_annot, f, indent=4, sort_keys=True)
		f.close()



# Save it
with open(os.path.join(save_f, datetime.now().strftime("%d-%m-%y_%Hh%M") + "_annot.json"),'w') as f:
	json.dump(dict_annot, f, indent=4, sort_keys=True)
f.close()

# remove temp file
os.remove(os.path.join(save_f, "_temp_annot.json"))

print("\nTHE END")
