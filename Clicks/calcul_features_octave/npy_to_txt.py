import os
import csv
import pandas as pd
import numpy as np

path = "./Results/peaks_02052022_txt/"
files = os.listdir(data_f + "Results/peaks_new_method_02052022")

#%% npy to txt
for file in files:
	data = np.load(data_f + "Results/peaks_02052022/" + file)
	if len(data) > 1:
		np.savetxt(path + file[:-4] + ".txt", data.astype(int), fmt='%i')


#%% list of files in txt
csv_f = "/../CSV_data" 
txt_files = os.listdir(data_f + "Results/peaks_02052022_txt")

check_in = os.listdir(path)
os.chdir(data_f)
from FuncUtils import get_csv
data_20_21, audio_paths = get_csv(data_f + csv_f, slash="/")


list_txt = []
for path in audio_paths:
	if path.split('/')[-1][:-4]+"_peaks.txt" in txt_files:
		list_txt.append(path)


np.savetxt("./calcul_features_octave/liste_fichiers_interessants.txt",
			np.array(list_txt[::-1]), fmt='%s')

