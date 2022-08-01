# Description of CSV data

|Colum Name|Description|
|:---| :---|
|"Fichier Audio"|Name of the recording ("ddmmyyyy/hydrophoneID_yyyymmdd_hhmmss.wav")|
|"Date"				|	Date ("dd/mm/yyyy")  |
|"Heure"			|	Time at the beginning of the record  |
|"T"				|	"Test" or "Control" sequences |
|"AV"				|	Before beacon's activation (translates to "BEF")|
|"AV+D"				|	At the beginning of beacon's emission sequence (translates to "BEF+DUR")  |
|"D"				|	During beacon's emission sequence (translates to "DUR") |
|"D+AP"				|	At the end of beacon's emission (translates to "DUR+AFT")  |
|"AP"				|	After beacon's emission (translates to "AFT") |
|"AP+AV"			|	Between emissions (translates to "AFT+BEF") |
|"F"				|	Presence of a fishing net  |
|"SSF"				|	Absence of a fishing net |
|"NSP"				|	Doubt on the presence of a fishing net  |
|"CHAS"				|	"Foraging"  |
|"SOCI"				|	"Socialising" |
|"DEPL"				|	"Travelling" |
|"SONAR"			|	Presence of a SONAR during sequences (nearby boat or experiment boat)  |
|"SIGNAL"			|	Type of signal used during emission sequence|
|"C-GR"				|	Count of dolphin observed in group  |
|"FILET"			|	Type of fishing net  |

*In cells, 0 = absence or false, 1 = presence of true*  
*Lines at the end enabled us to verify that there was no missing data*  

## Fishing net types
|Name (in french)|Description|
|:---| :---|
"tremail"			|	monkfish gillnet, nylon, mesh 220 mm  
"grand_filet"		|	hake and pollack gillnet, stretched mesh 136 mm, tread 0.6mm, with a weighted 12 mm-diameter bottom rope  
"chalut_vert"		|	trawl net, mesh 12 mm, thread 210/24/413, reinforced nylon  
"chalut_blanc"		|	trawl net, mesh 40 mm, thread 4mm, polyethylene PE  

*See publication for more details*

## Signal types

See table S1 in supplementary material