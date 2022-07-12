% Computation of the parameters of the clicks from the data set

clear all; clf;
close all;

tic();


%%% GLOBAL PARAMETERS

% high pass filter
fcut=100000;   %cutting freq
ordre=5;    % butterworth filter order

tick_en_temps=0.1;

% temps [tempsdetection-deltat ; tdetection+deltat] du fichier son qui est ouvert 
deltat=0.000500;

% initialisation compteur de clicks 
totalclics=0;

%%% Importation de la liste des fichiers à traiter 

%[NomFich] = textread('liste_fichiers_interessants_extrait.txt', '%s') ;
[NomFich] = textread('liste_fichiers_interessants.txt', '%s') ;
N=length(NomFich)
% N est le nombre de fichiers qui seront traités


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    BOUCLE PRINCIPALE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


frequencespeak_totales=[];
frequencescentr_totales=[];
DeltaF_3dB_totales=[];
Deltat_10dB_totales=[];
DeltaF_10dB_totales=[];
Deltat_20dB_totales=[];
DeltaF_rms_totales=[];
Deltat_rms_totales=[];
ICI_totales=[];
SNR_totales=[];

moyennefreqpeak=[];
moyennefreqcentr=[];
moyenneDeltaF_3dB=[];
moyenneDeltat_10dB=[];
moyenneDeltaF_10dB=[];
moyenneDeltat_20dB=[];
moyenneDeltaF_rms=[];
moyenneDeltat_rms=[];
moyenneICI=[];

stdfreqpeak=[];
stdfreqcentr=[];
stdDeltaF_3dB=[];
stdDeltat_10dB=[];
stdDeltaF_10dB=[];
stdDeltat_20dB=[];
stdDeltaF_rms=[];
stdDeltat_rms=[];
stdICI=[];

w=1;

for i=1:1:length(NomFich)
%for i=1:10


% ouverture fichier son + extraction parametres 
   nomfichierwav=NomFich{i}

   Chemin = '/media/malige/donnees/BIOACOUSTIQUE/Donnees_CIEP/fichiers_interessants/';
   fichierwav = [Chemin nomfichierwav];
   [taille_totale, CHANNELS] = wavread (fichierwav, "size");
   [s,fm,dyn] = wavread(fichierwav,[1 10]);
   clear s;
   Tm=1/fm;

% ouverture fichier detection associé au fichier son ci-dessus
   date_et_heure=nomfichierwav(1:15);
   nomfichierdetect=['Res_',date_et_heure,'.txt'];
   Chemin = '/home/malige/Documents/BALEINES/RECHERCHES/dauphins/detecteur_clicks/Resultats_PuertoCisnes1/';
   fichierdetect = [Chemin nomfichierdetect];

   [date_detection_clic] = textread(fichierdetect, '%f') ;
   M=length(date_detection_clic)

% test pour savoir si on est dans la même série
  
   if i==1

      msg='changement de serie'
      frequencepeak=[];
      frequencecentr=[];
      DeltaF_3dB=[];
      Deltat_10dB=[];
      DeltaF_10dB=[];
      Deltat_20dB=[];
      DeltaF_rms=[];
      Deltat_rms=[];
      ICI=[];
      SNR=[];
      test=1;

   else 
      heure_brut=str2num(NomFich{i}(10:11));
      min_brut=str2num(NomFich{i}(12:13));
      heure_m1_brut=str2num(NomFich{i-1}(10:11));
      min_m1_brut=str2num(NomFich{i-1}(12:13));
      datemin=heure_brut*60+min_brut;
      datemin_m1=heure_m1_brut*60+min_m1_brut;

      if abs(datemin-datemin_m1)>20

          msg='changement de serie'
          w=w+1;
          test=1;
      else 
          test=0;
      endif    
  
   endif


% seconde boucle sur chaque click dans chaque fichier

   for j=1:M
%  for j=1:10


% calcul de l'ICI

      if j>1
          ICI=[ICI date_detection_clic(j)-date_detection_clic(j-1)];
      endif

% ouverture fichier son click

      t_detect=date_detection_clic(j);
      t_entree=t_detect-deltat;
      t_sortie=t_detect+deltat;
      tfinal=t_sortie-t_entree;
      t=[0:Tm:Tm*floor(tfinal/Tm)]';

      inicioN = floor(t_entree * fm);
      finN = inicioN + length(t) - 1;

      extrait=wavread(fichierwav,[inicioN finN]);
      senal = extrait(:,1);


% filtre passe haut à fcut

      [butter1,butter2]=butter(ordre,fcut/(fm/2),'high');
      sfiltre=filter(butter1,butter2,senal);
      [val_max imax]=max(sfiltre);
      t0=imax*Tm;      
      L=length(sfiltre);

% calcul Delta t -10dB

       enveloppe=abs(hilbert(sfiltre));
       [valmax,indicemax]=max(enveloppe(200:300));
       indicemax=indicemax+199;

       valplus=valmax;
       incrplus=0;
       while valplus>valmax/sqrt(10)     % Delta t -10dB
            incrplus=incrplus+1;
            valplus=enveloppe(indicemax+incrplus);
       endwhile  

       valmoins=valmax;
       incrmoins=0;
       while valmoins>valmax/sqrt(10)    % Delta t -10 dB
            incrmoins=incrmoins-1;
            valmoins=enveloppe(indicemax+incrmoins);
       endwhile

       delta_t_10dB=(incrplus-incrmoins)*Tm;

% calcul Delta t -20dB

       enveloppe=abs(hilbert(sfiltre));
       [valmax,indicemax]=max(enveloppe(200:300));
       indicemax=indicemax+199;

       valplus=valmax;
       incrplus=0;
       while (valplus>valmax/10) && (incrplus<255)    % Delta t -20dB
            incrplus=incrplus+1;
            valplus=enveloppe(indicemax+incrplus);
       endwhile  

       valmoins=valmax;
       incrmoins=0;
       while (valmoins>valmax/10) && (incrmoins>-255)   % Delta t -20dB
            incrmoins=incrmoins-1;
            valmoins=enveloppe(indicemax+incrmoins);
       endwhile

       delta_t_20dB=(incrplus-incrmoins)*Tm;

% delta t rms

       energiet=std(sfiltre);
       MOM2t=sum((t-t0).^2.*(sfiltre.^2));
       delta_t_rms=sqrt(MOM2t/energiet); 

% bilan Delta t

       Deltat_10dB=[Deltat_10dB delta_t_10dB];
       Deltat_20dB=[Deltat_20dB delta_t_20dB];       
       Deltat_rms=[Deltat_rms delta_t_rms];       


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) estimation frequence fic et centroide par FFT 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fenetre

hamming=0.54-0.46*cos([0:L-1]*2*pi/(L-1));
fenetre=hamming;

%Transformee de Fourier 
%       sfiltre_fenetre=sfiltre.*fenetre;
       sfiltre_fenetre=sfiltre;
       Tfourier=fft(sfiltre_fenetre);


% zoom en frequence
       coefzoom=2;
       zoom_Tfourier=Tfourier(1:floor(L/coefzoom));
       f=[1:floor(L/coefzoom)]*fm/L;
       duree=Tm*L;

% freq peak

       energie=abs(zoom_Tfourier(100:200));
       [valmax,indicemax]=max(energie);
       fpeak=99000+indicemax*fm/L;

% freq centr
       E=sum(energie.^2);
       extraitf=f(100:200);
       MOM1=sum(extraitf.*energie'.^2);
       fcentr=MOM1/E;

% Delta f -3bB

       energieplus=valmax;
       incrplus=0;
       while energieplus>valmax/sqrt(2)   % Delta F -3dB
            incrplus=incrplus+1;
            energieplus=abs(zoom_Tfourier(indicemax+100+incrplus));
       endwhile  

       energiemoins=valmax;
       incrmoins=0;
       while energiemoins>valmax/sqrt(2)   % Delta F -3dB
            incrmoins=incrmoins-1;
            energiemoins=abs(zoom_Tfourier(indicemax+100+incrmoins));
       endwhile

       deltaf_3dB=incrplus-incrmoins;


% Delta f -10bB

       energieplus=valmax;
       incrplus=0;
       while energieplus>valmax/sqrt(10)   % Delta F -10dB
            incrplus=incrplus+1;
            energieplus=abs(zoom_Tfourier(indicemax+100+incrplus));
       endwhile  

       energiemoins=valmax;
       incrmoins=0;
       while energiemoins>valmax/sqrt(10)   % Delta F -10dB
            incrmoins=incrmoins-1;
            energiemoins=abs(zoom_Tfourier(indicemax+100+incrmoins));
       endwhile

       deltaf_10dB=incrplus-incrmoins;


% Delta f rms

       E=sum(energie.^2);
       MOM2f=sum((extraitf-fcentr).^2.*energie'.^2);
       deltaf_rms=sqrt(MOM2f/E);

% bilan

       frequencepeak=[frequencepeak fpeak];
       frequencecentr=[frequencecentr fcentr];
       DeltaF_3dB=[DeltaF_3dB deltaf_3dB];
       DeltaF_10dB=[DeltaF_10dB deltaf_10dB];
       DeltaF_rms=[DeltaF_rms deltaf_rms];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimation du signal sur bruit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      energie_signal=std(sfiltre);

% ouverture fichier bruit

      t_detect=date_detection_clic(j);
      t_entree=t_detect+0.000100;
      t_sortie=t_detect+0.000150;
      tfinal2=t_sortie-t_entree;
      x=[0:Tm:Tm*floor(tfinal2/Tm)]';

      inicioN = floor(t_entree * fm);
      finN = inicioN + length(x) - 1;

      extrait=wavread(fichierwav,[inicioN finN]);
      bruit = extrait(:,1);

      bruitfiltre=filter(butter1,butter2,bruit);
      energie_bruit=std(bruitfiltre);

      SNR=[SNR 20*log(energie_signal/energie_bruit)];
 


% representation graphique avec test.

%{
   if rem(j,100)==0

       figure(j)
       subplot(311);
       plot(t,sfiltre,"linewidth",1);
       grid;
       axis([0 duree -(val_max*1.1) val_max*1.1]);
       set(gca,'xtick',0:tick_en_temps:duree); 
       set(gca,'xticklabel',0:tick_en_temps:duree,'fontweight','bold')
       xlabel('time (s)','fontweight','bold','fontsize',16);
       ylabel('Amplitude','fontweight','bold','fontsize',16);
       subplot(312);
       plot(f/1000,abs(zoom_Tfourier),"linewidth",1);
       grid;
%       ylabel('intensity','fontweight','bold','fontsize',16)
       xlabel('frequency (kHz)','fontweight','bold','fontsize',16)
       set(gca,'fontweight','bold','fontsize',16); 
       subplot(313);
       plot(f/1000,real(zoom_Tfourier),"linewidth",1);
       grid;
%       ylabel('intensity','fontweight','bold','fontsize',16)
       xlabel('frequency (kHz)','fontweight','bold','fontsize',16)
       set(gca,'fontweight','bold','fontsize',16); 



    endif


   if rem(j,100)==0

       figure(1+j)
       subplot(211);
       plot(t,sfiltre,"linewidth",1);
       grid;
       axis([0 duree -(val_max*1.1) val_max*1.1]);
       set(gca,'xtick',0:tick_en_temps:duree); 
       set(gca,'xticklabel',0:tick_en_temps:duree,'fontweight','bold')
       xlabel('time (s)','fontweight','bold','fontsize',16);
       ylabel('Amplitude','fontweight','bold','fontsize',16);
       subplot(212);
       plot(t,abs(hilbert(sfiltre)),"linewidth",1);
       grid;
       xlabel('time (s)','fontweight','bold','fontsize',16);
       ylabel('Amplitude','fontweight','bold','fontsize',16);
       set(gca,'fontweight','bold','fontsize',16); 

    endif

%}

   endfor

% calcul des moyennes et ecartypes


%   Mtest=length(frequencepeak)

if (test==1) | (i==N)

   moyennefreqpeak=[moyennefreqpeak mean(frequencepeak/1000)];
   stdfreqpeak=[stdfreqpeak std(frequencepeak/1000)];   
   moyennefreqcentr=[moyennefreqcentr mean(frequencecentr/1000)];
   stdfreqcentr=[stdfreqcentr std(frequencecentr/1000)];   
   moyenneDeltaF_3dB=[moyenneDeltaF_3dB mean(DeltaF_3dB)];
   stdDeltaF_3dB=[stdDeltaF_3dB std(DeltaF_3dB)];   
   moyenneDeltaF_10dB=[moyenneDeltaF_10dB mean(DeltaF_10dB)];
   stdDeltaF_10dB=[stdDeltaF_10dB std(DeltaF_10dB)]; 
   moyenneDeltat_10dB=[moyenneDeltat_10dB mean(Deltat_10dB)];
   stdDeltat_10dB=[stdDeltat_10dB std(Deltat_10dB)];   
   moyenneDeltat_20dB=[moyenneDeltat_20dB mean(Deltat_20dB)];
   stdDeltat_20dB=[stdDeltat_20dB std(Deltat_20dB)]; 
   moyenneDeltaF_rms=[moyenneDeltaF_rms mean(DeltaF_rms)];
   stdDeltaF_rms=[stdDeltaF_rms std(DeltaF_rms)]; 
   moyenneDeltat_rms=[moyenneDeltat_rms mean(Deltat_rms)];
   stdDeltat_rms=[stdDeltat_rms std(Deltat_rms)]; 
   moyenneICI=[moyenneICI mean(ICI)];
   stdICI=[stdICI std(ICI)];

%   moyenneT=[moyenneT mean(T*1000)];
%   stdT=[stdT std(T*1000)];   

  frequencespeak_totales=[frequencespeak_totales frequencepeak];
length(frequencespeak_totales)
  frequencescentr_totales=[frequencescentr_totales frequencecentr];
  DeltaF_3dB_totales=[DeltaF_3dB_totales DeltaF_3dB];
  Deltat_10dB_totales=[Deltat_10dB_totales Deltat_10dB];
  DeltaF_10dB_totales=[DeltaF_10dB_totales DeltaF_10dB];
  Deltat_20dB_totales=[Deltat_20dB_totales Deltat_20dB];
  DeltaF_rms_totales=[DeltaF_rms_totales DeltaF_rms];
  Deltat_rms_totales=[Deltat_rms_totales Deltat_rms];
  ICI_totales=[ICI_totales ICI]; 
  SNR_totales=[SNR_totales SNR];
  produit_incertitude=DeltaF_rms_totales.*Deltat_rms_totales;
  pi4produitincertitude=4*pi*produit_incertitude; 

%  T_totales=[T_totales T];


% representation graphique de l'histogramme des frequences peak

%{

   freqkHz=frequencepeak/1000;
   centerfreq=[100:1:150];
   figure(100+w)
   hist(freqkHz,centerfreq)
%   hist(freqkHz,centerfreq,'fontsize',2)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('frequency (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',100:10:150); 
   set(gca,'xticklabel',100:10:150,'fontweight','bold')
   titre=['histogram_frequency_peak_of_clicks_serie_',num2str(w),'.jpg'];
   print(titre,'-djpg');

% representation graphique de l'histogramme des duree


   freqkHz=frequencecentr/1000;
   centerfreq=[100:1:150];
   figure(200+w)
   hist(freqkHz,centerfreq)
%   hist(freqkHz,centerfreq,'fontsize',2)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('frequency (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',100:10:150); 
   set(gca,'xticklabel',100:10:150,'fontweight','bold')
   titre=['histogram_frequency_centroide_of_clicks_serie_',num2str(w),'.jpg'];
   print(titre,'-djpg');

%}
% et les deux ensemble

%{

   freqpeakkHz=frequencepeak/1000;
   centerfreq=[100:1:150];
   freqcentrkHz=frequencecentr/1000;
    
   figure(300+w)

   subplot(211);
   hist(freqpeakkHz,centerfreq)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Peak frequency (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',100:10:150); 
   set(gca,'xticklabel',100:10:150,'fontweight','bold')
   
   subplot(212);
   hist(freqcentrkHz,centerfreq)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Centroid frequency (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',100:10:150); 
   set(gca,'xticklabel',100:10:150,'fontweight','bold')
   titre=['histogram_frequency_peak_and_centroide_of_clicks_serie_',num2str(w),'.jpg'];
   print(titre,'-djpg');

% representation graphique de l'histogramme des DeltaF

   freq=[0:1:50];
   figure(400+w)
   hist(DeltaF,freq)
%   hist(freqkHz,centerfreq,'fontsize',2)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('frequency band (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',0:10:50); 
   set(gca,'xticklabel',0:10:50,'fontweight','bold')
   titre=['histogram_frequency_band_-3dB_of_clicks_serie_',num2str(w),'.jpg'];
   print(titre,'-djpg');
%}

% representation graphique des 9 histogrammes

   freqpeakkHz=frequencepeak/1000;
   centerfreq=[100:1:200];
   freqcentrkHz=frequencecentr/1000;
   freq=[0:1:50];
   Deltat_rms_micros=Deltat_rms*1000000;
   Deltat_10dB_micros=Deltat_10dB*1000000;
   Deltat_20dB_micros=Deltat_20dB*1000000;
   DeltaF_rms=DeltaF_rms/1000;
   temps=[0:5:200];
   tempsbis=[0:5:400];
   ICIms=ICI*1000;
   tempsICI=[0:5:400];


   figure(100+w)

   subplot(331);
   hist(freqpeakkHz,centerfreq)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Peak frequency (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',100:50:200); 
   set(gca,'xticklabel',100:50:200,'fontweight','bold')
   
 
   subplot(332);
   hist(DeltaF_rms,freq)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Frequency band rms (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:20:50); 
   set(gca,'xticklabel',0:20:50,'fontweight','bold')
   

   subplot(333);
   hist(Deltat_rms_micros,temps)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Delta t rms (in micro s)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:100:200); 
   set(gca,'xticklabel',0:100:200,'fontweight','bold')


   subplot(334);
   hist(freqcentrkHz,centerfreq)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Centroid frequency (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',100:50:200); 
   set(gca,'xticklabel',100:50:200,'fontweight','bold')


   subplot(335);
   hist(DeltaF_3dB,freq)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Frequency band -3dB (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:20:50); 
   set(gca,'xticklabel',0:20:50,'fontweight','bold')
  
   subplot(336);
   hist(Deltat_10dB_micros,temps)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Duration -10dB (in micro s)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:100:200); 
   set(gca,'xticklabel',0:100:200,'fontweight','bold')

   subplot(337);
   hist(ICIms,tempsICI)
   axis([-20 300]);
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('ICI (in ms)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:100:300); 
   set(gca,'xticklabel',0:100:300,'fontweight','bold')

   subplot(338);
   hist(DeltaF_10dB,freq)
   hist(ICIms,tempsICI)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Frequency band -10dB (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:20:50); 
   set(gca,'xticklabel',0:20:50,'fontweight','bold')
  
   subplot(339);
   hist(Deltat_20dB_micros,tempsbis)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Duration -20dB (in micro s)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:200:400); 
   set(gca,'xticklabel',0:200:400,'fontweight','bold')


   titre=['histogram_9parameters_of_clicks_serie_',num2str(w),'.jpg'];
   print(titre,'-djpg');


  close all


  frequencepeak=[];
  frequencecentr=[];
  DeltaF_3dB=[];
  Deltat_10dB=[];
  DeltaF_10dB=[];
  Deltat_20dB=[];
  DeltaF_rms=[];
  Deltat_rms=[];
  ICI=[];
  SNR=[];


endif

% calcul du nombre total de clics
totalclics=totalclics+M;


endfor



% representation graphique de l'histogramme de toutes les frequences peak

   freq_peak_kHz=frequencespeak_totales/1000;
   centerfreq=[100:1:200];


%{
   figure(1000)
   hist(freq_peak_kHz,centerfreq)
%   hist(freqkHz,centerfreq,'fontsize',2)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Peak frequency (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',100:10:150); 
   set(gca,'xticklabel',100:10:150,'fontweight','bold')
   titre=['histogram_frequency_peak_of_clicks_serie_total.jpg'];
   print(titre,'-djpg');
%}


% representation graphique de l'histogramme de toutes les frequences centroides

   freq_centr_kHz=frequencescentr_totales/1000;
   centerfreq=[100:1:200];

%{
   figure(2000)
   hist(freq_centr_kHz,centerfreq)
%   hist(freqkHz,centerfreq,'fontsize',2)
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Centroid frequency (in kHz)','fontweight','bold','fontsize',16)
   set(gca,'xtick',100:10:150); 
   set(gca,'xticklabel',100:10:150,'fontweight','bold')
   titre=['histogram_frequency_centroid_of_clicks_serie_total.jpg'];
   print(titre,'-djpg');
%}


Deltat_rms_micros_totales=Deltat_rms_totales*1000000;
Deltat_10dB_micros_totales=Deltat_10dB_totales*1000000;
Deltat_20dB_micros_totales=Deltat_20dB_totales*1000000;
DeltaF_rms_totales=DeltaF_rms_totales/1000;
ICIms_totales=ICI_totales*1000;


   figure(1000)

   subplot(331);
   hist(freq_peak_kHz,centerfreq)
%   axis([100 150 0 700]);
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Peak frequency (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',100:50:200); 
   set(gca,'xticklabel',100:50:200,'fontweight','bold')
   

   subplot(332);
   hist(DeltaF_rms_totales,freq)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Frequency band rms (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:20:50); 
   set(gca,'xticklabel',0:20:50,'fontweight','bold')
   
   subplot(333);
   hist(Deltat_rms_micros_totales,temps)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Duration rms (in micro s)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:100:200); 
   set(gca,'xticklabel',0:100:200,'fontweight','bold')


   subplot(334);
   hist(freq_centr_kHz,centerfreq)
%   axis([100 150 0 700]);
   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Centroid frequency (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',100:50:200); 
   set(gca,'xticklabel',100:50:200,'fontweight','bold')


   subplot(335);
   hist(DeltaF_3dB_totales,freq)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Frequency band -3dB (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:20:50); 
   set(gca,'xticklabel',0:20:50,'fontweight','bold')
   
   subplot(336);
   hist(Deltat_10dB_micros_totales,temps)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Duration 10dB (in micro s)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:100:200); 
   set(gca,'xticklabel',0:100:200,'fontweight','bold')


   subplot(337);
   hist(ICIms_totales,tempsICI)
   axis([-20 300]);
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('ICI (in ms)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:100:300); 
   set(gca,'xticklabel',0:100:300,'fontweight','bold')


   subplot(338);
   hist(DeltaF_10dB_totales,freq)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Frequency band -10dB (in kHz)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:20:50); 
   set(gca,'xticklabel',0:20:50,'fontweight','bold')
   

   subplot(339);
   hist(Deltat_20dB_micros_totales,tempsbis)
%   ylabel('number of clicks','fontweight','bold','fontsize',16)
   xlabel('Duration 20dB (in micro s)','fontweight','bold','fontsize',10)
   set(gca,'xtick',0:200:400); 
   set(gca,'xticklabel',0:200:400,'fontweight','bold')


%   titre=['histogram_9parameters_of_clicks_serie_totale.jpg'];
%   print(titre,'-djpg');
   titre=['histogram_9parameters_of_clicks_serie_totale.eps'];
   print(titre,'-depsc');

% representation graphique de l'histogramme de toutes les durees

%{
   Tms=T_totales*1000;
   centerT=[0.005:0.005:0.12];
   figure(2000)
   hist(Tms,centerT)
   ylabel('number of clicks','fontweight','bold','fontsize',18)
   xlabel('characteristic duration T (in ms)','fontweight','bold','fontsize',18)
   %set(gca,'xtick',1:10:61); 
   %set(gca,'xticklabel',100:10:160,'fontweight','bold')
   titre=['histogram_duration_of_clicks_total_freqdebut',num2str(freqdebut),'.jpg'];
   print(titre,'-djpg');
%}

duree=toc

totalclics

% parametres moy/ecartype de chaque serie

%{
moyennefreqpeak
stdfreqpeak
moyennefreqcentr
stdfreqcentr
moyenneDeltaF_rms/1000
stdDeltaF_rms/1000
moyenneDeltaF_3dB
stdDeltaF_3dB
moyenneDeltaF_10dB
stdDeltaF_10dB
moyenneDeltat_rms*1000000
stdDeltat_rms*1000000
moyenneDeltat_10dB*1000000
stdDeltat_10dB*1000000
moyenneDeltat_20dB*1000000
stdDeltat_20dB*1000000
moyenneICI
stdICI
%}

% parametre pour toute la serie

mean(freq_peak_kHz)
std(freq_peak_kHz)

mean(freq_centr_kHz)
std(freq_centr_kHz)

mean(ICIms_totales)
std(ICIms_totales)

mean(DeltaF_rms_totales)
std(DeltaF_rms_totales)

mean(DeltaF_3dB_totales)
std(DeltaF_3dB_totales)

mean(DeltaF_10dB_totales)
std(DeltaF_10dB_totales)

mean(Deltat_rms_micros_totales)
std(Deltat_rms_micros_totales)

mean(Deltat_10dB_micros_totales)
std(Deltat_10dB_micros_totales)

mean(Deltat_20dB_micros_totales)
std(Deltat_20dB_micros_totales)

mean(pi4produitincertitude)
std(pi4produitincertitude)


% Ajustement de gaussiennes à l'histogramme des freq peak

% premier ajustement seul les deux gros peaks

%{
size(freq_peak_kHz)

figure(2)
hist(freq_peak_kHz,centerfreq)


freq_peak_kHz_sel=freq_peak_kHz(freq_peak_kHz>110);
freq_peak_kHz_selec=freq_peak_kHz_sel(freq_peak_kHz_sel<147);

K=length(freq_peak_kHz_selec)
centerfreq_selec=[101:1:150];

figure(3)
hist(freq_peak_kHz_selec,centerfreq_selec)

nbfpeak=zeros(1,50);

for i=1:K
    n=floor(freq_peak_kHz_selec(i)+0.5);
    nbfpeak(n-100)=nbfpeak(n-100)+1;
endfor


% Function that will be fit
function [y]=doublegauss(x,par)
  y=par(1) .* exp (-(x-par(2)).^2 / par(3)^2) + par(4) .* exp (-(x-par(5)).^2 / par(6)^2) ;
end

% fitting

weights=ones(size(nbfpeak));
pin=[600,126,3,800,135,3];
      dp=0.001 * ones (size (pin));  
      dFdp="dfdp";   
      options.bounds=[400 120 1 600 130 1; 700 130 6 900 140 6]';
      [f,p,cvg,iter,corp,covp]=leasqr(centerfreq_selec,nbfpeak,pin,"doublegauss",.000100,100,weights,dp,dFdp,options);
 


centerfreq_selec_precis=[101:0.1:150];
nbfpeak_reconstruit=p(1) .* exp (-(centerfreq_selec_precis-p(2)).^2 / p(3)^2) + p(4) .* exp (-(centerfreq_selec_precis-p(5)).^2 / p(6)^2) ;


fon=16

figure(4)
         plot(centerfreq_selec,nbfpeak,"linewidth",3)
         hold('on');
         plot(centerfreq_selec_precis,nbfpeak_reconstruit,'r',"linewidth",2)
         xlabel('fontweight','bold','fontsize',fon);
         titre=['fit_gaussiennes.eps'];
         print(titre,'-depsc');

figure(5)
         hist(freq_peak_kHz_selec,centerfreq_selec,"linewidth",3)
         hold('on');
         plot(centerfreq_selec_precis,nbfpeak_reconstruit,'r',"linewidth",2)
       set(gca,'xtick',100:10:150); 
       set(gca,'xticklabel',100:10:150'fontweight','bold')
       xlabel('Peak frequency (kHz)','fontweight','bold','fontsize',fon);
       ylabel('Number','fontweight','bold','fontsize',fon);
         titre=['histo_gaussiennes.eps'];
         print(titre,'-depsc');
%}
       


% deuxieme ajustement 

size(freq_peak_kHz)

figure(2)
hist(freq_peak_kHz,centerfreq)


K=length(freq_peak_kHz)
centerfreq_selec=[81:1:200];

figure(3)
hist(freq_peak_kHz,centerfreq_selec)

nbfpeak=zeros(1,120);

for i=1:K
    n=floor(freq_peak_kHz(i)+0.5);
    nbfpeak(n-80)=nbfpeak(n-80)+1;
endfor


% Function that will be fit
function [y]=quatregauss(x,par)
  y=par(1) .* exp (-(x-par(2)).^2 / par(3)^2) + par(4) .* exp (-(x-par(5)).^2 / par(6)^2) + par(7) .* exp (-(x-par(8)).^2 / par(9)^2) + par(10) .* exp (-(x-par(11)).^2 / par(12)^2) ;
end

% fitting

weights=ones(size(nbfpeak));
pin=[600,126,3,800,135,3,50,107,2,50,164,10];
      dp=0.001 * ones (size (pin));  
      dFdp="dfdp";   
      options.bounds=[400 120 1 600 130 1 10 100 1 10 160 1; 700 130 6 900 140 6 100 110 5 100 170 20]';
      [f,p,cvg,iter,corp,covp]=leasqr(centerfreq_selec,nbfpeak,pin,"quatregauss",.000100,100,weights,dp,dFdp,options);
 


centerfreq_selec_precis=[81:0.1:200];
nbfpeak_reconstruit=p(1) .* exp (-(centerfreq_selec_precis-p(2)).^2 / p(3)^2) + p(4) .* exp (-(centerfreq_selec_precis-p(5)).^2 / p(6)^2) + p(7) .* exp (-(centerfreq_selec_precis-p(8)).^2 / p(9)^2) + p(10) .* exp (-(centerfreq_selec_precis-p(11)).^2 / p(12)^2);


fon=16

figure(10)
         plot(centerfreq,nbfpeak,"linewidth",3)
         hold('on');
         plot(centerfreq_selec_precis,nbfpeak_reconstruit,'r',"linewidth",2)
         xlabel('fontweight','bold','fontsize',fon);
         titre=['fit_gaussiennes.eps'];
         print(titre,'-depsc');

figure(11)
         hist(freq_peak_kHz,centerfreq,"linewidth",3)
         hold('on');
         plot(centerfreq_selec_precis,nbfpeak_reconstruit,'r',"linewidth",2)
       set(gca,'xtick',100:10:200); 
       set(gca,'xticklabel',100:10:200,'fontweight','bold')
       xlabel('Peak frequency (kHz)','fontweight','bold','fontsize',fon);
       ylabel('Number','fontweight','bold','fontsize',fon);
         titre=['histo_4_gaussiennes.eps'];
         print(titre,'-depsc');


