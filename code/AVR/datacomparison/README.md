# CEAP_Analysis

This repository contains python scripts that were used to compare the _Continuous Physiological and Behavioral Emotion Annotation Dataset for 360° Videos (CEAP-360VR_, available here: https://github.com/cwi-dis/CEAP-360VR-Dataset) with the data from phase 1 of the _Affective VR_ study (not published yet). Both experiments were similar as in that they used 360° Virtual Reality (VR) videos with a length of 60 seconds each to stimulate emotional responses, and a continuous rating method based on the Circumplex Model of Emotion as affective measure. Their goal was to find out whether continuous emotional ratings can be seen as a more valid method to assess emotional states than post-hoc annotations.

## Overview

**01_preprocessing.py**: preprocesses annotation data from CEAP dataset  
**02_summary_stats.py**: calculates summary statistics for annotation data from Affective VR and CEAP dataset after preprocessing  
**03_plots_ceap_affectivevr.py**: plots preprocessed CEAP data against preprocessed Affective VR data  
**comparison_ceap_affectivevr.pdf**: file with a summary of the results obtained by the comparison of both datasets, including figures  
**README.md**: this readme file

## More detailed steps of the scripts

#### 01_preprocessing.py:

>**Step 1a:** Preprocess individual CEAP data sets to match AffectiveVR data  
**Step 1b:** Create a long format file of preprocessed CEAP data with the data of all participants  
**Step 1c:** Merge long CEAP data table with AffectiveVR data
 
>**Inputs:** Raw data from CEAP dataset (download here: https://github.com/cwi-dis/CEAP-360VR-Dataset)    
Preprocessed data from AffectiveVR dataset (ask for access) as csv file "cr_rs_clean.csv"
      
>**Outputs:**     
>>**1a:** Preprocessed data from CEAP dataset for each participant as csv files in "Preprocessed" directory  
**1b:** csv file with preprocessed data from CEAP dataset for all participants together  
**1c:** csv file from 1b merged with AffectiveVR data  

#### 02_summary_stats.py:

>**Inputs:** csv files created by 01_preprocessing.py with preprocessed data for each participant for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv)    

>**Step 2a:** creates csv files with descriptive statistics for each quadrant separately for each participant  
>>**Outputs:** cr_affectivevr_descriptive_individual.csv, cr_ceap_descriptive_individual.csv    

>**Step 2b:** creates csv file with descriptive statistics for each quadrant averaged over participants  
>>**Outputs:** cr_affectivevr_descriptive.csv, cr_ceap_descriptive.csv             

>**Step 2c:** creates csv file with descriptive statistics averaged over quadrants for each participant  
>>**Outputs:** cr_affectivevr_descriptive_avgvideos_individual.csv, cr_ceap_descriptive_avgvideos_individual.csv     

>**Step 2d:** creates csv file with descriptive statistics averaged over quadrants averaged over participants  
>>**Outputs:** cr_affectivevr_descriptive_avgvideos.csv, cr_ceap_descriptive_avgvideos.csv

>**Step 2e:** creates csv file with arousal and valence values for all timepoints, quadrants and rating methods across participants  
>>**Outputs:** cr_affectvevr_all.csv, cr_ceap_all.csv  
                
#### 03_plots_ceap_affectivevr.py:

>**Step 3a:** Create descriptive plots for CEAP data against AffectiveVR data  
>>**Inputs:** csv file created by 01_preprocessing.py with preprocessed data from CEAP and AffectiveVR dataset  
>>**Outputs:** 2 x 2 plot (four quadrants) for each dependent variable (Arousal, Valence, Distance, Angle)  

>**Step 3b:** Create descriptive plot for mean of CEAP data against mean of AffectiveVR data  
>>**Inputs:** two csv files created by 02_summary_stats.py with descriptive statistics averaged across timepoints and participants (cr_affectivevr_descriptive.csv and cr_ceap_descriptive.csv)   
>>**Output:** plot with coordinate system and mean arousal + valence values for all quadrants

>**Step 3c:** Create descriptive plots for CEAP / AffectiveVR data for all videos separately averaged across participants   
>>**Inputs:** csv files created by 02_summary_stats.py with averaged data across participants (cr_affectivevr_all.csv, cr_ceap_all.csv  
>>**Output:** 2 plots with coordinate system and averaged arousal + valence values for each video, one for CEAP and one for AffectiveVR  
