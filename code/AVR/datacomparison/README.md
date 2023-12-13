# AffectiveVR – **datacomparison**

    Last update:    December 13, 2023
    Status:         work in progress

***

## Description

This directory contains scripts that were used to compare the _Continuous Physiological and Behavioral Emotion Annotation Dataset for 360° Videos_ (**CEAP-360VR**, available [here](https://github.com/cwi-dis/CEAP-360VR-Dataset)) with the data from phase 1 of the AVR study. Both experiments were similar as in that they used 360° Virtual Reality (VR) videos with a length of 60 seconds each to stimulate emotional responses, and a continuous rating method based on the Circumplex Model of Emotion as affective measure. Their goal was to find out whether continuous emotional ratings can be seen as a more valid method to assess emotional states than post-hoc annotations.

## Overview


|               script               |                  contents                   |
| :--------------------------------: | :-----------------------------------------: |
|`./code/AVR/datacomparison/01_preprocessing.py`| Script to put CEAP and AVR phase 1 data in the same format so they can be compared later on.|
|`./code/AVR/datacomparison/02_summary_stats.py`| Script to calculate summary statistics for CEAP dataset.|
|`.code/AVR/datacomparison/03_plots_ceap_affectivevr.py`| Script to plot CEAP data against AVR phase 1 data in order to be able to compare both datasets. |

## Step by Step

#### `01_preprocessing.py`
|   step    |           |   input      |    output     |
|:--------: | :-------: | :-----------: | :------------: |
| 1a | Preprocess invididual CEAP datasets to match AVR data | Raw data from CEAP dataset | Preprocessed CAP data for each participant as csv file |
| 1b | Create a long format file of preprocessed CEAP data with the data of all participants | Preprocessed CEAP data | Csv fule with preprocessed data from CEAP datasets for all participants together |
| 1c | Merge long CEAP data table with AVR data | Preprocessed CEAP and AVR data | CSV file from 1b merged with AVR data |


#### `02_summary_stats.py`
|   step    |                        |   input      |    output     |
|:--------: | :--------------------: | :-----------: | :------------: |
| 2a | Creates csv files with descriptive statistics for each quadrant separately for each participant | csv files created by 01_preprocessing.py with preprocessed data for each participant for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv) | cr_affectivevr_descriptive_individual.csv, cr_ceap_descriptive_individual.csv   |
| 2b | Creates csv file with descriptive statistics for each quadrant averaged over participants | csv files created by 01_preprocessing.py with preprocessed data for each participant for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv) | cr_affectivevr_descriptive.csv, cr_ceap_descriptive.csv |
| 2c | Creates csv file with descriptive statistics averaged over quadrants for each participant | csv files created by 01_preprocessing.py with preprocessed data for each participant for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv) | cr_affectivevr_descriptive_avgvideos_individual.csv, cr_ceap_descriptive_avgvideos_individual.csv |
| 2d | Creates csv file with descriptive statistics averaged over quadrants averaged over participants | csv files created by 01_preprocessing.py with preprocessed data for each participant for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv) | cr_affectivevr_descriptive_avgvideos.csv, cr_ceap_descriptive_avgvideos.csv |
| 2e | creates csv file with arousal and valence values for all timepoints, quadrants and rating methods across participants  | csv files created by 01_preprocessing.py with preprocessed data for each participant for both CEAP and AffectiveVR data (all_participants_ceap_affectivevr.csv) | cr_affectvevr_all.csv, cr_ceap_all.csv  |
                
#### `02_plots_ceap_affectivevr.py`

|   step    |           |   input      |    output     |
|:--------: | :-------: | :-----------: | :------------: |
| 3a | Create descriptive plots for CEAP data against AffectiveVR data | csv file created by 01_preprocessing.py with preprocessed data from CEAP and AffectiveVR dataset | 2 x 2 plot (four quadrants) for each dependent variable (Arousal, Valence, Distance, Angle) |
| 3b | Create descriptive plot for mean of CEAP data against mean of AffectiveVR data  | two csv files created by 02_summary_stats.py with descriptive statistics averaged across timepoints and participants (cr_affectivevr_descriptive.csv and cr_ceap_descriptive.csv)   | plot with coordinate system and mean arousal + valence values for all quadrants |
| 3c | Create descriptive plots for CEAP / AffectiveVR data for all videos separately averaged across participants  |  csv files created by 02_summary_stats.py with averaged data across participants (cr_affectivevr_all.csv, cr_ceap_all.csv)  | 2 plots with coordinate system and averaged arousal + valence values for each video, one for CEAP and one for AffectiveVR  
 |