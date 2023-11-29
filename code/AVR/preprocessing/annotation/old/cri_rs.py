########################################################################################################################
# Script to preprocess CR data from AVR experiment into CR indices
# Need csv file containing CR data, from import_cr.py
# Output: csv file (dataframe) containing all CR indices and SR
# Author: Antonin Fourcade
# Last version: 15.08.2023
########################################################################################################################

# import packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

# Set paths and experiment parameters
data_path = 'E:/AffectiveVR/Data/'
blocks = ['Practice', 'Experiment', 'Assessment']
logging_freq = ['CR', 'SR']
test_site = ['Torino', 'Berlin']  # Torino (even SJ nb) = 0, Berlin (odd SJ nb) = 1
rating_method = ['Grid', 'Flubber', 'Proprioceptive', 'Baseline']
quadrant = ['HP', 'LP', 'LN', 'HN']
questionnaire = ['SUS', 'invasive_presence', 'Kunin']
cr_fs = 1/0.05  # sampling frequency CR in Hz
sess = 'S000'  # session of recording
plot = False  # plot individual CRs

debug = False  # debug mode

# get list of participants
sj_list = os.listdir(data_path + 'AVR/')
# sort participants in ascending order
sj_list.sort()
# exclude participants because no Grid CR log recorded
to_be_removed = {'01', '02', '03', '04', '05', '07', '09', '11', '13', '15', '17'}
sj_list = [item for item in sj_list if item not in to_be_removed]

# path to CR csv file and load data
filename_cr = 'cr_rs_clean.csv'
cr_path = data_path + filename_cr
cr_all = pd.read_csv(cr_path)

# Initialize variables
sub = []
site = []
rat_m = []
quad = []
srat_v = []
crat_last_v = []
crat_mean_v = []
crat_median_v = []
crat_mode_v = []
crat_max_v = []
crat_min_v = []
crat_std_v = []
crat_cv_v = []
crat_range_v = []
crat_iqr_v = []
crat_skew_v = []
crat_kurtosis_v = []
crat_auc_v = []
crat_cp_v = []
srat_a = []
crat_last_a = []
crat_mean_a = []
crat_median_a = []
crat_mode_a = []
crat_max_a = []
crat_min_a = []
crat_std_a = []
crat_cv_a = []
crat_range_a = []
crat_iqr_a = []
crat_skew_a = []
crat_kurtosis_a = []
crat_auc_a = []
crat_cp_a = []
srat_dist = []
crat_last_dist = []
crat_mean_dist = []
crat_median_dist = []
crat_mode_dist = []
crat_max_dist = []
crat_min_dist = []
crat_std_dist = []
crat_cv_dist = []
crat_range_dist = []
crat_iqr_dist = []
crat_skew_dist = []
crat_kurtosis_dist = []
crat_auc_dist = []
crat_cp_dist = []
srat_angle = []
crat_last_angle = []
crat_mean_angle = []
crat_median_angle = []
crat_mode_angle = []
crat_max_angle = []
crat_min_angle = []
crat_std_angle = []
crat_cv_angle = []
crat_range_angle = []
crat_iqr_angle = []
crat_skew_angle = []
crat_kurtosis_angle = []
crat_auc_angle = []
crat_cp_angle = []

# debug
if debug:
    sj_id = '19'
    rat_mtd = 'Flubber'
    vid = 'LP'

# Read and preprocess data
# Loop over participants
for sj, sj_id in enumerate(sj_list):
    # Read results in csv file
    sj_path = data_path + 'AVR/' + sj_id + '/' + sess + '/'
    trial_results_filename = sj_path + 'trial_results.csv'
    trials_results = pd.read_csv(trial_results_filename)
    cr_sj = cr_all[cr_all['sj_id'] == int(sj_id)]

    # Loop over rating methods
    for rm, rat_mtd in enumerate(rating_method):
        # Select data according to rat_mtd
        trials_rat_mtd = trials_results[trials_results['method_descriptor'] == rat_mtd]
        # Select Experiment data
        trials_exp = trials_rat_mtd[trials_rat_mtd['block_name'] == 'Experiment']
        cr_rm = cr_sj[cr_sj['rating_method'] == rat_mtd]

        # Loop over quadrants/videos
        for v, vid in enumerate(quadrant):
            sub = np.append(sub, sj_id)
            rat_m = np.append(rat_m, rat_mtd)
            quad = np.append(quad, vid)
            # Sjs with even number are in the first testing site, odd in the second
            if int(sj_id) % 2 == 0:
                site = np.append(site, test_site[0])
            else:
                site = np.append(site, test_site[1])
            # SR
            sr = trials_exp.loc[trials_exp['quadrant_descriptor'] == vid, ['SR_valence', 'SR_arousal']]
            srat_v = np.append(srat_v, sr['SR_valence'].values[0])
            srat_a = np.append(srat_a, sr['SR_arousal'].values[0])
            sr_dist = np.hypot(sr['SR_valence'].values[0], sr['SR_arousal'].values[0])
            sr_angle = np.arctan2(sr['SR_arousal'].values[0], sr['SR_valence'].values[0])
            srat_dist = np.append(srat_dist, sr_dist)
            srat_angle = np.append(srat_angle, sr_angle)
            # CR
            # Case of Baseline
            if rat_mtd == 'Baseline':
                #return nans because no CR
                crat_last_v = np.append(crat_last_v, np.nan)
                crat_mean_v = np.append(crat_mean_v, np.nan)
                crat_median_v = np.append(crat_median_v, np.nan)
                crat_mode_v = np.append(crat_mode_v, np.nan)
                crat_max_v = np.append(crat_max_v, np.nan)
                crat_min_v = np.append(crat_min_v, np.nan)
                crat_std_v = np.append(crat_std_v, np.nan)
                crat_cv_v = np.append(crat_cv_v, np.nan)
                crat_range_v = np.append(crat_range_v, np.nan)
                crat_iqr_v = np.append(crat_iqr_v, np.nan)
                crat_skew_v = np.append(crat_skew_v, np.nan)
                crat_kurtosis_v = np.append(crat_kurtosis_v, np.nan)
                crat_auc_v = np.append(crat_auc_v, np.nan)
                crat_cp_v = np.append(crat_cp_v, np.nan)
                crat_last_a = np.append(crat_last_a, np.nan)
                crat_mean_a = np.append(crat_mean_a, np.nan)
                crat_median_a = np.append(crat_median_a, np.nan)
                crat_mode_a = np.append(crat_mode_a, np.nan)
                crat_max_a = np.append(crat_max_a, np.nan)
                crat_min_a = np.append(crat_min_a, np.nan)
                crat_std_a = np.append(crat_std_a, np.nan)
                crat_cv_a = np.append(crat_cv_a, np.nan)
                crat_range_a = np.append(crat_range_a, np.nan)
                crat_iqr_a = np.append(crat_iqr_a, np.nan)
                crat_skew_a = np.append(crat_skew_a, np.nan)
                crat_kurtosis_a = np.append(crat_kurtosis_a, np.nan)
                crat_auc_a = np.append(crat_auc_a, np.nan)
                crat_cp_a = np.append(crat_cp_a, np.nan)
                crat_last_dist = np.append(crat_last_dist, np.nan)
                crat_mean_dist = np.append(crat_mean_dist, np.nan)
                crat_median_dist = np.append(crat_median_dist, np.nan)
                crat_mode_dist = np.append(crat_mode_dist, np.nan)
                crat_max_dist = np.append(crat_max_dist, np.nan)
                crat_min_dist = np.append(crat_min_dist, np.nan)
                crat_std_dist = np.append(crat_std_dist, np.nan)
                crat_cv_dist = np.append(crat_cv_dist, np.nan)
                crat_range_dist = np.append(crat_range_dist, np.nan)
                crat_iqr_dist = np.append(crat_iqr_dist, np.nan)
                crat_skew_dist = np.append(crat_skew_dist, np.nan)
                crat_kurtosis_dist = np.append(crat_kurtosis_dist, np.nan)
                crat_auc_dist = np.append(crat_auc_dist, np.nan)
                crat_cp_dist = np.append(crat_cp_dist, np.nan)
                crat_last_angle = np.append(crat_last_angle, np.nan)
                crat_mean_angle = np.append(crat_mean_angle, np.nan)
                crat_median_angle = np.append(crat_median_angle, np.nan)
                crat_mode_angle = np.append(crat_mode_angle, np.nan)
                crat_max_angle = np.append(crat_max_angle, np.nan)
                crat_min_angle = np.append(crat_min_angle, np.nan)
                crat_std_angle = np.append(crat_std_angle, np.nan)
                crat_cv_angle = np.append(crat_cv_angle, np.nan)
                crat_range_angle = np.append(crat_range_angle, np.nan)
                crat_iqr_angle = np.append(crat_iqr_angle, np.nan)
                crat_skew_angle = np.append(crat_skew_angle, np.nan)
                crat_kurtosis_angle = np.append(crat_kurtosis_angle, np.nan)
                crat_auc_angle = np.append(crat_auc_angle, np.nan)
                crat_cp_angle = np.append(crat_cp_angle, np.nan)
            else:
                # Other rating methods
                cr_vid = cr_rm[cr_rm['quadrant'] == vid]
                # Compute CR 'summary' indices
                # Dealing with missing data
                if cr_vid['cr_v'].isna().all() | cr_vid['cr_a'].isna().all():
                    crat_last_v = np.append(crat_last_v, np.nan)
                    crat_mean_v = np.append(crat_mean_v, np.nan)
                    crat_median_v = np.append(crat_median_v, np.nan)
                    crat_mode_v = np.append(crat_mode_v, np.nan)
                    crat_max_v = np.append(crat_max_v, np.nan)
                    crat_min_v = np.append(crat_min_v, np.nan)
                    crat_std_v = np.append(crat_std_v, np.nan)
                    crat_cv_v = np.append(crat_cv_v, np.nan)
                    crat_range_v = np.append(crat_range_v, np.nan)
                    crat_iqr_v = np.append(crat_iqr_v, np.nan)
                    crat_skew_v = np.append(crat_skew_v, np.nan)
                    crat_kurtosis_v = np.append(crat_kurtosis_v, np.nan)
                    crat_auc_v = np.append(crat_auc_v, np.nan)
                    crat_cp_v = np.append(crat_cp_v, np.nan)
                    crat_last_a = np.append(crat_last_a, np.nan)
                    crat_mean_a = np.append(crat_mean_a, np.nan)
                    crat_median_a = np.append(crat_median_a, np.nan)
                    crat_mode_a = np.append(crat_mode_a, np.nan)
                    crat_max_a = np.append(crat_max_a, np.nan)
                    crat_min_a = np.append(crat_min_a, np.nan)
                    crat_std_a = np.append(crat_std_a, np.nan)
                    crat_cv_a = np.append(crat_cv_a, np.nan)
                    crat_range_a = np.append(crat_range_a, np.nan)
                    crat_iqr_a = np.append(crat_iqr_a, np.nan)
                    crat_skew_a = np.append(crat_skew_a, np.nan)
                    crat_kurtosis_a = np.append(crat_kurtosis_a, np.nan)
                    crat_auc_a = np.append(crat_auc_a, np.nan)
                    crat_cp_a = np.append(crat_cp_a, np.nan)
                    crat_last_dist = np.append(crat_last_dist, np.nan)
                    crat_mean_dist = np.append(crat_mean_dist, np.nan)
                    crat_median_dist = np.append(crat_median_dist, np.nan)
                    crat_mode_dist = np.append(crat_mode_dist, np.nan)
                    crat_max_dist = np.append(crat_max_dist, np.nan)
                    crat_min_dist = np.append(crat_min_dist, np.nan)
                    crat_std_dist = np.append(crat_std_dist, np.nan)
                    crat_cv_dist = np.append(crat_cv_dist, np.nan)
                    crat_range_dist = np.append(crat_range_dist, np.nan)
                    crat_iqr_dist = np.append(crat_iqr_dist, np.nan)
                    crat_skew_dist = np.append(crat_skew_dist, np.nan)
                    crat_kurtosis_dist = np.append(crat_kurtosis_dist, np.nan)
                    crat_auc_dist = np.append(crat_auc_dist, np.nan)
                    crat_cp_dist = np.append(crat_cp_dist, np.nan)
                    crat_last_angle = np.append(crat_last_angle, np.nan)
                    crat_mean_angle = np.append(crat_mean_angle, np.nan)
                    crat_median_angle = np.append(crat_median_angle, np.nan)
                    crat_mode_angle = np.append(crat_mode_angle, np.nan)
                    crat_max_angle = np.append(crat_max_angle, np.nan)
                    crat_min_angle = np.append(crat_min_angle, np.nan)
                    crat_std_angle = np.append(crat_std_angle, np.nan)
                    crat_cv_angle = np.append(crat_cv_angle, np.nan)
                    crat_range_angle = np.append(crat_range_angle, np.nan)
                    crat_iqr_angle = np.append(crat_iqr_angle, np.nan)
                    crat_skew_angle = np.append(crat_skew_angle, np.nan)
                    crat_kurtosis_angle = np.append(crat_kurtosis_angle, np.nan)
                    crat_auc_angle = np.append(crat_auc_angle, np.nan)
                    crat_cp_angle = np.append(crat_cp_angle, np.nan)
                else:
                    # last rating
                    cr_last_v = cr_vid['cr_v'][~cr_vid['cr_v'].isna()].iloc[-1]
                    cr_last_a = cr_vid['cr_a'][~cr_vid['cr_a'].isna()].iloc[-1]
                    cr_last_dist = cr_vid['cr_dist'][~cr_vid['cr_dist'].isna()].iloc[-1]
                    cr_last_angle = cr_vid['cr_angle'][~cr_vid['cr_angle'].isna()].iloc[-1]
                    crat_last_v = np.append(crat_last_v, cr_last_v)
                    crat_last_a = np.append(crat_last_a, cr_last_a)
                    crat_last_dist = np.append(crat_last_dist, cr_last_dist)
                    crat_last_angle = np.append(crat_last_angle, cr_last_angle)
                    # mean
                    cr_mean_v = cr_vid['cr_v'].mean(skipna=True)
                    cr_mean_a = cr_vid['cr_a'].mean(skipna=True)
                    cr_mean_dist = np.nanmean(cr_vid['cr_dist'])
                    cr_mean_angle = np.nanmean(cr_vid['cr_angle'])
                    crat_mean_v = np.append(crat_mean_v, cr_mean_v)
                    crat_mean_a = np.append(crat_mean_a, cr_mean_a)
                    crat_mean_dist = np.append(crat_mean_dist, cr_mean_dist)
                    crat_mean_angle = np.append(crat_mean_angle, cr_mean_angle)
                    # median
                    cr_median_v = cr_vid['cr_v'].median(skipna=True)
                    cr_median_a = cr_vid['cr_a'].median(skipna=True)
                    cr_median_dist = cr_vid['cr_dist'].median(skipna=True)
                    cr_median_angle = cr_vid['cr_angle'].median(skipna=True)
                    crat_median_v = np.append(crat_median_v, cr_median_v)
                    crat_median_a = np.append(crat_median_a, cr_median_a)
                    crat_median_dist = np.append(crat_median_dist, cr_median_dist)
                    crat_median_angle = np.append(crat_median_angle, cr_median_angle)
                    # mode
                    cr_mode_v = cr_vid['cr_v'].mode(dropna=True)[0]
                    cr_mode_a = cr_vid['cr_a'].mode(dropna=True)[0]
                    cr_mode_dist = cr_vid['cr_dist'].mode(dropna=True)[0]
                    cr_mode_angle = cr_vid['cr_angle'].mode(dropna=True)[0]
                    crat_mode_v = np.append(crat_mode_v, cr_mode_v)
                    crat_mode_a = np.append(crat_mode_a, cr_mode_a)
                    crat_mode_dist = np.append(crat_mode_dist, cr_mode_dist)
                    crat_mode_angle = np.append(crat_mode_angle, cr_mode_angle)
                    # max
                    cr_max_v = cr_vid['cr_v'].max(skipna=True)
                    cr_max_a = cr_vid['cr_a'].max(skipna=True)
                    cr_max_dist = cr_vid['cr_dist'].max(skipna=True)
                    cr_max_angle = cr_vid['cr_angle'].max(skipna=True)
                    crat_max_v = np.append(crat_max_v, cr_max_v)
                    crat_max_a = np.append(crat_max_a, cr_max_a)
                    crat_max_dist = np.append(crat_max_dist, cr_max_dist)
                    crat_max_angle = np.append(crat_max_angle, cr_max_angle)
                    # min
                    cr_min_v = cr_vid['cr_v'].min(skipna=True)
                    cr_min_a = cr_vid['cr_a'].min(skipna=True)
                    cr_min_dist = cr_vid['cr_dist'].min(skipna=True)
                    cr_min_angle = cr_vid['cr_angle'].min(skipna=True)
                    crat_min_v = np.append(crat_min_v, cr_min_v)
                    crat_min_a = np.append(crat_min_a, cr_min_a)
                    crat_min_dist = np.append(crat_min_dist, cr_min_dist)
                    crat_min_angle = np.append(crat_min_angle, cr_min_angle)
                    # std
                    cr_std_v = cr_vid['cr_v'].std(skipna=True)
                    cr_std_a = cr_vid['cr_a'].std(skipna=True)
                    cr_std_dist = cr_vid['cr_dist'].std(skipna=True)
                    cr_std_angle = cr_vid['cr_angle'].std(skipna=True)
                    crat_std_v = np.append(crat_std_v, cr_std_v)
                    crat_std_a = np.append(crat_std_a, cr_std_a)
                    crat_std_dist = np.append(crat_std_dist, cr_std_dist)
                    crat_std_angle = np.append(crat_std_angle, cr_std_angle)
                    # cv: std/|mean|
                    cr_cv_v = cr_std_v/np.fabs(cr_mean_v)
                    cr_cv_a = cr_std_a/np.fabs(cr_mean_a)
                    cr_cv_dist = cr_std_dist/np.fabs(cr_mean_dist)
                    cr_cv_angle = cr_std_angle/np.fabs(cr_mean_angle)
                    crat_cv_v = np.append(crat_cv_v, cr_cv_v)
                    crat_cv_a = np.append(crat_cv_a, cr_cv_a)
                    crat_cv_dist = np.append(crat_cv_dist, cr_cv_dist)
                    crat_cv_angle = np.append(crat_cv_angle, cr_cv_angle)
                    # range
                    cr_range_v = cr_max_v - cr_min_v
                    cr_range_a = cr_max_a - cr_min_a
                    cr_range_dist = cr_max_dist - cr_min_dist
                    cr_range_angle = cr_max_angle - cr_min_angle
                    crat_range_v = np.append(crat_range_v, cr_range_v)
                    crat_range_a = np.append(crat_range_a, cr_range_a)
                    crat_range_dist = np.append(crat_range_dist, cr_range_dist)
                    crat_range_angle = np.append(crat_range_angle, cr_range_angle)
                    # interquartile range (iqr): Q3 -Q1
                    cr_iqr_v = stats.iqr(cr_vid['cr_v'].values, nan_policy='omit')
                    cr_iqr_a = stats.iqr(cr_vid['cr_a'].values, nan_policy='omit')
                    cr_iqr_dist = stats.iqr(cr_vid['cr_dist'], nan_policy='omit')
                    cr_iqr_angle = stats.iqr(cr_vid['cr_angle'], nan_policy='omit')
                    crat_iqr_v = np.append(crat_iqr_v, cr_iqr_v)
                    crat_iqr_a = np.append(crat_iqr_a, cr_iqr_a)
                    crat_iqr_dist = np.append(crat_iqr_dist, cr_iqr_dist)
                    crat_iqr_angle = np.append(crat_iqr_angle, cr_iqr_angle)
                    # skewness
                    cr_skew_v = cr_vid['cr_v'].skew(skipna=True)
                    cr_skew_a = cr_vid['cr_a'].skew(skipna=True)
                    cr_skew_dist = cr_vid['cr_dist'].skew(skipna=True)
                    cr_skew_angle = cr_vid['cr_angle'].skew(skipna=True)
                    crat_skew_v = np.append(crat_skew_v, cr_skew_v)
                    crat_skew_a = np.append(crat_skew_a, cr_skew_a)
                    crat_skew_dist = np.append(crat_skew_dist, cr_skew_dist)
                    crat_skew_angle = np.append(crat_skew_angle, cr_skew_angle)
                    # kurtosis
                    cr_kurtosis_v = cr_vid['cr_v'].kurtosis(skipna=True)
                    cr_kurtosis_a = cr_vid['cr_a'].kurtosis(skipna=True)
                    cr_kurtosis_dist = cr_vid['cr_dist'].kurtosis(skipna=True)
                    cr_kurtosis_angle = cr_vid['cr_angle'].kurtosis(skipna=True)
                    crat_kurtosis_v = np.append(crat_kurtosis_v, cr_kurtosis_v)
                    crat_kurtosis_a = np.append(crat_kurtosis_a, cr_kurtosis_a)
                    crat_kurtosis_dist = np.append(crat_kurtosis_dist, cr_kurtosis_dist)
                    crat_kurtosis_angle = np.append(crat_kurtosis_angle, cr_kurtosis_angle)
                    #AUC - use trapz or sum? ~ same results
                    cr_auc_v = np.trapz(cr_vid['cr_v'].dropna(), dx=1/cr_fs)
                    cr_auc_a = np.trapz(cr_vid['cr_a'].dropna(), dx=1/cr_fs)
                    cr_auc_dist = np.trapz(cr_vid['cr_dist'][~np.isnan(cr_vid['cr_dist'])], dx=1/cr_fs)
                    cr_auc_angle = np.trapz(cr_vid['cr_angle'][~np.isnan(cr_vid['cr_angle'])], dx=1/cr_fs)
                    crat_auc_v = np.append(crat_auc_v, cr_auc_v)
                    crat_auc_a = np.append(crat_auc_a, cr_auc_a)
                    crat_auc_dist = np.append(crat_auc_dist, cr_auc_dist)
                    crat_auc_angle = np.append(crat_auc_angle, cr_auc_angle)
                    # Cumulative power (cp)
                    # OK to drop nans? or would be better to interpolate?
                    f_v, psd_v = signal.welch(cr_vid['cr_v'].dropna(), cr_fs)
                    f_a, psd_a = signal.welch(cr_vid['cr_a'].dropna(), cr_fs)
                    f_dist, psd_dist = signal.welch(cr_vid['cr_dist'][~np.isnan(cr_vid['cr_dist'])], cr_fs)
                    f_angle, psd_angle = signal.welch(cr_vid['cr_angle'][~np.isnan(cr_vid['cr_angle'])], cr_fs)
                    if plot:
                        plt.figure()
                        plt.semilogy(f_v, psd_v, label='valence')
                        plt.semilogy(f_a, psd_a, label='arousal')
                        plt.semilogy(f_dist, psd_dist, label='distance')
                        plt.semilogy(f_angle, psd_angle, label='angle')
                        plt.legend()
                        plt.xlabel('frequency [Hz]')
                        plt.ylabel('PSD [V**2/Hz]')
                        plt.title('SJ' + sj_id + ' ' + rat_mtd + ' ' + vid)
                        fig_path = sj_path + 'figures/'
                        if not os.path.exists(fig_path):
                            os.makedirs(fig_path)
                        fig_name = 'SJ' + sj_id + '_' + rat_mtd + '_' + vid + '_VA_psd.png'
                        plt.savefig(fig_path + fig_name)
                        plt.close()
                    cr_cp_v = np.trapz(psd_v, f_v, dx=f_v[1])
                    cr_cp_a = np.trapz(psd_a, f_a, dx=f_a[1])
                    cr_cp_dist = np.trapz(psd_dist, f_dist, dx=f_dist[1])
                    cr_cp_angle = np.trapz(psd_angle, f_angle, dx=f_angle[1])
                    crat_cp_v = np.append(crat_cp_v, cr_cp_v)
                    crat_cp_a = np.append(crat_cp_a, cr_cp_a)
                    crat_cp_dist = np.append(crat_cp_dist, cr_cp_dist)
                    crat_cp_angle = np.append(crat_cp_angle, cr_cp_angle)
    print("SJ" + sj_id + " done")

# create dataframe with all CR indices
d = {'sj_id': sub, 'test_site': site, 'rating_method': rat_m, 'quadrant': quad,
     'sr_v': srat_v, 'cr_last_v': crat_last_v, 'cr_mean_v': crat_mean_v, 'cr_median_v': crat_median_v, 'cr_mode_v': crat_mode_v, 'cr_max_v': crat_max_v, 'cr_min_v': crat_min_v,
     'cr_std_v': crat_std_v, 'cr_cv_v': crat_cv_v, 'cr_range_v': crat_range_v, 'cr_iqr_v': crat_iqr_v, 'cr_skew_v': crat_skew_v, 'cr_kurtosis_v': crat_kurtosis_v, 'cr_auc_v': crat_auc_v, 'cr_cp_v': crat_cp_v,
     'sr_a': srat_a, 'cr_last_a': crat_last_a, 'cr_mean_a': crat_mean_a, 'cr_median_a': crat_median_a, 'cr_mode_a': crat_mode_a, 'cr_max_a': crat_max_a, 'cr_min_a': crat_min_a,
     'cr_std_a': crat_std_a, 'cr_cv_a': crat_cv_a, 'cr_range_a': crat_range_a, 'cr_iqr_a': crat_iqr_a, 'cr_skew_a': crat_skew_a, 'cr_kurtosis_a': crat_kurtosis_a, 'cr_auc_a': crat_auc_a, 'cr_cp_a': crat_cp_a,
     'sr_dist': srat_dist, 'cr_last_dist': crat_last_dist, 'cr_mean_dist': crat_mean_dist, 'cr_median_dist': crat_median_dist, 'cr_mode_dist': crat_mode_dist, 'cr_max_dist': crat_max_dist, 'cr_min_dist': crat_min_dist,
     'cr_std_dist': crat_std_dist, 'cr_cv_dist': crat_cv_dist, 'cr_range_dist': crat_range_dist, 'cr_iqr_dist': crat_iqr_dist, 'cr_skew_dist': crat_skew_dist, 'cr_kurtosis_dist': crat_kurtosis_dist, 'cr_auc_dist': crat_auc_dist, 'cr_cp_dist': crat_cp_dist,
     'sr_angle': srat_angle, 'cr_last_angle': crat_last_angle, 'cr_mean_angle': crat_mean_angle, 'cr_median_angle': crat_median_angle, 'cr_mode_angle': crat_mode_angle, 'cr_max_angle': crat_max_angle, 'cr_min_angle': crat_min_angle,
     'cr_std_angle': crat_std_angle, 'cr_cv_angle': crat_cv_angle, 'cr_range_angle': crat_range_angle, 'cr_iqr_angle': crat_iqr_angle, 'cr_skew_angle': crat_skew_angle, 'cr_kurtosis_angle': crat_kurtosis_angle, 'cr_auc_angle': crat_auc_angle, 'cr_cp_angle': crat_cp_angle
     }

filename = data_path + 'cr_sr_clean.csv'
df = pd.DataFrame(data=d)
# save dataframe
df.to_csv(filename, na_rep='NaN', index=False)
