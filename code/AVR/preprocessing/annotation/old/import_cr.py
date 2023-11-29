########################################################################################################################
# Script to import CR data from AVR experiment
# Preprocessing steps (optional): resampling and cut first seconds of data
# Output: csv file (dataframe)
# Author: Antonin Fourcade
# Last version: 15.08.2023
########################################################################################################################

# import packages
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

# Set paths and experiment parameters
data_path = '/Users/Lucy/Documents/Berlin/FU/MCNB/Praktikum/MPI_MBE/AffectiveVR/Phase1/AffectiveVR/data/'
blocks = ['Practice', 'Experiment', 'Assessment']
logging_freq = ['CR', 'SR']
test_site = ['Torino', 'Berlin']  # Torino (even SJ nb) = 0, Berlin (odd SJ nb) = 1
rating_method = ['Grid', 'Flubber', 'Proprioceptive', 'Baseline']
quadrant = ['HP', 'LP', 'LN', 'HN']
questionnaire = ['SUS', 'invasive_presence', 'Kunin']
cr_fs = 1/0.05  # sampling frequency CR in Hz
vid_len = 60  # length of the videos in s
sess = 'S000'  # session of recording
resample = True  # resample CR to cr_fs (if samples are not even)
debug = False  # debug mode
clean = True  # clean CR: remove first clean_s seconds of CR
clean_s = 5  # seconds to remove from CR

# get list of participants
sj_list = os.listdir(data_path + 'AVR/')
# delete the DS_Store file (I cannot see it in the folder, but it appears in the table for some reason)
if '.DS_Store' in sj_list:
    sj_list.remove('.DS_Store')
# sort participants in ascending order
sj_list.sort()
# exclude participants because no Grid CR log recorded
to_be_removed = {'01', '02', '03', '04', '05', '07', '09', '11', '13', '15', '17'}
sj_list = [item for item in sj_list if item not in to_be_removed]

# Initialize variables
sub = []
site = []
rat_m = []
quad = []
cr_v = []
cr_a = []
cr_dist = []
cr_angle = []
cr_time = []
cr_time_diff = []
nb_samples = []
sub2 = []
rat_m2 = []
quad2 = []

if debug:
    rh_time_diff = []
    sj_id = '19'
    rat_mtd = 'Flubber'
    vid = 'LP'

# Read and preprocess data
for sj, sj_id in enumerate(sj_list):
    # Read results in csv file
    sj_path = data_path + 'AVR/' + sj_id + '/' + sess + '/'
    trial_results_filename = sj_path + 'trial_results.csv'
    trials_results = pd.read_csv(trial_results_filename)

    # Loop over rating methods
    for rm, rat_mtd in enumerate(rating_method):
        # Select data according to rat_mtd
        trials_rat_mtd = trials_results[trials_results['method_descriptor'] == rat_mtd]
        # Select Experiment data
        trials_exp = trials_rat_mtd[trials_rat_mtd['block_name'] == 'Experiment']

        # Loop over quadrants/videos
        for v, vid in enumerate(quadrant):
            # if debug, check the framerate of right hand tracking
            if debug:
                rh_loc = 'righthand_movement_location_0'
                rh_path = data_path + trials_exp.loc[trials_exp['quadrant_descriptor'] == vid, rh_loc].item()
                rh = pd.read_csv(rh_path)
                rh_t = rh['time'].values
                rh_t_diff = np.append(np.nan, np.diff(rh_t))
                rh_time_diff = np.append(rh_time_diff, rh_t_diff)

            # get CR from file
            # rating methods other than Baseline
            if rat_mtd != 'Baseline':
                # location of CR file
                cr_loc = 'rating_' + rat_mtd + '_CR_location_0'
                cr_path = data_path + trials_exp.loc[trials_exp['quadrant_descriptor'] == vid, cr_loc].item()
                cr = pd.read_csv(cr_path)  # load CR file
                # rescale time from 0 to end of video
                cr['time'] = cr['time'] - cr['time'][0]
                # a little bit of cleaning name of column
                cr.rename(columns={'arousal ': 'arousal'}, inplace=True)
                cr_val = cr['valence'].values  # valence
                cr_aro = cr['arousal'].values  # arousal
                cr_d = np.hypot(cr_val, cr_aro)  # distance
                cr_ang = np.arctan2(cr_aro, cr_val)  # angle
                nb_cr = cr.__len__() # number of samples
                cr_t = cr['time'].values # times of samples
                if resample:
                    # resample cr to cr_fs (if samples are not even)
                    cr_t = np.arange(1/cr_fs, vid_len + 1/cr_fs, 1/cr_fs)
                    cr_v_interp = interp1d(cr['time'], cr_val, kind='linear', bounds_error=False,
                                           fill_value=(cr_val[0], cr_val[-1]))
                    cr_a_interp = interp1d(cr['time'], cr_aro, kind='linear', bounds_error=False,
                                           fill_value=(cr_aro[0], cr_aro[-1]))
                    cr_d_interp = interp1d(cr['time'], cr_d, kind='linear', bounds_error=False,
                                           fill_value=(cr_d[0], cr_d[-1]))
                    cr_ang_interp = interp1d(cr['time'], cr_ang, kind='linear', bounds_error=False,
                                             fill_value=(cr_ang[0], cr_ang[-1]))
                    cr_val = cr_v_interp(cr_t)
                    cr_aro = cr_a_interp(cr_t)
                    cr_d = cr_d_interp(cr_t)
                    cr_ang = cr_ang_interp(cr_t)
                    nb_cr = cr_t.__len__()
                    # debug plot
                    if debug:
                        plt.figure
                        plt.plot(cr['time'], cr['arousal'].values)
                        plt.plot(cr_t, cr_aro, 'r--')
                        plt.show()

                # compute time difference between samples
                cr_t_diff = np.append(np.nan, np.diff(cr_t))

                # save CR data of this trial in variables
                sub = np.append(sub, np.tile(sj_id, nb_cr))
                rat_m = np.append(rat_m, np.tile(rat_mtd, nb_cr))
                quad = np.append(quad, np.tile(vid, nb_cr))
                # Sjs with even number are in the first testing site, odd in the second
                if int(sj_id) % 2 == 0:
                    site = np.append(site, np.tile(test_site[0], nb_cr))
                else:
                    site = np.append(site, np.tile(test_site[1], nb_cr))

                sub2 = np.append(sub2, sj_id)
                rat_m2 = np.append(rat_m2, rat_mtd)
                quad2 = np.append(quad2, vid)
                cr_v = np.append(cr_v, cr_val)
                cr_a = np.append(cr_a, cr_aro)
                cr_dist = np.append(cr_dist, cr_d)
                cr_angle = np.append(cr_angle, cr_ang)
                cr_time = np.append(cr_time, cr_t)
                cr_time_diff = np.append(cr_time_diff, cr_t_diff)
                nb_samples = np.append(nb_samples, nb_cr)
    print("SJ" + sj_id + " done")  # show progress

# create dataframe with CR data
d_cr = {'sj_id': sub, 'test_site': site, 'rating_method': rat_m, 'quadrant': quad,
        'cr_v': cr_v, 'cr_a': cr_a, 'cr_dist': cr_dist, 'cr_angle': cr_angle, 'cr_time': cr_time
        }
df_cr = pd.DataFrame(data=d_cr)

# cut first seconds of data
if clean:
    df_cr = df_cr[df_cr['cr_time'] >= clean_s]

# save dataframe in csv file
if clean and resample:
    filename_cr = data_path + 'cr_rs_clean.csv'
    filename_nbs = data_path + 'cr_rs_clean_nb_samples.csv'
elif resample:
    filename_cr = data_path + 'cr_rs.csv'
    filename_nbs = data_path + 'cr_rs_nb_samples.csv'
elif clean:
    filename_cr = data_path + 'cr_clean.csv'
    filename_nbs = data_path + 'cr_clean_nb_samples.csv'
else:
    filename_cr = data_path + 'cr.csv'
    filename_nbs = data_path + 'cr_nb_samples.csv'
df_cr.to_csv(filename_cr, na_rep='NaN', index=False)

# create and save dataframe with number of samples per trial
d_nbs = {'sj_id': sub2, 'rating_method': rat_m2, 'quadrant': quad2, 'nb_samples': nb_samples}
df_nbs = pd.DataFrame(data=d_nbs)
df_nbs.to_csv(filename_nbs, na_rep='NaN', index=False)

# check time difference between samples (framerate)
plt.figure()
plt.hist(cr_time_diff, bins=50)
plt.show()
max_t_diff = np.nanmax(cr_time_diff)
min_t_diff = np.nanmin(cr_time_diff)
median_t_diff = np.nanmedian(cr_time_diff)
median_fs = 1/median_t_diff

if debug:
    # check the framerate of right hand tracking
    plt.figure()
    plt.hist(rh_time_diff, bins=500)
    plt.show()
    max_t_diff = np.nanmax(rh_time_diff)
    min_t_diff = np.nanmin(rh_time_diff)
    median_t_diff = np.nanmedian(rh_time_diff)
    median_fs = 1 / median_t_diff
    