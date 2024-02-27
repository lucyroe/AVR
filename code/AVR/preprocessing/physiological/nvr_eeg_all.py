import mne
import scipy
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import fooof
from scipy.io import loadmat, savemat
import pandas as pd

from nvr_eeg_functions_new import eeg_pow_extract
from nvr_eeg_functions_new import eeg_roipow_extract
from nvr_eeg_functions_new import NVR_EEG_pow
from nvr_eeg_functions_new import castToList

# Define frequency bands of interest
# Candia-Rivera (2021): delta (δ; 0–4 Hz), theta (θ; 4–8 Hz), alpha (α; 8–12 Hz), beta (β; 12–ƒ30 Hz) and gamma (γ; 30–45 Hz).
bands = fooof.bands.Bands({'delta' : [0.3, 4],
                           'theta' : [4, 8],
                           'alpha' : [8, 13],
                           'beta' : [13, 30],
                           'gamma': [30, 45]})
m_conds = ['nomov', 'mov']
a_conds = ['LA', 'HA']
c_style = 'SBA'

# Define ROI
# roi = ['Pz', 'P3', 'P4', 'P7', 'P8', 'O1', 'O2', 'Oz']  # occipito-parietal
# roi = ['C3', 'FC1', 'FC5', 'Fz', 'F3']  # HEP Cluster
roi = ['C4', 'FC2', 'FC6', 'Fz', 'F4']  # contralateral HEP cluster

# parameters for tfr and power integration
fs = 250  # Hz
fs_final = 1  # Hz
len_win = 2  # 2s window
overlap = 0.5  # 50% overlap
method_tfr = 'cwt'  # cwt or stft
method_int = 'trapezoid' # trapezoid or simpson
freqs = np.arange(0.5, 125.5, 0.5)  # resolution 0.5Hz
times = np.arange(1, 270, 1/fs_final)  # resolution 1s
pow_style = 'full'  # 'full' or 'flat' if AP regressed out
mirror_len = 80*fs_final
mirror_break = 30*fs_final
cut = 35*fs_final

save_powdir = 'mir'
save_powpath = 'E:/NeVRo/new_HEP_data_filtHP_0_3Hz/Frequency_Bands_Power/' + pow_style + '/' + save_powdir + 'EEG_pow_' + method_tfr + '/'
# save_roidir = 'mirROI'
save_roidir = 'mircontraHEPROI'
save_roipath = 'E:/NeVRo/new_HEP_data_filtHP_0_3Hz/Frequency_Bands_Power/' + pow_style + '/' + save_roidir + '_EEGpow_' + method_tfr + '/'

# All mov are included in nomov, but nomov have more SJs
SJs_nomov = [2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 34, 36, 37, 39, 44]
SJs_mov = [2, 4, 5, 6, 8, 9, 11, 13, 14, 17, 18, 20, 21, 22, 24, 25, 27, 28, 29, 31, 34, 36, 37, 39, 44]

EEG_pow = {
    'mov': [],
    'nomov': []
}


# Get the EEG_pow parameters for each SJ and each mov condition
# RUN ONLY ONCE if CWT! (TAKES 2.5H)
# Get the frequency bands power
for mc, m_cond in enumerate(m_conds):
    eeg_pow_extract(m_cond, c_style, fs, bands, freqs, len_win, overlap, method_tfr, method_int, pow_style, mirror_len,
                    mirror_break, cut)

# Get the ROI mean frequency bands power
for mc, m_cond in enumerate(m_conds):
    eeg_roipow_extract(m_cond, c_style, save_powpath, save_roipath, roi)

# Loop over the movement conditions:
for mc, m_cond in enumerate(m_conds):
    EEG_pow[m_cond] = NVR_EEG_pow(save_roipath, m_cond, c_style, pow_style, roi, cut)

# Save EEG_pow into file
filename_roipwr = save_roidir + '_' + pow_style + '_' + method_tfr + '.mat'

savemat(save_roipath + filename_roipwr, EEG_pow)

# Prepare data for Mixed design on R
## 2 factors: Movement & Arousal
## 2 levels : mov-nomov (rows) & LA-HA (columns)

filename_roipwr = save_roidir + '_' + pow_style + '_' + method_tfr + '.mat'

EEG_pow = loadmat(save_roipath + filename_roipwr, simplify_cells=True)

design = []
for i, subj in enumerate(SJs_nomov):
    if subj not in SJs_mov:
        mov_conds = ['nomov']
    else:
        mov_conds = m_conds
    for m in mov_conds:
        if m == 'mov':
            sj_idx = SJs_mov.index(subj)
        else:
            sj_idx = i
        for a in a_conds:
            delta = castToList(EEG_pow[m][sj_idx][a + '_delta_meanROI'])
            theta = castToList(EEG_pow[m][sj_idx][a + '_theta_meanROI'])
            alpha = castToList(EEG_pow[m][sj_idx][a + '_alpha_meanROI'])
            beta = castToList(EEG_pow[m][sj_idx][a + '_beta_meanROI'])
            gamma = castToList(EEG_pow[m][sj_idx][a + '_gamma_meanROI'])
            #
            z_delta = castToList(EEG_pow[m][sj_idx][a + '_z_delta_meanROI'])
            z_theta = castToList(EEG_pow[m][sj_idx][a + '_z_theta_meanROI'])
            z_alpha = castToList(EEG_pow[m][sj_idx][a + '_z_alpha_meanROI'])
            z_beta = castToList(EEG_pow[m][sj_idx][a + '_z_beta_meanROI'])
            z_gamma = castToList(EEG_pow[m][sj_idx][a + '_z_gamma_meanROI'])
            for s, s_delta in enumerate(delta):
                design.append([subj, m, a,
                               delta[s], theta[s], alpha[s], beta[s], gamma[s],
                               z_delta[s], z_theta[s], z_alpha[s], z_beta[s], z_gamma[s]
                               ])

design_df = pd.DataFrame(design, columns=['SJ', 'mov_cond', 'arousal',
                                          'delta', 'theta', 'alpha', 'beta', 'gamma',
                                          'z_delta', 'z_theta', 'z_alpha', 'z_beta', 'z_gamma'
                                          ])

filename = save_roipath + 'eegpowlahamovnomov_' + method_tfr + '.csv'
design_df.to_csv(filename, index=False, na_rep='NaN')

