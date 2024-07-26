import mne
import scipy
import glob
import numpy as np
import pandas as pd
import fooof
from scipy.stats import zscore
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#for debugging
mov_cond = 'mov'
c_style = 'SBA'
bands = fooof.bands.Bands({'delta': [0.3, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 45]})
freqs = np.arange(0.5, 125.5, 0.5)  # resolution 0.5Hz
times = np.arange(1, 270, 1)  # resolution 1s
fs = 250
len_win = 2  # 2s window
overlap = 0.5  # 50% overlap
method_tfr = 'cwt'
method_int = 'trapezoid'
pow_style = 'full'
roi = ['Pz', 'P3', 'P4', 'P7', 'P8', 'O1', 'O2', 'Oz']
isub = 0
clean = 1
clean_nb_s = 20
mirror_len = 80
mirror_break = 30
cut = 35

def eeg_pow_extract(mov_cond, c_style, fs, bands, freqs, len_win, overlap, method_tfr='cwt', method_int='trapezoid', pow_style ='full', mirror_len=80, mirror_break=30, cut=35):
    mon_path = 'E:/backup/NeVRo/Nevro_montage.loc'
    path_data = 'E:/NeVRo/new_HEP_data_filtHP_0_3Hz/'
    path_in_eeg = path_data + '13_eeglab2mne/' + mov_cond + '/' + c_style + '/'
    path_out_pwr = path_data + 'Frequency_Bands_Power/' + pow_style + '/newEEG_pow_' + method_tfr + '/' + mov_cond + '/' + c_style + '/'
    path_out_pwr_roi = path_data + 'Frequency_Bands_Power/' + pow_style + '/newROI_EEGpow_' + method_tfr + '/' + mov_cond + '/' + c_style + '/'
    files_eeg = glob.glob(path_in_eeg + '*.set')

    # For formatting the EEG file correctly (add montage of electrodes)
    hpf_freq = 0.3
    lpf_freq = 45

    # set tfr parameters
    win = len_win * fs
    noverlap = int(win * overlap)

    # set FOOOF parameters
    fooof_freq_range = [1, 40]
    peak_width_limits = [1.0, 12.0]
    min_peak_height = 0
    peak_threshold = 2.0

    for isub, filename_eeg in enumerate(files_eeg):
        # Import preprocessed EEG from EEGLAB
        montage = mne.channels.read_custom_montage(mon_path)
        ch_types = ['eeg'] * 33
        ch_types[4] = 'eog'
        ch_types[26] = 'eog'
        ch_types[32] = 'ecg'
        info_eeg = mne.create_info(ch_names=montage.ch_names, sfreq=fs, ch_types=ch_types)
        channel_map = {A: B for A, B in zip(montage.ch_names, ch_types)}
        preprocessed = mne.io.read_raw_eeglab(filename_eeg, eog=['HEOG', 'VEOG'], preload=True, verbose=False, uint16_codec=None)
        preprocessed.info = info_eeg
        #preprocessed.info['highpass'] = hpf_freq  # eeglab data already hp-filtered
        preprocessed.set_channel_types(channel_map)
        preprocessed.set_montage(montage)

        # Low-pass Filtering 45Hz
        preproc_filtered = preprocessed.filter(l_freq=hpf_freq, h_freq=lpf_freq, verbose=False)

        # pick EEG channels only and get just the data
        eeg = preproc_filtered.pick_types(eeg=True)
        eeg_data = eeg.get_data()
        # separate S,B,A sections in the data
        eeg_data_S = eeg_data[:, 0:148*fs]
        eeg_data_B = eeg_data[:, 148*fs:178*fs]
        eeg_data_A = eeg_data[:, 178*fs:]
        # Mirror data at beginning and end, symmetric padding
        eeg_data_S_mirrored = np.empty((eeg_data_S.shape[0], eeg_data_S.shape[1] + 2 * mirror_len * fs))
        eeg_data_B_mirrored = np.empty((eeg_data_B.shape[0], eeg_data_B.shape[1] + 2 * mirror_break * fs))
        eeg_data_A_mirrored = np.empty((eeg_data_A.shape[0], eeg_data_A.shape[1] + 2 * mirror_len * fs))
        for e in range(0, eeg_data.shape[0]):
            eeg_data_S_mirrored[e, :] = np.hstack(
                (np.flip(eeg_data_S[e])[-mirror_len * fs:], eeg_data_S[e], np.flip(eeg_data_S[e][-mirror_len * fs:])))
            eeg_data_B_mirrored[e, :] = np.hstack(
                (np.flip(eeg_data_B[e])[-mirror_break * fs:], eeg_data_B[e], np.flip(eeg_data_B[e][-mirror_break * fs:])))
            eeg_data_A_mirrored[e, :] = np.hstack(
                (np.flip(eeg_data_A[e])[-mirror_len * fs:], eeg_data_A[e], np.flip(eeg_data_A[e][-mirror_len * fs:])))

        # depending on tfr method
        if method_tfr == 'stft':
            #TODO: Adapt code like for CWT
            freqs, times, PSDs = scipy.signal.spectrogram(eeg_data, fs=fs, window='hann', nperseg=win, noverlap=noverlap,
                                                          nfft=None, detrend=False, return_onesided=True,
                                                          scaling='density', axis=- 1, mode='psd')
            # Output units [V**2/Hz]
            # scipy.signal.spectrogram returns values for f=0
            # Exclude f = 0 Hz (first freq) because it can cause pbs with other functions
            freqs = freqs[1::]
            PSDs = PSDs[:, 1::, :]

        elif method_tfr == 'cwt':
            # CWT for each S, B, A sections separately
            Ws = mne.time_frequency.morlet(fs, freqs, n_cycles=7.0, sigma=None, zero_mean=False)
            tfr_S = mne.time_frequency.tfr.cwt(eeg_data_S_mirrored, Ws, use_fft=True, mode='same', decim=1)
            tfr_S_pow = (np.abs(tfr_S)) ** 2
            tfr_B = mne.time_frequency.tfr.cwt(eeg_data_B_mirrored, Ws, use_fft=True, mode='same', decim=1)
            tfr_B_pow = (np.abs(tfr_B)) ** 2
            tfr_A = mne.time_frequency.tfr.cwt(eeg_data_A_mirrored, Ws, use_fft=True, mode='same', decim=1)
            tfr_A_pow = (np.abs(tfr_A)) ** 2

            # Cut the TFR and stack it together again
            tfr_S_pow = tfr_S_pow[:, :, 0:mirror_len * fs + eeg_data_S.shape[1]]
            tfr_B_pow = tfr_B_pow[:, :, mirror_break * fs:mirror_break * fs + eeg_data_B.shape[1]]
            tfr_A_pow = tfr_A_pow[:, :, mirror_len * fs:2 * mirror_len * fs + eeg_data_A.shape[1]]
            tfr_pow_mirrored = np.concatenate((tfr_S_pow, tfr_B_pow, tfr_A_pow), axis=2)

            # Smoothing: Average over 2s with 50% overlap
            PSDs = np.empty(
                (tfr_pow_mirrored.shape[0], tfr_pow_mirrored.shape[1], int(tfr_pow_mirrored.shape[2] / fs) - 1))
            for e in range(0, tfr_pow_mirrored.shape[0]):
                for f in range(0, tfr_pow_mirrored.shape[1]):
                    window_avg = [np.mean(tfr_pow_mirrored[e, f, i:i + win]) for i in
                                  range(0, len(tfr_pow_mirrored[e, f]), noverlap)
                                  if i + win <= len(tfr_pow_mirrored[e, f])]
                    PSDs[e, f] = np.asarray(window_avg)

        # FOOOF Modeling - or not
        # Initialize power variables
        if pow_style == 'flat':
            PSDs_flat = np.empty(PSDs.shape)

        power_delta = np.empty((PSDs.shape[2], PSDs.shape[0],))
        power_theta = np.empty((PSDs.shape[2], PSDs.shape[0],))
        power_alpha = np.empty((PSDs.shape[2], PSDs.shape[0],))
        power_beta = np.empty((PSDs.shape[2], PSDs.shape[0],))
        power_gamma = np.empty((PSDs.shape[2], PSDs.shape[0],))

        for t in range(0, PSDs.shape[2]):
            PSD = PSDs[:, :, t]
            if pow_style == 'flat':  # if regress out EEG aperiodic component
                #TODO: Adapt code for mirrored data

                # FOOOF fitting, for each times
                fg = fooof.FOOOFGroup(peak_width_limits=peak_width_limits, min_peak_height=min_peak_height, peak_threshold=peak_threshold,
                                      aperiodic_mode='fixed')
                fg.fit(freqs, PSD, freq_range=fooof_freq_range)
                # get the aperiodic parameters
                exps = fg.get_params('aperiodic_params', 'exponent')
                offs = fg.get_params('aperiodic_params', 'offset')
                # regress out the aperiodic component from the PSD
                PSDs_flat[:, :, t] = PSD - aperiodic(freqs, exps, offs)
                # Integrate flat PSD over the different frequency bands
                power_delta[t] = integrate_pow(freqs, PSDs_flat[:, :, t], bands.delta, method_int)
                power_theta[t] = integrate_pow(freqs, PSDs_flat[:, :, t], bands.theta, method_int)
                power_alpha[t] = integrate_pow(freqs, PSDs_flat[:, :, t], bands.alpha, method_int)
                power_beta[t] = integrate_pow(freqs, PSDs_flat[:, :, t], bands.beta, method_int)
                power_gamma[t] = integrate_pow(freqs, PSDs_flat[:, :, t], bands.gamma, method_int)
            elif pow_style == 'full':
                # Integrate full PSD over the different frequency bands
                power_delta[t] = integrate_pow(freqs, PSD, bands.delta, method_int)
                power_theta[t] = integrate_pow(freqs, PSD, bands.theta, method_int)
                power_alpha[t] = integrate_pow(freqs, PSD, bands.alpha, method_int)
                power_beta[t] = integrate_pow(freqs, PSD, bands.beta, method_int)
                power_gamma[t] = integrate_pow(freqs, PSD, bands.gamma, method_int)

        # Cut mirrored data +/- cut s (data w/o artifacts, similar to ECG pipeline)
        power_delta_mirrored = power_delta[mirror_len - 1 - cut:270 + mirror_len - 1 + cut]
        power_theta_mirrored = power_theta[mirror_len - 1 - cut:270 + mirror_len - 1 + cut]
        power_alpha_mirrored = power_alpha[mirror_len - 1 - cut:270 + mirror_len - 1 + cut]
        power_beta_mirrored = power_beta[mirror_len - 1 - cut:270 + mirror_len - 1 + cut]
        power_gamma_mirrored = power_gamma[mirror_len - 1 - cut:270 + mirror_len - 1 + cut]
        times_mirrored = np.arange(0, power_delta_mirrored.shape[0]) - cut

        # Save frequency bands' powers into mat files
        filename_pwr = filename_eeg.rsplit('\\')[1].rsplit('_')[1] + '_eeg_pow_' + pow_style + '.mat'
        d = {'times': times_mirrored, 'chans': eeg.ch_names, 'delta_power_mirrored': power_delta_mirrored,
             'theta_power_mirrored': power_theta_mirrored,
             'alpha_power_mirrored': power_alpha_mirrored,
             'beta_power_mirrored': power_beta_mirrored, 'gamma_power_mirrored': power_gamma_mirrored}
        savemat(path_out_pwr + filename_pwr, d)
        print(filename_eeg.rsplit('\\')[1].rsplit('_')[1] + ' done ...')

        # Average over ROI
        idx_roi = np.isin(eeg.ch_names, roi)
        roi_delta = power_delta_mirrored[:, idx_roi]
        roi_theta = power_theta_mirrored[:, idx_roi]
        roi_alpha = power_alpha_mirrored[:, idx_roi]
        roi_beta = power_beta_mirrored[:, idx_roi]
        roi_gamma = power_gamma_mirrored[:, idx_roi]
        delta_mean_roi = np.mean(roi_delta, 1)
        theta_mean_roi = np.mean(roi_theta, 1)
        alpha_mean_roi = np.mean(roi_alpha, 1)
        beta_mean_roi = np.mean(roi_beta, 1)
        gamma_mean_roi = np.mean(roi_gamma, 1)

        # Save ROI frequency bands' powers into mat files
        filename_pwr_roi = filename_eeg.rsplit('\\')[1].rsplit('_')[1] + '_eeg_pow_roi_mirrored_' + pow_style + '.mat'
        d = {'times': times_mirrored, 'roi': roi, 'delta_power_roi_mirrored': delta_mean_roi,
             'theta_power_roi_mirrored': theta_mean_roi,
             'alpha_power_roi_mirrored': alpha_mean_roi,
             'beta_power_roi_mirrored': beta_mean_roi, 'gamma_power_roi_mirrored': gamma_mean_roi}
        savemat(path_out_pwr_roi + filename_pwr_roi, d)
        print(filename_eeg.rsplit('\\')[1].rsplit('_')[1] + ' done ...')

def castToList(x):
    # Casts x to a list
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    try:
        return list(x)
    except TypeError:
        return [x]

def aperiodic_log(frs, exps, offs, knees=np.zeros((30, 250))):
    # Compute log10 of aperiodic component for all electrodes (nb_ch PSDs)
    # return.shape = 2D (nb_ch, nb_fr)
    nb_ch = exps.shape[0]
    nb_fr = frs.shape[0]
    freqs = np.vstack([frs] * nb_ch)  # make frs same shape as output
    offsets = np.vstack([offs] * nb_fr).T  # make offs same shape as output
    frs_exp = np.zeros((nb_ch, nb_fr))
    for i, exp in enumerate(exps):
        frs_exp[i] = freqs[i] ** exp
    return offsets - np.log10(knees + frs_exp)


def aperio(frs, expo, off, knee=0):  # return.shape = 1D (nb_fr)
    # Compute aperiodic component for one electrode (one PSD)
    return (10 ** off) * (1/(knee + frs ** expo))


def aperiodic(frs, exps, offs, knee=0):
    # Compute aperiodic component for all electrodes (nb_ch PSDs)
    # return.shape = 2D (nb_ch, nb_fr)
    ap = np.empty((exps.shape[0], frs.shape[0]))
    for e, exp in enumerate(exps):
        ap[e] = aperio(frs, exps[e], offs[e], knee)
    return ap


def integrate_pow(frs, PSD, band, method='trapezoid'):
    # Integrate PSD over frequency band, using trapezoid rule
    # PSD shape: (nb_ch, nb_fr)
    power = np.zeros(PSD.shape[0])
    frs_band = frs[np.logical_and(frs >= band[0], frs <= band[1])]
    for i in range(PSD.shape[0]):
        psd = PSD[i]
        psd_band = psd[np.logical_and(frs >= band[0], frs <= band[1])]
        if method == 'trapezoid':
            # The trapezoidal rule approximates the function as a straight line between adjacent points
            power[i] = scipy.integrate.trapezoid(psd_band, frs_band)
        elif method == 'simpson':
            power[i] = scipy.integrate.simpson(psd_band, frs_band)
    return power


def NVR_EEG_pow(mov_cond, c_style, pow_style, method_tfr, roi, clean=0, clean_nb_s=20):
    # 1.1 Set different paths:
    # input paths:
    path_data = 'E:/NeVRo/new_HEP_data_filtHP_0_3Hz/'
    path_in_eeg_pow = path_data + 'Frequency_Bands_Power/' + pow_style + '/newROI_EEGpow_' + method_tfr + '/' + mov_cond + '/' + c_style + '/'
    path_in_ar = path_data + 'ratings/class_bins/' + mov_cond + '/' + c_style + '/'

    # 1.2 Get data files
    files_eeg_pow = glob.glob(path_in_eeg_pow + '*.mat')
    files_ar = glob.glob(path_in_ar + '*.txt')

    # Initialize EEGpow_all
    EEGpow_all = []

    # Loop for all subjects
    for isub, file in enumerate(files_eeg_pow):
        # 1.3 Set filename:
        filename_eeg_pow = files_eeg_pow[isub]
        filename_ar = files_ar[isub]

        # 2. Import ECG and arousal ratings data
        eeg_pow = loadmat(filename_eeg_pow, simplify_cells=True)
        aro_rat = pd.read_csv(filename_ar, header=0, names=["latency", "class_aro"])

        # Get the different EEG metrics
        # only 269 values for powers, because 2s averaging window -> miss 1 sample at the end
        # -> add last value of times pow
        times_pow = np.append(eeg_pow['times'], 270)
        if pow_style == 'full':
            delta_meanROI = eeg_pow['delta_power_roi_mirrored'].T
            theta_meanROI = eeg_pow['theta_power_roi_mirrored'].T
            alpha_meanROI = eeg_pow['alpha_power_roi_mirrored'].T
            beta_meanROI = eeg_pow['beta_power_roi_mirrored'].T
            gamma_meanROI = eeg_pow['gamma_power_roi_mirrored'].T
        elif pow_style == 'flat':
            delta_meanROI = eeg_pow['delta_power_roi_mirrored'].T - np.min(eeg_pow['delta_power_roi_mirrored'])
            theta_meanROI = eeg_pow['theta_power_roi_mirrored'].T - np.min(eeg_pow['theta_power_roi_mirrored'])
            alpha_meanROI = eeg_pow['alpha_power_roi_mirrored'].T - np.min(eeg_pow['alpha_power_roi_mirrored'])
            beta_meanROI = eeg_pow['beta_power_roi_mirrored'].T - np.min(eeg_pow['beta_power_roi_mirrored'])
            gamma_meanROI = eeg_pow['gamma_power_roi_mirrored'].T - np.min(eeg_pow['gamma_power_roi_mirrored'])

        # remove padding
        delta_meanROI = delta_meanROI[cut:270 + cut]
        theta_meanROI = theta_meanROI[cut:270 + cut]
        alpha_meanROI = alpha_meanROI[cut:270 + cut]
        beta_meanROI = beta_meanROI[cut:270 + cut]
        gamma_meanROI = gamma_meanROI[cut:270 + cut]

        # Clean data without edge(beginning and end)  and transition (break) artifacts
        # Select data at middle of the rollercoasters +/- clean_nb_s
        # length Space (S) = 148 s, length Andes (A) = 92 s
        m_S = 74
        m_A = 46 + 148 + 30  # SBA
        if clean == 1:
            aro_rat = aro_rat[m_S - clean_nb_s:m_S + clean_nb_s].append(aro_rat[m_A - clean_nb_s:m_A + clean_nb_s])
            times_pow = np.append(times_pow[m_S - clean_nb_s:m_S + clean_nb_s],
                                  times_pow[m_A - clean_nb_s:m_A + clean_nb_s])
            delta_meanROI = np.append(delta_meanROI[m_S - clean_nb_s:m_S + clean_nb_s],
                                      delta_meanROI[m_A - clean_nb_s:m_A + clean_nb_s])
            theta_meanROI = np.append(theta_meanROI[m_S - clean_nb_s:m_S + clean_nb_s],
                                      theta_meanROI[m_A - clean_nb_s:m_A + clean_nb_s])
            alpha_meanROI = np.append(alpha_meanROI[m_S - clean_nb_s:m_S + clean_nb_s],
                                      alpha_meanROI[m_A - clean_nb_s:m_A + clean_nb_s])
            beta_meanROI = np.append(beta_meanROI[m_S - clean_nb_s:m_S + clean_nb_s],
                                     beta_meanROI[m_A - clean_nb_s:m_A + clean_nb_s])
            gamma_meanROI = np.append(gamma_meanROI[m_S - clean_nb_s:m_S + clean_nb_s],
                                      gamma_meanROI[m_A - clean_nb_s:m_A + clean_nb_s])

        # Z-score the time-series
        z_delta_meanROI = zscore(delta_meanROI, nan_policy='omit')
        z_theta_meanROI = zscore(theta_meanROI, nan_policy='omit')
        z_alpha_meanROI = zscore(alpha_meanROI, nan_policy='omit')
        z_beta_meanROI = zscore(beta_meanROI, nan_policy='omit')
        z_gamma_meanROI = zscore(gamma_meanROI, nan_policy='omit')

        # Select the HA and LA samples
        LA_delta_meanROI = delta_meanROI[aro_rat['class_aro'] == 1]
        LA_theta_meanROI = theta_meanROI[aro_rat['class_aro'] == 1]
        LA_alpha_meanROI = alpha_meanROI[aro_rat['class_aro'] == 1]
        LA_beta_meanROI = beta_meanROI[aro_rat['class_aro'] == 1]
        LA_gamma_meanROI = gamma_meanROI[aro_rat['class_aro'] == 1]

        HA_delta_meanROI = delta_meanROI[aro_rat['class_aro'] == 3]
        HA_theta_meanROI = theta_meanROI[aro_rat['class_aro'] == 3]
        HA_alpha_meanROI = alpha_meanROI[aro_rat['class_aro'] == 3]
        HA_beta_meanROI = beta_meanROI[aro_rat['class_aro'] == 3]
        HA_gamma_meanROI = gamma_meanROI[aro_rat['class_aro'] == 3]

        # Select the HA and LA samples from the z-scores
        LA_z_delta_meanROI = z_delta_meanROI[aro_rat['class_aro'] == 1]
        LA_z_theta_meanROI = z_theta_meanROI[aro_rat['class_aro'] == 1]
        LA_z_alpha_meanROI = z_alpha_meanROI[aro_rat['class_aro'] == 1]
        LA_z_beta_meanROI = z_beta_meanROI[aro_rat['class_aro'] == 1]
        LA_z_gamma_meanROI = z_gamma_meanROI[aro_rat['class_aro'] == 1]

        HA_z_delta_meanROI = z_delta_meanROI[aro_rat['class_aro'] == 3]
        HA_z_theta_meanROI = z_theta_meanROI[aro_rat['class_aro'] == 3]
        HA_z_alpha_meanROI = z_alpha_meanROI[aro_rat['class_aro'] == 3]
        HA_z_beta_meanROI = z_beta_meanROI[aro_rat['class_aro'] == 3]
        HA_z_gamma_meanROI = z_gamma_meanROI[aro_rat['class_aro'] == 3]

        # Store relevant variables in eeg_dict
        eeg_dict = {'filename': filename_eeg_pow,
                    'aro_ratings_bins': aro_rat['class_aro'].to_numpy(),
                    'aro_ratings_latency': aro_rat['latency'].to_numpy(),
                    'times': times_pow,
                    'roi': roi,
                    'delta_meanROI': delta_meanROI,
                    'theta_meanROI': theta_meanROI,
                    'alpha_meanROI': alpha_meanROI,
                    'beta_meanROI': beta_meanROI,
                    'gamma_meanROI': gamma_meanROI,
                    'z_delta_meanROI': z_delta_meanROI,
                    'z_theta_meanROI': z_theta_meanROI,
                    'z_alpha_meanROI': z_alpha_meanROI,
                    'z_beta_meanROI': z_beta_meanROI,
                    'z_gamma_meanROI': z_gamma_meanROI,
                    'LA_delta_meanROI': LA_delta_meanROI,
                    'LA_theta_meanROI': LA_theta_meanROI,
                    'LA_alpha_meanROI': LA_alpha_meanROI,
                    'LA_beta_meanROI': LA_beta_meanROI,
                    'LA_gamma_meanROI': LA_gamma_meanROI,
                    'HA_delta_meanROI': HA_delta_meanROI,
                    'HA_theta_meanROI': HA_theta_meanROI,
                    'HA_alpha_meanROI': HA_alpha_meanROI,
                    'HA_beta_meanROI': HA_beta_meanROI,
                    'HA_gamma_meanROI': HA_gamma_meanROI,
                    'LA_z_delta_meanROI': LA_z_delta_meanROI,
                    'LA_z_theta_meanROI': LA_z_theta_meanROI,
                    'LA_z_alpha_meanROI': LA_z_alpha_meanROI,
                    'LA_z_beta_meanROI': LA_z_beta_meanROI,
                    'LA_z_gamma_meanROI': LA_z_gamma_meanROI,
                    'HA_z_delta_meanROI': HA_z_delta_meanROI,
                    'HA_z_theta_meanROI': HA_z_theta_meanROI,
                    'HA_z_alpha_meanROI': HA_z_alpha_meanROI,
                    'HA_z_beta_meanROI': HA_z_beta_meanROI,
                    'HA_z_gamma_meanROI': HA_z_gamma_meanROI,
                    }
        EEGpow_all.append(eeg_dict)
        print(filename_eeg_pow.rsplit('\\')[1].rsplit('_')[0] + ' done ...')
    return EEGpow_all


