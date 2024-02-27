import glob
import os
import neurokit2 as nk
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import zscore
import matlab.engine

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

def LP(ibi):
    # Compute Local Power (LP) form the ibi time series
    # Adaptation Matlab code from
    data = np.empty((ibi.shape[0], 3))
    data[:, 2] = np.nan
    data[:, 1] = 0
    data[:, 0] = ibi
    # We go through the data and check for every IBI whether it was an increase
    # (mark: 1), or a decrease (mark: -1) as compared to the previous IBI
    for i in range(1, data.shape[0]):
        if data[i, 0] > data[i - 1, 0]:
            data[i, 1] = 1
        if data[i, 0] < data[i - 1, 0]:
            data[i, 1] = -1
    # We look for the first position where we can assign
    # an LP value, that is, the point where we have the first turn in
    # the IBI curve
    i = 1
    oldSign = data[1, 1]
    actSign = oldSign
    while oldSign == actSign:
        i = i + 1
        actSign = data[i, 1]
        data[:, 2] = np.nan
    # having reached the point where the first turn appears...
    startComputation = i + 1  # we save the following IBI as a starting point for the computation
    parcelStart = data[startComputation - 2, 0]
    # now we compute the distances from bottom to peak, from peak to bottom
    startPos = startComputation - 1
    for i in range(startComputation, data.shape[0]):
        if data[i, 1] != data[i - 1, 1]:
            parcelEnd = data[i - 1, 0]
            RSA = np.abs(parcelEnd - parcelStart)
            data[startPos:i + 1, 2] = RSA  # assign the RSA beside the lag in which the LP occured
            startPos = i
            parcelStart = parcelEnd
            parcelEnd = np.nan
    lp = data[:, 2]
    return lp

def r_peak_extract(m_cond, c_style, sampling_freq):
    mov_cond = m_cond
    fs = sampling_freq
    # 1.1 Set different paths:
    # input paths:
    path_data = 'D:/NeVRo/new_HEP_data_filtHP_0_3Hz/ECG/'
    path_in_ecg = path_data + mov_cond + '/' + c_style + '/'
    path_out_r_peaks = path_in_ecg + 'R-peak_time_series/'
    # 1.2 Get data files
    files_ecg = glob.glob(path_in_ecg + '*.txt')
    for isub, file in enumerate(files_ecg):
        filename_ecg = files_ecg[isub]
        ecg_signal = pd.read_csv(filename_ecg, header=0, names=["ECG"])
        # Process the ecg with Neurokit -> get the r_peaks
        ecg_processed, info = nk.ecg_process(ecg_signal["ECG"], sampling_rate=fs, method='neurokit')
        r_peaks = info["ECG_R_Peaks"] / fs
        filename_r_peaks = filename_ecg.rsplit('_')[6] + '_Rpeaks_' + mov_cond + '.txt'
        r_peaks.tofile(path_out_r_peaks + filename_r_peaks, sep='\n')

#for debug
# mov_cond = 'nomov'
# c_style = 'SBA'
# fs = 1
# lf = [0.04, 0.15]
# hf = [0.15, 0.4]
# isub = 3
# method = 'spwvd' # 'cwt'; 'spwvd'
# mirror_len = 80
# cut = 35
# fig = 1

def heart_extract(mov_cond, c_style='SBA', fs=1, method='cwt', lf=[0.04, 0.15], hf=[0.15, 0.4], mirror_len=80, cut=35, fig=0):
    # set paths
    path_data = 'E:/NeVRo/new_HEP_data_filtHP_0_3Hz/ECG/mov_nomov_ECG/'
    path_in_rpeak = path_data + 'R-peak_time_series/' + mov_cond + '/'
    path_out_rs_ibi = path_data + 'rs_IBI_mir/' + mov_cond + '/' + c_style + '/'
    path_out_ibi = path_data + 'IBI_mir/' + mov_cond + '/' + c_style + '/'
    path_out_pwr = path_data + 'HRV_mir_' + method + '/' + mov_cond + '/' + c_style + '/'
    files_rpeak = glob.glob(path_in_rpeak + '*.txt')

    # create directories if not exist
    if not os.path.exists(path_out_rs_ibi):
        os.makedirs(path_out_rs_ibi)
    if not os.path.exists(path_out_ibi):
        os.makedirs(path_out_ibi)
    if not os.path.exists(path_out_pwr):
        os.makedirs(path_out_pwr)

    # loop over participants
    for isub, file in enumerate(files_rpeak):
        # get rpeaks
        filename_rpeak = files_rpeak[isub].rsplit('\\')[1] #to get the SJ nb for saving...
        r_peaks = pd.read_csv(files_rpeak[isub], sep='\t', usecols=[0], names=['r_peak'], squeeze=True)
        r_peaks = r_peaks.to_numpy()

        #calculate ibi for each S,B,A sections separately
        r_peaks_S = r_peaks[r_peaks < 148]
        r_peaks_A = r_peaks[r_peaks > 178]
        r_peaks_b = r_peaks[r_peaks > 148]
        r_peaks_B = r_peaks_b[r_peaks_b < 178]
        ibi_S = np.diff(r_peaks_S)
        ibi_A = np.diff(r_peaks_A)
        ibi_B = np.diff(r_peaks_B)
        ibi = np.append(ibi_S, np.append(ibi_B, ibi_A))
        # reformat r_peaks to exclude first R_peak of each S,B,A
        r_peaks_clean = np.append(r_peaks_S[1:], np.append(r_peaks_B[1:], r_peaks_A[1:]))
        # resample ibi to fs (1Hz)
        samples_new_ibi = np.arange(0, 270.25, 1/fs)
        ibi_interp = interp1d(r_peaks_clean, ibi, kind='cubic', bounds_error=False, fill_value=(ibi[0], ibi[-1]))
        rs_ibi = ibi_interp(samples_new_ibi)
        # resample ibi to 4 Hz for TFR computation
        fs2 = 4
        samples_new = np.arange(0, 270.25, 1 / fs2)
        ibi_interp = interp1d(r_peaks_clean, ibi, kind='cubic', bounds_error=False, fill_value=(ibi[0], ibi[-1]))
        rs_ibi2 = ibi_interp(samples_new)
        # mirror 4 Hz ibi at beginning and end, symmetric padding
        sp_ibi_end = rs_ibi2[-mirror_len * fs2:]
        sp_ibi_end = sp_ibi_end[::-1]
        sp_ibi_begin = rs_ibi2[:mirror_len * fs2]
        sp_ibi_begin = sp_ibi_begin[::-1]
        rs_ibi2_mirrored = np.concatenate([sp_ibi_begin, rs_ibi2, sp_ibi_end])

        # compute TFR
        if method == 'spwvd':
            # compute spwvd on mirrored data
            # DOESN'T SEEM TO WORK WITH MIRRORED DATA!!!
            # freqs, times, tfr = nk.signal_timefrequency(rs_ibi2_mirrored, sampling_rate=fs2, method="spwvd")
            # Run MATLAB script spwvd.m instead
            eng = matlab.engine.start_matlab()
            eng.cd(r'E:\NeVRo\Analysis\new_HEP_filtHP_0_3Hz\BHImodel', nargout=0)
            ibi_mat = matlab.double(rs_ibi2_mirrored.tolist())
            fs_mat = matlab.double(fs2)
            mo = eng.spwvd(ibi_mat, fs_mat, 0, nargout=3)
            eng.quit()
            #Reformat output
            freqs = np.array(mo[0]).transpose()[:, 0]
            times = np.array(mo[1]).transpose()[:, 0]
            tfr = np.array(mo[2])
            # Reduce the TFR to HRV frequencies
            hrv_idx = np.logical_and(freqs >= lf[0], freqs <= hf[1])
            freqs = freqs[hrv_idx]
            tfr = tfr[hrv_idx, :]
        elif method == 'cwt':
            # compute cwt on mirrored data
            freqs, times, tfr = nk.signal_timefrequency(rs_ibi2_mirrored, fs2, method="cwt", nfreqbin=50, min_frequency=lf[0], max_frequency=hf[1], show=False)
        elif method == 'stft':
            # compute stft on mirrored data
            freqs, times, pwvd = nk.signal_timefrequency(rs_ibi2_mirrored, fs2, method="stft", window_type="hann", mode='psd', min_frequency=lf[0], max_frequency=hf[1], show=False)

        # debug
        if fig == 1:
            plt.pcolormesh(times, freqs, tfr, shading='gouraud')
            plt.colorbar()
            plt.hlines(lf[1], xmin=times[0], xmax=times[-1], colors='white', linestyles='dotted', linewidth=1)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('TFR smoothed HRV (' + method + ')')
            plt.show()

        # Smoothing: Average over 2s with 50% overlap
        tfr_smooth = np.empty((tfr.shape[0], int(tfr.shape[1] / fs2) - 1))
        win = 2
        overlap = 0.5
        win_len = win * fs2
        i_win = int(win_len * overlap)
        for f in range(0, tfr.shape[0]):
            window_avg = [np.mean(tfr[f, i:i + win_len]) for i in range(0, len(tfr[f]), i_win)
                          if i + win_len <= len(tfr[f])]
            tfr_smooth[f] = np.asarray(window_avg)
        times = np.arange(0, tfr_smooth.shape[1])

        # Get LF and HF power -> integrate over frequencies
        lf_idx = np.logical_and(freqs >= lf[0], freqs <= lf[1])
        hf_idx = np.logical_and(freqs >= hf[0], freqs <= hf[1])
        hrv_idx = np.logical_and(freqs >= lf[0], freqs <= hf[1])
        frs_lf = freqs[lf_idx]
        frs_hf = freqs[hf_idx]
        frs_hrv = freqs[hrv_idx]
        psd_lf = tfr_smooth[lf_idx, :]
        psd_hf = tfr_smooth[hf_idx, :]
        psd_hrv = tfr_smooth[hrv_idx, :]
        power_lf = scipy.integrate.trapezoid(psd_lf.transpose(), frs_lf)
        power_hf = scipy.integrate.trapezoid(psd_hf.transpose(), frs_hf)
        power_hrv = scipy.integrate.trapezoid(psd_hrv.transpose(), frs_hrv)

        # Add one NaN at the end, because only N-1 values because of smoothing
        power_lf = np.append(power_lf, np.nan)
        power_hf = np.append(power_hf, np.nan)
        power_hrv = np.append(power_hrv, np.nan)
        times = np.append(times, times[-1]+1)

        # Plotting
        if fig == 1:
            plt.plot(samples_new_ibi, rs_ibi, c='orange')
            plt.title('rs_IBI 1Hz')
            plt.xlim([0, samples_new_ibi[-1]])
            plt.plot(r_peaks_clean, ibi, c='blue', linestyle='dotted', linewidth=1)
            plt.axvline(x=148, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=178, color='grey', linestyle='dotted', linewidth=1)
            plt.show()
            plt.plot(np.arange(-mirror_len*fs2, rs_ibi2_mirrored.shape[0]-mirror_len*fs2), rs_ibi2_mirrored, c='orange', linewidth=1)
            plt.axvline(x=148*fs2, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=178*fs2, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=0, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=270, color='grey', linestyle='dotted', linewidth=1)
            plt.title('rs_IBI 4Hz mirrored')
            plt.xlim([-mirror_len*fs2, rs_ibi2_mirrored.shape[0]-mirror_len*fs2])
            plt.show()
            for i, hrv in enumerate(['power_lf', 'power_hf']):
                if hrv == 'power_lf':
                    plt.plot(times, power_lf, c='orange')
                    plt.title('Low Frequency HRV')
                elif hrv == 'power_hf':
                    plt.plot(times, power_hf, c='orange')
                    plt.title('High Frequency HRV')
                plt.xlim([0, power_hf.shape[0]])
                plt.ylim([0, 0.14])
                plt.axvline(x=mirror_len-1, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=270+mirror_len-1, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=148+mirror_len-1, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=148+30+mirror_len-1, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=mirror_len-1-cut, color='red', linestyle='dotted', linewidth=1)
                plt.axvline(x=270+mirror_len-1+cut, color='red', linestyle='dotted', linewidth=1)
                plt.ylabel('Time [sec]')
                plt.xlabel('Time [sec]')
                plt.show()

        # Cut mirrored data +/- cut s (data w/o artifacts)
        power_lf_mirrored = power_lf[(mirror_len-1-cut)*fs:(270+mirror_len-1+cut)*fs]
        power_hf_mirrored = power_hf[(mirror_len-1-cut)*fs:(270+mirror_len-1+cut)*fs]
        power_hrv_mirrored = power_hrv[(mirror_len-1-cut)*fs:(270+mirror_len-1+cut)*fs]
        times_mirrored = times[(mirror_len-1-cut)*fs:(270+mirror_len-1+cut)*fs]
        times_mirrored = times_mirrored - times_mirrored[0] - cut*fs

        # mirror rs_ibi 1Hz at beginning and end, symmetric padding +/- cut s
        rs_ibi_sl = rs_ibi[:-1]  # remove last dummy value (not important)
        sp_ibi_end = rs_ibi_sl[-cut*fs:]
        sp_ibi_end = sp_ibi_end[::-1]
        sp_ibi_begin = rs_ibi_sl[:cut*fs]
        sp_ibi_begin = sp_ibi_begin[::-1]
        rs_ibi_mirrored = np.concatenate([sp_ibi_begin, rs_ibi_sl, sp_ibi_end])

        # mirror ibi and r_peaks at beginning and end, symmetric padding +/- cut s
        sp_rpeak_begin = r_peaks_clean[r_peaks_clean <= cut]
        sp_rpeak_end = r_peaks_clean[r_peaks_clean >= 270 - cut]
        sp_rpeak_begin = -sp_rpeak_begin[::-1]
        sp_rpeak_end = sp_rpeak_end[::-1]
        sp_rpeak_end2 = np.zeros(sp_rpeak_end.shape[0])
        for idx, i in enumerate(sp_rpeak_end):
            sp_rpeak_end2[idx] = 270 + (270-i)
        rpeak_mirrored = np.concatenate([sp_rpeak_begin, r_peaks_clean, sp_rpeak_end2])

        sp_ibi_end = ibi[r_peaks_clean >= 270 - cut]
        sp_ibi_end = sp_ibi_end[::-1]
        sp_ibi_begin = ibi[r_peaks_clean <= cut]
        sp_ibi_begin = sp_ibi_begin[::-1]
        ibi_mirrored = np.concatenate([sp_ibi_begin, ibi, sp_ibi_end])

        #Plotting
        if fig == 1:
            plt.plot(times_mirrored, rs_ibi_mirrored, c='orange', linewidth=1)
            plt.plot(rpeak_mirrored, ibi_mirrored, c='red', linestyle='dotted', linewidth=1)
            plt.plot(r_peaks_clean, ibi, c='blue', linestyle='dotted', linewidth=1)
            plt.title('rs_IBI 1Hz mirrored cut')
            plt.xlim([times_mirrored[0], times_mirrored[-1]])
            plt.axvline(x=0, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=270, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=148, color='grey', linestyle='dotted', linewidth=1)
            plt.axvline(x=178, color='grey', linestyle='dotted', linewidth=1)
            plt.show()
            for i, hrv in enumerate(['power_lf', 'power_hf']):
                if hrv == 'power_lf':
                    plt.plot(times_mirrored, power_lf_mirrored, c='orange', linewidth=1)
                    plt.title('clean mirrored Low Frequency HRV')
                elif hrv == 'power_hf':
                    plt.plot(times_mirrored, power_hf_mirrored, c='orange', linewidth=1)
                    plt.title('clean mirrored High Frequency HRV')
                plt.xlim([times_mirrored[0], times_mirrored[-1]])
                plt.ylim([0, 0.14])
                plt.axvline(x=0, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=270, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=148, color='grey', linestyle='dotted', linewidth=1)
                plt.axvline(x=178, color='grey', linestyle='dotted', linewidth=1)
                plt.ylabel('Time [sec]')
                plt.xlabel('Time [sec]')
                plt.show()

        # Save R-peaks, IBI and the LF and HF power into files
        filename_pwr = filename_rpeak.rsplit('_')[0] + '_hrv_' + mov_cond + '.txt'
        d = {'times': times_mirrored, 'LF_mirrored': power_lf_mirrored, 'HF_mirrored': power_hf_mirrored, 'HRV_all_mirrored': power_hrv_mirrored}
        df = pd.DataFrame(data=d)
        df.to_csv(path_out_pwr + filename_pwr, sep='\t', na_rep=np.nan, index=False)
        filename_rs_ibi = filename_rpeak.rsplit('_')[0] + '_rs_ibi_' + mov_cond + '.txt'
        d = {'times': times_mirrored, 'rs_ibi_mirrored': rs_ibi_mirrored}
        df = pd.DataFrame(data=d)
        df.to_csv(path_out_rs_ibi + filename_rs_ibi, sep='\t', na_rep=np.nan, index=False)
        filename_ibi = filename_rpeak.rsplit('_')[0] + '_ibi_' + mov_cond + '.txt'
        d = {'rpeaks_mirrored': rpeak_mirrored, 'ibi_mirrored': ibi_mirrored}
        df = pd.DataFrame(data=d)
        df.to_csv(path_out_ibi + filename_ibi, sep='\t', na_rep=np.nan, index=False)

#for debug
# mov_cond = 'nomov'
# c_style = 'SBA'
# method='cwt'
# fig = 0
# isub = 0
# cut = 35

def NVR_ECG_SAIPAInew(mov_cond, c_style, method='cwt', cut=35, fig=0):
    # 1.1 Set different paths:
    # input paths:
    path_data = 'E:/NeVRo/new_HEP_data_filtHP_0_3Hz/ECG/mov_nomov_ECG/'
    path_in_rpeak = path_data + 'R-peak_time_series/' + mov_cond + '/'
    path_in_ibi = path_data + 'rs_IBI_mir/' + mov_cond + '/' + c_style + '/'
    path_in_ar = 'E:/NeVRo/new_HEP_Data/ratings/class_bins/' + mov_cond + '/' + c_style + '/'
    path_in_saipai = path_data + 'SAI-PAI/' + mov_cond + '/'
    path_in_hrv = path_data + 'HRV_mir_' + method + '/' + mov_cond + '/' + c_style + '/'
    path_in_kal = path_data + 'Kalman/' + mov_cond + '/'

    # 1.2 Get data files
    files_rpeak = glob.glob(path_in_rpeak + '*.txt')
    files_ibi = glob.glob(path_in_ibi + '*.txt')
    files_ar = glob.glob(path_in_ar + '*.txt')
    files_saipai = glob.glob(path_in_saipai + '*.txt')
    files_hrv = glob.glob(path_in_hrv + '*.txt')
    files_kal = glob.glob(path_in_kal + '*.csv')
    # Initialize ECG_all
    ECG_all = []

    # Loop for all subjects
    for isub, file in enumerate(files_rpeak):
        # 1.3 Set filename:
        filename_rpeak = files_rpeak[isub]
        filename_ibi = files_ibi[isub]
        filename_ar = files_ar[isub]
        filename_saipai = files_saipai[isub]
        filename_hrv = files_hrv[isub]
        filename_kal = files_kal[isub]

        # 2. Import ECG and arousal ratings data
        r_peaks = pd.read_csv(filename_rpeak, sep='\t', usecols=[0], names=['r_peak'], squeeze=True)
        aro_rat = pd.read_csv(filename_ar, header=0, names=["latency", "class_aro"])
        ibi = pd.read_csv(filename_ibi, header=0, sep='\t', names=["rs_ibi_mirrored"])
        saipai = pd.read_csv(filename_saipai, sep='\t', na_values='NaN', names=["SAI", "PAI", "ratio_SAIPAI"])
        hrv = pd.read_csv(filename_hrv, header=0, sep='\t', names=["LF_mirrored", "HF_mirrored", "HRV_mirrored"], na_values='NaN')
        kal = pd.read_csv(filename_kal, sep=';', header=0, na_values='NaN',
                          names=["SAI", "PAI", "ratio_SAIPAI", "hrv_lf", "hrv_hf", "ratioLFHF"])

        # Get the different ECG metrics
        r_peaks = r_peaks.to_numpy()
        ibi = ibi['rs_ibi_mirrored'].to_numpy()

        # PB WITH SAIPAI FILES: 2 LESS THAN RPEAK VALUES AT THE END...
        # so add 2 NaNs at the end... or beginning?
        sai = saipai['SAI']
        # sai = np.append(sai.values, [np.nan, np.nan])
        sai = np.insert(sai.values, [0, 0], [np.nan, np.nan])
        pai = saipai['PAI']
        # pai = np.append(pai.values, [np.nan, np.nan])
        pai = np.insert(pai.values, [0, 0], [np.nan, np.nan])
        ratio_saipai = saipai['ratio_SAIPAI']
        # ratio_saipai = np.append(ratio_saipai.values, [np.nan, np.nan])
        ratio_saipai = np.insert(ratio_saipai.values, [0, 0], [np.nan, np.nan])
        hrv_lf = hrv["LF_mirrored"].to_numpy()
        hrv_hf = hrv["HF_mirrored"].to_numpy()
        #remove padding IBI, LF & HF-HRV
        ibi2 = ibi[cut:270+cut]
        hrv_lf2 = hrv_lf[cut:270 + cut]
        hrv_hf2 = hrv_hf[cut:270 + cut]

        ratio_lfhf = hrv_lf2 / hrv_hf2

        # hrv_all = hrv["HRV_all"].to_numpy()
        sai_kal = kal["SAI"].to_numpy()
        pai_kal = kal["PAI"].to_numpy()
        ratio_saipai_kal = kal["ratio_SAIPAI"].to_numpy()
        hrv_lf_kal = kal["hrv_lf"].to_numpy()
        hrv_hf_kal = kal["hrv_hf"].to_numpy()
        ratio_lfhf_kal = kal["ratioLFHF"].to_numpy()

        # Resampling to 1Hz -> 270 samples
        # always n-1 ibi and lp compared to r_peaks (can't compute diff on first peak), so r_peaks[1:]
        # for sai, pai and ratio_saipai -> correction above
        # if interpolation outside of r_peaks time range -> nan
        samples_new = np.arange(0.5, 270, 1)  # because latency aro_ratings = 0.5s
        # sai
        sai_interp = interp1d(r_peaks, sai, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_sai = sai_interp(samples_new)
        # pai
        pai_interp = interp1d(r_peaks, pai, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_pai = pai_interp(samples_new)
        # ratio_saipai
        ratio_saipai_interp = interp1d(r_peaks, ratio_saipai, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_ratio_saipai = ratio_saipai_interp(samples_new)
        # HRV already at 1Hz
        rs_hrv_lf = hrv_lf2
        rs_hrv_hf = hrv_hf2
        rs_ratio_lfhf = ratio_lfhf
        rs_ibi = ibi2

        # sai_kal
        sai_kal_interp = interp1d(r_peaks[1:], sai_kal, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_sai_kal = sai_kal_interp(samples_new)
        # pai_kal
        pai_kal_interp = interp1d(r_peaks[1:], pai_kal, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_pai_kal = pai_kal_interp(samples_new)
        # ratio_saipai_kal
        ratio_saipai_kal_interp = interp1d(r_peaks[1:], ratio_saipai_kal, kind='slinear', bounds_error=False,
                                           fill_value=np.nan)
        rs_ratio_saipai_kal = ratio_saipai_kal_interp(samples_new)
        # hrv_kal
        hrv_lf_kal_interp = interp1d(r_peaks[1:], hrv_lf_kal, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_hrv_lf_kal = hrv_lf_kal_interp(samples_new)
        hrv_hf_kal_interp = interp1d(r_peaks[1:], hrv_hf_kal, kind='slinear', bounds_error=False, fill_value=np.nan)
        rs_hrv_hf_kal = hrv_hf_kal_interp(samples_new)
        ratio_lfhf_kal_interp = interp1d(r_peaks[1:], ratio_lfhf_kal, kind='slinear', bounds_error=False,
                                         fill_value=np.nan)
        rs_ratio_lfhf_kal = ratio_lfhf_kal_interp(samples_new)

        if fig == 1:
            plt.figure()
            plt.plot(samples_new, rs_ibi, 'ro--', r_peaks[1:], ibi)
            plt.plot(samples_new, rs_sai, 'co--', r_peaks, sai)
            plt.plot(samples_new, rs_pai, 'mo--', r_peaks, pai)
            plt.plot(samples_new, rs_ratio_saipai, 'yo--', r_peaks, ratio_saipai)
            plt.plot(np.arange(0, 270), aro_rat['class_aro'], 'b--')
            plt.show()
            plt.figure()
            plt.plot(samples_new, rs_ibi, 'ro--', r_peaks[1:], ibi)
            plt.plot(samples_new, rs_sai_kal, 'co--', r_peaks[1:], sai_kal)
            plt.plot(samples_new, rs_pai_kal, 'mo--', r_peaks[1:], pai_kal)
            plt.plot(samples_new, rs_ratio_saipai_kal, 'yo--', r_peaks[1:], ratio_saipai_kal)
            plt.plot(np.arange(0, 270), aro_rat['class_aro'], 'b--')
            plt.show()
            plt.figure()
            plt.plot(samples_new, rs_ibi, 'ro--')
            plt.plot(samples_new, rs_hrv_lf, 'ro--', samples_new, rs_hrv_lf_kal, 'rx-')
            plt.plot(samples_new, rs_hrv_hf, 'bo--', samples_new, rs_hrv_hf_kal, 'bx-')
            plt.show()

        # Select the HA and LA samples
        LA_ibi = rs_ibi[aro_rat['class_aro'] == 1]
        LA_sai = rs_sai[aro_rat['class_aro'] == 1]
        LA_pai = rs_pai[aro_rat['class_aro'] == 1]
        LA_ratio_saipai = rs_ratio_saipai[aro_rat['class_aro'] == 1]
        LA_hrv_lf = rs_hrv_lf[aro_rat['class_aro'] == 1]
        LA_hrv_hf = rs_hrv_hf[aro_rat['class_aro'] == 1]
        LA_ratio_lfhf = rs_ratio_lfhf[aro_rat['class_aro'] == 1]
        LA_sai_kal = rs_sai_kal[aro_rat['class_aro'] == 1]
        LA_pai_kal = rs_pai_kal[aro_rat['class_aro'] == 1]
        LA_ratio_saipai_kal = rs_ratio_saipai_kal[aro_rat['class_aro'] == 1]
        LA_hrv_lf_kal = rs_hrv_lf_kal[aro_rat['class_aro'] == 1]
        LA_hrv_hf_kal = rs_hrv_hf_kal[aro_rat['class_aro'] == 1]
        LA_ratio_lfhf_kal = rs_ratio_lfhf_kal[aro_rat['class_aro'] == 1]

        HA_ibi = rs_ibi[aro_rat['class_aro'] == 3]
        HA_sai = rs_sai[aro_rat['class_aro'] == 3]
        HA_pai = rs_pai[aro_rat['class_aro'] == 3]
        HA_ratio_saipai = rs_ratio_saipai[aro_rat['class_aro'] == 3]
        HA_hrv_lf = rs_hrv_lf[aro_rat['class_aro'] == 3]
        HA_hrv_hf = rs_hrv_hf[aro_rat['class_aro'] == 3]
        HA_ratio_lfhf = rs_ratio_lfhf[aro_rat['class_aro'] == 3]
        HA_sai_kal = rs_sai_kal[aro_rat['class_aro'] == 3]
        HA_pai_kal = rs_pai_kal[aro_rat['class_aro'] == 3]
        HA_ratio_saipai_kal = rs_ratio_saipai_kal[aro_rat['class_aro'] == 3]
        HA_hrv_lf_kal = rs_hrv_lf_kal[aro_rat['class_aro'] == 3]
        HA_hrv_hf_kal = rs_hrv_hf_kal[aro_rat['class_aro'] == 3]
        HA_ratio_lfhf_kal = rs_ratio_lfhf_kal[aro_rat['class_aro'] == 3]

        # Z-transform the time-series
        # nan policy: ‘propagate’ returns nan; ‘omit’ performs the calculations ignoring nan values
        z_ibi = zscore(rs_ibi, nan_policy='omit')
        z_sai = zscore(rs_sai, nan_policy='omit')
        z_ratio_saipai = zscore(rs_ratio_saipai, nan_policy='omit')
        z_pai = zscore(rs_pai, nan_policy='omit')
        z_hrv_hf = zscore(rs_hrv_hf, nan_policy='omit')
        z_hrv_lf = zscore(rs_hrv_lf, nan_policy='omit')
        z_ratio_lfhf = zscore(rs_ratio_lfhf, nan_policy='omit')
        z_sai_kal = zscore(rs_sai_kal, nan_policy='omit')
        z_ratio_saipai_kal = zscore(rs_ratio_saipai_kal, nan_policy='omit')
        z_pai_kal = zscore(rs_pai_kal, nan_policy='omit')
        z_hrv_hf_kal = zscore(rs_hrv_hf_kal, nan_policy='omit')
        z_hrv_lf_kal = zscore(rs_hrv_lf_kal, nan_policy='omit')
        z_ratio_lfhf_kal = zscore(rs_ratio_lfhf_kal, nan_policy='omit')

        # Select the HA and LA samples from the z-scores
        LA_z_ibi = z_ibi[aro_rat['class_aro'] == 1]
        LA_z_sai = z_sai[aro_rat['class_aro'] == 1]
        LA_z_pai = z_pai[aro_rat['class_aro'] == 1]
        LA_z_ratio_saipai = z_ratio_saipai[aro_rat['class_aro'] == 1]
        LA_z_hrv_lf = z_hrv_lf[aro_rat['class_aro'] == 1]
        LA_z_hrv_hf = z_hrv_hf[aro_rat['class_aro'] == 1]
        LA_z_ratio_lfhf = z_ratio_lfhf[aro_rat['class_aro'] == 1]
        LA_z_sai_kal = z_sai_kal[aro_rat['class_aro'] == 1]
        LA_z_pai_kal = z_pai_kal[aro_rat['class_aro'] == 1]
        LA_z_ratio_saipai_kal = z_ratio_saipai_kal[aro_rat['class_aro'] == 1]
        LA_z_hrv_lf_kal = z_hrv_lf_kal[aro_rat['class_aro'] == 1]
        LA_z_hrv_hf_kal = z_hrv_hf_kal[aro_rat['class_aro'] == 1]
        LA_z_ratio_lfhf_kal = z_ratio_lfhf_kal[aro_rat['class_aro'] == 1]

        HA_z_ibi = z_ibi[aro_rat['class_aro'] == 3]
        HA_z_sai = z_sai[aro_rat['class_aro'] == 3]
        HA_z_pai = z_pai[aro_rat['class_aro'] == 3]
        HA_z_ratio_saipai = z_ratio_saipai[aro_rat['class_aro'] == 3]
        HA_z_hrv_lf = z_hrv_lf[aro_rat['class_aro'] == 3]
        HA_z_hrv_hf = z_hrv_hf[aro_rat['class_aro'] == 3]
        HA_z_ratio_lfhf = z_ratio_lfhf[aro_rat['class_aro'] == 3]
        HA_z_sai_kal = z_sai_kal[aro_rat['class_aro'] == 3]
        HA_z_pai_kal = z_pai_kal[aro_rat['class_aro'] == 3]
        HA_z_ratio_saipai_kal = z_ratio_saipai_kal[aro_rat['class_aro'] == 3]
        HA_z_hrv_lf_kal = z_hrv_lf_kal[aro_rat['class_aro'] == 3]
        HA_z_hrv_hf_kal = z_hrv_hf_kal[aro_rat['class_aro'] == 3]
        HA_z_ratio_lfhf_kal = z_ratio_lfhf_kal[aro_rat['class_aro'] == 3]

        # Store relevant variables in ecg_dict
        ecg_dict = {'filename': filename_rpeak,
                    'r_peaks': r_peaks,
                    'aro_ratings_bins': aro_rat['class_aro'].to_numpy(),
                    'aro_ratings_latency': aro_rat['latency'].to_numpy(),
                    'ibi': ibi,
                    'SAI': sai,
                    'PAI': pai,
                    'ratioSAIPAI': ratio_saipai,
                    'hrv_lf': hrv_lf,
                    'hrv_hf': hrv_hf,
                    'ratioLFHF': ratio_lfhf,
                    'SAI_kal': sai_kal,
                    'PAI_kal': pai_kal,
                    'ratioSAIPAI_kal': ratio_saipai_kal,
                    'hrv_lf_kal': hrv_lf_kal,
                    'hrv_hf_kal': hrv_hf_kal,
                    'ratioLFHF_kal': ratio_lfhf_kal,
                    'rs_samples': samples_new,
                    'rs_ibi': rs_ibi,
                    'rs_SAI': rs_sai,
                    'rs_PAI': rs_pai,
                    'rs_ratioSAIPAI': rs_ratio_saipai,
                    'rs_hrv_lf': rs_hrv_lf,
                    'rs_hrv_hf': rs_hrv_hf,
                    'rs_ratioLFHF': rs_ratio_lfhf,
                    'rs_SAI_kal': rs_sai_kal,
                    'rs_PAI_kal': rs_pai_kal,
                    'rs_ratioSAIPAI_kal': rs_ratio_saipai_kal,
                    'rs_hrv_lf_kal': rs_hrv_lf_kal,
                    'rs_hrv_hf_kal': rs_hrv_hf_kal,
                    'rs_ratioLFHF_kal': rs_ratio_lfhf_kal,
                    'z_ibi': z_ibi,
                    'z_SAI': z_sai,
                    'z_PAI': z_pai,
                    'z_ratioSAIPAI': z_ratio_saipai,
                    'z_hrv_lf': z_hrv_lf,
                    'z_hrv_hf': z_hrv_hf,
                    'z_ratioLFHF': z_ratio_lfhf,
                    'z_SAI_kal': z_sai_kal,
                    'z_PAI_kal': z_pai_kal,
                    'z_ratioSAIPAI_kal': z_ratio_saipai_kal,
                    'z_hrv_lf_kal': z_hrv_lf_kal,
                    'z_hrv_hf_kal': z_hrv_hf_kal,
                    'z_ratioLFHF_kal': z_ratio_lfhf_kal,
                    'LA_ibi': LA_ibi,
                    'HA_ibi': HA_ibi,
                    'LA_SAI': LA_sai,
                    'HA_SAI': HA_sai,
                    'LA_PAI': LA_pai,
                    'HA_PAI': HA_pai,
                    'LA_ratioSAIPAI': LA_ratio_saipai,
                    'HA_ratioSAIPAI': HA_ratio_saipai,
                    'LA_hrv_lf': LA_hrv_lf,
                    'HA_hrv_lf': HA_hrv_lf,
                    'LA_hrv_hf': LA_hrv_hf,
                    'HA_hrv_hf': HA_hrv_hf,
                    'LA_ratioLFHF': LA_ratio_lfhf,
                    'HA_ratioLFHF': HA_ratio_lfhf,
                    'LA_SAI_kal': LA_sai_kal,
                    'HA_SAI_kal': HA_sai_kal,
                    'LA_PAI_kal': LA_pai_kal,
                    'HA_PAI_kal': HA_pai_kal,
                    'LA_ratioSAIPAI_kal': LA_ratio_saipai_kal,
                    'HA_ratioSAIPAI_kal': HA_ratio_saipai_kal,
                    'LA_hrv_lf_kal': LA_hrv_lf_kal,
                    'HA_hrv_lf_kal': HA_hrv_lf_kal,
                    'LA_hrv_hf_kal': LA_hrv_hf_kal,
                    'HA_hrv_hf_kal': HA_hrv_hf_kal,
                    'LA_ratioLFHF_kal': LA_ratio_lfhf_kal,
                    'HA_ratioLFHF_kal': HA_ratio_lfhf_kal,
                    'LA_z_ibi': LA_z_ibi,
                    'HA_z_ibi': HA_z_ibi,
                    'LA_z_SAI': LA_z_sai,
                    'HA_z_SAI': HA_z_sai,
                    'LA_z_PAI': LA_z_pai,
                    'HA_z_PAI': HA_z_pai,
                    'LA_z_ratioSAIPAI': LA_z_ratio_saipai,
                    'HA_z_ratioSAIPAI': HA_z_ratio_saipai,
                    'LA_z_hrv_lf': LA_z_hrv_lf,
                    'HA_z_hrv_lf': HA_z_hrv_lf,
                    'LA_z_hrv_hf': LA_z_hrv_hf,
                    'HA_z_hrv_hf': HA_z_hrv_hf,
                    'LA_z_ratioLFHF': LA_z_ratio_lfhf,
                    'HA_z_ratioLFHF': HA_z_ratio_lfhf,
                    'LA_z_SAI_kal': LA_z_sai_kal,
                    'HA_z_SAI_kal': HA_z_sai_kal,
                    'LA_z_PAI_kal': LA_z_pai_kal,
                    'HA_z_PAI_kal': HA_z_pai_kal,
                    'LA_z_ratioSAIPAI_kal': LA_z_ratio_saipai_kal,
                    'HA_z_ratioSAIPAI_kal': HA_z_ratio_saipai_kal,
                    'LA_z_hrv_lf_kal': LA_z_hrv_lf_kal,
                    'HA_z_hrv_lf_kal': HA_z_hrv_lf_kal,
                    'LA_z_hrv_hf_kal': LA_z_hrv_hf_kal,
                    'HA_z_hrv_hf_kal': HA_z_hrv_hf_kal,
                    'LA_z_ratioLFHF_kal': LA_z_ratio_lfhf_kal,
                    'HA_z_ratioLFHF_kal': HA_z_ratio_lfhf_kal,
                    }

        ECG_all.append(ecg_dict)
        print(filename_rpeak.rsplit('\\')[1] + ' done ...')
    return ECG_all
