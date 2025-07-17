import os
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import hilbert, butter, filtfilt, find_peaks
from scipy.stats import circmean, circstd
import pickle
import matplotlib.pyplot as plt
import pickle as pkl
from general_utils import *
from scipy.signal import resample


# Directories
base_directory = '/home/melma31/Documents/time_project_grin_lens'
data_directory = os.path.join(base_directory, 'data')
figure_directory = os.path.join(base_directory, 'figures')
output_directory = os.path.join(base_directory, 'output')

# Your bands
bands = {
    'infra-slow':(0.01,0.1),
    'delta': (1, 3),
    'theta': (8, 12),
    'slow-gamma': (40, 90),
    'ripple-band': (100, 250),
    'MUA': (300, 1000)
}

from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis = 0)

def detect_events(signal, height=None, distance=None):
    """
    Detect events as peaks in the signal.
    Returns event indices and their amplitudes.
    """
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    calcium_events = peaks
    calcium_amplitude = signal[peaks]
    return calcium_events, calcium_amplitude

def extract_event_phase(events_idx, phase_signal):
    """
    Extract phase at event indices.
    """
    phases = phase_signal[events_idx]
    return phases


mice_list = ['CalbEphys1GRIN1', 'CalbEphys1GRIN2', 'Thy1Ephys1GRIN1', 'Thy1Ephys1GRIN2']
mice_list = ['CalbEphys1GRIN1']

mi_dict = dict()
for mouse in mice_list:

    input_file = mouse + '_all_df_dict.pkl'
    input_file = os.path.join(data_directory, input_file)

    with open(input_file, 'rb') as file:
        mouse_data = pickle.load(file)

    mi_dict[mouse] = dict()


    signal = mouse_data['clean_traces']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 3)[0]
    signal = signal[valid_index, :]
    speed = speed[valid_index]
    pos = mouse_data['position'][valid_index, :]
    trial_id = mouse_data['trial_id'][valid_index]
    mov_dir = mouse_data['mov_direction'][valid_index]
    pos_dir = pos[:, 0] * mov_dir
    inner_time = compute_inner_time(trial_id)
    time = np.arange(0, pos.shape[0])

    behaviours_list = [pos[:, 0], pos_dir, mov_dir, speed, time, inner_time, trial_id]
    beh_names = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']

    behaviour_dict = {
        'position': behaviours_list[0],
        '(pos,dir)': behaviours_list[1],
        'mov_dir': behaviours_list[2],
        'speed': behaviours_list[3],
        'time': behaviours_list[4],
        'inner_time': behaviours_list[5],
        'trial_id': behaviours_list[6]
    }

    mi_all = []
    mi_bands_all = {'amplitude': [], 'phase': [], 'pl_mean': [], 'pl_std': []}
    fs = 20
    target_length = signal.shape[0]
    for beh_index, beh in enumerate(behaviours_list):
        print('MI for variable:' + beh_names[beh_index])
        mi = []
        for neui in range(signal.shape[1]):
            neuron_trace = signal[:, neui]
            neuron_mi = mutual_info_regression(neuron_trace.reshape(-1, 1), beh, n_neighbors=50, random_state=16)[0]
            mi.append(neuron_mi)

            if beh_index == 0:
                events_idx, _ = detect_events(neuron_trace)
                events_idx = events_idx * fs / 20
                events_idx = events_idx.astype(int)

                mi_amp_bands = []
                mi_phase_bands = []
                pl_bands = []
                pl_bands_std = []

                lfp_data = mouse_data['LFP']['LFP']  # shape (samples, channels)
                # Compute the mean over channels (axis 1 is time, axis 0 is samples!)
                mean_lfp = np.mean(lfp_data, axis=1)  # shape: (2262699,)
                #mean_lfp = lfp_data[:,0]  # shape: (2262699,)

                fs = mouse_data['LFP']['sf'][0]

                for band_name, (low, high) in bands.items():
                    nyquist = 0.5 * fs
                    if low <= 0:
                        print(f"Band {band_name} has lowcut <= 0, skipping.")
                        continue
                    if high >= nyquist:
                        print(f"Band {band_name} has highcut >= Nyquist, skipping.")
                        continue

                    filtered = bandpass_filter(mean_lfp, low, high, fs, order = 2)

                    analytic_signal = hilbert(filtered)
                    amp = np.abs(analytic_signal)
                    phase = np.angle(analytic_signal)

                    pl_neuron_phases = extract_event_phase(events_idx, phase)

                    pl_mean = circmean(pl_neuron_phases, high=np.pi, low=-np.pi)
                    pl_std = circstd(pl_neuron_phases, high=np.pi, low=-np.pi)

                    pl_bands.append(pl_mean)
                    pl_bands_std.append(pl_std)

                mi_bands_all['amplitude'].append(mi_amp_bands)
                mi_bands_all['phase'].append(mi_phase_bands)
                mi_bands_all['pl_mean'].append(pl_bands)
                mi_bands_all['pl_std'].append(pl_bands_std)

        mi_all.append(mi)

    mi_dict[mouse]['behaviour'] = behaviour_dict
    mi_dict[mouse]['signal'] = signal
    mi_dict[mouse]['valid_index'] = valid_index
    mi_dict[mouse]['MIR'] = mi_all
    mi_dict[mouse]['MIR_bands'] = mi_bands_all
    mi_dict[mouse]['lfp'] = mouse_data['lfp']
    mi_dict[mouse]['cwt_freqs'] = mouse_data['cwt_freqs']
    mi_dict[mouse]['cwt'] = mouse_data['cwt']

with open(os.path.join(output_directory, 'mi_beh_mi_band_pl.pkl'), 'wb') as f:
    pkl.dump(mi_dict, f)

for mouse in mice_list:
    mouse_dict = dict()
    mouse_dict['lt'] = dict()
    mouse_dict['lt']['behaviour'] = mi_dict[mouse]['behaviour']
    mouse_dict['lt']['signal'] =  mi_dict[mouse]['signal']
    mouse_dict['lt']['valid_index'] = mi_dict[mouse]['valid_index']
    mouse_dict['lt']['MIR'] = mi_dict[mouse]['MIR']
    mouse_dict['lt']['MIR_bands'] =  mi_dict[mouse]['MIR_bands']
    moutput_directory = os.path.join(output_directory, mouse)
    if not os.path.isdir(moutput_directory): os.makedirs(moutput_directory)

    with open(os.path.join(output_directory, mouse +'mi_beh_mi_band_pl.pkl'), 'wb') as f:
        pkl.dump(mouse_dict, f)

