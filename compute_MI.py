import os
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from general_utils import *

import pickle
import matplotlib.pyplot as plt
import pickle as pkl
import random

def circular_shuffle(data, split_idx):
    return np.concatenate((data[split_idx:], data[:split_idx]), axis=0)


base_directory = '/home/melma31/Documents/time_project_grin_lens'
data_directory = os.path.join(base_directory, 'data')
figure_directory = os.path.join(base_directory, 'figures')
output_directory = os.path.join(base_directory, 'output')

input_file = 'lfp_mice_all_data_dict.pkl'
input_file = os.path.join(data_directory, input_file)

with open(input_file, 'rb') as file:
    data = pickle.load(file)

import numpy as np

# Your bands
bands = {
    'infra-slow':(0.01,0.1),
    'delta': (1, 3),
    'theta': (8, 12),
    'slow-gamma': (40, 90),
    'ripple-band': (100, 250),
    'MUA': (300, 1000)
}

from scipy.signal import hilbert, butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


mice_list = list(data.keys())
mi_dict = dict()
for mouse in mice_list:
    mi_dict[mouse] = dict()
    mouse_data = data[mouse]

    signal = mouse_data['clean_traces']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 3)[0]
    signal = signal[valid_index,:]
    speed = speed[valid_index]
    pos = mouse_data['position'][valid_index,:]
    trial_id = mouse_data['trial_id'][valid_index]
    mov_dir = mouse_data['mov_direction'][valid_index]
    pos_dir = pos[:,0] * mov_dir
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
    mi_bands_all = []
    lfp = mouse_data['lfp'][valid_index]
    fs = 20
    n_channels = lfp.shape[1]
    mi_bands_all = {'amplitude': [], 'phase': []}

    for beh_index, beh in enumerate(behaviours_list):
        print('MI for variable:' + beh_names[beh_index])
        mi = []
        for neui in range(signal.shape[1]):
            neuron_mi = mutual_info_regression(signal[:, neui].reshape(-1, 1), beh, n_neighbors=50,
                                                           random_state=16)[0]
            mi.append(neuron_mi)
            if beh_index == 0:  # Only once per neuron
                mi_amp_bands = []
                mi_phase_bands = []
                lfp_freqs = mouse_data['cwt_freqs']
                lfp_wavelet = mouse_data['cwt']
                lfp_wavelet = lfp_wavelet[valid_index,:]
                for band_name, (fmin, fmax) in bands.items():
                    # Filter LFP (actually take the )
                    band_mask = (lfp_freqs >= fmin) & (lfp_freqs <= fmax)
                    cwt_band = lfp_wavelet[:, band_mask, :]  # shape: [T, band_freqs, chans]

                    # Mean across frequencies and channels
                    cwt_band_mean = np.mean(cwt_band, axis=(1, 2))
                    amp = cwt_band_mean # Equivalent to Hilbert envelope
                    phase = np.angle(hilbert(cwt_band_mean))
                    # Compute MI with raw trace (not shuffle)
                    amp_mi = mutual_info_regression(signal[:, neui].reshape(-1, 1), amp, n_neighbors=50,
                                                                random_state=16)[0]
                    phase_mi = mutual_info_regression(signal[:, neui].reshape(-1, 1), phase, n_neighbors=50,
                                                                  random_state=16)[0]
                    mi_amp_bands.append(amp_mi)
                    mi_phase_bands.append(phase_mi)
                mi_bands_all['amplitude'].append(mi_amp_bands)
                mi_bands_all['phase'].append(mi_phase_bands)
        mi_all.append(mi)

    n_shuffles = 100
    mir_shuffle_all = []
    mir_bands_shuffle_all = {'amplitude': [], 'phase': []}

    for shuffle_idx in range(n_shuffles):
        print(f"Shuffling iteration {shuffle_idx + 1}/{n_shuffles}")
        #cut_idx = np.random.randint(1, signal.shape[0] - 1)
        #shuffled_signal = circular_shuffle(signal, cut_idx)

        shuffled_signal = signal.copy()
        np.random.shuffle(shuffled_signal)

        mi_shuffle_behaviors = []
        mi_amp_band_shuffle = []
        mi_phase_band_shuffle = []

        for beh in behaviours_list:
            mi_shuffle = []
            for neui in range(signal.shape[1]):
                neuron_mi = mutual_info_regression(shuffled_signal[:, neui].reshape(-1, 1), beh, n_neighbors=50,
                                                   random_state=16)[0]
                mi_shuffle.append(neuron_mi)
            mi_shuffle_behaviors.append(mi_shuffle)

        for neui in range(signal.shape[1]):
            mi_amp_bands = []
            mi_phase_bands = []
            for band_name, (fmin, fmax) in bands.items():
                band_mask = (lfp_freqs >= fmin) & (lfp_freqs <= fmax)
                cwt_band = lfp_wavelet[:, band_mask, :]  # shape: [T, band_freqs, chans]
                cwt_band_mean = np.mean(cwt_band, axis=(1, 2))
                amp = cwt_band_mean
                phase = np.angle(hilbert(cwt_band_mean))

                amp_mi = mutual_info_regression(shuffled_signal[:, neui].reshape(-1, 1), amp, n_neighbors=50,
                                                random_state=16)[0]
                phase_mi = mutual_info_regression(shuffled_signal[:, neui].reshape(-1, 1), phase, n_neighbors=16)[0]

                mi_amp_bands.append(amp_mi)
                mi_phase_bands.append(phase_mi)
            mi_amp_band_shuffle.append(mi_amp_bands)
            mi_phase_band_shuffle.append(mi_phase_bands)

        mir_shuffle_all.append(mi_shuffle_behaviors)
        mir_bands_shuffle_all['amplitude'].append(mi_amp_band_shuffle)
        mir_bands_shuffle_all['phase'].append(mi_phase_band_shuffle)

    # Save the mean over all 100 shuffles
    mi_dict[mouse]['MIR_shuffle'] = np.array(mir_shuffle_all)
    mi_dict[mouse]['MIR_bands_shuffle'] = {
        'amplitude': np.array(mir_bands_shuffle_all['amplitude']),
        'phase': np.array(mir_bands_shuffle_all['phase'])
    }

    # mi_final = np.hstack(mi_all)
    mi_dict[mouse]['behaviour'] = behaviour_dict
    mi_dict[mouse]['signal'] = signal
    mi_dict[mouse]['valid_index'] = valid_index
    mi_dict[mouse]['MIR'] = mi_all
    mi_dict[mouse]['MIR_bands'] = mi_bands_all
    mi_dict[mouse]['lfp'] = mouse_data['lfp']
    mi_dict[mouse]['cwt_freqs'] = mouse_data['cwt_freqs']
    mi_dict[mouse]['cwt'] = mouse_data['cwt']

    #mouse_dict = dict()
    #mouse_dict['lt'] = dict()
    #mouse_dict['lt']['behaviour'] = behaviour_dict
    #mouse_dict['lt']['signal'] = signal
    #mouse_dict['lt']['valid_index'] = valid_index
    #mouse_dict['lt']['MIR'] = mi_all

    #moutput_directory = os.path.join(output_directory, mouse)
    #if not os.path.isdir(moutput_directory): os.makedirs(moutput_directory)

    #with open(os.path.join(output_directory, mouse +'_mi_clean_traces_dict_alldir.pkl'), 'wb') as f:
    #    pkl.dump(mouse_dict, f)

with open(os.path.join(output_directory,'mi_beh_mi_band_hilbert_shuffle.pkl'), 'wb') as f:
    pkl.dump(mi_dict, f)

for mouse in mice_list:
    mouse_dict = dict()
    mouse_dict['lt'] = dict()
    mouse_dict['lt']['behaviour'] = mi_dict[mouse]['behaviour']
    mouse_dict['lt']['signal'] =  mi_dict[mouse]['signal']
    mouse_dict['lt']['valid_index'] = mi_dict[mouse]['valid_index']
    mouse_dict['lt']['MIR'] = mi_dict[mouse]['MIR']
    mouse_dict['lt']['MIR_bands'] =  mi_dict[mouse]['MIR_bands']
    # Save the mean over all 100 shuffles
    mouse_dict['lt']['MIR_shuffle'] = mi_dict[mouse]['MIR_shuffle']
    mouse_dict['lt']['MIR_bands_shuffle'] = {
        'amplitude': np.array(mi_dict[mouse]['MIR_bands_shuffle']['amplitude']),
        'phase': np.array(mi_dict[mouse]['MIR_bands_shuffle']['amplitude'])
    }
    moutput_directory = os.path.join(output_directory, mouse)
    if not os.path.isdir(moutput_directory): os.makedirs(moutput_directory)

    with open(os.path.join(output_directory, mouse +'_mi_beh_mi_band_hilbert_shuffle.pkl'), 'wb') as f:
        pkl.dump(mouse_dict, f)

########################################################################################################
#
# CREATING DF AND PLOTTING
#
########################################################################################################
import pandas as pd
from scipy.stats import zscore
import numpy as np
import os
import pickle as pkl

with open(os.path.join(output_directory,'mi_beh_mi_band_hilbert_shuffle.pkl'), 'rb') as file:
    data = pkl.load(file)

rows = []
for mouse, mouse_data in data.items():
    mir = np.array(mouse_data['MIR']).T
    mir_band_amp = np.array(mouse_data['MIR_bands']['amplitude'])
    mir_band_phase = np.array(mouse_data['MIR_bands']['phase'])

    mir_shuffle = np.array(mouse_data['MIR_shuffle'])  # shape: (100, n_neurons, 7)
    mir_band_amp_shuffle = np.array(mouse_data['MIR_bands_shuffle']['amplitude'])  # (100, n_neurons, 6)
    mir_band_phase_shuffle = np.array(mouse_data['MIR_bands_shuffle']['phase'])  # (100, n_neurons, 6)
    n_neurons = mir.shape[0]

    if 'Calb' in mouse:
        area = 'sup'
    elif 'Thy' in mouse:
        area = 'deep'
    else:
        area = 'unknown'

    for neuron_idx in range(n_neurons):
        row = {
            'mouse': mouse,
            'neuron_idx': neuron_idx,
            'area': area
        }

        # Behavior-related MI
        mi_values = []
        for i, (beh_name, mi_list) in enumerate(zip(beh_names, mir.T)):
            mi_value = mi_list[neuron_idx]
            shuffle_vals = mir_shuffle[:, i,i]
            signif = 's' if mi_value > np.percentile(shuffle_vals, 95) else 'n'
            row[beh_name] = mi_value
            row[f'{beh_name}_significance'] = signif
            mi_values.append(mi_value)

        # Band amplitude MI
        mi_values_band = []
        for i, (band_name, mi_list_band) in enumerate(zip(bands.keys(), mir_band_amp.T)):
            mi_value = mi_list_band[neuron_idx]
            shuffle_vals = mir_band_amp_shuffle[:, neuron_idx,i]
            signif = 's' if mi_value > np.percentile(shuffle_vals, 95) else 'n'
            row[band_name + '_amp'] = mi_value
            row[band_name + '_amp_significance'] = signif
            mi_values_band.append(mi_value)

        # Band phase MI
        mi_values_band_phase = []
        for i, (band_name, mi_list_band) in enumerate(zip(bands.keys(), mir_band_phase.T)):
            mi_value = mi_list_band[neuron_idx]
            shuffle_vals = mir_band_phase_shuffle[:, neuron_idx,i]
            signif = 's' if mi_value > np.percentile(shuffle_vals, 95) else 'n'
            row[band_name + '_phase'] = mi_value
            row[band_name + '_phase_significance'] = signif
            mi_values_band_phase.append(mi_value)

        # Z-scoring
        mi_zscore = zscore(np.array(mi_values), nan_policy='omit')
        mi_zscore_band = zscore(np.array(mi_values_band), nan_policy='omit')
        mi_zscore_band_phase = zscore(np.array(mi_values_band_phase), nan_policy='omit')

        for beh_name, z_value in zip(beh_names, mi_zscore):
            row[f'{beh_name}_zscore'] = z_value
        for freq_band, z_val in zip(bands.keys(), mi_zscore_band):
            row[f'{freq_band}_amp_zscore'] = z_val
        for freq_band, z_val in zip(bands.keys(), mi_zscore_band_phase):
            row[f'{freq_band}_phase_zscore'] = z_val

        rows.append(row)

neuron_mi_df = pd.DataFrame(rows)




#####################################################################
#### PLOTING
import seaborn as sns
import matplotlib.pyplot as plt
# Define MI variables and corresponding significance columns
mi_vars = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID',
           'infra-slow_amp', 'delta_amp', 'theta_amp', 'slow-gamma_amp', 'ripple-band_amp', 'MUA_amp',
           'infra-slow_phase', 'delta_phase', 'theta_phase', 'slow-gamma_phase', 'ripple-band_phase', 'MUA_phase']
signif_vars = [f'{v}_significance' for v in mi_vars]
# Keep only significant values
signif_mask = np.full(len(neuron_mi_df), False)
for mi, sig in zip(mi_vars, signif_vars):
    signif_mask |= (neuron_mi_df[sig] == 's')
signif_df = neuron_mi_df[signif_mask].copy()
# Melt to long format
long_df = signif_df.melt(id_vars='area', value_vars=mi_vars,
                         var_name='variable', value_name='MI_value')
# Keep only 'sup' and 'deep' areas
long_df = long_df[long_df['area'].isin(['sup', 'deep'])]
# Plot
plt.figure(figsize=(2 * len(mi_vars), 6))
sns.boxenplot(
    data=long_df,
    x='variable',
    y='MI_value',
    hue='area',
    palette={'sup': 'purple', 'deep': 'gold'}
)
plt.title("Significant MI Distributions by Variable and Area")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mutual Information")
plt.xlabel("Variable")
plt.tight_layout()
# Save
violin_path = os.path.join(figure_directory, "MI_Significant_Only_ViolinPlot.png")
plt.savefig(violin_path, dpi=300)
plt.show()
plt.close()
print(f"Saved violin plot to: {violin_path}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Frequency band order
freq_order = bands.keys()
n_beh = 7
n_freq = 6
lfp_variable = '_amp'

# Prepare output directory
figure_directory = os.path.join(base_directory, 'figures')
os.makedirs(figure_directory, exist_ok=True)
# Prepare to store per-area Pearson r values
detailed_pearson_data = []
fig, axs = plt.subplots(n_beh, n_freq, figsize=(4 * n_freq, 4 * n_beh), sharex=False, sharey=False)

for i, beh in enumerate(beh_names):
    for j, freq in enumerate(freq_order):
        ax = axs[i, j] if n_beh > 1 else axs[j]

        # Subset
        df = neuron_mi_df[[beh, freq + lfp_variable, 'area']].dropna()
        x = df[beh].values
        y = df[freq + lfp_variable].values
        area = df['area'].values

        # Scatter colored by area
        for group in ['sup', 'deep']:
            mask = (area == group)
            ax.scatter(x[mask], y[mask], s=5, alpha=0.5, color=palette[group], label=f'{group}')

            # Regression per area
            if np.sum(mask) > 2:
                slope, intercept, r, _, _ = linregress(x[mask], y[mask])
                x_fit = np.linspace(np.min(x[mask]), np.max(x[mask]), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, color=palette[group], linewidth=1.2, label=f'{group} (r={r:.2f})')

                detailed_pearson_data.append({
                    'behavior': beh, 'frequency': freq,
                    'area': group, 'pearson_r': r
                })

        # Global regression
        slope, intercept, r_all, _, _ = linregress(x, y)
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='black', linewidth=1, linestyle='--', label=f'all (r={r_all:.2f})')

        detailed_pearson_data.append({
            'behavior': beh, 'frequency': freq,
            'area': 'all', 'pearson_r': r_all
        })

        # Labels
        ax.set_title(f"{beh} vs {freq}")
        ax.set_xlabel(f"{beh} MI")
        ax.set_ylabel(f"{freq} MI")
        ax.legend(fontsize=7)
        ax.grid(False)

# Save scatter plot
plt.tight_layout()
fig.suptitle("MI vs Frequency Band MI by Area", fontsize=16, y=1.02)
scatter_path = os.path.join(figure_directory,'MI_vs_Frequency_'+ lfp_variable+'.png')
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved color-coded scatter grid with Pearson r to: {scatter_path}")

detailed_r_df = pd.DataFrame(detailed_pearson_data)

n_beh = len(beh_names)
fig, axs = plt.subplots(n_beh, 1, figsize=(8, 3 * n_beh), sharex=True)

if n_beh == 1:
    axs = [axs]

for i, beh in enumerate(beh_names):
    ax = axs[i]

    for group in ['sup', 'deep', 'all']:
        subset = detailed_r_df[(detailed_r_df['behavior'] == beh) &
                               (detailed_r_df['area'] == group)]
        subset = subset.set_index('frequency').reindex(freq_order)

        label = f"{group}"
        color = palette[group] if group in palette else 'black'
        ax.plot(freq_order, subset['pearson_r'], marker='o', label=label, color=color)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(f"{beh} MI vs Frequency Band")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1, 1)
    ax.grid(True)
    ax.legend(title="Area")

axs[-1].set_xlabel("Frequency Band")

plt.tight_layout()
summary_path = os.path.join(figure_directory, 'Pearson_r_vs_Frequency_ByArea_Subplots'+ lfp_variable +'.png')
plt.savefig(summary_path, dpi=300)
plt.close()

print(f"Saved Pearson r per-behavior subplot figure to: {summary_path}")
