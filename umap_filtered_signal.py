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
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

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
    'slow': (0.1, 0.5),
    'slow2': (0.4, 0.9),
    'delta': (0.9, 1.5),
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
mice_list = ['Thy1Ephys1GRIN1']

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
    signal = gaussian_filter1d(signal, sigma=5, axis=0)

    import matplotlib.pyplot as plt
    import umap
    from matplotlib import cm

    fs = 20  # Sampling rate
    umap_model = umap.UMAP(n_neighbors=60, n_components=3, min_dist=0.1, random_state=42)


    # Include 'raw' as first entry
    all_bands = {'raw': None, **bands}
    band_names = list(all_bands.keys())
    n_bands = len(band_names)

    fig = plt.figure(figsize=(18, 5 * n_bands))
    fig.suptitle(f'3D UMAP Embeddings for {mouse}', fontsize=20)

    for i, band_name in enumerate(band_names):
        print(f'Processing band: {band_name}')

        if band_name == 'raw':
            filtered = signal.copy()
        else:
            lowcut, highcut = bands[band_name]
            filtered = bandpass_filter(signal, lowcut, highcut, fs)

        # Z-score across time
        filtered_z = (filtered - np.mean(filtered, axis=0)) / np.std(filtered, axis=0)

        # UMAP embedding (3D)
        embedding = umap_model.fit_transform(filtered_z)

        for j, (var_name, cmap) in enumerate([
            ('position', 'magma'),
            ('mov_dir', 'Blues'),
            ('time', 'YlGn')
        ]):
            ax = fig.add_subplot(n_bands, 3, i * 3 + j + 1, projection='3d')
            var = behaviour_dict[var_name]

            p = ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=var, cmap=cmap, s=1, alpha=0.6
            )

            band_label = 'Raw signal' if band_name == 'raw' else band_name
            ax.set_title(f'{band_label} - {var_name}')
            fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1)

            # Remove grid lines
            ax.grid(False)

            # Turn off background panes
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_name = mouse + '_umap_fitlers'
    plt.savefig(os.path.join(figure_directory, save_name + '.png'), dpi=300)
    plt.show()
