import sys, copy, os
import numpy as np
import pandas as pd
import sys, copy, os
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

#%% PARAMS
params = {
    'verbose': True,
    'signal_field': 'rawProb',
    'speed_th': 3,
    'sigma_up': 4,
    'sigma_down': 12,
    'peak_th': 0.05,
    'sig_filt': 6,
    'columns_to_rename': {'Fs':'sf','pos':'position', 'vel':'speed'}
}

def preprocess_traces(traces, sig_filt=6, sig_up=4, sig_down=12, peak_th=0.1):
    lp_traces = uniform_filter1d(traces, size=4000, axis=0)
    clean_traces = gaussian_filter1d(traces, sigma=sig_filt, axis=0)

    for cell in range(clean_traces.shape[1]):
        bleaching = np.histogram(traces[:, cell], 100)
        bleaching = bleaching[1][np.argmax(bleaching[0])]
        bleaching = bleaching + lp_traces[:, cell] - np.min(lp_traces[:, cell])

        clean_traces[:, cell] = clean_traces[:, cell] - bleaching
        clean_traces[:, cell] = clean_traces[:, cell] / np.max(clean_traces[:, cell], axis=0)

    clean_traces[clean_traces < 0] = 0

    conv_traces = np.zeros(clean_traces.shape)

    gaus = lambda x, sig, amp, vo: amp * np.exp(-(((x) ** 2) / (2 * sig ** 2))) + vo;
    x = np.arange(-5 * sig_down, 5 * sig_down, 1);
    left_gauss = gaus(x, sig_up, 1, 0);
    left_gauss[5 * sig_down + 1:] = 0
    right_gauss = gaus(x, sig_down, 1, 0);
    right_gauss[:5 * sig_down + 1] = 0
    gaus_kernel = right_gauss + left_gauss;

    for cell in range(clean_traces.shape[1]):
        peak_idx, _ = find_peaks(clean_traces[:, cell], height=peak_th)
        conv_traces[peak_idx, cell] = clean_traces[peak_idx, cell]
        conv_traces[:, cell] = np.convolve(conv_traces[:, cell], gaus_kernel, 'same')

    return conv_traces


### here we load two or three different datesets

#Information about dataset1 : learning dataset
learning_dir = '/home/melma31/Documents/learning_project/'
deep_sup_dir = '/home/melma31/Documents/deepsup_project/'
data_dir = os.path.join(learning_dir, 'processed_data')
save_dir = os.path.join(learning_dir, 'mutual_info')
if not os.path.isdir(save_dir): os.makedirs(save_dir)
learning_condition = 'learners'
use_values = 'z'
reduce = 'tSNE' ### 'PCA', 'UMAP'
if learning_condition == 'learners':
    mice_list = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
    learning_name_output = learning_condition
    learners = [0, 1, 2, 3, 4]
    learners_names = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
    non_learners = []
    non_learners_names = []
else:
    mice_list = ['M2019', 'M2021', 'M2022', 'M2023', 'M2024', 'M2025', 'M2026']
    learning_name_output = learning_condition + '_non_learners'
    learners = [0,1,2,3,4]
    learners_names = ['M2019', 'M2023', 'M2024', 'M2025', 'M2026']
    non_learners = [5,6]
    non_learners_names = ['M2021', 'M2022']

# information about dataset2

imaging_data = 'egrin'
if imaging_data == 'egrin':
    mice_dict = {'superficial': ['CalbEphys1GRIN1', 'CalbEphys1GRIN2'],
            'deep':['Thy1Ephys1GRIN1', 'Thy1Ephys1GRIN2']
           }

### loading dataset 1
mice_list = learners + non_learners
mice_names = learners_names + non_learners_names
case = 'mov_same_length'
cells = 'all_cells'
signal_name = 'clean_traces'
### re-arange dictionaries for session ordering.

MIR_dict = {'session1':{},'session2':{},'session3':{},'session4':{}}
MI_dict = {'session1':{},'session2':{},'session3':{},'session4':{}}
NMI_dict = {'session1':{},'session2':{},'session3':{},'session4':{}}
MI_total_dict =  {'session1':{},'session2':{},'session3':{},'session4':{}}
# Define learners and non-learners
behavior_labels = ['pos', 'posdir', 'dir', 'speed', 'time', 'inner_trial_time', 'trial_id']
##### create joint organized dictionary
sessions_names = ['session1','session2','session3','session4']
for mouse in mice_names:
    msave_dir = os.path.join(save_dir, mouse) #mouse save dir
    pickle_file = os.path.join(msave_dir,f"{mouse}_mi_{case}_{cells}_{signal_name}_dict.pkl")
    with open(pickle_file, 'rb') as f:
        mi_dict = pickle.load(f)

    session_names = list(mi_dict.keys())
    session_names.sort()
    for idx, session in enumerate(session_names):
        MIR = np.vstack(mi_dict[session]['MIR']).T
        MIR_dict[sessions_names[idx]][mouse] = MIR

MI_session_dict =  {'session1':{},'session2':{},'session3':{},'session4':{}}
mouse_session_list =  {'session1':{},'session2':{},'session3':{},'session4':{}}
for session in sessions_names:
    MIR_list =[]
    mouse_session = []
    for idx,mouse in enumerate(mice_names):
        from scipy.stats import zscore
        n_cells = MIR_dict[session][mouse].shape[0]
        MIR_list.append(MIR_dict[session][mouse])
        mouse_array = np.ones((n_cells,1))*idx
        mouse_session.append(mouse_array)
    MIR = np.vstack(MIR_list)
    mouse_session_final = np.vstack(mouse_session)
    mouse_session_list[session]['mice'] = mouse_session_final
    MI_session_dict[session]['MIR']=MIR

# create dataframe for dataset1
mouse_name_list, session_list = [], []
raw_mi_values = {key: [] for key in behavior_labels}
z_mi_values = {f'z_{key}': [] for key in behavior_labels}
# Create DataFrame similar to mi_pd
for session in sessions_names:
    for mouse in mice_names:
        MIR = MIR_dict[session][mouse]  # shape: (n_cells, n_features)
        data_z = zscore(MIR, axis=1)  # z-score per neuron

        for neuron in range(MIR.shape[0]):
            mouse_name_list.append(mouse)
            session_list.append(session)
            for i, key in enumerate(behavior_labels):
                raw_mi_values[key].append(MIR[neuron, i])
                z_mi_values[f'z_{key}'].append(data_z[neuron, i])
# Combine all into a DataFrame
mi_pd_learners = pd.DataFrame({
    'mouse': mouse_name_list,
    'session': session_list,
    **raw_mi_values,
    **z_mi_values
})

#### subdataselection and parameters for clustering
# Parameters
session_to_use = 'session4'
k = 3
unassigned_cluster_id = -10
# 1. Filter session
mi_pd_learners = mi_pd_learners[mi_pd_learners['session'] == session_to_use].copy()
# 2. Extract z-scored MI features
z_cols = [f'z_{key}' for key in behavior_labels]
mi_raw = mi_pd_learners[z_cols].values
# 3. Standardize features
mi_scaled = StandardScaler().fit_transform(mi_raw)
kmeans = KMeans(n_clusters=k, random_state=42)
initial_clusters = kmeans.fit_predict(mi_scaled)
centroids = kmeans.cluster_centers_
# 6. Keep only 75% closest neurons per cluster
final_cluster_labels = np.full(mi_scaled.shape[0], unassigned_cluster_id)
for cid in range(k):
    cluster_indices = np.where(initial_clusters == cid)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_points = mi_scaled[cluster_indices]
    centroid = centroids[cid]
    dists = np.linalg.norm(cluster_points - centroid, axis=1)
    threshold = np.percentile(dists, 75)
    keep_mask = dists <= threshold
    keep_indices = cluster_indices[keep_mask]
    final_cluster_labels[keep_indices] = cid
# 7. Store results in the DataFrame
mi_pd_learners['cluster'] = final_cluster_labels
########################################################################################################################
base_directory = '/home/melma31/Documents/time_project_grin_lens'
# Setup
data_dir = os.path.join(base_directory, 'output')
signal_name = 'beh_mi_band_hilbert'
mice_files = [f for f in os.listdir(data_dir) if f.endswith(f'_mi_{signal_name}.pkl')]
# Behavioral keys expected
behavior_keys = ['position', '(pos,dir)', 'mov_dir', 'speed', 'time', 'inner_time', 'trial_id']
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}
# Phase and amplitude band MI values
band_names = ['delta', 'theta', 'beta', 'gamma', 'high_gamma']
amp_bands = {f'amp_{b}': [] for b in band_names}
phase_bands = {f'phase_{b}': [] for b in band_names}
# Metadata
mouse_list, area_list, session_type_list = [], [], []
# Loop through files
for file in mice_files:
    mouse = file.split('_mi_')[0]
    with open(os.path.join(data_dir, file), 'rb') as f:
        mouse_dict = pickle.load(f)
    if 'lt' not in mouse_dict:
        continue
    data = np.array(mouse_dict['lt']['MIR'])  # shape: [num_behaviors, num_neurons]
    data_z = zscore(data, axis=1)

    mir_bands = mouse_dict['lt']['MIR_bands']  # dict with 'amplitude' and 'phase'
    amp = np.array(mir_bands['amplitude'])  # shape: [num_neurons, num_bands]
    phase = np.array(mir_bands['phase'])    # shape: [num_neurons, num_bands]

    num_neurons = data.shape[1]
    for neuron in range(num_neurons):
        mouse_list.append(mouse)
        if 'Calb' in mouse:
            area_list.append('superficial')
        elif 'Thy1' in mouse:
            area_list.append('deep')
        session_type_list.append('lt')
        for i, key in enumerate(behavior_keys):
            raw_mi_values[key].append(data[i][neuron])
            z_mi_values[f'z_{key}'].append(data_z[i][neuron])

        for j, b in enumerate(band_names):
            amp_bands[f'amp_{b}'].append(amp[neuron][j])
            phase_bands[f'phase_{b}'].append(phase[neuron][j])

# Create DataFrame
mi_pd = pd.DataFrame({
    'mouse': mouse_list,
    'area': area_list,
    'session_type': session_type_list,
    **raw_mi_values,
    **z_mi_values,
    **amp_bands,
    **phase_bands
})

# Total MI (sum over behavioral labels)
mi_pd['total_MI'] = mi_pd[[*raw_mi_values]].sum(axis=1)
# Optionally filter by session
mi_pd_lt = mi_pd[mi_pd['session_type'] == 'lt']

X_target = mi_pd_lt[[f'z_{key}' for key in behavior_keys]].values
X_target_scaled = StandardScaler().fit_transform(X_target)  # use same preprocessing type
# 5--- Predict cluster assignments using trained KMeans ---
pred_clusters = kmeans.predict(X_target_scaled)
pred_centroids = kmeans.cluster_centers_
# 6 Assign only top 75% closest points per cluster
final_labels = np.full(X_target_scaled.shape[0], -10)  # default: unassigned
for cid in range(k):
    cluster_indices = np.where(pred_clusters == cid)[0]
    if len(cluster_indices) == 0:
        continue
    cluster_points = X_target_scaled[cluster_indices]
    centroid = pred_centroids[cid]
    dists = np.linalg.norm(cluster_points - centroid, axis=1)
    threshold = np.percentile(dists, 75)
    keep_mask = dists <= threshold
    keep_indices = cluster_indices[keep_mask]
    final_labels[keep_indices] = cid
# 7 Store cluster labels in the dataframe
mi_pd_lt['transferred_cluster'] = final_labels

import os
import pickle
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
import umap
from scipy.signal import hilbert
from scipy.stats import pearsonr


# --- Settings ---
base_directory = '/home/melma31/Documents/time_project_grin_lens'
data_directory = os.path.join(base_directory, 'data')
figure_directory = os.path.join(base_directory, 'figures')
os.makedirs(figure_directory, exist_ok=True)

input_file = 'lfp_mice_all_data_dict.pkl'
input_file = os.path.join(data_directory, input_file)
fs = 20
# --- Bands ---
bands = {
    'infra-slow':(0.01,0.1),
    'delta': (1, 3),
    'theta': (8, 12),
    'slow-gamma': (40, 90),
    'high-gamma': (100, 250),
    'MUA': (300, 1000)
}

# --- Helper to save plots ---
def save_figure(fig, name, mouse_name, figure_directory):
    filename = f"{mouse_name}_{name}.png"
    filepath = os.path.join(figure_directory, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"✅ Saved figure: {filepath}")

# --- Load Data ---
with open(input_file, 'rb') as file:
    data = pickle.load(file)

###### first UMAP check

from scipy.ndimage import gaussian_filter1d
umap_view_elev = 30
umap_view_azim = 90
# --- UMAP 3D Exploration by Position and Time ---
fig = plt.figure(figsize=(10, 5 * len(data)))
for i, mouse in enumerate(data.keys()):
    mouse_data = data[mouse]
    signal_all = mouse_data['clean_traces']
    position = mouse_data['position']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 3)[0]
    signal = gaussian_filter1d(signal_all[valid_index], sigma=6, axis=0)
    position = position[valid_index]
    #preprocess_traces(signal,
    #                  sig_filt=params['sig_filt'],
    #                  sig_up=params['sigma_up'],
    #                  sig_down=params['sigma_down'],
    #                  peak_th=params['peak_th'])
    time = np.arange(len(signal)) / fs

    reducer = umap.UMAP(n_neighbors=60, n_components=3, min_dist=0.1, random_state=42)
    signal_umap = reducer.fit_transform(signal)

    ax1 = fig.add_subplot(len(data), 2, 2 * i + 1, projection='3d')
    ax1.view_init(elev=umap_view_elev, azim=umap_view_azim)
    p1 = ax1.scatter(signal_umap[:, 0], signal_umap[:, 1], signal_umap[:, 2], c=time, cmap='viridis', s=3)
    ax1.set_title(f'{mouse} - Colored by Time')
    fig.colorbar(p1, ax=ax1, label='Time (s)')

    ax2 = fig.add_subplot(len(data), 2, 2 * i + 2, projection='3d')
    ax2.view_init(elev=umap_view_elev, azim=umap_view_azim)
    p2 = ax2.scatter(signal_umap[:, 0], signal_umap[:, 1], signal_umap[:, 2], c=position[:,0], cmap='plasma', s=3)
    ax2.set_title(f'{mouse} - Colored by Position')
    fig.colorbar(p2, ax=ax2, label='Position')

plt.tight_layout()
fig.savefig(os.path.join(figure_directory, 'umap3d_all_mice_time_position.png'))

# --- UMAP 2D Exploration by Position and Time ---
fig2 = plt.figure(figsize=(10, 5 * len(data)))
for i, mouse in enumerate(data.keys()):
    mouse_data = data[mouse]
    signal_all = mouse_data['clean_traces']
    position = mouse_data['position']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 3)[0]
    signal = gaussian_filter1d(signal_all[valid_index], sigma=6, axis=0)
    position = position[valid_index]
    #preprocess_traces(signal,
    #                  sig_filt=params['sig_filt'],
    #                  sig_up=params['sigma_up'],
    #                  sig_down=params['sigma_down'],
    #                  peak_th=params['peak_th'])
    time = np.arange(len(signal)) / fs

    reducer = umap.UMAP(n_neighbors=60, n_components=2, min_dist=0.1, random_state=42)
    signal_umap = reducer.fit_transform(signal)

    ax1 = fig2.add_subplot(len(data), 2, 2 * i + 1)
    p1 = ax1.scatter(signal_umap[:, 0], signal_umap[:, 1], c=time, cmap='viridis', s=3)
    ax1.set_title(f'{mouse} - Colored by Time')
    fig2.colorbar(p1, ax=ax1, label='Time (s)')

    ax2 = fig2.add_subplot(len(data), 2, 2 * i + 2)
    p2 = ax2.scatter(signal_umap[:, 0], signal_umap[:, 1], c=position[:,0], cmap='plasma', s=3)
    ax2.set_title(f'{mouse} - Colored by Position')
    fig2.colorbar(p2, ax=ax2, label='Position')

plt.tight_layout()
fig2.savefig(os.path.join(figure_directory, 'umap2d_all_mice_time_position.png'))
# ... (rest of your original code follows here)



pop_amp_dict = defaultdict(dict)
t_dict = defaultdict(dict)
mi_dict = {}
fs = 20  # LFP sampling rate

mi_dict = {}
pearson_dict = {}
plv_dict = {}


for mouse in data.keys():
    mouse_data = data[mouse]
    signal_all = mouse_data['clean_traces']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 3)[0]
    lfp_freqs = mouse_data['cwt_freqs']
    lfp_wavelet = mouse_data['cwt'][valid_index, :, :]
    n_channels = lfp_wavelet.shape[2]

    cluster_list = list(mi_pd_lt[(mi_pd_lt['mouse'] == mouse) & (mi_pd_lt['transferred_cluster'] != -10)]['transferred_cluster'].unique())
    cluster_list.append('all_cells')

    umap_2d_store = {}
    umap_3d_store = {}

    for cluster_id in cluster_list:
        if cluster_id == 'all_cells':
            cluster_indices = mi_pd_lt[mi_pd_lt['mouse'] == mouse].reset_index(drop=True).index.to_numpy()
        else:
            cluster_indices = mi_pd_lt[(mi_pd_lt['mouse'] == mouse) & (mi_pd_lt['transferred_cluster'] == cluster_id)].reset_index(drop=True).index.to_numpy()

        if len(cluster_indices) < 2:
            continue

        signal = signal_all[valid_index][:, cluster_indices]
        if signal.shape[1] < 2:
            continue
        signal = gaussian_filter1d(signal, sigma=6, axis=0)

        #signal = preprocess_traces(signal,sig_filt=params['sig_filt'],
        #              sig_up=params['sigma_up'],
        #              sig_down=params['sigma_down'],
        #              peak_th=params['peak_th'])

        neural_data = signal.T
        pca = PCA(n_components=2)
        neural_pca = pca.fit_transform(neural_data)
        angles = np.arctan2(neural_pca[:, 1], neural_pca[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_signal = signal[:, sorted_indices].T

        reducer_3d = umap.UMAP(random_state=42, n_neighbors=60, n_components=3, min_dist=0.1)
        time_proj_3d = reducer_3d.fit_transform(signal)
        population_amplitude = np.linalg.norm(time_proj_3d, axis=1)
        umap_3d_store[cluster_id] = (time_proj_3d, np.arange(len(signal)) / fs)

        reducer_2d = umap.UMAP(random_state=42, n_neighbors=60, n_components=2, min_dist=0.1)
        time_proj_2d = reducer_2d.fit_transform(signal)
        population_angles = np.arctan2(time_proj_2d[:, 0] - np.mean(time_proj_2d[:, 0]),
                                       time_proj_2d[:, 1] - np.mean(time_proj_2d[:, 1]))
        umap_2d_store[cluster_id] = (time_proj_2d, np.arange(len(signal)) / fs)

        pop_amp_dict[mouse][cluster_id] = population_amplitude
        t_dict[mouse][cluster_id] = np.arange(0, len(population_amplitude))

        lfp_wavelet_mean = np.mean(lfp_wavelet, axis=2)
        band_amplitude_dict = {}
        band_phase_dict = {}
        for band_name, (fmin, fmax) in bands.items():
            band_indices = np.where((lfp_freqs >= fmin) & (lfp_freqs <= fmax))[0]
            if len(band_indices) == 0:
                continue
            band_wavelet = lfp_wavelet_mean[:, band_indices]
            band_signal = np.mean(np.abs(band_wavelet), axis=1)
            band_amplitude_dict[band_name] = band_signal
            band_phase_dict[band_name] = np.angle(hilbert(band_signal))

        def clip_and_normalize(x, q=95):
            clip_val = np.percentile(x, q)
            x = np.clip(x, None, clip_val)
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

        t = np.arange(signal.shape[0]) / fs
        offset = 0
        amplitude_traces = []
        labels = []
        amp = clip_and_normalize(population_amplitude)
        amplitude_traces.append(amp + offset)
        labels.append('Population Amplitude')
        offset += 1.2
        for band_name, band_amp in band_amplitude_dict.items():
            amp = clip_and_normalize(band_amp)
            amplitude_traces.append(amp + offset)
            labels.append(f'{band_name} Amplitude')
            offset += 1.2

        amplitude_traces = np.vstack(amplitude_traces)
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        axs[0].imshow(sorted_signal, aspect='auto', cmap='viridis')
        axs[0].set_title(f'{mouse} - Cluster {cluster_id} - Neural Activity (Sorted)')
        axs[1].set_title(f'{mouse} - Cluster {cluster_id} - Amplitude Signals')
        for i, trace in enumerate(amplitude_traces):
            axs[1].plot(t, trace, label=labels[i])
        axs[1].legend(loc='upper right', ncol=2)
        axs[1].set_xlim([0, t[-1]])
        save_figure(fig, f'cluster{cluster_id}_amplitude_trace_plot', mouse, figure_directory)

        # Scatter time-colored
        fig, axs = plt.subplots(1, len(band_amplitude_dict), figsize=(5 * len(band_amplitude_dict), 4), squeeze=False)
        axs = axs[0]
        norm = plt.Normalize(vmin=0, vmax=len(population_amplitude) / fs)
        for i, (band_name, band_amp) in enumerate(band_amplitude_dict.items()):
            ax = axs[i]
            sc = ax.scatter(band_amp, population_amplitude, c=t, cmap='viridis', alpha=0.7, edgecolors='none')
            ax.set_xlabel(f'{band_name} Amplitude')
            ax.set_ylabel('Population Amplitude')
            ax.set_title(f'{mouse} - Cluster {cluster_id}')
            fig.colorbar(sc, ax=ax, label='Time (s)')
        plt.tight_layout()
        save_figure(fig, f'cluster{cluster_id}_pop_vs_band_scatter_time', mouse, figure_directory)

        # Scatter time-colored
        fig, axs = plt.subplots(1, len(band_amplitude_dict), figsize=(5 * len(band_amplitude_dict), 4), squeeze=False)
        axs = axs[0]
        norm = plt.Normalize(vmin=0, vmax=len(population_amplitude) / fs)
        for i, (band_name, band_phase) in enumerate(band_phase_dict.items()):
            ax = axs[i]
            sc = ax.scatter(band_phase, population_angles, c=t, cmap='viridis', alpha=0.7, edgecolors='none')
            ax.set_xlabel(f'{band_name} Amplitude')
            ax.set_ylabel('Population Angles')
            ax.set_title(f'{mouse} - Cluster {cluster_id}')
            fig.colorbar(sc, ax=ax, label='Time (s)')
        plt.tight_layout()
        save_figure(fig, f'cluster{cluster_id}_pop_vs_band_phases_scatter_time', mouse, figure_directory)


        if mouse not in mi_dict:
            mi_dict[mouse] = {}
            pearson_dict[mouse] = {}
            plv_dict[mouse] = {}
        mi_dict[mouse][cluster_id] = {}
        pearson_dict[mouse][cluster_id] = {}
        plv_dict[mouse][cluster_id] = {}

        #pop_phase = np.angle(hilbert(population_amplitude))
        pop_phase = population_angles
        for band_name in band_amplitude_dict:
            band_amp = band_amplitude_dict[band_name]
            mi_val = mutual_info_regression(band_amp.reshape(-1, 1), population_amplitude, discrete_features=False, random_state=42)[0]
            mi_dict[mouse][cluster_id][band_name] = mi_val
            pearson_val = pearsonr(band_amp, population_amplitude)[0]
            pearson_dict[mouse][cluster_id][band_name] = pearson_val

            band_phase = band_phase_dict[band_name]
            plv = np.abs(np.mean(np.exp(1j * (pop_phase - band_phase))))
            plv_dict[mouse][cluster_id][band_name] = plv

        # --- New Figure: Raw Amplitudes and Phases ---
        from matplotlib.gridspec import GridSpec
        n_bands = len(band_phase_dict)
        height_ratios = [0.5] + [0.5 / n_bands] * n_bands
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(n_bands + 1, 1, height_ratios=height_ratios)
        # Amplitude plot (top 50%)
        ax0 = fig.add_subplot(gs[0])
        offset = 0
        for i, (label, signal_data) in enumerate(
                [('Population Amplitude', population_amplitude)] + list(band_amplitude_dict.items())):
            percentile_range = np.percentile(signal_data, [5, 95])
            scaled = np.clip(signal_data, *percentile_range)
            ax0.plot(t, scaled + offset, label=label, linewidth=1.2)
            offset += np.max(scaled) - np.min(scaled) + 0.5
        ax0.set_ylabel('Amplitude + Offset')
        ax0.set_title(f'{mouse} - Cluster {cluster_id} - Raw Amplitudes (90% range, shifted)')
        ax0.legend(loc='upper right')
        ax0.tick_params(labelbottom=False)

        for i, (band_name, band_phase) in enumerate(band_phase_dict.items()):
            ax = fig.add_subplot(gs[i + 1], sharex=ax0)
            ax.plot(t, pop_phase, label='Population Phase', linewidth=1.2)
            ax.plot(t, band_phase, label=f'{band_name} Phase', linewidth=1.2)
            ax.set_ylabel('Phase (rad)')
            ax.legend(loc='upper right')

            if i != n_bands - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time (s)')

            if i == 0:
                ax.set_title(f'{mouse} - Cluster {cluster_id} - Phase Signals by Band')

        plt.tight_layout()
        save_figure(fig, f'cluster{cluster_id}_raw_amp_phase', mouse, figure_directory)

    # UMAP 2D summary plot
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, (cluster_id, (coords, t_umap)) in enumerate(umap_2d_store.items()):
        if i >= 4:
            break
        axs[i].scatter(coords[:, 0], coords[:, 1], c=t_umap, cmap='viridis', s=5)
        axs[i].set_title(f'{mouse} - {cluster_id}')
        axs[i].set_xlabel('UMAP 1')
        axs[i].set_ylabel('UMAP 2')
    plt.tight_layout()
    save_figure(fig, f'{mouse}_umap_2d_summary', mouse, figure_directory)

    # UMAP 3D summary plot
    fig = plt.figure(figsize=(16, 4))
    for i, (cluster_id, (coords, t_umap)) in enumerate(umap_3d_store.items()):
        if i >= 4:
            break
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        p = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=t_umap, cmap='viridis', s=3)
        ax.set_title(f'{mouse} - {cluster_id}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
    plt.tight_layout()
    save_figure(fig, f'{mouse}_umap_3d_summary', mouse, figure_directory)

# --- Summary Plots with Boxplot + Stripplot ---
all_records = []
for mouse in mi_dict:
    for cluster_id in mi_dict[mouse]:
        for band in mi_dict[mouse][cluster_id]:
            all_records.append({
                'mouse': mouse,
                'cluster': str(cluster_id),
                'band': band,
                'mi': mi_dict[mouse][cluster_id][band],
                'pearson': pearson_dict[mouse][cluster_id][band],
                'plv': plv_dict[mouse][cluster_id][band]
            })

all_df = pd.DataFrame(all_records)

for metric in ['mi', 'pearson', 'plv']:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=all_df, x='band', y=metric, hue='cluster', ax=ax, showfliers=False)
    sns.stripplot(data=all_df, x='band', y=metric, hue='cluster', dodge=True, jitter=True,color = 'k', ax=ax, marker='o', alpha=0.6)
    ax.set_ylabel(f'{metric.upper()} Value')
    ax.set_title(f'{metric.upper()}: Pop Amp vs Band (Mice + Clusters)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_directory, f'summary_{metric}_by_cluster_and_band_box_strip.png'))
