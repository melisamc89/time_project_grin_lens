import os
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from general_utils import *

import pickle
import matplotlib.pyplot as plt
import pickle as pkl


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
    'delta': (1.2, 4),
    'theta': (6, 12),
    'beta': (20, 40),
    'gamma': (60, 150),
    'high_gamma': (150, 250)
}

mice_list = list(data.keys())
mi_dict = dict()
for mouse in mice_list:
    mi_dict[mouse] = dict()
    mouse_data = data[mouse]

    signal = mouse_data['clean_traces']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 6)[0]
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
    for beh_index, beh in enumerate(behaviours_list):
        print('MI for variable:' + beh_names[beh_index])
        mi = []
        for neui in range(signal.shape[1]):
            neuron_mi = \
                mutual_info_regression(signal[:, neui].reshape(-1, 1), beh, n_neighbors=50,
                                       random_state=16)[0]
            mi.append(neuron_mi)
            # Store coherence per neuron and per band
            if beh_index == 0:
                frequencies = mouse_data['cwt_freqs']
                lfp_wavelet = np.mean(mouse_data['cwt'],axis = 2)
                lfp_wavelet = lfp_wavelet[valid_index,:]
                mi_bands = []
                for band_name, band_range in bands.items():
                    band_min, band_max = bands[band_name]
                    band_mask = (frequencies >= band_min) & (frequencies <= band_max)

                    # Select only frequencies in the band
                    lfp_band = lfp_wavelet[:,band_mask]
                    lfp_band = np.mean(lfp_band,axis = 1)
                    neuron_mi_band = \
                        mutual_info_regression(signal[:, neui].reshape(-1, 1), lfp_band, n_neighbors=50,
                                               random_state=16)[0]
                    mi_bands.append(neuron_mi_band)
                mi_bands_all.append(mi_bands)
        mi_all.append(mi)
        mi_bands_all = np.array(mi_bands_all)

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

with open(os.path.join(output_directory,'mi_beh_mi_band.pkl'), 'wb') as f:
    pkl.dump(mi_dict, f)


########################################################################################################
#
########################################################################################################
import pandas as pd
from scipy.stats import zscore
import numpy as np

with open(os.path.join(output_directory,'mi_beh_mi_band.pkl'), 'rb') as file:
    data = pkl.load(file)

rows = []

for mouse, mouse_data in mi_dict.items():
    mir = mouse_data['MIR']  # List: one list per behavior, each list has MI values for all neurons
    mir_band = mouse_data['MIR_bands'].T
    n_neurons = len(mir[0])  # Number of neurons

    # Assign area based on mouse name
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
        # Collect MI values for this neuron across behaviors
        mi_values = []
        for beh_name, mi_list in zip(beh_names, mir):
            mi_value = mi_list[neuron_idx]
            row[beh_name] = mi_value
            mi_values.append(mi_value)
        mi_values_band = []
        for band_name, mi_list_band in zip(bands.keys(), mir_band):
            mi_value = mi_list_band[neuron_idx]
            row[band_name] = mi_value
            mi_values_band.append(mi_value)

        # Compute z-scored MI values for this neuron
        mi_values = np.array(mi_values)
        mi_zscore = zscore(mi_values, nan_policy='omit')  # safe if NaNs exist

        # Compute z-scored MI values for this neuron
        mi_values_band = np.array(mi_values_band)
        mi_zscore_band = zscore(mi_values_band, nan_policy='omit')  # safe if NaNs exist

        # Add z-scored MI values
        for beh_name, z_value in zip(beh_names, mi_zscore):
            row[f'{beh_name}_zscore'] = z_value

        for freq_band, z_val in zip(bands.keys(), mi_zscore_band):
            row[f'{freq_band}_zscore'] = z_val

        rows.append(row)

# Create the DataFrame
neuron_mi_df = pd.DataFrame(rows)




#####################################################################
#### PLOTING
import seaborn as sns
import matplotlib.pyplot as plt

# Define MI variables (behavior + frequency)
mi_vars = beh_names + list(bands.keys())

# Melt to long format for seaborn
long_df = neuron_mi_df.melt(id_vars='area', value_vars=mi_vars,
                            var_name='variable', value_name='MI_value')

# Keep only 'sup' and 'deep' areas
long_df = long_df[long_df['area'].isin(['sup', 'deep'])]

# Define custom palette
palette = {'sup': 'purple', 'deep': 'gold'}

# Plot
plt.figure(figsize=(2 * len(mi_vars), 6))
sns.boxenplot(
    data=long_df,
    x='variable',
    y='MI_value',
    hue='area',
    palette=palette
)

plt.title("MI Distributions by Variable and Area")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mutual Information")
plt.xlabel("Variable")
plt.tight_layout()

# Save
violin_path = os.path.join(figure_directory, "MI_ViolinPlot_ByArea_SplitColor.png")
plt.savefig(violin_path, dpi=300)
plt.close()

print(f"Saved violin plot to: {violin_path}")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Frequency band order
freq_order = bands.keys()

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
        df = neuron_mi_df[[beh, freq, 'area']].dropna()
        x = df[beh].values
        y = df[freq].values
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
scatter_path = os.path.join(figure_directory, "MI_vs_Frequency_ScatterGrid_PearsonByArea.png")
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved color-coded scatter grid with Pearson r to: {scatter_path}")

# Create DataFrame of Pearson r values
detailed_r_df = pd.DataFrame(detailed_pearson_data)

# Set up subplot grid
n_beh = len(beh_names)
fig, axs = plt.subplots(n_beh, 1, figsize=(8, 3 * n_beh), sharex=True)

# Ensure axs is iterable even if n_beh = 1
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
summary_path = os.path.join(figure_directory, "Pearson_r_vs_Frequency_ByArea_Subplots.png")
plt.savefig(summary_path, dpi=300)
plt.close()

print(f"Saved Pearson r per-behavior subplot figure to: {summary_path}")

###########################CLUSTERS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os

# Behavioral and frequency MI variable names
zscore_beh_cols = [f"{b}_zscore" for b in beh_names]
freq_cols = list(bands.keys())

# Prepare valid data: drop NaNs in behavioral zscores
zscore_data = neuron_mi_df[zscore_beh_cols].dropna()
valid_indices = zscore_data.index

# Get t-SNE embedding
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embedding = tsne.fit_transform(zscore_data.values)

# Get area
area = neuron_mi_df.loc[valid_indices, 'area'].values
palette = {'sup': 'purple', 'deep': 'gold'}
area_colors = [palette.get(a, 'gray') for a in area]

# Run KMeans clustering
kmeans_labels = {}
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42)
    kmeans_labels[k] = km.fit_predict(zscore_data.values)

# Total subplots
n_beh = len(beh_names)
n_freq = len(freq_cols)
n_clusters = len(kmeans_labels)
n_extra = 1  # for area
n_total = n_beh + n_freq + n_clusters + n_extra

# Grid dimensions
n_cols = 4
n_rows = int(np.ceil(n_total / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

# --- Plot behavioral MI coloring ---
for i, beh in enumerate(beh_names):
    ax = axs[i // n_cols, i % n_cols]
    values = neuron_mi_df.loc[valid_indices, beh]
    sc = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=values,
                    cmap='coolwarm', s=10, vmin=0, vmax=0.4)
    ax.set_title(f"t-SNE colored by {beh}")
    plt.colorbar(sc, ax=ax)

# --- Plot frequency MI coloring ---
offset = len(beh_names)
for j, freq in enumerate(freq_cols):
    ax = axs[(j + offset) // n_cols, (j + offset) % n_cols]
    values = neuron_mi_df.loc[valid_indices, freq]
    sc = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=values,
                    cmap='coolwarm', s=10, vmin=0, vmax=0.1)
    ax.set_title(f"t-SNE colored by {freq}")
    plt.colorbar(sc, ax=ax)

# --- KMeans clusters (k = 2–5) ---
offset += len(freq_cols)
for idx, (k, labels) in enumerate(kmeans_labels.items()):
    ax = axs[(idx + offset) // n_cols, (idx + offset) % n_cols]
    ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=labels, cmap='tab10', s=10)
    ax.set_title(f"KMeans Clusters (k={k})")

# --- Area coloring ---
final_idx = offset + len(kmeans_labels)
ax = axs[final_idx // n_cols, final_idx % n_cols]
ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=area_colors, s=10)
ax.set_title("t-SNE colored by Area")

# Clean up all axes
for row in axs:
    for ax in row:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
tsne_path = os.path.join(figure_directory, "tSNE_AllMI_Colored.png")
plt.savefig(tsne_path, dpi=300)
plt.close()

print(f"Saved t-SNE figure with all MI variables to: {tsne_path}")
