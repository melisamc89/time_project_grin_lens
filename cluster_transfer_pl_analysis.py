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
if imaging_data == 'miniscope_and_egrin':
    mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9',
                                 'CalbEphys1GRIN1', 'CalbEphys1GRIN2'],
                'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1',
                        'Thy1Ephys1GRIN1', 'Thy1Ephys1GRIN2']
                }
if imaging_data == 'egrin':
    mice_dict = {'superficial': ['CalbEphys1GRIN1', 'CalbEphys1GRIN2'],
            'deep':['Thy1Ephys1GRIN1', 'Thy1Ephys1GRIN2']
           }
if imaging_data == 'miniscope':
    mice_dict = {'superficial': ['CGrin1','CZ3','CZ4','CZ6','CZ8','CZ9'],
          'deep':['ChZ4','ChZ7','ChZ8','GC2','GC3','GC7','GC5_nvista','TGrin1']
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
# 4. Dimensionality reduction
if reduce == 'PCA':
    reducer = PCA(n_components=2)
    mi_reduced = reducer.fit_transform(mi_scaled)
    reducer_name = 'PC'
if reduce == 'tSNE':
    from openTSNE import TSNE as openTSNE
    from openTSNE import affinity, initialization
    # Assume mi_scaled is already StandardScaler()'d from dataset 1
    aff = affinity.PerplexityBasedNN(mi_scaled, perplexity=50, metric="euclidean")
    init = initialization.pca(mi_scaled)
    tsne_model = openTSNE(n_components=2, perplexity=50, initialization=init, random_state=42)
    # Learn t-SNE embedding on dataset 1
    tsne_embedding_learners = tsne_model.fit(mi_scaled)
    mi_pd_learners['tSNE1'] = tsne_embedding_learners[:, 0]
    mi_pd_learners['tSNE2'] = tsne_embedding_learners[:, 1]
    reducer_name = 'tSNE'
# 5. Clustering in original feature space
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
if reduce == 'PCA':
    mi_pd_learners[f'{reducer_name}1'] = mi_reduced[:, 0]
    mi_pd_learners[f'{reducer_name}2'] = mi_reduced[:, 1]


########################################################################################################################
import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore

base_directory = '/home/melma31/Documents/time_project_grin_lens'

# Define frequency bands
bands = {
    'infra-slow': (0.01, 0.1),
    'delta': (1, 3),
    'theta': (8, 12),
    'slow-gamma': (40, 90),
    'high-gamma': (100, 250),
    'MUA': (300, 1000)
}

# Setup
data_dir = os.path.join(base_directory, 'output')
signal_name = 'mi_beh_mi_band_pl'
mice_files = [f for f in os.listdir(data_dir) if f.endswith(f'{signal_name}.pkl')]

# Behavioral keys
behavior_keys = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']

# Prepare storage
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}
sig_flags = {f'{key}_sig': [] for key in behavior_keys}

band_names = list(bands.keys())
amp_bands = {f'{b}_pl_mean': [] for b in band_names}
phase_bands = {f'{b}_pl_std': [] for b in band_names}
sig_amp_flags = {f'{b}_pl_mean_sig': [] for b in band_names}
sig_phase_flags = {f'{b}_pl_std_sig': [] for b in band_names}

# Metadata
mouse_list, area_list, session_type_list = [], [], []

# Process each file
for file in mice_files:
    mouse = file.split('_mi_')[0]
    with open(os.path.join(data_dir, file), 'rb') as f:
        mouse_dict = pickle.load(f)

    if 'lt' not in mouse_dict:
        continue

    data = np.array(mouse_dict['lt']['MIR'])  # shape: (7, n_neurons)
    data_z = zscore(data, axis=1)

    mir_bands = mouse_dict['lt']['MIR_bands']
    amp = np.array(mir_bands['amplitude'])  # shape: (n_neurons, 6)
    phase = np.array(mir_bands['phase'])    # shape: (n_neurons, 6)
    pl_mean = np.array(mir_bands['pl_mean'])
    pl_std = np.array(mir_bands['pl_std'])

    num_neurons = data.shape[1]

    for neuron in range(num_neurons):
        mouse_list.append(mouse)
        area_list.append('superficial' if 'Calb' in mouse else 'deep')
        session_type_list.append('lt')

        # Behavioral MI values
        for i, key in enumerate(behavior_keys):
            real_val = data[i][neuron]

            raw_mi_values[key].append(real_val)
            z_mi_values[f'z_{key}'].append(data_z[i][neuron])
            sig_flags[f'{key}_sig'].append('s')

        # Band amplitude and phase MI values
        for j, b in enumerate(band_names):
            real_amp = pl_mean[neuron][j]
            real_phase = pl_std[neuron][j]

            amp_bands[f'{b}_pl_mean'].append(real_amp)
            phase_bands[f'{b}_pl_std'].append(real_phase)

            sig_amp_flags[f'{b}_pl_mean_sig'].append('s')
            sig_phase_flags[f'{b}_pl_std_sig'].append('s')

# Combine all into a single DataFrame
mi_pd = pd.DataFrame({
    'mouse': mouse_list,
    'area': area_list,
    'session_type': session_type_list,
    **raw_mi_values,
    **z_mi_values,
    **amp_bands,
    **phase_bands,
    **sig_flags,
    **sig_amp_flags,
    **sig_phase_flags
})

# Optional: save to file
# mi_pd.to_pickle(os.path.join(base_directory, 'mi_significance_flags.pkl'))
# Optional: save to disk
# mi_pd.to_pickle(os.path.join(base_directory, 'mi_significant_df.pkl'))

# Total MI (sum over behavioral labels)
mi_pd['total_MI'] = mi_pd[[*raw_mi_values]].sum(axis=1)
# Optionally filter by session
mi_pd_lt = mi_pd[mi_pd['session_type'] == 'lt']

X_target = mi_pd_lt[[f'z_{key}' for key in behavior_keys]].values
X_target_scaled = StandardScaler().fit_transform(X_target)  # use same preprocessing type
# 4--- Apply PCA transformation learned from dataset 1 ---
if reduce == 'PCA':
    X_target_pca = reducer.transform(X_target_scaled)
    mi_pd_lt[f'{reducer_name}1'] = X_target_pca[:, 0]
    mi_pd_lt[f'{reducer_name}2'] = X_target_pca[:, 1]
if reduce == 'tSNE':
    # Use the same preprocessing as dataset 1 (StandardScaler)
    X_target = mi_pd_lt[[f'z_{key}' for key in behavior_keys]].values
    X_target_scaled = StandardScaler().fit_transform(X_target)  # same method as mi_scaled
    # Transform into dataset 1's t-SNE space
    tsne_embedding_target = tsne_embedding_learners.transform(X_target_scaled)
    mi_pd_lt['tSNE1'] = tsne_embedding_target[:, 0]
    mi_pd_lt['tSNE2'] = tsne_embedding_target[:, 1]
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
    threshold = np.percentile(dists, 99)
    keep_mask = dists <= threshold
    keep_indices = cluster_indices[keep_mask]
    final_labels[keep_indices] = cid

# 7 Store cluster labels in the dataframe
mi_pd_lt['transferred_cluster'] = final_labels

################################################################################################################
#################                        STARTS PLOTTING                                ########################
################################################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# Define amplitude columns
amp_columns = ['infra-slow_pl_mean', 'delta_pl_mean', 'theta_pl_mean', 'slow-gamma_pl_mean', 'high-gamma_pl_mean', 'MUA_pl_mean']
sig_columns = [f'{col}_sig' for col in amp_columns]
# Custom cluster color palette
custom_cluster_palette = {
    0: '#bce784ff',  # light green
    1: '#66cef4ff',  # light blue
    2: '#ec8ef8ff'   # light pink
}
# Prepare long-format dataframe with per-feature significance filtering
plot_data = []
for amp_col, sig_col in zip(amp_columns, sig_columns):
    sub_df = mi_pd_lt[
        (mi_pd_lt['transferred_cluster'].isin(custom_cluster_palette.keys())) &
        (mi_pd_lt[sig_col] == 's') &
        (~mi_pd_lt[amp_col].isna())
    ]
    melted = pd.DataFrame({
        'transferred_cluster': sub_df['transferred_cluster'],
        'Frequency Band': amp_col,
        'Phase': sub_df[amp_col]
    })
    plot_data.append(melted)
df_amp_melted = pd.concat(plot_data, ignore_index=True)
# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_amp_melted, x='Frequency Band', y='Phase',
                 hue='transferred_cluster', palette=custom_cluster_palette)
# Create all cluster pair combinations per band
cluster_vals = sorted(df_amp_melted['transferred_cluster'].unique())
pairs = []
for band in amp_columns:
    for i in range(len(cluster_vals)):
        for j in range(i + 1, len(cluster_vals)):
            pairs.append(((band, cluster_vals[i]), (band, cluster_vals[j])))
# Annotate stats
annotator = Annotator(ax, pairs, data=df_amp_melted,
                      x='Frequency Band', y='Phase', hue='transferred_cluster')
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
annotator.apply_and_annotate()
plt.title('Pl MEAN by Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_{signal_name}_PL_mean_by_cluster_{imaging_data}.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_{signal_name}_PL_mean_by_cluster_{imaging_data}.svg'), dpi=400, bbox_inches="tight")

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# Define amplitude columns
amp_columns = ['infra-slow_pl_std', 'delta_pl_std', 'theta_pl_std', 'slow-gamma_pl_std', 'high-gamma_pl_std', 'MUA_pl_std']
sig_columns = [f'{col}_sig' for col in amp_columns]
# Custom cluster color palette
custom_cluster_palette = {
    0: '#bce784ff',  # light green
    1: '#66cef4ff',  # light blue
    2: '#ec8ef8ff'   # light pink
}
# Prepare long-format dataframe with per-feature significance filtering
plot_data = []
for amp_col, sig_col in zip(amp_columns, sig_columns):
    sub_df = mi_pd_lt[
        (mi_pd_lt['transferred_cluster'].isin(custom_cluster_palette.keys())) &
        (mi_pd_lt[sig_col] == 's') &
        (~mi_pd_lt[amp_col].isna())
    ]
    melted = pd.DataFrame({
        'transferred_cluster': sub_df['transferred_cluster'],
        'Frequency Band': amp_col,
        'PhaseSTD': sub_df[amp_col]
    })
    plot_data.append(melted)
df_amp_melted = pd.concat(plot_data, ignore_index=True)
# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_amp_melted, x='Frequency Band', y='PhaseSTD',
                 hue='transferred_cluster', palette=custom_cluster_palette)
# Create all cluster pair combinations per band
cluster_vals = sorted(df_amp_melted['transferred_cluster'].unique())
pairs = []
for band in amp_columns:
    for i in range(len(cluster_vals)):
        for j in range(i + 1, len(cluster_vals)):
            pairs.append(((band, cluster_vals[i]), (band, cluster_vals[j])))
# Annotate stats
annotator = Annotator(ax, pairs, data=df_amp_melted,
                      x='Frequency Band', y='PhaseSTD', hue='transferred_cluster')
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
annotator.apply_and_annotate()
plt.title('Pl STD by Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_{signal_name}_PL_std_by_cluster_{imaging_data}.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_{signal_name}_PL_std_by_cluster_{imaging_data}.svg'), dpi=400, bbox_inches="tight")

plt.show()


# Define MIR columns and their significance labels
mir_columns = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']
mir_sig_columns = [f'{col}_sig' for col in mir_columns]

# Prepare long-format dataframe with per-feature significance filtering
plot_data_mir = []

for mir_col, sig_col in zip(mir_columns, mir_sig_columns):
    sub_df = mi_pd_lt[
        (mi_pd_lt['transferred_cluster'].isin(custom_cluster_palette.keys())) &
        (mi_pd_lt[sig_col] == 's') &
        (~mi_pd_lt[mir_col].isna())
    ]
    melted = pd.DataFrame({
        'transferred_cluster': sub_df['transferred_cluster'],
        'Feature': mir_col,
        'MI': sub_df[mir_col]
    })
    plot_data_mir.append(melted)

df_mir_melted = pd.concat(plot_data_mir, ignore_index=True)

# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_mir_melted, x='Feature', y='MI',
                 hue='transferred_cluster', palette=custom_cluster_palette)

# Cluster pairs for annotations
cluster_vals = sorted(df_mir_melted['transferred_cluster'].unique())
pairs = []
for feat in mir_columns:
    for i in range(len(cluster_vals)):
        for j in range(i + 1, len(cluster_vals)):
            pairs.append(((feat, cluster_vals[i]), (feat, cluster_vals[j])))

# Annotate
annotator = Annotator(ax, pairs, data=df_mir_melted,
                      x='Feature', y='MI', hue='transferred_cluster')
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
annotator.apply_and_annotate()

plt.title('Behavioral MI by Cluster (Only Feature-Specific Significant Values)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_beh_{imaging_data}.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_beh_{imaging_data}.svg'), dpi=400, bbox_inches="tight")
plt.show()

####################################################################

import matplotlib.pyplot as plt

# Decide which embedding to use
x_axis = 'tSNE1' if reduce == 'tSNE' else f'{reducer_name}1'
y_axis = 'tSNE2' if reduce == 'tSNE' else f'{reducer_name}2'

fig, axes = plt.subplots(2, len(band_names), figsize=(4 * len(band_names), 8))
fig.suptitle('t-SNE embedding colored by MI (amplitude and phase)', fontsize=16)

for i, band in enumerate(band_names):
    # Plot amplitude
    ax_amp = axes[0, i]
    sc_amp = ax_amp.scatter(
        mi_pd_lt[x_axis],
        mi_pd_lt[y_axis],
        c=mi_pd_lt[f'{band}_pl_mean'],
        cmap='coolwarm',
        s=10,
#        vmin = 0,
#        vmax = 0.05
    )
    ax_amp.set_title(f'Amplitude: {band}')
    ax_amp.set_xlabel(x_axis)
    ax_amp.set_ylabel(y_axis)
    plt.colorbar(sc_amp, ax=ax_amp, fraction=0.046, pad=0.04)

    # Plot phase
    ax_phase = axes[1, i]
    sc_phase = ax_phase.scatter(
        mi_pd_lt[x_axis],
        mi_pd_lt[y_axis],
        c=mi_pd_lt[f'{band}_pl_std'],
        cmap='twilight',
#        vmin=0,
#        vmax=0.005,
        s=10
    )
    ax_phase.set_title(f'Phase: {band}')
    ax_phase.set_xlabel(x_axis)
    ax_phase.set_ylabel(y_axis)
    plt.colorbar(sc_phase, ax=ax_phase, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_phase_locking_mean_std_{imaging_data}.png'), dpi=400, bbox_inches="tight")

plt.show()
