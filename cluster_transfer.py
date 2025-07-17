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
signal_name = 'beh_mi_band_hilbert_shuffle'
mice_files = [f for f in os.listdir(data_dir) if f.endswith(f'_mi_{signal_name}.pkl')]

# Behavioral keys
behavior_keys = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']

# Prepare storage
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}
sig_flags = {f'{key}_sig': [] for key in behavior_keys}

band_names = list(bands.keys())
amp_bands = {f'{b}_amp': [] for b in band_names}
phase_bands = {f'{b}_phase': [] for b in band_names}
sig_amp_flags = {f'{b}_amp_sig': [] for b in band_names}
sig_phase_flags = {f'{b}_phase_sig': [] for b in band_names}

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

    mir_shuffle = np.array(mouse_dict['lt']['MIR_shuffle'])  # shape: (100, n_neurons, 7)
    amp_shuffle = np.array(mouse_dict['lt']['MIR_bands_shuffle']['amplitude'])  # shape: (100, n_neurons, 6)
    phase_shuffle = np.array(mouse_dict['lt']['MIR_bands_shuffle']['phase'])    # shape: (100, n_neurons, 6)

    num_neurons = data.shape[1]

    for neuron in range(num_neurons):
        mouse_list.append(mouse)
        area_list.append('superficial' if 'Calb' in mouse else 'deep')
        session_type_list.append('lt')

        # Behavioral MI values
        for i, key in enumerate(behavior_keys):
            real_val = data[i][neuron]
            shuffle_vals = mir_shuffle[:, i, neuron]
            thresh = np.percentile(shuffle_vals, 95)

            raw_mi_values[key].append(real_val)
            z_mi_values[f'z_{key}'].append(data_z[i][neuron])
            sig_flags[f'{key}_sig'].append('s' if real_val > thresh else 'n')

        # Band amplitude and phase MI values
        for j, b in enumerate(band_names):
            real_amp = amp[neuron][j]
            real_phase = phase[neuron][j]
            amp_shuffle_vals = amp_shuffle[:, neuron, j]
            phase_shuffle_vals = phase_shuffle[:, neuron, j]

            amp_bands[f'{b}_amp'].append(real_amp)
            phase_bands[f'{b}_phase'].append(real_phase)

            sig_amp_flags[f'{b}_amp_sig'].append('s' if real_amp > np.percentile(amp_shuffle_vals, 95) else 'n')
            sig_phase_flags[f'{b}_phase_sig'].append('s' if real_phase > np.percentile(phase_shuffle_vals, 95) else 'n')

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
amp_columns = ['infra-slow_amp', 'delta_amp', 'theta_amp', 'slow-gamma_amp', 'high-gamma_amp', 'MUA_amp']
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
        'Amplitude': sub_df[amp_col]
    })
    plot_data.append(melted)
df_amp_melted = pd.concat(plot_data, ignore_index=True)
# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_amp_melted, x='Frequency Band', y='Amplitude',
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
                      x='Frequency Band', y='Amplitude', hue='transferred_cluster')
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
annotator.apply_and_annotate()
plt.title('Amplitude by Cluster (Only Feature-Specific Significant Values)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_lfp_amp_{imaging_data}.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_lfp_amp_{imaging_data}.svg'), dpi=400, bbox_inches="tight")

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

# Define phase columns and significance labels
phase_columns = ['infra-slow_phase', 'delta_phase', 'theta_phase',
                 'slow-gamma_phase', 'high-gamma_phase', 'MUA_phase']
phase_sig_columns = [f'{col}_sig' for col in phase_columns]

# Prepare long-format dataframe with per-feature significance filtering
plot_data_phase = []

for phase_col, sig_col in zip(phase_columns, phase_sig_columns):
    sub_df = mi_pd_lt[
        (mi_pd_lt['transferred_cluster'].isin(custom_cluster_palette.keys())) &
        (mi_pd_lt[sig_col] == 's') &
        (~mi_pd_lt[phase_col].isna())
    ]
    melted = pd.DataFrame({
        'transferred_cluster': sub_df['transferred_cluster'],
        'Frequency Band': phase_col,
        'Phase MI': sub_df[phase_col]
    })
    plot_data_phase.append(melted)

df_phase_melted = pd.concat(plot_data_phase, ignore_index=True)

# Plot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=df_phase_melted, x='Frequency Band', y='Phase MI',
                 hue='transferred_cluster', palette=custom_cluster_palette)

# Cluster pairs for annotations
cluster_vals = sorted(df_phase_melted['transferred_cluster'].unique())
pairs = []
for band in phase_columns:
    for i in range(len(cluster_vals)):
        for j in range(i + 1, len(cluster_vals)):
            pairs.append(((band, cluster_vals[i]), (band, cluster_vals[j])))

# Annotate
annotator = Annotator(ax, pairs, data=df_phase_melted,
                      x='Frequency Band', y='Phase MI', hue='transferred_cluster')
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
annotator.apply_and_annotate()

plt.title('Phase MI by Cluster (Only Feature-Specific Significant Values)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_phase_{imaging_data}.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_phase_{imaging_data}.svg'), dpi=400, bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt

mir_columns = ['Position', 'DirPosition', 'MovDir', 'Speed', 'Time', 'InnerTime', 'TrialID']
cluster_ids = sorted(mi_pd_lt['transferred_cluster'].unique())
cluster_ids = [cid for cid in cluster_ids if cid in custom_cluster_palette]  # exclude -10, -1, etc.

mir_sig_df = []

for feature in mir_columns:
    sig_col = f'{feature}_sig'
    for cluster in cluster_ids:
        cluster_mask = mi_pd_lt['transferred_cluster'] == cluster
        total = cluster_mask.sum()
        sig_count = ((mi_pd_lt[sig_col] == 's') & cluster_mask).sum()
        perc = 100 * sig_count / total if total > 0 else 0
        mir_sig_df.append({'Feature': feature, 'Cluster': cluster, 'Percentage': perc})

mir_sig_df = pd.DataFrame(mir_sig_df)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=mir_sig_df, x='Feature', y='Percentage', hue='Cluster', palette=custom_cluster_palette)
plt.title('Percentage of Significant Behavioral MI Neurons per Cluster')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_behaviour_{imaging_data}_pecentage.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_behaviour_{imaging_data}_percentage.svg'), dpi=400, bbox_inches="tight")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Amplitude features
amp_columns = ['infra-slow_amp', 'delta_amp', 'theta_amp',
               'slow-gamma_amp', 'high-gamma_amp', 'MUA_amp']
amp_sig_df = []
for feature in amp_columns:
    sig_col = f'{feature}_sig'
    for cluster in cluster_ids:
        cluster_mask = mi_pd_lt['transferred_cluster'] == cluster
        total = cluster_mask.sum()
        sig_count = ((mi_pd_lt[sig_col] == 's') & cluster_mask).sum()
        perc = 100 * sig_count / total if total > 0 else 0
        amp_sig_df.append({'Band': feature.replace('_amp', ''), 'Cluster': cluster, 'Percentage': perc})
amp_sig_df = pd.DataFrame(amp_sig_df)
# Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=amp_sig_df, x='Band', y='Percentage', hue='Cluster', palette=custom_cluster_palette)
plt.title('Percentage of Neurons with Significant Amplitude MI per Cluster')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_amplitude_{imaging_data}_pecentage.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_amplitude_{imaging_data}_percentage.svg'), dpi=400, bbox_inches="tight")
plt.show()

# Phase features
phase_columns = ['infra-slow_phase', 'delta_phase', 'theta_phase',
                 'slow-gamma_phase', 'high-gamma_phase', 'MUA_phase']
phase_sig_df = []
for feature in phase_columns:
    sig_col = f'{feature}_sig'
    for cluster in cluster_ids:
        cluster_mask = mi_pd_lt['transferred_cluster'] == cluster
        total = cluster_mask.sum()
        sig_count = ((mi_pd_lt[sig_col] == 's') & cluster_mask).sum()
        perc = 100 * sig_count / total if total > 0 else 0
        phase_sig_df.append({'Band': feature.replace('_phase', ''), 'Cluster': cluster, 'Percentage': perc})
phase_sig_df = pd.DataFrame(phase_sig_df)
# Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=phase_sig_df, x='Band', y='Percentage', hue='Cluster', palette=custom_cluster_palette)
plt.title('Percentage of Neurons with Significant Phase MI per Cluster')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_phase_{imaging_data}_pecentage.png'), dpi=400, bbox_inches="tight")
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_phase_{imaging_data}_percentage.svg'), dpi=400, bbox_inches="tight")
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
        c=mi_pd_lt[f'{band}_amp'],
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
        c=mi_pd_lt[f'{band}_phase'],
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
plt.savefig(os.path.join(data_dir, f'MI_transferred_cluster_all_{signal_name}_area_mouse_lfp_amp_phase_{imaging_data}.png'), dpi=400, bbox_inches="tight")

plt.show()
#################################################################3
############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from itertools import product
# Behavioral keys expected
raw_mi_values = {key: [] for key in behavior_keys}
z_mi_values = {f'z_{key}': [] for key in behavior_keys}

# Prepare directory for figures
fig_dir = os.path.join(data_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

color_all = '#bbbcc0ff'
color_deep = '#cc9900'
color_sup = '#9900ff'

# Storage for correlations
corrs_amp = {k: {'all': [], 'deep': [], 'sup': []} for k in behavior_keys}
corrs_phase = {k: {'all': [], 'deep': [], 'sup': []} for k in behavior_keys}

# Function to compute and plot scatter with fits
def scatter_fits(x, y, area, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))
    palette = {'deep': color_deep, 'superficial': color_sup}

    # Scatter with hue
    sns.scatterplot(x=x, y=y, hue=area, palette=palette, s=10, alpha=0.6)

    # Fit lines
    for group, color in zip(['all', 'deep', 'superficial'], [color_all, color_deep, color_sup]):
        if group == 'all':
            x_vals, y_vals = x, y
        else:
            mask = area == group
            x_vals, y_vals = x[mask], y[mask]
        if len(x_vals) > 2:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = slope * xs + intercept
            plt.plot(xs, ys, color=color, label=f'{group} r={pearsonr(x_vals, y_vals)[0]:.2f}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Loop over behavior keys vs amplitude and phase
for beh_key in behavior_keys:
    for band_key in amp_bands.keys():
        x = mi_pd_lt[beh_key]
        y = mi_pd_lt[band_key]
        area = mi_pd_lt['area']

        # Store correlations
        corrs_amp[beh_key]['all'].append(pearsonr(x, y)[0])
        corrs_amp[beh_key]['deep'].append(pearsonr(x[area == 'deep'], y[area == 'deep'])[0])
        corrs_amp[beh_key]['sup'].append(pearsonr(x[area == 'superficial'], y[area == 'superficial'])[0])

        save_path = os.path.join(fig_dir, f'scatter_amp_{beh_key}_vs_{band_key}.png')
        scatter_fits(x, y, area, beh_key, band_key, f'{beh_key} vs {band_key} (Amplitude)', save_path)

    for band_key in phase_bands.keys():
        x = mi_pd_lt[beh_key]
        y = mi_pd_lt[band_key]
        area = mi_pd_lt['area']

        # Store correlations
        corrs_phase[beh_key]['all'].append(pearsonr(x, y)[0])
        corrs_phase[beh_key]['deep'].append(pearsonr(x[area == 'deep'], y[area == 'deep'])[0])
        corrs_phase[beh_key]['sup'].append(pearsonr(x[area == 'superficial'], y[area == 'superficial'])[0])

        save_path = os.path.join(fig_dir, f'scatter_phase_{beh_key}_vs_{band_key}.png')
        scatter_fits(x, y, area, beh_key, band_key, f'{beh_key} vs {band_key} (Phase)', save_path)

# Plot evolution of correlations
def plot_corr_evolution(corr_dict, title, save_name):
    plt.figure(figsize=(12, 8))
    for group, color in zip(['all', 'deep', 'sup'], [color_all, color_deep, color_sup]):
        for i, beh_key in enumerate(behavior_keys):
            plt.subplot(2, 4, i+1)
            plt.plot(band_names, corr_dict[beh_key][group], label=group, color=color, marker='o')
            plt.title(beh_key)
            plt.ylim(-1, 1)
            plt.axhline(0, color='gray', linestyle='--')
            if i == 0:
                plt.legend()
            if i >= 4:
                plt.xlabel('Band')
            if i % 4 == 0:
                plt.ylabel('Pearson r')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, save_name), dpi=300)
    plt.close()

plot_corr_evolution(corrs_amp, 'Correlation: MIR_behavior vs MIR_amplitude', 'correlation_evolution_amp.png')
plot_corr_evolution(corrs_phase, 'Correlation: MIR_behavior vs MIR_phase', 'correlation_evolution_phase.png')

# Save correlation dictionaries
with open(os.path.join(fig_dir, 'correlations_amp.pkl'), 'wb') as f:
    pickle.dump(corrs_amp, f)
with open(os.path.join(fig_dir, 'correlations_phase.pkl'), 'wb') as f:
    pickle.dump(corrs_phase, f)

#############################################################

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import pickle
valid_mask = mi_pd_lt['transferred_cluster'] != -10
mi_pd_lt= mi_pd_lt[valid_mask].copy()

# Define color map for clusters
cluster_palette = custom_cluster_palette

# Make sure the output directory exists
cluster_fig_dir = os.path.join(data_dir, 'figures', 'by_cluster')
os.makedirs(cluster_fig_dir, exist_ok=True)

# Storage for correlations by cluster
corrs_amp_cluster = {k: {cid: [] for cid in [0, 1, 2]} for k in behavior_keys}
corrs_phase_cluster = {k: {cid: [] for cid in [0, 1, 2]} for k in behavior_keys}

# Function to compute and plot scatter with fits for clusters
def scatter_fits_by_cluster(x, y, cluster_ids, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))

    # Scatter with cluster color
    sns.scatterplot(x=x, y=y, hue=cluster_ids, palette=cluster_palette, s=10, alpha=0.6, legend=False)

    # Fit and plot regression lines per cluster
    for cid, color in cluster_palette.items():
        mask = cluster_ids == cid
        x_vals, y_vals = x[mask], y[mask]
        if len(x_vals) > 2:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = slope * xs + intercept
            r_val = pearsonr(x_vals, y_vals)[0]
            plt.plot(xs, ys, color=color, label=f'Cluster {cid} r={r_val:.2f}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Loop over behavior keys vs amplitude and phase, per cluster
cluster_ids = mi_pd_lt['transferred_cluster'].values

for beh_key in behavior_keys:
    for band_key in amp_bands.keys():
        x = mi_pd_lt[beh_key].values
        y = mi_pd_lt[band_key].values

        # Store correlations per cluster
        for cid in [0, 1, 2]:
            mask = cluster_ids == cid
            if np.sum(mask) > 2:
                r = pearsonr(x[mask], y[mask])[0]
            else:
                r = np.nan
            corrs_amp_cluster[beh_key][cid].append(r)

        save_path = os.path.join(cluster_fig_dir, f'scatter_amp_{beh_key}_vs_{band_key}_by_cluster.png')
        scatter_fits_by_cluster(x, y, cluster_ids, beh_key, band_key, f'{beh_key} vs {band_key} (Amp)', save_path)

    for band_key in phase_bands.keys():
        x = mi_pd_lt[beh_key].values
        y = mi_pd_lt[band_key].values

        # Store correlations per cluster
        for cid in [0, 1, 2]:
            mask = cluster_ids == cid
            if np.sum(mask) > 2:
                r = pearsonr(x[mask], y[mask])[0]
            else:
                r = np.nan
            corrs_phase_cluster[beh_key][cid].append(r)

        save_path = os.path.join(cluster_fig_dir, f'scatter_phase_{beh_key}_vs_{band_key}_by_cluster.png')
        scatter_fits_by_cluster(x, y, cluster_ids, beh_key, band_key, f'{beh_key} vs {band_key} (Phase)', save_path)

# Plot evolution of correlations by cluster
def plot_corr_evolution_by_cluster(corr_dict, title, save_name):
    plt.figure(figsize=(12, 8))
    for i, beh_key in enumerate(behavior_keys):
        plt.subplot(2, 4, i+1)
        for cid, color in cluster_palette.items():
            plt.plot(band_names, corr_dict[beh_key][cid], label=f'Cluster {cid}', color=color, marker='o')
        plt.title(beh_key)
        plt.ylim(-1, 1)
        plt.axhline(0, color='gray', linestyle='--')
        if i == 0:
            plt.legend()
        if i >= 4:
            plt.xlabel('Band')
        if i % 4 == 0:
            plt.ylabel('Pearson r')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(cluster_fig_dir, save_name), dpi=300)
    plt.close()

plot_corr_evolution_by_cluster(corrs_amp_cluster, 'Correlation by Cluster: Behavior vs Amplitude MI', 'correlation_by_cluster_amp.png')
plot_corr_evolution_by_cluster(corrs_phase_cluster, 'Correlation by Cluster: Behavior vs Phase MI', 'correlation_by_cluster_phase.png')

# Save correlation results
with open(os.path.join(cluster_fig_dir, 'correlations_amp_by_cluster.pkl'), 'wb') as f:
    pickle.dump(corrs_amp_cluster, f)
with open(os.path.join(cluster_fig_dir, 'correlations_phase_by_cluster.pkl'), 'wb') as f:
    pickle.dump(corrs_phase_cluster, f)

