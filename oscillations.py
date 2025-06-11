import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import hilbert
import pickle

# --- Settings ---
base_directory = '/home/melma31/Documents/time_project_grin_lens'
data_directory = os.path.join(base_directory, 'data')
figure_directory = os.path.join(base_directory, 'figures')
os.makedirs(figure_directory, exist_ok=True)

input_file = 'lfp_mice_all_data_dict.pkl'
input_file = os.path.join(data_directory, input_file)

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

fs = 20  # LFP sampling rate
# Dictionary to store population amplitudes for all mice
pop_amp_dict = {}
t_dict = {}
# --- Main Analysis Loop ---
for mouse in data.keys():
    mouse_data = data[mouse]

    signal = mouse_data['clean_traces']
    speed = mouse_data['speed']
    valid_index = np.where(speed > 8)[0]

    signal = signal[valid_index, :]
    lfp_freqs = mouse_data['cwt_freqs']
    lfp_wavelet = mouse_data['cwt'][valid_index, :, :]
    n_channels = lfp_wavelet.shape[2]

    # --- PCA of neural activity (neurons x time) ---
    neural_data = signal.T
    pca = PCA(n_components=2)
    neural_pca = pca.fit_transform(neural_data)
    angles = np.arctan2(neural_pca[:, 1], neural_pca[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_signal = signal[:, sorted_indices].T

    # --- PCA over time ---
    #pca_time = PCA(n_components=2)
    #time_proj = pca_time.fit_transform(signal)
    #population_amplitude = np.linalg.norm(time_proj, axis=1)
    # --- UMAP over time ---
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    time_proj = reducer.fit_transform(signal)
    population_amplitude = np.linalg.norm(time_proj, axis=1)

    # Store population amplitude for summary plot
    pop_amp_dict[f"{mouse}"] = population_amplitude
    t_dict[f"{mouse}"] = t

    # --- Band amplitude and phase ---
    lfp_wavelet_mean = np.mean(lfp_wavelet, axis=2)
    band_amplitude_dict = {}
    band_phase_dict = {}

    for band_name, (fmin, fmax) in bands.items():
        band_indices = np.where((lfp_freqs >= fmin) & (lfp_freqs <= fmax))[0]
        if len(band_indices) == 0:
            continue
        band_wavelet = lfp_wavelet_mean[:, band_indices]
        band_amplitude = np.mean(np.abs(band_wavelet), axis=1)
        band_amp_hilbert = hilbert(band_amplitude)
        band_phase = np.mod(np.angle(band_amp_hilbert), 2 * np.pi)
        band_amplitude_dict[band_name] = band_amplitude
        band_phase_dict[band_name] = band_phase

    # --- Plot sorted signal + amplitude traces ---
    def clip_and_normalize(x, q=95):
        clip_val = np.percentile(x, q)
        x = np.clip(x, None, clip_val)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    t = np.arange(signal.shape[0]) / fs
    offset = 0
    amplitude_traces = []
    labels = []
    #amp = clip_and_normalize(population_amplitude)
    amp = population_amplitude
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
    axs[0].set_title(f'{mouse} - Neural Activity (Sorted by PCA Angle)')
    axs[1].set_title(f'{mouse} - Clipped & Normalized Amplitude Signals')
    for i, trace in enumerate(amplitude_traces):
        axs[1].plot(t, trace, label=labels[i])
    axs[1].legend(loc='upper right', ncol=2)
    axs[1].set_xlim([0, t[-1]])
    save_figure(fig, 'amplitude_trace_plot', mouse, figure_directory)

    # --- Zero-lag cross-correlation ---
    pop_amp = population_amplitude - np.mean(population_amplitude)
    correlation_scores = []
    band_labels = []
    for band_name, band_amp in band_amplitude_dict.items():
        band_amp_zm = band_amp - np.mean(band_amp)
        xcorr = np.correlate(pop_amp, band_amp_zm, mode='full') / len(pop_amp)
        peak_corr = np.max(np.abs(xcorr))
        correlation_scores.append(peak_corr)
        band_labels.append(band_name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(band_labels, correlation_scores, color='slateblue')
    ax.set_ylabel('Peak Cross-Correlation')
    ax.set_ylim([0, 1])
    ax.set_title(f'{mouse} - Cross-Correlation: Population vs Band Amplitudes')
    save_figure(fig, 'zero_lag_crosscorr_bar', mouse, figure_directory)

    # --- Time-lagged cross-correlation ---
    max_lag = len(pop_amp) - 1
    lags = np.arange(-max_lag, max_lag + 1) / fs
    fig = plt.figure(figsize=(12, 10))
    for i, (band_name, band_amp) in enumerate(band_amplitude_dict.items()):
        ax = fig.add_subplot(len(band_amplitude_dict), 1, i + 1)
        y = band_amp - np.mean(band_amp)
        xcorr = np.correlate(pop_amp, y, mode='full') / len(pop_amp)
        ax.plot(lags, xcorr, label=band_name)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_title(f'{mouse} - Time-lagged Cross-Corr: Pop vs {band_name}')
        if i == len(band_amplitude_dict) - 1:
            ax.set_xlabel('Lag (s)')
    save_figure(fig, 'time_lagged_crosscorr', mouse, figure_directory)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    # --- Full time-lagged xcorr per channel per frequency ---
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    lags = np.arange(-len(pop_amp) + 1, len(pop_amp)) / fs
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    colormap = plt.get_cmap('hsv')
    norm = Normalize(vmin=0, vmax=2 * np.pi)
    colors = [colormap(norm(a)) for a in angles]
    fig, axs = plt.subplots(len(bands), 1, figsize=(12, 3 * len(bands)), sharex=True, sharey=True)
    if len(bands) == 1:
        axs = [axs]
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        band_indices = np.where((lfp_freqs >= fmin) & (lfp_freqs <= fmax))[0]
        if len(band_indices) == 0:
            continue
        ax = axs[i]
        for ch in range(n_channels):
            band_wavelet = lfp_wavelet[:, band_indices, ch]  # shape: (time, freqs)
            band_amp = np.mean(np.abs(band_wavelet), axis=1)
            y = band_amp - np.mean(band_amp)
            xcorr = np.correlate(pop_amp, y, mode='full') / len(pop_amp)
            ax.plot(lags, xcorr, color=colors[ch], alpha=0.5, linewidth=1.2)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f'{mouse} - {band_name} Band')
        if i == len(bands) - 1:
            ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Cross-Corr')
    plt.tight_layout()
    save_figure(fig, 'time_lagged_xcorr_bands_channels', mouse, figure_directory)

    # --- Spatial profile per band ---
    spatial_corr = {}
    for band_name, (fmin, fmax) in bands.items():
        band_indices = np.where((lfp_freqs >= fmin) & (lfp_freqs <= fmax))[0]
        if len(band_indices) == 0:
            continue
        band_corr = []
        for ch in range(n_channels):
            band_wavelet = lfp_wavelet[:, band_indices, ch]
            band_amp = np.mean(np.abs(band_wavelet), axis=1)
            y = band_amp - np.mean(band_amp)
            xcorr = np.correlate(pop_amp, y, mode='full') / len(pop_amp)
            band_corr.append(np.max(np.abs(xcorr)))
        spatial_corr[band_name] = np.array(band_corr)

    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    fig, axs = plt.subplots(1, len(spatial_corr), subplot_kw={'projection': 'polar'}, figsize=(4 * len(spatial_corr), 4))
    if len(spatial_corr) == 1:
        axs = [axs]
    for ax, (band_name, corr_vals) in zip(axs, spatial_corr.items()):
        ax.bar(angles, corr_vals, width=2 * np.pi / n_channels, color='royalblue')
        ax.set_title(f'{mouse} - {band_name} Band')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
    plt.suptitle(f'{mouse} - Spatial Profile of Cross-Correlation', fontsize=14)
    save_figure(fig, 'spatial_profile_polar', mouse, figure_directory)


# --- After the loop ends ---
# Plot population amplitude across all mice
n_mice = len(pop_amp_dict)
fig, axs = plt.subplots(n_mice, 1, figsize=(10, 3 * n_mice), sharex=False)
if n_mice == 1:
    axs = [axs]
from sklearn.linear_model import LinearRegression

for i, (key, pop_amp) in enumerate(pop_amp_dict.items()):
    pop_amp = pop_amp.reshape(-1, 1)
    t = np.arange(0,pop_amp.shape[0]).reshape(-1,1)
    model = LinearRegression().fit(t, pop_amp)
    r = np.corrcoef(t.squeeze(), pop_amp.squeeze())[0, 1]
    line = model.predict(t)

    axs[i].plot(t, pop_amp, label='Pop Amplitude')
    axs[i].plot(t, line, '--', label=f'Linear Fit (R={r:.2f})')
    axs[i].set_title(f'{key} - Population Amplitude')
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel('Amplitude')
    axs[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(figure_directory, 'population_amplitude_all_mice.png'), dpi=300)
plt.close()
print("✅ Saved population amplitude summary plot for all mice.")
