
import numpy as np
from scipy.signal import cwt, morlet2

def compute_wavelet(signal, frequencies, fs, n_cycles=7):
    """
    Compute Morlet wavelet transform for a given signal.

    Parameters:
    - signal: 1D numpy array (calcium trace)
    - frequencies: 1D numpy array (frequencies in Hz)
    - fs: sampling frequency in Hz
    - n_cycles: number of cycles per wavelet (controls time/freq tradeoff)

    Returns:
    - wavelet_transform: 2D numpy array (n_freqs, n_times), complex
    """
    widths = n_cycles * fs / (2 * np.pi * frequencies)
    wavelet_transform = cwt(signal, morlet2, widths, w=n_cycles)
    return wavelet_transform

# Function to compute coherence
def compute_coherence(lfp_wavelet, calcium_wavelet, frequencies, band):
    band_min, band_max = band
    band_mask = (frequencies >= band_min) & (frequencies <= band_max)

    # Select only frequencies in the band
    lfp_band = lfp_wavelet[band_mask,:]
    cal_band = calcium_wavelet[band_mask,:]

    # Cross-spectra and auto-spectra
    Sxy = np.mean(lfp_band * np.conj(cal_band), axis=1)  # per frequency
    Sxx = np.mean(np.abs(lfp_band) ** 2, axis=1)
    Syy = np.mean(np.abs(cal_band) ** 2, axis=1)

    # Coherence per frequency
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy)

    # Average coherence across frequencies
    coherence_band = np.mean(coh)

    return coherence_band

import numpy as np

def compute_inner_time(trial_id):
    trial_id = np.array(trial_id)
    inner_time = np.zeros_like(trial_id)
    counter = 0

    for i in range(1, len(trial_id)):
        if trial_id[i] == trial_id[i - 1]:
            counter += 1
        else:
            counter = 0
        inner_time[i] = counter

    return inner_time
