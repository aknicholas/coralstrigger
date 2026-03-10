import numpy as np
import matplotlib.pyplot as plt
from . import CoRaLs_geometry as corals

def get_Shannon_Whitaker_coeffs(fs=corals.ritc_sample_rate*1e9, fc=1.5e9):
    """
    Get the Shannon-Whitaker FIR lowpass filter coefficients.
    Returns normalized tap coefficients as 1D array (length 33).

    Args:
        fs: sampling frequency in Hz (default: RITC sample rate)
        fc: cutoff frequency in Hz (default: 1.5 GHz)
    """
    n_taps = 33
    n = np.arange(n_taps) - (n_taps - 1) / 2
    fc_norm = 2.0 * fc / fs  # normalized cutoff (0–2, where 2 = fs)
    h = fc_norm * np.sinc(fc_norm * n) * np.hamming(n_taps)
    return h


def apply_Shannon_Whitaker_filter(waveforms, fs=corals.ritc_sample_rate*1e9, fc=1.5e9):
    """
    Apply Shannon-Whitaker FIR lowpass filter to waveforms.

    This is the digital anti-aliasing filter implemented in hardware.
    Should be applied to both signal and noise after beamforming.

    Args:
        waveforms: 1D or 2D array (samples) or (channels, samples)
        fs: sampling frequency in Hz (default: 4 GHz)
        fc: lowpass cutoff frequency in Hz (default: 1.5 GHz)

    Returns:
        filtered: same shape as input
    """
    from scipy.signal import lfilter

    h = get_Shannon_Whitaker_coeffs(fs, fc)
    
    # Handle both 1D and 2D arrays
    if waveforms.ndim == 1:
        return lfilter(h, 1.0, waveforms)
    elif waveforms.ndim == 2:
        # Apply filter to each channel independently
        filtered = np.zeros_like(waveforms)
        for i in range(waveforms.shape[0]):
            filtered[i] = lfilter(h, 1.0, waveforms[i])
        return filtered
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {waveforms.shape}")


def Shannon_Whitaker(fs=corals.ritc_sample_rate*1e9, fc=1.5e9, plot=True):
    """
    Compute the frequency response of the Shannon–Whitaker FIR filter.
    Returns (freqs, H_mag_db, cutoff_freq).
    """
    # Get filter coefficients
    h = get_Shannon_Whitaker_coeffs(fs, fc)

    # 3) Choose a fine FFT length for smooth plot:
    M = 4096
    H = np.fft.fft(h, n=M)
    # Frequency axis: from 0 to fs (positive half)
    freqs = np.fft.fftfreq(M, d=1/fs)
    pos = freqs >= 0
    freqs = freqs[pos]/1e9
    H_mag = np.abs(H[pos])

    # 4) Normalize magnitude to 0 dB at DC:
    H_mag_db = 20 * np.log10(H_mag / np.max(H_mag))

    # 5) Plot:
    if plot:
        plt.plot(freqs, H_mag_db)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Shannon–Whitaker FIR Frequency Response')
        plt.axhline(-3, color='red', linestyle='--', label='-3 dB')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 6) Estimate cutoff: find freq where H_mag_db crosses -3 dB:
    idx = np.where(H_mag_db <= -3)[0]
    if len(idx):
        cutoff_freq = freqs[idx[0]]
        print(f"Cutoff Frequency: {cutoff_freq} GHz")
    else:
        cutoff_freq = None

    return freqs, H_mag_db, cutoff_freq

def get_bandpass_coeffs(f_low, f_high, n_taps=33, fs=corals.ritc_sample_rate*1e9, window='hamming'):
    """
    Compute windowed-sinc bandpass FIR filter coefficients.

    Constructs a bandpass filter as the difference of two lowpass sinc filters:
        h[n] = h_high[n] - h_low[n]
    where each h_x[n] = 2*fx/fs * sinc(2*fx/fs * n), then windowed.

    Args:
        f_low:   lower passband edge in Hz (e.g. 300e6 for 300 MHz)
        f_high:  upper passband edge in Hz (e.g. 1200e6 for 1200 MHz)
        n_taps:  number of filter taps (should be odd for symmetric/linear phase)
        fs:      sampling frequency in Hz (default: RITC sample rate)
        window:  window function name: 'hamming', 'hann', 'blackman', or 'rectangular'

    Returns:
        h: 1D numpy array of length n_taps
    """
    if n_taps % 2 == 0:
        n_taps += 1  # force odd for linear phase

    # Centered sample indices
    n = np.arange(n_taps) - (n_taps - 1) / 2

    # Normalized cutoff frequencies (0 to 1 relative to fs)
    fc_low  = 2.0 * f_low  / fs
    fc_high = 2.0 * f_high / fs

    # Lowpass sinc kernels; np.sinc(x) = sin(pi*x)/(pi*x)
    h_low  = fc_low  * np.sinc(fc_low  * n)
    h_high = fc_high * np.sinc(fc_high * n)

    # Bandpass = difference of lowpass filters
    h = h_high - h_low

    # Apply window
    windows = {
        'hamming':     np.hamming(n_taps),
        'hann':        np.hanning(n_taps),
        'blackman':    np.blackman(n_taps),
        'rectangular': np.ones(n_taps),
    }
    if window not in windows:
        raise ValueError(f"Unknown window '{window}'. Choose from {list(windows.keys())}")
    h *= windows[window]

    return h


def apply_bandpass_filter(waveforms, f_low = 1e6, f_high=1200e6, n_taps=33,
                          fs=corals.ritc_sample_rate*1e9, window='hamming'):
    """
    Apply a windowed-sinc bandpass FIR filter to waveforms.

    Args:
        waveforms: 1D or 2D array (samples,) or (channels, samples)
        f_low:     lower passband edge in Hz  (e.g. 300e6)
        f_high:    upper passband edge in Hz  (e.g. 1200e6)
        n_taps:    number of filter taps (odd preferred)
        fs:        sampling frequency in Hz
        window:    'hamming', 'hann', 'blackman', or 'rectangular'

    Returns:
        filtered: same shape as input
    """
    from scipy.signal import lfilter

    h = get_bandpass_coeffs(f_low, f_high, n_taps=n_taps, fs=fs, window=window)

    if waveforms.ndim == 1:
        return lfilter(h, 1.0, waveforms)
    elif waveforms.ndim == 2:
        filtered = np.zeros_like(waveforms)
        for i in range(waveforms.shape[0]):
            filtered[i] = lfilter(h, 1.0, waveforms[i])
        return filtered
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {waveforms.shape}")


def plot_bandpass_response(f_low, f_high, n_taps=33,
                           fs=corals.ritc_sample_rate*1e9, window='hamming'):
    """
    Plot the frequency response of the bandpass filter and return
    (freqs_GHz, H_mag_db).
    """
    h = get_bandpass_coeffs(f_low, f_high, n_taps=n_taps, fs=fs, window=window)

    M = 4096
    H = np.fft.fft(h, n=M)
    freqs = np.fft.fftfreq(M, d=1.0 / fs)
    pos = freqs >= 0
    freqs_GHz = freqs[pos] / 1e9
    H_mag = np.abs(H[pos])
    H_mag_db = 20 * np.log10(H_mag / np.max(H_mag) + 1e-12)

    plt.figure()
    plt.plot(freqs_GHz * 1e3, H_mag_db)  # x-axis in MHz
    plt.axvline(f_low  / 1e6, color='green', linestyle='--', label=f'f_low  = {f_low /1e6:.0f} MHz')
    plt.axvline(f_high / 1e6, color='blue',  linestyle='--', label=f'f_high = {f_high/1e6:.0f} MHz')
    plt.axhline(-3, color='red', linestyle=':', label='-3 dB')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Bandpass FIR ({f_low/1e6:.0f}–{f_high/1e6:.0f} MHz), {n_taps} taps, {window} window')
    plt.ylim(-80, 5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("Bandpass_Response.png",dpi = 400)

    return freqs_GHz, H_mag_db


if __name__=="__main__":
    Shannon_Whitaker(fs=4e9, plot=True)
    # Example: 300–1200 MHz bandpass at 4 GSa/s
    plot_bandpass_response(300e6, 1200e6, n_taps=65, fs=4e9, window='hamming')