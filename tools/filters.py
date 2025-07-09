import numpy as np
import matplotlib.pyplot as plt

def Shannon_Whitaker(fs=3e9, plot=True):
    """
    Compute the frequency response of the Shannon–Whitaker FIR filter.
    Returns (freqs, H_mag_db, cutoff_freq).
    """
    # 1) Define the integer coefficients from PDF:
    b = np.array([
         0,  -23,   0,  105,   0,  -263,   0,   526,
         0,  -949,   0,  1672,   0, -3216,   0, 10342,
      16384, 10342,   0, -3216,   0,  1672,   0,  -949,
         0,   526,   0,  -263,   0,   105,   0,   -23
    ], dtype=float)

    # 2) Normalize to real taps:
    h = b / 32768.0  # array length 33

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

if __name__=="__main__":
    Shannon_Whitaker(fs=3e9, plot=True)