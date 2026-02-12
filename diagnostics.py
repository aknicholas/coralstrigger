import matplotlib.pyplot as plt
import tools.CoRaLs_geometry as aso_geometry
import tools.constants as constants
import matplotlib.gridspec as gridspec
import tools.waveform as waveform
import tools.filters as filters
import payload_signal as payload
import math
import noise
import os
import coherent_sum as sum
from scipy import ndimage
from skimage.measure import label, regionprops
import pickle
import json
import os
import sys
import time
import argparse
from tools import delays as delays_tools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------- Diagnostics helpers (ADDED) ----------------

def _ref_power_sum(trace, window, step, normalize=True, use_abs2=True):
    """
    Reference sliding power: sum(|x|^2) over each window, stepped by 'step'.
    normalize=True divides by window.
    """
    x = np.asarray(trace)
    x = np.abs(x)**2 if use_abs2 else (x.real**2)
    n = len(x)
    if n < window or step <= 0:
        raise ValueError(f"Invalid window/step/len: {window}/{step}/{n}")
    num = 1 + (n - window) // step
    # Strided view
    s0 = x.strides[0]
    view = np.lib.stride_tricks.as_strided(x, shape=(num, window),
                                           strides=(s0*step, s0))
    p = view.sum(axis=1)
    return (p / window) if normalize else p

def _compare_power(time_trace, window, step, label='trace', ax=None):
    """
    Compare coherent_sum.powerSum vs reference. Returns dict with arrays and stats, and plots if ax provided.
    """
    from coherent_sum import powerSum as mod_powerSum

    # Module output
    mod_power, _ = mod_powerSum(time_trace, window=window, step=step)
    # Two references (with/without window normalization)
    ref_norm = _ref_power_sum(time_trace, window, step, normalize=True, use_abs2=True)
    ref_raw  = _ref_power_sum(time_trace, window, step, normalize=False, use_abs2=True)

    # Align lengths (module vs reference can differ by 1 when (N-window) % step != 0)
    L = min(len(mod_power), len(ref_norm), len(ref_raw))
    if (len(mod_power), len(ref_norm), len(ref_raw)) != (L, L, L):
        print(f"[diagnostics] Aligning lengths: module={len(mod_power)} ref_norm={len(ref_norm)} ref_raw={len(ref_raw)} -> {L}")
    mod_power = mod_power[:L]
    ref_norm  = ref_norm[:L]
    ref_raw   = ref_raw[:L]

    # Decide which reference matches module best
    def stats(a, b):
        # robust scale-free mismatch
        r = np.median(a / np.maximum(b, 1e-30))
        rmse = np.sqrt(np.mean((a - b)**2))
        return r, rmse
    r_norm, rmse_norm = stats(mod_power, ref_norm)
    r_raw,  rmse_raw  = stats(mod_power, ref_raw)
    match = 'normalized' if rmse_norm <= rmse_raw else 'raw'
    ref = ref_norm if match == 'normalized' else ref_raw
    ratio = np.median(mod_power / np.maximum(ref, 1e-30))

    if ax is not None:
        ax.plot(mod_power, '-', lw=1.2, label=f'module powerSum ({label})')
        ax.plot(ref, '--', lw=1.0, label=f'reference ({match})')
        ax.set_title(f'Window={window} Step={step} | match={match}, ratio≈{ratio:.3f}')
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Power')
        ax.grid(alpha=0.3)
        ax.legend()

    return {
        'module': mod_power,
        'ref': ref,
        'which_ref': match,
        'ratio': ratio,
        'rmse_norm': rmse_norm,
        'rmse_raw': rmse_raw
    }

def power_v_amplitude(power, noise):
    """
    Quick diagnostic: power vs sample amplitude distribution.
    """
    noise = np.asarray(noise).real
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].hist(noise, bins=100, alpha=0.7)
    axes[0].set_title('Noise amplitude histogram')
    axes[0].set_xlabel('Voltage')
    axes[0].set_ylabel('Count')
    axes[0].grid(alpha=0.3)

    axes[1].hist(power, bins=100, alpha=0.7, log=True)
    axes[1].set_title('Windowed power histogram')
    axes[1].set_xlabel('Power')
    axes[1].set_ylabel('Count (log)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig

def diagnose_thresholds(noise_file=None, power_file=None, window=160, step=40):
    """
    - If power_file is provided, uses it directly; optionally also rebuild from noise_file (if provided) for cross-check.
    - If only noise_file given, builds power via coherent_sum.powerSum.
    Plots: time snippet, power overlay (module vs reference), hist.
    """
    # Load inputs
    trace = None
    mod_power = None

    if noise_file:
        raw = np.load(noise_file)
        trace = raw.flatten().real if raw.ndim == 2 else raw.real

    if power_file and os.path.exists(power_file):
        mod_power = np.load(power_file)

    # Build power from noise if needed
    if mod_power is None:
        if trace is None:
            raise ValueError("Provide either power_file or noise_file")
        from coherent_sum import powerSum as mod_powerSum
        mod_power, _ = mod_powerSum(trace, window=window, step=step)

    # If we have a trace, compare with reference
    fig, axes = plt.subplots(2, 1, figsize=(10,7))
    if trace is not None:
        # Show a small time snippet
        ns = min(5000, len(trace))
        axes[0].plot(trace[:ns], lw=0.8)
        axes[0].set_title(f'Time trace (first {ns} samples)')
        axes[0].set_xlabel('Sample'); axes[0].set_ylabel('Voltage')
        axes[0].grid(alpha=0.3)

        _compare_power(trace, window, step, label='thresholds-path', ax=axes[1])
    else:
        axes[1].plot(mod_power, '-', lw=1.0)
        axes[1].set_title(f'Power sequence (module) Window={window} Step={step}')
        axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Power')
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    # Power histogram
    fig2 = plt.figure(figsize=(6,4))
    plt.hist(mod_power, bins=100, alpha=0.8, log=True)
    plt.xlabel('Power'); plt.ylabel('Count (log)'); plt.grid(alpha=0.3)
    plt.title('Power histogram (thresholds path)')
    plt.tight_layout()
    plt.show()

def diagnose_power_cuts(phi=90.0, theta=0.0, window=256, step=128, apply_filter=False):
    """
    Build a coherent sum from payload.getPayloadWaveforms at one angle, then compare powerSum vs reference.
    """
    trigger_sectors_phi = [1,2,3,4]
    ringmask = [1,1,1,1]
    eplane  = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane  = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff=2)
    waveforms, timebase, _ = payload.getPayloadWaveforms(phi, theta, trigger_sectors_phi, impulse, (eplane, hplane),
                                                         downsample=True, snr=5, plot=False)
    if apply_filter:
        lp_freq, lp_mult, _ = filters.Shannon_Whitaker(fs=4e9, plot=False)
        lp_gain = 10**(lp_mult/20)
        # Apply filter per channel in frequency domain
        for i in range(waveforms.shape[0]):
            for j in range(waveforms.shape[1]):
                wf = waveform.Waveform(waveforms[i,j], timebase)
                wf.fft()
                fft_len = len(wf.ampl)
                gain = np.interp(np.linspace(0,1,fft_len),
                                 np.linspace(0,1,len(lp_gain)),
                                 lp_gain) if len(lp_gain)!=fft_len else lp_gain
                wf.ampl *= gain
                wf.voltage = np.fft.irfft(wf.ampl, n=wf.n)
                waveforms[i,j] = wf.voltage

    delays = payload.getRemappedDelays(phi, theta, trigger_sectors_phi)
    coh, _ = sum.coherentSum(waveforms, timebase, delays, False, ringmask)

    fig, ax = plt.subplots(1,1, figsize=(10,4))
    _compare_power(coh, window, step, label='power_cuts path', ax=ax)
    plt.tight_layout(); plt.show()

def diagnose_snr_scan(noise_file=None, window=160, step=40):
    """
    If snr_scan module exists and has a power path, compare; otherwise just reuse threshold path on noise.
    """
    try:
        import snr_scan as snr
        print("[diagnostics] snr_scan imported; reusing thresholds diagnostic on noise")
    except ImportError:
        print("[diagnostics] snr_scan not found; reusing thresholds diagnostic on noise")
    if noise_file is None:
        print("Provide --noise-file to check snr_scan-like power path.")
        return
    diagnose_thresholds(noise_file=noise_file, power_file=None, window=window, step=step)

def main():
    ap = argparse.ArgumentParser(description="Diagnostics for power summing consistency")
    sub = ap.add_subparsers(dest='cmd')

    th = sub.add_parser('thresholds', help='Check generate_threshold_curves power path')
    th.add_argument('--noise-file', default=None, help='Path to simulated noise .npy')
    th.add_argument('--power-file', default='noise/power_160_40.npy', help='Path to precomputed power file')
    th.add_argument('--window', type=int, default=160)
    th.add_argument('--step', type=int, default=40)

    pc = sub.add_parser('power_cuts', help='Check power_cuts coherent->power path at one pointing')
    pc.add_argument('--phi', type=float, default=90.0)
    pc.add_argument('--theta', type=float, default=0.0)
    pc.add_argument('--window', type=int, default=256)
    pc.add_argument('--step', type=int, default=128)
    pc.add_argument('--filter', action='store_true', help='Apply Shannon-Whitaker front-end filter')

    sc = sub.add_parser('snr_scan', help='Check snr_scan power path using a noise file')
    sc.add_argument('--noise-file', required=True)
    sc.add_argument('--window', type=int, default=160)
    sc.add_argument('--step', type=int, default=40)

    args = ap.parse_args()

    if args.cmd == 'thresholds':
        diagnose_thresholds(noise_file=args.noise_file, power_file=args.power_file,
                            window=args.window, step=args.step)
    elif args.cmd == 'power_cuts':
        diagnose_power_cuts(phi=args.phi, theta=args.theta,
                            window=args.window, step=args.step,
                            apply_filter=args.filter)
    elif args.cmd == 'snr_scan':
        diagnose_snr_scan(noise_file=args.noise_file, window=args.window, step=args.step)
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
# ---------------- End diagnostics helpers ----------------
