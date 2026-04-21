"""
SNR efficiency scan for dual-polarization CoRaLS trigger

Generates S-curves (efficiency vs SNR) for LHCP, RHCP, and coincidence triggers.
Tests signal detection efficiency at various SNR levels and angles.

Usage:
    python snr_scan_dualpol.py --threshold-lhcp 3.23 --threshold-rhcp 3.23 
    python snr_scan_dualpol.py --angle-scan --phi0 0 --theta0 0 --span-phi 10 --span-theta 10
"""

import numpy as np
import matplotlib.pyplot as plt
import tools.CoRaLs_geometry as aso_geometry
import tools.constants as constants
import payload_signal as payload
import coherent_sum as trigger
import noise
import tools.filters as filters
import json
import argparse
import sys
import os


def run_snr_scan_dualpol(phi, theta, psi, threshold_lhcp, threshold_rhcp, coincidence_window,
                         snr_grid, n_trials, window, step, save_filename=None,
                         single_ant_threshold=None, antennas=None,
                         apply_second_filter=None, fc_second=None):
    """
    Run SNR efficiency scan for dual-pol at a single direction.

    The x-axis is SNR = Vpp / (2*sigma), where the signal
    is injected with the same Vpp at every antenna before beamforming.
    The coherent sum provides sqrt(n_ant) improvement in SNR.

    Parameters:
    -----------
    phi, theta, psi : float
        Signal arrival angles (deg)
    threshold_lhcp, threshold_rhcp : float
        Normalized power thresholds (per-antenna power after beamforming / n_ant)
    coincidence_window : float
        Coincidence time window (ns)
    snr_grid : array
        Per-antenna SNR values to test
    n_trials : int
        Number of trials per SNR point
    window, step : int
        Power sum window and step (samples)
    save_filename : str, optional
        Base filename for saving results
    single_ant_threshold : float or None
        Unused (kept for backward compatibility; use antennas=[0] instead).
    antennas : list of int or None
        Physical antenna indices to use, e.g. [0] for single-antenna or
        [0,1,2,3] for full array (default: all four).
    apply_second_filter : bool or None
        Apply second 750 MHz lowpass filter after beamforming.  None means
        use the module-level toggle payload.APPLY_SECOND_FILTER.

    Returns:
    --------
    dict with keys: snr, eff_lhcp, eff_rhcp, eff_coinc, n_ant, apply_second_filter
    """
    if apply_second_filter is None:
        apply_second_filter = payload.APPLY_SECOND_FILTER
    if fc_second is None:
        fc_second = payload.FC_SECOND_FILTER

    print(f"\n=== SNR Scan at (phi={phi:.1f}°, theta={theta:.1f}°, psi={psi:.1f}°) ===")
    print(f"LHCP threshold: {threshold_lhcp:.3f}")
    print(f"RHCP threshold: {threshold_rhcp:.3f}")
    print(f"Coincidence window: {coincidence_window:.1f} ns")
    print(f"SNR range: {snr_grid[0]:.2f} to {snr_grid[-1]:.2f} ({len(snr_grid)} points)")
    print(f"Trials per SNR: {n_trials}")
    
    # Configuration
    if antennas is None:
        antennas = [0, 1, 2, 3]
    n_ant = len(antennas)

    print(f"Using {n_ant} antennas per polarization: {antennas}")
    print(f"Second filter ({fc_second/1e6:.0f} MHz): {'ENABLED' if apply_second_filter else 'DISABLED'}")
    if single_ant_threshold is not None:
        print(f"  (--single-ant-threshold is deprecated; use --ant-scan instead)")
    
    # Load beam patterns correctly for each polarization.
    # H-pol: E-plane spans azimuth [-180,180], H-plane spans elevation [-90,90]
    # V-pol: E-plane spans elevation [-90,90], H-plane spans azimuth [-180,180]
    # (V-pol element is physically rotated 90° around boresight, so E/H planes swap.)
    eplane_h = payload.beamPattern(plot=False, which_plane='E', which_pol='H')
    hplane_h = payload.beamPattern(plot=False, which_plane='H', which_pol='H')
    eplane_v = payload.beamPattern(plot=False, which_plane='E', which_pol='V')
    hplane_v = payload.beamPattern(plot=False, which_plane='H', which_pol='V')

    # Load and prepare impulse
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse)
    
    # Generate waveforms for H and V polarizations simultaneously
    delays_array = payload.getRemappedDelays(phi, theta, antennas)
    
    waveforms, timebase, multipliers = payload.getPayloadWaveforms_dualpol(
        phi, theta, impulse,
        (eplane_h, hplane_h),  # H-pol: [0] eval at phi (az), [1] eval at theta (el)
        (eplane_v, hplane_v),  # V-pol: [0] eval at theta (el), [1] eval at phi (az)
        antennas=antennas,
        psi=psi,
        plot=False
    )
    
    # Select only the requested antenna channels.
    # getPayloadWaveforms_dualpol places H-pol at waveforms[ant_num] and
    # V-pol at waveforms[ant_num+4] for each ant_num in antennas.
    waveforms_h = waveforms[antennas, :]               # shape (n_ant, n_samp)
    waveforms_v = waveforms[[a + 4 for a in antennas], :]  # shape (n_ant, n_samp)
    
    # waveforms shape: (n_antennas, n_samples)
    n_ant = waveforms_h.shape[0]
    n_samp = waveforms_h.shape[1]
    
    print(f"Signal dimensions: {n_ant} antennas × {n_samp} samples")
    print(f"Active antennas: {n_ant}")
    print(f"Note: No power normalization needed - thresholds determined with {n_ant} antennas")
    
    # Load pre-generated noise from files
    print("Loading pre-generated noise...")
    from pathlib import Path
    noise_path = Path('noise')
    h_file = noise_path / 'dualpol_noise_hpol.npy'
    v_file = noise_path / 'dualpol_noise_vpol.npy'
    
    if not h_file.exists() or not v_file.exists():
        raise FileNotFoundError(
            f"Noise files not found in {noise_path}/\n"
            "Run generate_dualpol_noise_chunked.py first to generate noise files."
        )
    
    # Load metadata
    metadata_file = noise_path / 'dualpol_noise_metadata.npy'
    if metadata_file.exists():
        metadata = np.load(metadata_file, allow_pickle=True).item()
        print(f"  Metadata: {metadata['duration_sec']} sec, {metadata['sample_rate_GHz']} GHz")
    
    # Load noise (memory-mapped for efficiency)
    noise_h_full = np.load(h_file, mmap_mode='r')
    noise_v_full = np.load(v_file, mmap_mode='r')
    print(f"  Noise shape: {noise_h_full.shape} (antennas, samples)")

    # Compute noise RMS from a small sample to avoid loading the full array into RAM
    rms_sample_size = 100_000
    noise_rms_h = np.sqrt(np.mean(noise_h_full[:, :rms_sample_size].astype(np.float64)**2))
    noise_rms_v = np.sqrt(np.mean(noise_v_full[:, :rms_sample_size].astype(np.float64)**2))
    noise_rms = (noise_rms_h + noise_rms_v) / 2
    print(f"  Noise RMS raw (H): {noise_rms_h:.4f}, (V): {noise_rms_v:.4f}, (avg): {noise_rms:.4f}")

    # Compute noise RMS after the 1.5 GHz Shannon-Whitaker filter (= analog filter reference).
    # This is the noise floor at the ADC input in hardware, and the physically meaningful
    # SNR denominator: SNR = Vpp / (2 * sigma_f1).
    filt_sample_size = 200_000
    noise_h_f1 = filters.apply_Shannon_Whitaker_filter(
        noise_h_full[:, :filt_sample_size].astype(np.float64))
    noise_v_f1 = filters.apply_Shannon_Whitaker_filter(
        noise_v_full[:, :filt_sample_size].astype(np.float64))
    noise_rms_f1_h = np.sqrt(np.mean(noise_h_f1**2))
    noise_rms_f1_v = np.sqrt(np.mean(noise_v_f1**2))
    noise_rms_f1 = (noise_rms_f1_h + noise_rms_f1_v) / 2
    print(f"  Noise RMS post-filt1 (H): {noise_rms_f1_h:.4f}, (V): {noise_rms_f1_v:.4f}, (avg): {noise_rms_f1:.4f}")
    print(f"  SNR axis: per-antenna Vpp/(2*sigma_filt1)  [analog filter reference]")
    noise_rms = noise_rms_f1  # use filtered RMS as SNR reference going forward

    # Also measure noise RMS after the optional second filter, so the comparison
    # plot can rescale the x-axis to the filt2 native noise floor.
    if apply_second_filter:
        noise_h_f2 = filters.apply_Shannon_Whitaker_filter(
            noise_h_f1, fc=fc_second)
        noise_v_f2 = filters.apply_Shannon_Whitaker_filter(
            noise_v_f1, fc=fc_second)
        noise_rms_f2_h = np.sqrt(np.mean(noise_h_f2**2))
        noise_rms_f2_v = np.sqrt(np.mean(noise_v_f2**2))
        noise_rms_f2 = (noise_rms_f2_h + noise_rms_f2_v) / 2
        print(f"  Noise RMS post-filt2 (H): {noise_rms_f2_h:.4f}, (V): {noise_rms_f2_v:.4f}, (avg): {noise_rms_f2:.4f}")
    else:
        noise_rms_f2 = None

    # Extract random segments for trials
    total_samples = noise_h_full.shape[1]
    max_start = total_samples - n_samp
    
    print(f"  Extracting {n_trials} random noise segments ({n_samp} samples each)...")
    noise_h_flat = np.zeros((n_trials * n_ant, n_samp))
    noise_v_flat = np.zeros((n_trials * n_ant, n_samp))
    
    for trial in range(n_trials):
        start_idx = np.random.randint(0, max_start)
        for i, ant_phys in enumerate(antennas):
            noise_h_flat[trial * n_ant + i] = noise_h_full[ant_phys, start_idx:start_idx + n_samp]
            noise_v_flat[trial * n_ant + i] = noise_v_full[ant_phys, start_idx:start_idx + n_samp]
    
    # Scan over SNR values
    eff_lhcp = []
    eff_rhcp = []
    eff_coinc = []

    # Debug: track max power values
    max_powers_lhcp_debug = []
    max_powers_rhcp_debug = []

    print("\nRunning SNR scan...")
    for snr in snr_grid:
        hits_lhcp = 0
        hits_rhcp = 0
        hits_coinc = 0
        
        max_lhcp_this_snr = []
        max_rhcp_this_snr = []
        
        for i in range(n_trials):
            # Extract noise for this trial
            start = i * n_ant
            stop = start + n_ant
            
            event_noise_h = np.reshape(noise_h_flat[start:stop], (n_ant, n_samp))
            event_noise_v = np.reshape(noise_v_flat[start:stop], (n_ant, n_samp))
            
            # Inject signal at specified per-antenna SNR, SNR = Vpp / (2 * noise_rms)
            # Vpp=1 after prepImpulse, so amplitude scale = snr * 2 * noise_rms
            sig_scale = snr * 2 * noise_rms
            injected_h = waveforms_h * sig_scale + event_noise_h
            injected_v = waveforms_v * sig_scale + event_noise_v
            
            # Beamform and convert to circular polarizations
            # Hardware signal chain: digitize → filter @ 4 GHz → sum → circular
            lhcp, rhcp, tb_coinc = trigger.coherentSum_dualpol(
                injected_h, injected_v, timebase, delays_array,
                downsample=True, apply_filter=True, digitize_first=True,
                apply_second_filter=apply_second_filter,
                fc_second=fc_second,
                output='circular'
            )
            
            # Compute power sums.
            # Power is in the same units as the threshold files generated by
            # generate_power_dualpol.py: mean(|v|^2) over each window, where v is
            # the 4-antenna coherently beamformed LHCP/RHCP stream.
            # Do NOT normalize by n_ant here — the threshold JSON values are calibrated
            # on the un-normalized (summed) beamformed power distribution.
            power_lhcp, _ = trigger.powerSum(lhcp, window=window, step=step)
            power_rhcp, _ = trigger.powerSum(rhcp, window=window, step=step)
            
            # Check individual triggers
            max_lhcp = np.max(power_lhcp)
            max_rhcp = np.max(power_rhcp)
            
            max_lhcp_this_snr.append(max_lhcp)
            max_rhcp_this_snr.append(max_rhcp)
            
            lhcp_triggered = max_lhcp > threshold_lhcp
            rhcp_triggered = max_rhcp > threshold_rhcp
            
            if lhcp_triggered:
                hits_lhcp += 1
            if rhcp_triggered:
                hits_rhcp += 1
            
            # Check coincidence: any frame where BOTH LHCP and RHCP exceed
            # threshold simultaneously.  LHCP and RHCP are derived from the
            # same beamformed H/V streams, so no cross-channel time offset
            # exists; the correct model is a frame-by-frame AND gate.
            # (argmax-based time-window checks fail with wider-bandwidth
            # filters whose dispersed impulse response shifts each channel's
            # peak independently due to noise.)
            if lhcp_triggered and rhcp_triggered:
                coinc_frames = (power_lhcp > threshold_lhcp) & (power_rhcp > threshold_rhcp)
                if np.any(coinc_frames):
                    hits_coinc += 1

        # single_ant_threshold path removed – use antennas=[0] instead

        # Calculate efficiencies
        eff_lhcp.append(hits_lhcp / n_trials)
        eff_rhcp.append(hits_rhcp / n_trials)
        eff_coinc.append(hits_coinc / n_trials)

        # Store debug info
        max_powers_lhcp_debug.append(np.mean(max_lhcp_this_snr))
        max_powers_rhcp_debug.append(np.mean(max_rhcp_this_snr))

        # Print with power info for first few SNR points
        if snr <= snr_grid[2] or snr >= snr_grid[-3]:
            print(f"SNR {snr:5.2f}  LHCP {eff_lhcp[-1]:.3f} (pow={max_powers_lhcp_debug[-1]:.3f})  "
                  f"RHCP {eff_rhcp[-1]:.3f} (pow={max_powers_rhcp_debug[-1]:.3f})  Coinc {eff_coinc[-1]:.3f}")
        else:
            print(f"SNR {snr:5.2f}  LHCP {eff_lhcp[-1]:.3f}  RHCP {eff_rhcp[-1]:.3f}  Coinc {eff_coinc[-1]:.3f}")
    
    # Convert to arrays
    results = {
        'snr': snr_grid,
        'eff_lhcp': np.array(eff_lhcp),
        'eff_rhcp': np.array(eff_rhcp),
        'eff_coinc': np.array(eff_coinc),
        'phi': phi,
        'theta': theta,
        'psi': psi,
        'threshold_lhcp': threshold_lhcp,
        'threshold_rhcp': threshold_rhcp,
        'coincidence_window': coincidence_window,
        'window': window,
        'step': step,
        'n_trials': n_trials,
        'n_ant': n_ant,
        'antennas': antennas,
        'apply_second_filter': apply_second_filter,
        'fc_second': fc_second,
        'noise_rms_f1': noise_rms_f1,
        'noise_rms_f2': noise_rms_f2,
    }
    
    # Save data
    if save_filename:
        os.makedirs('plots', exist_ok=True)
        np.save(f'plots/{save_filename}.npy', results)
        
        # Save as text for easy reading
        data = np.column_stack([snr_grid, eff_lhcp, eff_rhcp, eff_coinc])
        header = f"SNR efficiency scan at (phi={phi:.1f}, theta={theta:.1f}, psi={psi:.1f})\n"
        header += f"Thresholds: LHCP={threshold_lhcp:.3f}, RHCP={threshold_rhcp:.3f}\n"
        header += f"Window={window}, Step={step}, Coinc={coincidence_window:.1f}ns, Trials={n_trials}\n"
        header += "SNR\tEff_LHCP\tEff_RHCP\tEff_Coinc"
        np.savetxt(f'plots/{save_filename}.txt', data, fmt='%.6g', header=header)
        
        print(f"\nSaved: plots/{save_filename}.npy and .txt")
    
    return results


def plot_scurves(results, save_filename=None):
    """
    Plot S-curves showing efficiency vs SNR for LHCP, RHCP, and coincidence
    """
    snr = results['snr']
    eff_lhcp = results['eff_lhcp']
    eff_rhcp = results['eff_rhcp']
    eff_coinc = results['eff_coinc']
    
    phi = results['phi']
    theta = results['theta']
    psi = results['psi']
    thr_l = results['threshold_lhcp']
    thr_r = results['threshold_rhcp']
    
    n_ant = results.get('n_ant', 4)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(snr, eff_coinc, '^-', label=f'{n_ant}-ant coincidence (T={thr_l:.2f})',
            color='green', markersize=5, linewidth=2)

    # Mark 50% efficiency point
    for label, eff, color in [('Coinc', eff_coinc, 'green')]:
        idx = np.where(eff >= 0.5)[0]
        if len(idx) > 0:
            k = idx[0]
            if k > 0:
                # Interpolate
                x0, x1 = snr[k-1], snr[k]
                y0, y1 = eff[k-1], eff[k]
                if y1 != y0:
                    snr_50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
                else:
                    snr_50 = x1
            else:
                snr_50 = snr[0]
            
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.axvline(snr_50, color=color, linestyle=':', alpha=0.5, linewidth=1)
            ax.text(snr_50, 0.05, f'{label}\nSNR={snr_50:.2f}', 
                   ha='center', fontsize=9, color=color, bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.7, edgecolor=color))
    
    snr_mode = results.get('snr_mode', 'per_antenna')
    if snr_mode == 'beamformed':
        xlabel = f'Beamformed SNR = Vpp / (2\u03c3) at coherent sum output '
    else:
        xlabel = f'Per-antenna SNR = Vpp / (2\u03c3) '
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Trigger Efficiency', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(snr[0] - 0.1, snr[-1] + 0.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    title = f'Dual-Pol S-Curves: (φ={phi:.1f}°, θ={theta:.1f}°, ψ={psi:.1f}°)'
    ax.set_title(title, fontsize=13)
    
    plt.tight_layout()
    
    if save_filename:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{save_filename}_scurve.png', dpi=150)
        print(f"Saved: plots/{save_filename}_scurve.png")
    
    plt.show()
    
    return fig, ax


def plot_multi_scurves(results_list, save_filename=None):
    """
    Overlay coincidence S-curves on one figure.

    Color encodes number of antennas; line style encodes target rate rank.
    Up to 4 distinct rates and 4 distinct antenna counts are supported.

    Parameters
    ----------
    results_list : list of dict
        Each entry is a result dict from run_snr_scan_dualpol().
    save_filename : str or None
        Base filename for saving (appends '_multi_scurve.png').
    """
    # --- Style tables ---------------------------------------------------------
    # rate-rank → linestyle (shared by both vary_by_filter and non-filter modes)
    RATE_LINESTYLES = {0: '-', 1: '--', 2: '-.', 3: ':'}
    # rate-rank → alpha (non-filter mode only, kept for backward compat)
    RATE_STYLES = {
        0: dict(linestyle='-',  alpha=1.0),
        1: dict(linestyle='--', alpha=1.0),
        2: dict(linestyle='-',  alpha=0.45),
        3: dict(linestyle=':',  alpha=1.0),
    }
    # One colour per unique antenna count (tab10 first four: blue, orange, green, red)
    ANT_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}
    # When comparing filter states:
    #   color  = filter config (two hues: blue / red)
    #   shade  = rate rank (full-brightness → progressively darker)
    #   linestyle = rate rank (solid → dashed → dash-dot → dotted)
    FILTER_BASE_COLORS = {False: (31, 119, 180), True: (214, 39, 40)}   # RGB ints
    NANT_STYLES        = {1: ':', 2: '--', 3: '-.', 4: '-'}

    def _shade(base_rgb, rank, n_ranks):
        """Darken base colour for higher rate-ranks (rank 0 = brightest)."""
        n = max(n_ranks, 1)
        factor = 1.0 - 0.55 * rank / n   # 1.0 down to ~0.45
        r, g, b = (c / 255.0 * factor for c in base_rgb)
        return (r, g, b)

    fig, ax = plt.subplots(figsize=(11, 7))

    phi   = results_list[0]['phi']
    theta = results_list[0]['theta']
    psi   = results_list[0]['psi']

    # Build ordered unique lists so rank is stable
    unique_n_ants  = sorted(set(r.get('n_ant', 4) for r in results_list))
    unique_rates   = list(dict.fromkeys(
        r.get('target_global_rate_hz', None) for r in results_list
    ))
    unique_filters = sorted(set(r.get('apply_second_filter', False) for r in results_list))
    vary_by_filter = len(unique_filters) > 1
    n_unique_rates = len(unique_rates)

    for res in results_list:
        snr        = res['snr']
        eff_coinc  = res['eff_coinc']
        thr        = res['threshold_lhcp']
        n_ant_r    = res.get('n_ant', 4)
        rate       = res.get('target_global_rate_hz', None)
        filt2      = res.get('apply_second_filter', False)
        rate_rank  = unique_rates.index(rate) if rate in unique_rates else 0

        if vary_by_filter:
            # Color hue = filter config; shade + linestyle = rate rank
            base = FILTER_BASE_COLORS.get(filt2, (128, 128, 128))
            color = _shade(base, rate_rank, n_unique_rates)
            ls    = RATE_LINESTYLES.get(rate_rank, '-')
            style = dict(linestyle=ls, alpha=1.0)
            filt_str  = '1.5G+0.75G' if filt2 else '1.5G only'
            rate_str  = f'{rate:.3g} Hz' if rate is not None else ''
            lbl = f'{n_ant_r}-ant  {filt_str}  {rate_str}  (T={thr:.3f})'
        else:
            color     = ANT_COLORS.get(n_ant_r, '#888888')
            style     = RATE_STYLES.get(rate_rank, RATE_STYLES[0])
            if 'label' in res:
                lbl = res['label']
            else:
                rate_str = f'{rate:.3g} Hz' if rate is not None else 'T only'
                lbl = f'{n_ant_r}-ant  {rate_str}  (T={thr:.3f})'

        ax.plot(snr, eff_coinc, color=color, linewidth=2, label=lbl, **style)

        # Dotted vertical at SNR-50
        idx = np.where(eff_coinc >= 0.5)[0]
        if len(idx) > 0:
            k = idx[0]
            if k > 0:
                x0, x1 = snr[k-1], snr[k]
                y0, y1 = eff_coinc[k-1], eff_coinc[k]
                snr_50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
            else:
                snr_50 = snr[0]
            ax.axvline(snr_50, color=color, linestyle=':', alpha=0.35, linewidth=1)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Per-antenna SNR = Vpp / (2\u03c3)', fontsize=12)
    ax.set_ylabel('Trigger Efficiency', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(results_list[0]['snr'][0] - 0.1, results_list[0]['snr'][-1] + 0.1)
    ax.grid(True, alpha=0.3)

    # Sort legend: primary key = n_ant, secondary = filter state
    handles, labels = ax.get_legend_handles_labels()
    sort_keys = []
    for res in results_list:
        n_ant_r  = res.get('n_ant', 4)
        rate     = res.get('target_global_rate_hz', None)
        filt2    = int(res.get('apply_second_filter', False))
        rate_rank = unique_rates.index(rate) if rate in unique_rates else 0
        sort_keys.append((n_ant_r, filt2, rate_rank))
    order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    legend_title = ('filter  rate  (threshold)  [color=filter, shade+style=rate]'
                    if vary_by_filter else 'N-ant  rate  (threshold)')
    ax.legend(handles, labels, loc='upper left', fontsize=9, title=legend_title)

    vary_by_nant = len(unique_n_ants) > 1
    if vary_by_filter and vary_by_nant:
        subtitle = 'vs. Filter & N Antennas'
    elif vary_by_filter:
        subtitle = 'vs. Filter Stage'
    elif vary_by_nant:
        subtitle = 'vs. N Antennas & Trigger Rate'
    else:
        subtitle = 'vs. Trigger Rate'
    title = (f'Dual-Pol S-Curves {subtitle}\n'
             f'(\u03c6={phi:.1f}\u00b0, \u03b8={theta:.1f}\u00b0, \u03c8={psi:.1f}\u00b0)')
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if save_filename:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{save_filename}_multi_scurve.png', dpi=150)
        print(f"Saved: plots/{save_filename}_multi_scurve.png")

    plt.show()

    return fig, ax


def plot_filter_comparison(res_f1, res_f2, save_filename=None):
    """
    Two-panel figure comparing filt1 vs filt2 S-curves under two SNR axis definitions:

    Left panel  — Common σ_f1 axis: both curves plotted against the same
                  Vpp/(2*σ_f1) denominator (what the scan outputs directly).
                  The two curves nearly overlap, showing that on a common
                  physical input scale the threshold improvement is invisible.

    Right panel — Native σ axis: filt1 unchanged (σ_f1 denominator); filt2
                  x-values rescaled by σ_f2/σ_f1 so the x-axis reads
                  Vpp/(2*σ_f2).  The filt2 curve shifts left by that ratio,
                  revealing the intrinsic sensitivity gain of the second filter.
    """
    snr_f1       = res_f1['snr']
    eff_f1       = res_f1['eff_coinc']
    thr_f1       = res_f1['threshold_lhcp']

    snr_f2_common = res_f2['snr']         # same σ_f1 denominator as scanned
    eff_f2        = res_f2['eff_coinc']
    thr_f2        = res_f2['threshold_lhcp']

    sigma_f1 = res_f1.get('noise_rms_f1') or res_f2.get('noise_rms_f1') or 0.852
    sigma_f2 = res_f2.get('noise_rms_f2') or 0.592
    ratio    = sigma_f2 / sigma_f1        # e.g. 0.695

    # Rescaled x-axis: each filt2 SNR value expressed in its own σ_f2 units
    snr_f2_native = snr_f2_common * ratio

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    c1, c2 = '#1f77b4', '#d62728'   # blue = filt1, red = filt2

    def mark_50(ax, snr_arr, eff, color):
        idx = np.where(eff >= 0.5)[0]
        if len(idx) == 0:
            return
        k = idx[0]
        if k > 0:
            x0, x1 = snr_arr[k-1], snr_arr[k]
            y0, y1 = eff[k-1], eff[k]
            snr_50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
        else:
            snr_50 = snr_arr[0]
        ax.axvline(snr_50, color=color, linestyle=':', alpha=0.6, linewidth=1.2)
        ax.text(snr_50, 0.04, f'{snr_50:.2f}', ha='center', fontsize=9,
                color=color, bbox=dict(boxstyle='round', facecolor='white',
                                       alpha=0.8, edgecolor=color, linewidth=0.7))

    # ---- Left panel: common σ_f1 axis ----------------------------------------
    ax = axes[0]
    ax.plot(snr_f1,       eff_f1, '-',  color=c1, linewidth=2,
            label=f'filt1  1.5 GHz   T={thr_f1:.3f}')
    ax.plot(snr_f2_common, eff_f2, '--', color=c2, linewidth=2,
            label=f'filt2  750 MHz   T={thr_f2:.3f}')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    mark_50(ax, snr_f1,        eff_f1, c1)
    mark_50(ax, snr_f2_common, eff_f2, c2)
    ax.set_xlabel(r'Per-antenna SNR $= V_{pp}\,/\,(2\,\sigma_{f1})$  [common axis]',
                  fontsize=11)
    ax.set_ylabel('Coincidence Trigger Efficiency', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(snr_f1[0] - 0.05, snr_f1[-1] + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title(f'Common $\\sigma_{{f1}}$ reference  ($\\sigma_{{f1}}={sigma_f1:.3f}$)\n'
                 'Curves nearly overlap — improvement hidden in threshold value',
                 fontsize=10)

    # ---- Right panel: each curve on its own native σ axis -------------------
    ax = axes[1]
    ax.plot(snr_f1,        eff_f1, '-',  color=c1, linewidth=2,
            label=f'filt1   $\\sigma_{{f1}}={sigma_f1:.3f}$')
    ax.plot(snr_f2_native, eff_f2, '--', color=c2, linewidth=2,
            label=f'filt2   $\\sigma_{{f2}}={sigma_f2:.3f}$')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    mark_50(ax, snr_f1,        eff_f1, c1)
    mark_50(ax, snr_f2_native, eff_f2, c2)
    ax.set_xlabel(r'Per-antenna SNR $= V_{pp}\,/\,(2\,\sigma)$  '
                  r"[each filter's own $\sigma$]", fontsize=11)
    x_max = max(snr_f1[-1], snr_f2_native[-1])
    ax.set_xlim(min(snr_f1[0], snr_f2_native[0]) - 0.05, x_max + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title(f'Native $\\sigma$ reference  '
                 f'(filt2 axis $\\times\\,\\sigma_{{f2}}/\\sigma_{{f1}} = {ratio:.3f}$)\n'
                 'Leftward shift reveals intrinsic sensitivity gain', fontsize=10)

    phi   = res_f1['phi']
    theta = res_f1['theta']
    psi   = res_f1['psi']
    fig.suptitle(
        f'Filter Comparison: Effect of SNR-axis Definition on S-Curve\n'
        f'($\\varphi={phi:.1f}^\\circ$, '
        f'$\\theta={theta:.1f}^\\circ$, '
        f'$\\psi={psi:.1f}^\\circ$)',
        fontsize=13)

    plt.tight_layout()

    if save_filename:
        os.makedirs('plots', exist_ok=True)
        out = f'plots/{save_filename}_filter_comparison.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")

    plt.show()
    return fig, axes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dual-pol SNR efficiency scan")
    
    # Direction parameters
    parser.add_argument('--phi', type=float, default=0.0, help='Azimuth angle (deg)')
    parser.add_argument('--theta', type=float, default=0.0, help='Elevation angle (deg, -90=nadir, 0=horizon, +90=zenith)')
    parser.add_argument('--psi', type=float, default=45.0, help='Polarization angle (deg, 0=H-pol, 90=V-pol)')
    
    # Threshold parameters
    parser.add_argument('--threshold-lhcp', type=float, default=3.23, 
                       help='LHCP normalized power threshold')
    parser.add_argument('--threshold-rhcp', type=float, default=3.23,
                       help='RHCP normalized power threshold')
    parser.add_argument('--coincidence-window', type=float, default=100.0,
                       help='Coincidence time window (ns)')
    parser.add_argument('--threshold-json', type=str, default=None,
                       help='Path to threshold_analysis_dualpol.json; overrides '
                            '--threshold-lhcp, --threshold-rhcp, and --coincidence-window')
    parser.add_argument('--filt2-threshold-json', type=str, default=None,
                       help='Path to a second threshold JSON used for the filt2=True run '
                            'when --filter-scan is active. If omitted, the same JSON '
                            '(--threshold-json) is used for both filter configs.')
    parser.add_argument('--threshold-index', type=int, default=None,
                       help='Index into the "thresholds" list in the JSON. '
                            'Omit to run all entries; pass an integer to run only that entry.')
    
    # SNR scan parameters
    parser.add_argument('--snr-min', type=float, default=0.2, help='Minimum SNR')
    parser.add_argument('--snr-max', type=float, default=6.0, help='Maximum SNR')
    parser.add_argument('--snr-step', type=float, default=0.2, help='SNR step size')
    parser.add_argument('--trials', type=int, default=1000, help='Trials per SNR point')
    
    # Power sum parameters
    parser.add_argument('--window', type=int, default=160, help='Power sum window (samples)')
    parser.add_argument('--step', type=int, default=40, help='Power sum step (samples)')
    
    # Output
    parser.add_argument('--save', type=str, default='snr_scan_dualpol',
                       help='Output base filename')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    parser.add_argument('--single-ant-threshold', type=float, default=None,
                       help='Deprecated. Use --ant-scan instead.')

    # Antenna configuration
    parser.add_argument('--antennas', type=int, nargs='+', default=None,
                       help='Physical antenna indices to use, e.g. --antennas 0 for single-antenna '
                            'or --antennas 0 1 2 3 for full array (default: all four)')
    parser.add_argument('--ant-scan', action='store_true',
                       help='Run scans for 1, 2, 3, and 4 antennas and overlay S-curves '
                            'to show beamforming gain')
    parser.add_argument('--filter-scan', action='store_true',
                       help='Run scans with and without the second 750 MHz filter and '
                            'overlay the S-curves side by side')
    parser.add_argument('--second-filter', dest='second_filter', action='store_true', default=None,
                       help='Force second 750 MHz filter ON (overrides payload.APPLY_SECOND_FILTER)')
    parser.add_argument('--no-second-filter', dest='second_filter', action='store_false',
                       help='Force second filter OFF')

    # Angular scan mode (future enhancement)
    parser.add_argument('--angle-scan', action='store_true',
                       help='Enable angular scan (not yet implemented)')
    
    args = parser.parse_args()

    # Build SNR grid
    snr_grid = np.arange(args.snr_min, args.snr_max + 0.5*args.snr_step, args.snr_step)

    # Resolve filter toggle
    if args.filter_scan:
        filter_configs = [False, True]   # run without then with second filter
    elif args.second_filter is None:
        filter_configs = [payload.APPLY_SECOND_FILTER]
    else:
        filter_configs = [args.second_filter]

    # -------------------------------------------------------------------------
    # JSON mode: run one scan per threshold entry and overlay all S-curves
    # -------------------------------------------------------------------------
    if args.threshold_json is not None:
        with open(args.threshold_json, 'r') as f:
            thr_data = json.load(f)

        coinc_window_json = thr_data.get('coincidence_window_ns', args.coincidence_window)
        thr_entries = thr_data['thresholds']

        # If a specific index was requested, use only that entry
        if args.threshold_index is not None:
            thr_entries = [thr_entries[args.threshold_index]]

        print(f"Loaded {len(thr_entries)} threshold entr{'y' if len(thr_entries)==1 else 'ies'} "
              f"from {args.threshold_json}")
        print(f"Coincidence window: {coinc_window_json:.1f} ns")

        # --ant-scan: iterate over antenna counts for each threshold entry
        ant_configs = ([[0], [0,1], [0,1,2], [0,1,2,3]] if args.ant_scan
                       else [args.antennas])  # None = default (all four)

        # Build a second set of threshold entries for filt2 when a separate JSON is given
        if args.filt2_threshold_json is not None and args.filter_scan:
            with open(args.filt2_threshold_json, 'r') as _f2:
                _thr_data2 = json.load(_f2)
            _thr_entries2 = _thr_data2['thresholds']
            if args.threshold_index is not None:
                _thr_entries2 = [_thr_entries2[args.threshold_index]]
            if len(_thr_entries2) < len(thr_entries):
                # Warn rather than silently padding — mismatched JSON entries
                print(f"WARNING: filt2 JSON has {len(_thr_entries2)} entries but filt1 has "
                      f"{len(thr_entries)}. Extra filt1 entries will be skipped.")
                thr_entries = thr_entries[:len(_thr_entries2)]
            filt2_thr_entries = _thr_entries2
            print(f"Loaded filt2 thresholds from {args.filt2_threshold_json}")
        else:
            filt2_thr_entries = thr_entries  # same thresholds for both configs

        all_results = []
        for i, entry in enumerate(thr_entries):
            rate_label = entry.get('target_global_rate_hz', None)

            for ant_cfg in ant_configs:
                for filt2 in filter_configs:
                    # Pick the threshold entry for the current filter config
                    thr_entry = filt2_thr_entries[i] if filt2 else entry
                    t_lhcp = thr_entry['threshold_lhcp']
                    t_rhcp = thr_entry['threshold_rhcp']

                    n = len(ant_cfg) if ant_cfg is not None else 4
                    rate_tag = f"_rate{i}" if len(thr_entries) > 1 else ''
                    ant_tag  = f"_ant{n}"  if args.ant_scan else ''
                    filt_tag = f"_filt{'2' if filt2 else '1'}" if args.filter_scan else ''
                    save_tag = f"{args.save}{rate_tag}{ant_tag}{filt_tag}"

                    print(f"\n--- Entry {i}: LHCP={t_lhcp:.4f}  target={rate_label} Hz  "
                          f"antennas={ant_cfg if ant_cfg is not None else [0,1,2,3]}  "
                          f"2nd-filter={filt2} ---")

                    res = run_snr_scan_dualpol(
                        phi=args.phi,
                        theta=args.theta,
                        psi=args.psi,
                        threshold_lhcp=t_lhcp,
                        threshold_rhcp=t_rhcp,
                        coincidence_window=coinc_window_json,
                        snr_grid=snr_grid,
                        n_trials=args.trials,
                        window=args.window,
                        step=args.step,
                        save_filename=save_tag,
                        antennas=ant_cfg,
                        apply_second_filter=filt2,
                    )
                    if rate_label is not None:
                        res['target_global_rate_hz'] = rate_label
                    all_results.append(res)

        if not args.no_plot:
            if len(all_results) == 1:
                plot_scurves(all_results[0], save_filename=args.save)
            else:
                plot_multi_scurves(all_results, save_filename=args.save)
            # For --filter-scan: produce one comparison plot per threshold entry,
            # pairing filt1 and filt2 results that share the same rate label.
            if args.filter_scan:
                filt1_results = [r for r in all_results if not r.get('apply_second_filter', False)]
                filt2_results = [r for r in all_results if     r.get('apply_second_filter', False)]
                for rf1, rf2 in zip(filt1_results, filt2_results):
                    rate = rf1.get('target_global_rate_hz')
                    rate_tag = f'_rate{rate:.3g}Hz'.replace('.', 'p') if rate is not None else ''
                    plot_filter_comparison(rf1, rf2,
                                          save_filename=f'{args.save}{rate_tag}')

    # -------------------------------------------------------------------------
    # Single-threshold mode (original behaviour)
    # -------------------------------------------------------------------------
    else:
        ant_configs = ([[0], [0,1], [0,1,2], [0,1,2,3]] if args.ant_scan
                       else [args.antennas])

        all_results = []
        for ant_cfg in ant_configs:
            for filt2 in filter_configs:
                n = len(ant_cfg) if ant_cfg is not None else 4
                ant_tag  = f"_ant{n}"             if args.ant_scan    else ''
                filt_tag = f"_filt{'2' if filt2 else '1'}" if args.filter_scan else ''
                save_tag = f"{args.save}{ant_tag}{filt_tag}"

                res = run_snr_scan_dualpol(
                    phi=args.phi,
                    theta=args.theta,
                    psi=args.psi,
                    threshold_lhcp=args.threshold_lhcp,
                    threshold_rhcp=args.threshold_rhcp,
                    coincidence_window=args.coincidence_window,
                    snr_grid=snr_grid,
                    n_trials=args.trials,
                    window=args.window,
                    step=args.step,
                    save_filename=save_tag,
                    antennas=ant_cfg,
                    apply_second_filter=filt2,
                )
                all_results.append(res)

        if not args.no_plot:
            if len(all_results) == 1:
                plot_scurves(all_results[0], save_filename=args.save)
            else:
                plot_multi_scurves(all_results, save_filename=args.save)
            if args.filter_scan and len(all_results) == 2:
                res_f1 = next((r for r in all_results if not r.get('apply_second_filter', False)), None)
                res_f2 = next((r for r in all_results if     r.get('apply_second_filter', False)), None)
                if res_f1 is not None and res_f2 is not None:
                    plot_filter_comparison(res_f1, res_f2, save_filename=args.save)

    print("\n=== Scan Complete ===")
