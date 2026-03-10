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
                         apply_second_filter=None):
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
    print(f"Second filter (750 MHz): {'ENABLED' if apply_second_filter else 'DISABLED'}")
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
    print(f"  Noise RMS (H): {noise_rms_h:.4f}, (V): {noise_rms_v:.4f}, (avg): {noise_rms:.4f}")
    print(f"  SNR axis: per-antenna Vpp/(2σ)")
    
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
                fc_second=payload.FC_SECOND_FILTER,
                output='circular'
            )
            
            # Compute power sums
            # IMPORTANT: Normalize by n_ant to get power per antenna
            # - Coherent sum grows signal voltage by n_ant → power by n_ant²
            # - Incoherent noise sum grows noise power by n_ant
            # - After beamforming: effective SNR improves by n_ant
            # - Thresholds are set on n_ant-beamformed noise, so we normalize by n_ant
            #   to express SNR relative to single-antenna threshold
            power_lhcp, _ = trigger.powerSum(lhcp, window=window, step=step)
            power_rhcp, _ = trigger.powerSum(rhcp, window=window, step=step)
            
            # Normalize power by number of antennas per polarization
            power_lhcp = power_lhcp / n_ant
            power_rhcp = power_rhcp / n_ant
            
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
            
            # Check coincidence: both must trigger within time window
            if lhcp_triggered and rhcp_triggered:
                # Find trigger times
                idx_lhcp = np.argmax(power_lhcp)
                idx_rhcp = np.argmax(power_rhcp)
                
                # Convert frame indices to time (ns)
                dt_lhcp = idx_lhcp * step * aso_geometry.ritc_sample_step * 1e9
                dt_rhcp = idx_rhcp * step * aso_geometry.ritc_sample_step * 1e9
                
                time_diff = abs(dt_lhcp - dt_rhcp)
                
                if time_diff <= coincidence_window:
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
    # --- Style tables (rate-rank → linestyle/alpha, n_ant → color) -----------
    # Up to 4 rates: solid full, dashed, solid half-opacity, dotted
    RATE_STYLES = {
        0: dict(linestyle='-',  alpha=1.0),
        1: dict(linestyle='--', alpha=1.0),
        2: dict(linestyle='-',  alpha=0.45),
        3: dict(linestyle=':',  alpha=1.0),
    }
    # One colour per unique antenna count (tab10 first four: blue, orange, green, red)
    ANT_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}
    # When comparing filter states, use color=filter state, linestyle=n_ant
    FILTER_COLORS = {False: '#1f77b4', True: '#d62728'}     # blue=no 2nd filter, red=with 2nd filter
    NANT_STYLES   = {1: ':', 2: '--', 3: '-.', 4: '-'}

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

    for res in results_list:
        snr        = res['snr']
        eff_coinc  = res['eff_coinc']
        thr        = res['threshold_lhcp']
        n_ant_r    = res.get('n_ant', 4)
        rate       = res.get('target_global_rate_hz', None)
        filt2      = res.get('apply_second_filter', False)

        if vary_by_filter:
            # Color = filter state; linestyle = n_ant
            color = FILTER_COLORS.get(filt2, '#888888')
            ls    = NANT_STYLES.get(n_ant_r, '-')
            style = dict(linestyle=ls, alpha=1.0)
            filt_str = '1.5G+0.75G' if filt2 else '1.5G only'
            lbl = f'{n_ant_r}-ant  {filt_str}  (T={thr:.3f})'
        else:
            color     = ANT_COLORS.get(n_ant_r, '#888888')
            rate_rank = unique_rates.index(rate) if rate in unique_rates else 0
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
    legend_title = ('N-ant  filter  (threshold)' if vary_by_filter
                    else 'N-ant  rate  (threshold)')
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
    parser.add_argument('--threshold-index', type=int, default=0,
                       help='Index into the "thresholds" list in the JSON (default: 0)')
    
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
        if args.threshold_index != 0 or len(thr_entries) == 1:
            thr_entries = [thr_entries[args.threshold_index]]

        print(f"Loaded {len(thr_entries)} threshold entr{'y' if len(thr_entries)==1 else 'ies'} "
              f"from {args.threshold_json}")
        print(f"Coincidence window: {coinc_window_json:.1f} ns")

        # --ant-scan: iterate over antenna counts for each threshold entry
        ant_configs = ([[0], [0,1], [0,1,2], [0,1,2,3]] if args.ant_scan
                       else [args.antennas])  # None = default (all four)

        all_results = []
        for i, entry in enumerate(thr_entries):
            t_lhcp = entry['threshold_lhcp']
            t_rhcp = entry['threshold_rhcp']
            rate_label = entry.get('target_global_rate_hz', None)

            for ant_cfg in ant_configs:
                for filt2 in filter_configs:
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

    print("\n=== Scan Complete ===")
