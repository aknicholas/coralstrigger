"""
SNR efficiency scan for dual-polarization CoRaLS trigger

Generates S-curves (efficiency vs SNR) for LHCP, RHCP, and coincidence triggers.
Tests signal detection efficiency at various SNR levels and angles.

Usage:
    python snr_scan_dualpol.py --threshold-lhcp 3.23 --threshold-rhcp 3.23 --n-beams 100
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
                         snr_grid, n_trials, window, step, save_filename=None):
    """
    Run SNR efficiency scan for dual-pol at a single direction
    
    Parameters:
    -----------
    phi, theta, psi : float
        Signal arrival angles (deg)
    threshold_lhcp, threshold_rhcp : float
        Normalized power thresholds for LHCP and RHCP
    coincidence_window : float
        Coincidence time window (ns)
    snr_grid : array
        SNR values to test
    n_trials : int
        Number of trials per SNR point
    window, step : int
        Power sum window and step (samples)
    save_filename : str, optional
        Base filename for saving results
        
    Returns:
    --------
    dict with keys: snr, eff_lhcp, eff_rhcp, eff_coinc
    """
    
    print(f"\n=== SNR Scan at (phi={phi:.1f}°, theta={theta:.1f}°, psi={psi:.1f}°) ===")
    print(f"LHCP threshold: {threshold_lhcp:.3f}")
    print(f"RHCP threshold: {threshold_rhcp:.3f}")
    print(f"Coincidence window: {coincidence_window:.1f} ns")
    print(f"SNR range: {snr_grid[0]:.2f} to {snr_grid[-1]:.2f} ({len(snr_grid)} points)")
    print(f"Trials per SNR: {n_trials}")
    
    # Load beam patterns - NOTE: which_pol parameter does NOT mean polarization response!
    # It's a confusing naming convention where 'V' means "use standard pattern"
    # The actual H-pol vs V-pol response comes from the psi projection, not beam patterns
    eplane = payload.beamPattern(plot=False, which_plane='E', which_pol='V')
    hplane = payload.beamPattern(plot=False, which_plane='H', which_pol='V')
    
    # Load and prepare impulse
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    # Note: prepImpulse no longer applies analog filters - just normalizes
    impulse = payload.prepImpulse(impulse)
    
    # Configuration
    antennas = [0, 1, 2, 3]
    ringmask = [1, 1, 1, 1]
    
    # Generate waveforms for H and V polarizations simultaneously
    delays_array = payload.getRemappedDelays(phi, theta, antennas)
    
    waveforms_h, waveforms_v, timebase, mult_h, mult_v = payload.getPayloadWaveforms_dualpol(
        phi, theta, impulse,
        (eplane, hplane),  # Same beam patterns for both polarizations
        (eplane, hplane),  # Polarization response comes from psi projection
        antennas=antennas,
        psi=psi,
        plot=False,
        downsample=True
    )
    
    # waveforms shape: (n_antennas, n_samples)
    n_ant = waveforms_h.shape[0]
    n_samp = waveforms_h.shape[1]
    norm = np.sqrt(n_ant)
    
    print(f"Signal dimensions: {n_ant} antennas × {n_samp} samples")
    print(f"Active antennas: {n_ant}, normalization: {norm:.2f}")
    
    # Generate white noise (will be filtered digitally after beamforming)
    thermal_noise = noise.ThermalNoise(
        0.26, 1.05, filter_order=(0, 0),  # White noise (no analog filtering)
        v_rms=1.0,
        fbins=int(n_samp),
        time_domain_sampling_rate=aso_geometry.ritc_sample_step
    )
    
    # Pre-generate noise for all trials (separate for H and V)
    print("Generating noise waveforms...")
    n_traces = n_trials * n_ant
    noise_h = thermal_noise.makeNoiseWaveform(ntraces=n_traces)
    noise_v = thermal_noise.makeNoiseWaveform(ntraces=n_traces)
    noise_h_flat = np.real(noise_h[2])
    noise_v_flat = np.real(noise_v[2])
    
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
            
            # Inject signal at specified SNR
            injected_h = waveforms_h * snr + event_noise_h
            injected_v = waveforms_v * snr + event_noise_v
            
            # Beamform and convert to circular polarizations
            # Shannon-Whitaker filter applied inside coherentSum_dualpol
            lhcp, rhcp, tb_coinc = trigger.coherentSum_dualpol(
                injected_h, injected_v, timebase, delays_array, apply_filter=True
            )
            
            # Compute power sums (NO normalization - threshold determined on raw power)
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
        'n_trials': n_trials
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(snr, eff_lhcp, 'o-', label=f'LHCP (T={thr_l:.3f})', color='blue', markersize=4)
    ax.plot(snr, eff_rhcp, 's-', label=f'RHCP (T={thr_r:.3f})', color='red', markersize=4)
    ax.plot(snr, eff_coinc, '^-', label='Coincidence', color='green', markersize=5, linewidth=2)
    
    # Mark 50% efficiency points
    for label, eff, color in [('LHCP', eff_lhcp, 'blue'), 
                               ('RHCP', eff_rhcp, 'red'),
                               ('Coinc', eff_coinc, 'green')]:
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
    
    ax.set_xlabel('Injected SNR (coherent amplitude scale)', fontsize=12)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dual-pol SNR efficiency scan")
    
    # Direction parameters
    parser.add_argument('--phi', type=float, default=0.0, help='Azimuth angle (deg)')
    parser.add_argument('--theta', type=float, default=-30.0, help='Elevation angle (deg, -90=nadir, 0=horizon, +90=zenith)')
    parser.add_argument('--psi', type=float, default=45.0, help='Polarization angle (deg, 0=H-pol, 90=V-pol)')
    
    # Threshold parameters
    parser.add_argument('--threshold-lhcp', type=float, default=3.23, 
                       help='LHCP normalized power threshold')
    parser.add_argument('--threshold-rhcp', type=float, default=3.23,
                       help='RHCP normalized power threshold')
    parser.add_argument('--coincidence-window', type=float, default=100.0,
                       help='Coincidence time window (ns)')
    
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
    
    # Angular scan mode (future enhancement)
    parser.add_argument('--angle-scan', action='store_true',
                       help='Enable angular scan (not yet implemented)')
    
    args = parser.parse_args()
    
    # Build SNR grid
    snr_grid = np.arange(args.snr_min, args.snr_max + 0.5*args.snr_step, args.snr_step)
    
    # Run scan
    results = run_snr_scan_dualpol(
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
        save_filename=args.save
    )
    
    # Plot results
    if not args.no_plot:
        plot_scurves(results, save_filename=args.save)
    
    print("\n=== Scan Complete ===")
