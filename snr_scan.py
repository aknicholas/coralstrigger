import numpy
import myplot
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


#pick phi, theta (incoming wave angle)
phi = 90
theta = 0

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="SNR efficiency scan")
    parser.add_argument('--fit-json', type=str, help='JSON file with A,b,window,step from threshold fit')
    parser.add_argument('--target-global-rate', type=float, default=0.1, help='Target global false rate (Hz)')
    parser.add_argument('--n-beams', type=int, default=400, help='Total active beams for global rate calc')
    parser.add_argument('--window', type=int, default=256, help='Power sum window (samples)')
    parser.add_argument('--step', type=int, default=None, help='Power sum step (samples, default window/2)')
    parser.add_argument('--save', type=str, default='snr_scan_data', help='Output base filename')

    # New: angle scan options
    parser.add_argument('--angle-scan', action='store_true',
                        help='Enable angular scan around (phi,theta) and build a 50%%-efficiency SNR heatmap')
    parser.add_argument('--phi0', type=float, default=phi, help='Center phi (deg)')
    parser.add_argument('--theta0', type=float, default=theta, help='Center theta (deg)')
    parser.add_argument('--span-phi', type=float, default=10.0, help='+/- span in phi (deg) around center')
    parser.add_argument('--span-theta', type=float, default=10.0, help='+/- span in theta (deg) around center')
    parser.add_argument('--ang-res', type=float, default=1.0, help='Angular resolution (deg)')

    # New: SNR scan controls for heatmap
    parser.add_argument('--snr-min', type=float, default=0.2)
    parser.add_argument('--snr-max', type=float, default=6.0)
    parser.add_argument('--snr-step', type=float, default=0.1)
    parser.add_argument('--trials', type=int, default=400, help='Trials per SNR point (per angle)')
    parser.add_argument('--target-eff', type=float, default=0.5, help='Target efficiency for SNR* (e.g., 0.5)')

    args_cli = parser.parse_args()

    save_filename = args_cli.save
    window = args_cli.window
    step = window//2 if args_cli.step is None else args_cli.step

    # --- Load threshold fit (optional) ---
    dynamic_threshold = None
    if args_cli.fit_json:
        with open(args_cli.fit_json) as f:
            fit_data = json.load(f)

        # Extract coefficients robustly
        def _get_coeff(name):
            if name in fit_data:
                return fit_data[name]
            if 'fit_coefficients' in fit_data and name in fit_data['fit_coefficients']:
                return fit_data['fit_coefficients'][name]
            return None

        A = _get_coeff('A')
        b = _get_coeff('b')
        if A is None or b is None:
            print("ERROR: Could not find A/b in JSON (looked at top level and fit_coefficients).")
            sys.exit(1)

        fit_window = fit_data.get('window', window)
        fit_step   = fit_data.get('step', step)

        if fit_window != window or fit_step != step:
            print(f"WARNING: Fit window/step ({fit_window},{fit_step}) != current ({window},{step}). Consider refitting or matching parameters.")

        # Compute threshold: R_global = n_beams * A * exp(b T)
        # => T = (ln(R_global) - ln(A * n_beams)) / b
        if b > 0:
            print(f"WARNING: Fit b={b:.4g} > 0 (expected negative). Thresholds may invert.")

        T_needed = (numpy.log(args_cli.target_global_rate) - numpy.log(A * args_cli.n_beams)) / b
        dynamic_threshold = float(T_needed)
        print(f"Computed threshold for global {args_cli.target_global_rate} Hz with {args_cli.n_beams} beams: {dynamic_threshold:.3f}")

    # (retain rest of original setup, but replace hard-coded threshold list)
    lp_freq, lp_mult, _ = filters.Shannon_Whitaker(fs=4e9, plot=False)
    lp_gain = 10**(lp_mult/20)
    eplane  = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
    trigger_sectors_phi = [1,2,3,4]
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff=2)

    phi_sectors = [1,2,3,4]
    ringmask=[1,1,1,1]

    # Threshold list: if dynamic threshold computed, use same for all noise profiles
    if dynamic_threshold is not None:
        threshold = [dynamic_threshold]
    else:
        threshold = [4.2]  # fallback manual
        print(f"WARNING: Using fallback threshold {threshold[0]:.3f} (no --fit-json provided).")

    # --- Angle scan mode: build 50% efficiency SNR heatmap ---
    if args_cli.angle_scan:
        # Configure SNR grid for per-angle scans
        snr_grid = numpy.arange(args_cli.snr_min, args_cli.snr_max + 0.5*args_cli.snr_step, args_cli.snr_step)
        trials = int(args_cli.trials)
        target_eff = float(args_cli.target_eff)

        # Angular grid
        phi_vals = numpy.arange(args_cli.phi0 - args_cli.span_phi,
                                args_cli.phi0 + args_cli.span_phi + 1e-9, args_cli.ang_res)
        theta_vals = numpy.arange(args_cli.theta0 - args_cli.span_theta,
                                  args_cli.theta0 + args_cli.span_theta + 1e-9, args_cli.ang_res)
        PHI, THETA = numpy.meshgrid(phi_vals, theta_vals)
        snr50_map = numpy.full_like(PHI, numpy.nan, dtype=float)


        print(f"[Angle Scan] Grid: {len(theta_vals)}x{len(phi_vals)} (res={args_cli.ang_res} deg). "
              f"SNR grid {snr_grid[0]}..{snr_grid[-1]} step {args_cli.snr_step}, trials={trials}, target eff={target_eff}")

        # Loop over angles
        for it, th in enumerate(theta_vals):
            for ip, ph in enumerate(phi_vals):
                # Build waveforms and delays for this direction
                delays = payload.getRemappedDelays(ph, th, phi_sectors)
                waveforms, timebase, multiplier = payload.getPayloadWaveforms(
                    ph, th, phi_sectors, impulse, (eplane, hplane), plot=False, downsample=True)

                n_phi = waveforms.shape[0]
                n_ring = waveforms.shape[1]
                n_samp = waveforms.shape[2]
                # Determine actual antenna channels (collapse sector/ring fiction)
                wf_flat = waveforms.reshape(-1, n_samp)
                active_mask = numpy.any(wf_flat != 0, axis=1)
                n_ant = int(active_mask.sum()) if active_mask.any() else wf_flat.shape[0]
                norm = numpy.sqrt(n_ant)

                # Build noise model for this n_samp
                thermal_noise_v2 = noise.ThermalNoise(
                    0.26, 1.05, filter_order=(10,10), v_rms=1.0,
                    fbins=int(n_samp),
                    time_domain_sampling_rate=aso_geometry.ritc_sample_step
                )


                # Pre-generate noise for all trials once for this angle (reused across SNRs)
                noise_pack = thermal_noise_v2.makeNoiseWaveform(
                    ntraces=trials * n_phi * n_ring)
                noise_flat = numpy.real(noise_pack[2])

                # Compute efficiency curve vs SNR for this angle
                effs = []
                for snr in snr_grid:
                    _hits = 0
                    for i in range(trials):
                        start = i * n_phi * n_ring
                        stop  = start + n_phi * n_ring
                        event_noise = numpy.reshape(noise_flat[start:stop], (n_phi, n_ring, n_samp))
                        injected = waveforms * snr + event_noise
                        coh_sum, _  = trigger.coherentSum(injected, timebase, delays, ringmask=ringmask)
                        power,_ = trigger.powerSum(coh_sum / norm, window=window, step=step)
                        if numpy.max(power) > threshold[0]:
                            _hits += 1
                    effs.append(_hits / trials)

                effs = numpy.array(effs)
                # Find SNR at target efficiency by linear interpolation
                idx = numpy.where(effs >= target_eff)[0]
                if idx.size == 0:
                    # Did not reach target; leave NaN (consider increasing snr_max)
                    snr_star = numpy.nan
                else:
                    k = idx[0]
                    if k == 0:
                        snr_star = snr_grid[0]
                    else:
                        x0, x1 = snr_grid[k-1], snr_grid[k]
                        y0, y1 = effs[k-1], effs[k]
                        # Prevent divide by zero
                        if y1 == y0:
                            snr_star = x1
                        else:
                            snr_star = x0 + (target_eff - y0) * (x1 - x0) / (y1 - y0)
                snr50_map[it, ip] = snr_star
                print(f"[Angle ({ph:.1f},{th:.1f})] SNR@{target_eff:.2f} ≈ {snr_star if numpy.isfinite(snr_star) else float('nan'):.3f}")

        # Plot heatmap
        import os
        os.makedirs('plots', exist_ok=True)
        masked = numpy.ma.masked_invalid(snr50_map)
        plt.figure(figsize=(9,7))
        pcm = plt.pcolormesh(PHI, THETA, masked, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=f'SNR at {int(target_eff*100)}% efficiency')
        plt.xlabel('Phi [deg]')
        plt.ylabel('Theta [deg]')
        plt.title(f'SNR@{int(target_eff*100)}% Efficiency Heatmap (win={window}, step={step})')
        plt.tight_layout()
        out_png = f'plots/efficiency_{int(target_eff*100)}pct_heatmap_phi{args_cli.phi0}_th{args_cli.theta0}_res{args_cli.ang_res}.png'
        plt.savefig(out_png, dpi=150)
        print(f"Saved heatmap: {out_png}")

        # Save data
        out_npy = f'plots/efficiency_{int(target_eff*100)}pct_heatmap.npy'
        numpy.save(out_npy, {'phi': phi_vals, 'theta': theta_vals, 'snr_star': snr50_map,
                             'window': window, 'step': step,
                             'threshold': threshold[0],
                             'target_eff': target_eff})
        print(f"Saved heatmap data: {out_npy}")

        sys.exit(0)

    # ----------------- Original single-direction SNR scan (unchanged) -----------------
    num_of_events_per_snr_step = 1000
    snr_scan = numpy.arange(0.2, 4.0, 0.1)
    print(f"snr scan array {snr_scan}")
    delays = payload.getRemappedDelays(phi, theta, phi_sectors)
    waveforms, timebase, multiplier = payload.getPayloadWaveforms(phi, theta, phi_sectors, impulse, (eplane, hplane), plot=False, downsample=True)
    # Actual antenna count (ignore phi sectors)
    wf_flat = waveforms.reshape(-1, waveforms.shape[2])
    active_mask = numpy.any(wf_flat != 0, axis=1)
    n_ant = int(active_mask.sum()) if active_mask.any() else wf_flat.shape[0]
    norm = 4

    thermal_noise_v2 = noise.ThermalNoise(0.26, 1.05, filter_order=(10,10), v_rms=1.0,
                                          fbins=waveforms.shape[2],
                                          time_domain_sampling_rate=aso_geometry.ritc_sample_step)

    noise_list=[]
    noise_list.append(thermal_noise_v2.makeNoiseWaveform(
        ntraces=num_of_events_per_snr_step*waveforms.shape[0]*waveforms.shape[1]))
    print(f"noise lise array {noise_list}")
    data_to_save = []
    data_to_save.append(snr_scan)

    for j in range(len(noise_list)):
        hits=[]
        for snr in snr_scan:
            _hits = 0
            for i in range(num_of_events_per_snr_step):
                start = i*waveforms.shape[0]*waveforms.shape[1]
                stop  = start + waveforms.shape[0]*waveforms.shape[1]
                event_noise = numpy.reshape(
                    numpy.real(noise_list[j][2][start:stop]),
                    (waveforms.shape[0], waveforms.shape[1], waveforms.shape[2])
                )
                injected = waveforms*snr + event_noise
                coh_sum, timebase_coh_sum  = trigger.coherentSum(
                    injected, timebase, delays, ringmask=ringmask)
                power,_ = trigger.powerSum(coh_sum / norm, window=window, step=step)
                if numpy.max(power) > threshold[j]:
                    _hits += 1
            eff = _hits/num_of_events_per_snr_step
            print(f"SNR {snr:.2f}  Eff {eff:.3f}")
            hits.append(eff)

        hits = numpy.array(hits)
        data_to_save.append(hits)
        data_to_save.append(hits * (numpy.max(multiplier)))

        plt.plot(snr_scan, hits, label=f'Noise')

    plt.ylim(-0.05,1.05)
    plt.xlabel("Injected SNR (coherent amplitude scale)")
    plt.ylabel("Trigger Efficiency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    numpy.savetxt(save_filename+'.txt', numpy.array(data_to_save), fmt="%.6g")
    plt.tight_layout()
    plt.savefig("plots/SNR_scurve.png", dpi =150)
    plt.show()