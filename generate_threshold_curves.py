import numpy
from scipy.optimize import curve_fit
import coherent_sum
import tools.CoRaLs_geometry as aso_geometry
import json
import myplot
import matplotlib.pyplot as plt
import sys, os, argparse

def generatePowerSums(noise, window=160, step=40, save=True):
    power = coherent_sum.powerSum(noise, window, step)
    if save:
        numpy.save('/home/aknicholas/Python Projects/Simulation/coralstrigger/noise/power_'+str(window)+'_'+str(step)+'.npy', power[0])

    return power


if __name__=='__main__':



    parser = argparse.ArgumentParser(description="Generate threshold curve from a power sums .npy file")
    default_power = os.path.join(os.path.dirname(__file__), 'noise', 'power_160_40.npy')
    parser.add_argument('--power-file', nargs='?', const=None,
                        default=default_power,
                        help='Optional path to power_<window>_<step>.npy. If used without value, uses default.')
    parser.add_argument('--window', type=int, default=160, help='Window (samples) used to make that file (informational)')
    parser.add_argument('--step', type=int, default=40, help='Step (samples) used in power sums')
    parser.add_argument('--fs', type=float, default=4e9, help='Sample rate (Hz)')
    parser.add_argument('--fit-min-count', type=float, default=50, help='Min exceedance count to include in fit')
    parser.add_argument('--fit-max-count', type=float, default=5e5, help='Max exceedance count to include in fit')
    parser.add_argument('--thr-min', type=float, default=0.2)
    parser.add_argument('--thr-max', type=float, default=8.0)
    parser.add_argument('--thr-step', type=float, default=0.1)
    parser.add_argument('--n-beams', type=int, default=1, help='Number of active beams (for global false rate)')
    parser.add_argument('--target-global-rate', type=float, nargs='*',
                        default=[1000,100,10,1],
                        help='List of target global false rates (Hz) to solve thresholds for')
    args = parser.parse_args()
    if args.power_file is None:
        args.power_file = default_power
        print(f"No path supplied after --power-file; using default: {args.power_file}")
    # If you just want to (re)generate power sums from raw noise, do that first then exit:
    # noise_data = numpy.load('noise/simulated_noise.npy').flatten()
    # generatePowerSums(numpy.real(noise_data), window=args.window, step=args.step)
    # sys.exit(0)

    # -------- Load power sums ----------
    if not os.path.exists(args.power_file):
        print(f"Power file not found: {args.power_file}")
        sys.exit(1)
    powersums = numpy.load(args.power_file)
    print(f"Loaded power sums: {powersums.shape[0]} windows from {args.power_file}")
    print(f"Window={args.window} samples  Step={args.step} samples  fs={args.fs/1e9:.3f} GHz")

    # -------- Threshold scan ----------
    thresh_array = numpy.arange(args.thr_min, args.thr_max + 0.5*args.thr_step, args.thr_step)
    hits = numpy.array([(powersums >= thr).sum() for thr in thresh_array])

    total_windows = len(powersums)
    dt = args.step / args.fs          # seconds between window starts
    obs_time = total_windows * dt     # total time represented
    per_beam_rate = hits / obs_time   # Hz per beam

    # -------- Select fit region (by raw counts) ----------
    fit_mask = (hits >= args.fit_min_count) & (hits <= args.fit_max_count)
    fit_th = thresh_array[fit_mask]
    fit_counts = hits[fit_mask]

    if fit_th.size < 3:
        print("Not enough points in fit range; adjust --fit-min-count / --fit-max-count.")
        sys.exit(1)

    # Convert to rate and log
    fit_rates = fit_counts / obs_time
    log_rates = numpy.log(fit_rates)

    # Weighted linear fit log(rate) = a + b * T (expect b negative)
    # Poisson sigma ~ sqrt(count); weight ~ 1/sigma => use counts**0.5
    weights = numpy.sqrt(fit_counts)
    # polyfit with weights on y -> pass w=weights
    coeff = numpy.polyfit(fit_th, log_rates, 1, w=weights)
    b, a = coeff[0], coeff[1]
    A = numpy.exp(a)      # prefactor
    k = -b if b < 0 else -b  # just notation; model R(T)=A*exp(b*T); usually b<0 so k=-b>0

    print(f"Fit: log(R) = {a:.3f} + ({b:.3f})*T  =>  R(T)= {A:.3e} * exp({b:.3f} T)")
    # -------- Solve thresholds for target global rates ----------
    # global_rate = per_beam_rate * n_beams = A*exp(b*T)*n_beams
    # T = (ln(global_rate) - ln(A*n_beams)) / b
    def solve_threshold(target_global):
        return (numpy.log(target_global) - numpy.log(A * args.n_beams)) / b

    print("Target global false rates:")
    for gr in args.target_global_rate:
        T_needed = solve_threshold(gr)
        print(f"  {gr:>10g} Hz  =>  threshold ≈ {T_needed:.3f}")
    # -------- Plot ----------
    model_rate = A * numpy.exp(b * thresh_array)
    model_output = {
        "version": 2,
        # Top-level (flat) copies so downstream tools can read directly
        "a": float(a),
        "b": float(b),
        "A": float(A),
        "k": float(k),
        "window": args.window,
        "step": args.step,
        "fs": args.fs,
        "n_beams": args.n_beams,
        "power_file": args.power_file,
        # Original nested block retained for backwards compatibility
        "fit_coefficients": {
            "a": float(a),
            "b": float(b),
            "A": float(A),
            "k": float(k)
        },
        "target_global_rates": [
            {
                "rate": float(gr),
                "threshold": float(solve_threshold(gr))
            } for gr in args.target_global_rate
        ]
    }

    os.makedirs('plots', exist_ok=True)
    out_json = f'plots/threshold_model_{args.n_beams}_beams.json'
    with open(out_json, 'w') as f:
        json.dump(model_output, f, indent=2)
    print(f"Saved threshold fit JSON: {out_json}")
    # Dynamic x-limits based on data (nonzero hits)
    nonzero_idx = numpy.where(hits > 0)[0]
    if nonzero_idx.size:
        x_left = thresh_array[nonzero_idx[0]]
        x_right = thresh_array[nonzero_idx[-1]]
        if x_left == x_right:
            # Single populated bin; widen slightly
            x_left -= args.thr_step
            x_right += args.thr_step
    else:
        # Fallback: entire scanned range
        x_left = thresh_array[0]
        x_right = thresh_array[-1]

    # Small padding
    pad = 0.05 * (x_right - x_left) if (x_right - x_left) > 0 else args.thr_step
    x_min_plot = max(thresh_array[0], x_left - pad)
    x_max_plot = min(thresh_array[-1], x_right + pad)

    plt.figure(figsize=(7,5))
    plt.semilogy(thresh_array, per_beam_rate, 'o', ms=4, label='Per-beam empirical')
    plt.semilogy(thresh_array, model_rate, '-', label='Fit')
    plt.semilogy(fit_th, fit_rates, 's', ms=5, label='Fit region')
    plt.xlabel('Normalized Power Threshold  (P / (N σ²))')
    plt.ylabel('Per-beam Noise Rate [Hz]')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.xlim(x_min_plot, x_max_plot)
    plt.ylim(bottom=0.01)
    plt.tight_layout()
    plt.savefig(f'plots/threshold_fit_{args.n_beams}_beams.jpg', dpi = 150)

    # Histogram (log y)
    plt.figure(figsize=(7,5))
    plt.hist(powersums, bins=thresh_array, alpha=0.6)
    plt.yscale('log')
    plt.xlabel('Power')
    plt.ylabel('Counts')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(x_min_plot, x_max_plot)
    plt.tight_layout()
    plt.savefig(f'plots/threshold_{args.n_beams}_beams.jpg', dpi = 150)
    plt.show()