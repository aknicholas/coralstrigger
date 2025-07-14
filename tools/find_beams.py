import numpy as np
from sklearn.cluster import DBSCAN
import coherent_sum as sum
import payload_signal as payload

# ─── Prep once ───────────────────────────────────────────────────────────────
eplane = payload.beamPattern(plot=False, which_plane='E', which_pol='V')
hplane = payload.beamPattern(plot=False, which_plane='H', which_pol='V')
sectors = [1,2,3,4,5,6,7,8]
ringmask = [1,1,0,0,0,0,0,1]

impulse_raw = payload.loadImpulse('impulse/corals_impulse_sci.txt')
impulse   = payload.prepImpulse(impulse_raw,
                                highpass_cutoff=0.15,
                                lowpass_cutoff=2)

def make_get_power(impulse, eplane, hplane, sectors, ringmask):
    def get_power(phi, theta):
        delays = payload.getRemappedDelays(phi, theta, sectors)
        wfs, tb, _ = payload.getPayloadWaveforms(
            phi, theta, sectors, impulse,
            (eplane, hplane),
            downsample=True, snr=5, plot=False
        )
        coh, _ = sum.coherentSum(wfs, tb, delays, False, ringmask)
        pwr, _ = sum.powerSum(coh, window=128, step=32)
        return float(np.max(pwr))
    return get_power

get_power = make_get_power(impulse, eplane, hplane, sectors, ringmask)

def descend_to_valley(phi0, theta0, lr=1.0, max_iter=20):
    φ, θ = phi0, theta0
    for _ in range(max_iter):
        f0 = get_power(φ, θ)
        # finite‐difference gradient
        dφ = (get_power(φ+1e-1, θ) - f0) / 1e-1
        dθ = (get_power(φ, θ+1e-1) - f0) / 1e-1
        norm = np.hypot(dφ, dθ)
        if norm < 1e-3:
            break
        # step *opposite* to gradient
        φ -= lr * dφ / norm
        θ -= lr * dθ / norm
    return φ, θ


def find_new_beams(existing_beams, R=4.0, n_samples=32,
                   step_size=1.0, max_iter=20,
                   eps=1.0, min_samples=3,
                   min_dist=2.0):
    """
    existing_beams: list of (phi,theta) already discovered
    returns: list of NEW dark‐spot centers (local minima) found this generation
    """
    # 1) generate ring of starts around each existing beam
    starts = []
    for phi0, theta0 in existing_beams:
        # only sample the “right” half‐circle
        angles = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=False)
        for alpha in angles:
            starts.append((phi0 + R*np.cos(alpha),
                        theta0 + R*np.sin(alpha)))

    starts = np.array(starts)

    # 2) descend each to its local minimum
    valleys = np.array([
        descend_to_valley(phi, theta, lr=step_size, max_iter=max_iter)
        for phi, theta in starts
    ])

    # 3) cluster the valleys
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(valleys)
    labels = db.labels_
    clusters = set(labels) - {-1}

    new_beams = []
    for lab in clusters:
        pts = valleys[labels == lab]
        center = pts.mean(axis=0)

        # 4) prune any that lie too close to existing beams
        dists = np.hypot(center[0] - np.array([b[0] for b in existing_beams]),
                         center[1] - np.array([b[1] for b in existing_beams]))
        if np.all(dists > min_dist):
            new_beams.append(tuple(center))
    # mirror across phi=0, but don’t duplicate phi=0 points
    mirrored = [(-phi, theta) for phi,theta in new_beams if phi != 0]
    return new_beams + mirrored



# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 0) start with your boresight
    beams = [(0.0, -45.0)]

    # parameters you’ll want to tune:
    max_generations = 5
    R0            = 4.0    # initial sampling radius [°]
    n_samples     = 32
    step_size     = 0.5
    max_iter      = 30
    eps           = 1.0    # clustering radius [°]
    min_samples   = 3
    min_dist      = 2.5    # prune beams < 2.5° apart

    for gen in range(max_generations):
        new = find_new_beams(beams,
                             R=R0,
                             n_samples=n_samples,
                             step_size=step_size,
                             max_iter=max_iter,
                             eps=eps,
                             min_samples=min_samples,
                             min_dist=min_dist)
        if not new:
            print(f"No new beams found at generation {gen}.")
            break

        print(f"Generation {gen}: found {len(new)} new beams.")
        beams += new

    phi_list = [f"{φ:.2f}" for φ, _ in beams]
    theta_list = [f"{θ:.2f}" for _, θ in beams]
    print(f"phi_values = [{', '.join(phi_list)}]")
    print(f"theta_values = [{', '.join(theta_list)}]")

    # You now have `beams` to feed into your heatmap‐summing routine.
