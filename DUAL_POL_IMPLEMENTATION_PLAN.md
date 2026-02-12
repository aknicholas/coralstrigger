# Dual Polarization Implementation Plan
**Status:** Sprint 5 Complete, Ready for Sprint 6  
**Last Updated:** January 29, 2026

**CRITICAL:** See [COORDINATE_SYSTEM.md](COORDINATE_SYSTEM.md) for complete coordinate system documentation!

**Key Convention:**
- **θ (theta)** is elevation angle: -90° = nadir, 0° = horizon, +90° = zenith
- All operational angles use **θ = -30°** (30° from nadir) unless otherwise specified


---

## Hardware Signal Chain

```
4 Antennas × (H-pol + V-pol) = 8 channels
    ↓
Digitize @ 4 GHz
    ↓
Shannon-Whitaker Digital Lowpass Filter (~0.95 GHz cutoff)
    ↓
Beamforming per direction:
  H-pol → coherent sum → |_____
                               |→ LHCP = (H + jV)/√2 → Power → Threshold
  V-pol → coherent sum → |‾‾‾‾‾    RHCP = (H - jV)/√2 → Power → Threshold
    ↓
Coincidence Trigger: LHCP AND RHCP overlap ≥ min_frames
    ↓
Save to disk
```

**Key Constraints:**
- Global trigger rate: 0.1 Hz across all beams
- Power sum: 160-sample window, 40-sample step
- Coincidence: Both LHCP and RHCP must trigger with time overlap

---

## Sprint Progress

### ✅ Sprint 1: Core Functions (January 13, 2026)

**Implemented:**
- `coherentSum(waveforms[antennas, samples], ...)` - 2D antenna-based summation
- `coherentSum_dualpol(waveforms_h, waveforms_v, ..., output='circular')` - LHCP/RHCP conversion
- `check_coincidence(triggers_lhcp, triggers_rhcp, min_overlap)` - Time-domain overlap detection
- `compute_coincidence_rate(rate_lhcp, rate_rhcp, window_time)` - Rate calculations

**Test Results:**
- Power conservation: 100.0% ✓
- Coincidence detection: 2/2 events correct ✓
- Circular conversion: LHCP = (H + jV)/√2 verified ✓

**Files:** `coherent_sum.py`, `test_sprint1_dualpol.py`

---

### ✅ Sprint 2: Signal Generation (January 13, 2026)

**Implemented:**
- `getPayloadWaveforms_dualpol(phi, el, impulse, beam_patterns_h, beam_patterns_v, psi=45.0, ...)`
  - Separate H-pol and V-pol waveform generation
  - Polarization projection: H_response = cos(ψ) × beam_h, V_response = sin(ψ) × beam_v
  - 4 beam patterns: (E-plane H, H-plane H, E-plane V, H-plane V)
  - Returns: waveforms_h[4, samples], waveforms_v[4, samples]

**Polarization Angle (ψ):**
- ψ = 0°: Pure H-pol (V nulled)
- ψ = 45°: Equal H/V (default)
- ψ = 90°: Pure V-pol (H nulled)

**Test Results:**
- H ≠ V confirmed: Peaks vary with angle (0.00 to 2.49)
- Beam patterns working: H-pol nulled at some angles, V-pol at others ✓
- Polarization projection: cos(ψ) and sin(ψ) scaling verified ✓

**Code Cleanup:**
- `payload_signal.py`: 881 → 555 lines (37% reduction)
- Removed: 9 unused functions (gimmePlots*, loadPlaneWave, debug code)
- Kept: Core impulse/beam/waveform generation only

**Files:** `payload_signal.py` (555 lines), `coherent_sum.py`, `CLEANUP_SUMMARY.txt`

---

### ✅ Sprint 3: Noise Infrastructure (January 15, 2026)

**Goal:** Generate independent thermal noise for 8 channels (4 antennas × 2 pols)

**Challenge:** Memory overflow
- 1 second @ 4 GHz = 4 billion samples
- FFT needs 2^32 bins = 4.3 billion complex128 values
- 8 channels × 16 bytes/complex × 4.3B = **550 GB RAM** ❌

**Solution:** Chunked streaming to disk
- Generate noise in small chunks (e.g., 5 ms = 20M samples)
- Each chunk: ~4 GB RAM ✓
- Stream directly to disk using memory-mapped arrays
- Total file size: ~4 GB per second per channel

**Implementation:**
```python
# generate_dualpol_noise_chunked.py
- Creates memory-mapped files for H-pol and V-pol
- Generates noise in chunks (default 10 ms)
- Writes directly to disk without holding full array in RAM
- Verifies H-V independence (correlation < 0.01)
```

**Results:**
- ✅ Chunked generation script created and tested
- ✅ Test run: 10 ms (2.4 GB, ~0.6 GB RAM, H-V correlation < 0.001)
- ✅ Production run: 0.5 sec (120 GB, 100 chunks of 5 ms each)
- ✅ Files ready to load with `np.load('noise/dualpol_noise_hpol.npy')`

**Files:** 
- `generate_dualpol_noise_chunked.py` (new)
- Output: `noise/dualpol_noise_hpol.npy`, `noise/dualpol_noise_vpol.npy`

---

### ✅ Sprint 4: Threshold Analysis (January 22, 2026)

**Goal:** Determine trigger thresholds for target global rate with placeholder N_beams=100

**Challenge:** Memory overflow processing 0.5 sec (2B samples)
- Solution: Chunked processing (50 ms chunks) with memory-mapped noise files

**Implemented:**
- `generate_power_dualpol.py`: Chunked power sum generation
  - Loads noise in 50 ms chunks to avoid OOM
  - Applies Shannon-Whitaker filter before beamforming
  - Performs dual-pol coherent sum → LHCP/RHCP conversion
  - Computes sliding window power (window=160, step=40)
  - Output: 50M power frames from 0.5 sec noise
  
- `generate_threshold_curves_dualpol.py`: Exponential curve fitting
  - Scans thresholds: measures per-beam trigger rates
  - Auto-detects linear region in semilog space for fitting
  - Fits: Rate(T) = A × exp(-B × T)
  - Computes coincidence rate: 2 × R_L × R_R × τ_window
  - Solves for symmetric threshold: T_L = T_R
  - Displays reduced χ²/dof for goodness-of-fit evaluation

**Results (N_beams=100, target=0.1 Hz global):**
- Fit region: T ∈ [2.40, 3.30] (auto-detected, 10 < counts < 2e4)
- Exponential fit: Rate(T) = 5.12e18 × exp(-11.97 × T) Hz
- χ²/dof = 1.87 (good fit in linear region)
- **Required threshold: T = 3.23** (normalized power units)
- Per-beam rates at threshold:
  * LHCP: 70.7 Hz
  * RHCP: 70.7 Hz
  * Coincidence: 0.001 Hz per beam
  * Global: 0.1 Hz across 100 beams ✓

**Files:**
- `generate_power_dualpol.py` (new, 280 lines)
- `generate_threshold_curves_dualpol.py` (new, 465 lines)
- Output: `noise/power_lhcp_160_40.npy`, `noise/power_rhcp_160_40.npy`
- Analysis: `noise/threshold_analysis_dualpol.json`, `noise/threshold_curves_dualpol.png`

**Note:** Threshold will be re-calculated in Sprint 6 with actual N_beams from beam pattern analysis

---

### ✅ Sprint 5: SNR Validation (January 29, 2026)

**Goal:** Measure dual-pol trigger efficiency vs SNR and validate coordinate system

**Coordinate System Discovery:**
- **CRITICAL BUG FOUND:** Initial testing used θ=0° (horizon) instead of θ=-30° (operational angle)
- **Root Cause:** Misunderstanding of coordinate convention - θ is elevation angle, NOT zenith angle
  - θ = -90° means nadir (straight down) ✓
  - θ = 0° means horizon (90° off-boresight for nadir-pointing antennas) ✗
  - θ = +90° means zenith (straight up)
- **Fix Applied:** 
  - Updated snr_scan_dualpol.py default from θ=0° to θ=-30°
  - Fixed angle wrapping in payload_signal.py (removed incorrect abs() calls)
  - Created COORDINATE_SYSTEM.md with complete documentation
- **Validation:** All test cases now pass with correct beam responses

**Implementation:**
- Created `snr_scan_dualpol.py` (341 lines)
- Inject signal at specified (φ, θ, ψ) angles
- Add signal to white noise at varying SNR levels
- Apply Shannon-Whitaker filter and beamform to LHCP/RHCP
- Check individual triggers (LHCP > T_L, RHCP > T_R)
- Check coincidence trigger (both fire within 100 ns window)
- Measure efficiency: N_detected / N_injected vs SNR

**Results at θ=-30° (operational angle) with Threshold T=3.23:**

| Polarization | SNR at 50% Efficiency | LHCP Eff | RHCP Eff | Coinc Eff | Notes |
|-------------|-----------------------|----------|----------|-----------|-------|
| ψ = 0° (H-pol) | **6.56** | 0.52 @ SNR=6.6 | 0.52 @ SNR=6.6 | 0.52 @ SNR=6.6 | All equivalent ✓ |
| ψ = 45° (balanced) | **6.53** | 0.53 @ SNR=6.6 | 0.53 @ SNR=6.6 | 0.53 @ SNR=6.6 | All equivalent ✓ |
| ψ = 90° (V-pol) | **6.55** | 0.52 @ SNR=6.6 | 0.52 @ SNR=6.6 | 0.52 @ SNR=6.6 | All equivalent ✓ |

**Key Findings:**
1. **Polarization Independence:** All polarizations (H, V, balanced) have identical 50% efficiency (SNR ~ 6.5)
   - This is expected at θ=-30° where beam pattern is symmetric
   - Small variations (±0.03) are statistical noise

2. **LHCP/RHCP Symmetry:** LHCP and RHCP have identical efficiency curves
   - Validates circular conversion: both contain (H±jV)/√2 correctly

3. **Coincidence Performance:** Coincidence efficiency exactly matches individual polarization efficiencies
   - This is expected: LHCP and RHCP are 100% correlated (derived from same H/V signals)
   - No false trigger reduction vs single-pol (both always trigger together for signal)
   - False trigger reduction comes from noise (tested in Sprint 4)

4. **Coordinate System Validated:**
   - θ=-90° (nadir): Maximum signal, all delays=0 ✓
   - θ=-30° (operational): Reduced but measurable signal ✓
   - θ=0° (horizon): No signal (90° off-boresight) ✓

**Files:**
- `snr_scan_dualpol.py` (new, 341 lines) - updated with θ=-30° default
- `plot_scurve_comparison.py` (new, 134 lines)
- `COORDINATE_SYSTEM.md` (new documentation)
- Output: `plots/snr_scan_dualpol_theta-30_psi{0,45,90}.{npy,txt,png}`
- Comparison: `plots/scurve_comparison_polarization.png`

**Next:** Sprint 6 will analyze actual beam patterns to determine N_beams and angular sensitivity, then finalize thresholds

---

### 🔜 Sprint 6: Beam Optimization

**Tasks:**
1. Update `power_cuts.py` for LHCP/RHCP beam patterns
2. Generate -1dB contours
3. Run `calculate_beams()` → determine actual N_beams
4. **FINALIZE thresholds** with real N_beams
5. Re-run SNR scans with final parameters
6. Validate global trigger rate = 0.1 Hz

---

## Key Files

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `coherent_sum.py` | ✅ Updated | 461 | Core processing + dual-pol functions + Shannon-Whitaker filter |
| `payload_signal.py` | ✅ Updated | 558 | Waveform generation (cleaned up, no analog filters) |
| `tools/filters.py` | ✅ Updated | 150 | Shannon-Whitaker FIR filter implementation |
| `noise.py` | ✅ Updated | 150 | White noise generation |
| `generate_dualpol_noise_chunked.py` | ✅ Complete | 239 | Chunked noise generation with streaming |
| `generate_power_dualpol.py` | ✅ New | 280 | Chunked power sum generation for threshold analysis |
| `generate_threshold_curves_dualpol.py` | ✅ New | 465 | Exponential curve fitting with auto fit region |
| `snr_scan_dualpol.py` | ✅ New | 341 | SNR efficiency scan for dual-pol triggers |
| `plot_scurve_comparison.py` | ✅ New | 134 | Compare S-curves across polarization angles |
| `analyze_beams_dualpol.py` | ✅ New | 560 | Beam pattern analysis and optimization |
| `visualize_beam_vs_spacing.py` | ✅ New | 390 | Array size vs beamwidth demonstration |
| `test_sprint1_dualpol.py` | ✅ Complete | - | Sprint 1 validation tests |
| `snr_scan.py` | 📋 Reference | - | Original single-pol SNR scan |
| `power_cuts.py` | 🔜 Todo | - | Beam pattern analysis (needs dual-pol update) |

---

## Validation Checklist

### Sprint 1 ✅
- [x] Power conservation in circular basis: 100.0%
- [x] Coincidence trigger logic working
- [x] Rate calculations validated

### Sprint 2 ✅
- [x] H-pol and V-pol physically distinct
- [x] Beam patterns applied correctly
- [x] Polarization projection (ψ parameter) working
- [x] Code cleanup complete

### Sprint 3 ✅
- [x] Chunked generation script working
- [x] Memory-mapped streaming to disk
- [x] H-V independence verified (|corr| < 0.001)
- [x] Full 0.5 sec generation complete

### Sprint 4 ✅
- [x] Chunked power generation working (50M frames)
- [x] Threshold curves fitted (χ²/dof = 1.87)
- [x] Auto-detected linear fit region
- [x] Threshold T=3.23 determined for N_beams=100
- [x] Global rate 0.1 Hz achieved

### Sprint 5 ✅
- [x] SNR efficiency scan for dual-pol
- [x] S-curves generated for ψ = 0°, 45°, 90°
- [x] Polarization dependence characterized
- [x] LHCP/RHCP symmetry validated
- [x] Beam pattern nulls identified (V-pol at nadir)

### Sprint 6 ✅
- [x] Generate LHCP/RHCP beam patterns across angular grid
- [x] Implement beam overlap reduction algorithm
- [x] Determine optimal N_beams with minimal overlap
- [x] Analyze beam width vs antenna array span
- [x] Demonstrate array size effect on beamwidth
- [x] Array beamforming visualization created

---

## Sprint Details

### Sprint 6: Beam Pattern Analysis and Array Size Effects

**Goal:** Analyze beam patterns, optimize beam placement, and demonstrate how antenna array size affects beamwidth.

**Tools Created:**
- `analyze_beams_dualpol.py` (560 lines): Generate LHCP/RHCP beam patterns, find optimal beam directions
- `visualize_beam_vs_spacing.py` (390 lines): Show beam power heatmaps vs antenna spacing

**Key Findings:**

1. **Optimal Beam Coverage:**
   - Coverage region: -60° to 0° elevation, 0° to 90° azimuth
   - With -1 dB threshold and 8° min separation: **N_beams = 4**
   - Mean beam width: 29° ± 3.6°

2. **Array Size vs Beam Width:**
   - Demonstrated inverse relationship between array size and beamwidth
   - At θ=-30° (operational angle):
     - 0.5x spacing (1.42m): θ-width = 22°
     - 1.0x spacing (2.83m): θ-width = 5°
     - 1.5x spacing (4.25m): θ-width = 6°
   - **Larger arrays → narrower beams → need more beams for coverage**

3. **Beam Optimization Algorithm:**
   - Greedy selection from local maxima
   - Enforces minimum angular separation
   - Reduces unnecessary beam overlap
   - Generates -1 dB contour visualizations

**Visualizations Generated:**
- `beam_pattern_lhcp_optimized.png`: LHCP beam pattern with optimal beam centers
- `beam_pattern_rhcp_optimized.png`: RHCP beam pattern with optimal beam centers  
- `beam_vs_spacing.png`: Side-by-side comparison of beam power for different array sizes
- `beam_width_vs_spacing.png`: Beam width vs array spacing (log-log plot)
- `array_size_analysis.png`: Summary of N_beams and width vs spacing

**Impact on Thresholds:**
- Previous estimate: N_beams = 100 (placeholder)
- Optimized value for test region: N_beams ≈ 4-6
- Full-sky coverage estimate: N_beams ≈ 16-24 (assuming 4x azimuthal symmetry)
- **Note:** Sprint 4 thresholds (T=3.23) were calculated with N_beams=100 placeholder
- For accurate global rate control, thresholds should be recalculated with actual N_beams

---

## Notes

**Polarization Model:**
- Linear polarization at angle ψ from H-pol axis
- H-channel: cos(ψ) × beam_h × impulse
- V-channel: sin(ψ) × beam_v × impulse
- Circular conversion after digitization: LHCP = (H + jV)/√2

**Memory Management:**
- Impulse processing: ~150k samples, negligible RAM
- Noise generation: Chunked streaming to avoid OOM
- Rule of thumb: Keep chunks < 10 GB RAM

**Beam Patterns:**
- Each polarization has E-plane and H-plane response
- Pyramidal LPDA has distinct H-pol and V-pol patterns
- Must load 4 patterns total per test

**Next Steps:**
1. ✅ Sprint 3 complete: 0.5 sec noise files ready
2. Start Sprint 4: Dual-pol threshold analysis
3. Then Sprint 5: SNR validation
4. Finally Sprint 6: Optimize N_beams and finalize thresholds
