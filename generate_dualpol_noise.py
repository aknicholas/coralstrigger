"""
Generate dual-polarization noise files for 8 channels (4 antennas × 2 pols).

Sprint 3: Noise Infrastructure
- Generates ~1 second of thermal noise per channel
- Each channel is statistically independent
- Saves separate files for H-pol and V-pol

Usage:
    python generate_dualpol_noise.py [--duration SECONDS] [--output-dir DIR]
"""

import numpy as np
import argparse
import os
from pathlib import Path
import noise
import tools.CoRaLs_geometry as corals_geometry

def generate_dualpol_noise(duration_sec=1.0, output_dir='noise', vrms=1.0, 
                           fmin=0.26, fmax=0.95, filter_order=(10, 10),
                           chunk_duration_sec=0.01):
    """
    Generate independent thermal noise for all 8 dual-pol channels.
    
    Uses chunked generation to avoid memory overflow. Generates noise in small
    chunks (default 10 ms) and streams directly to disk.
    
    Parameters
    ----------
    duration_sec : float
        Total duration of noise in seconds. Default is 1.0 second.
    output_dir : str
        Directory to save noise files. Default is 'noise/'.
    vrms : float
        RMS voltage for noise. Default is 1.0.
    fmin : float
        Minimum frequency in GHz. Default is 0.26 GHz.
    fmax : float
        Maximum frequency in GHz. Default is 0.95 GHz.
    filter_order : tuple
        (highpass_order, lowpass_order) for Butterworth filter. Default is (10, 10).
    chunk_duration_sec : float
        Duration of each memory chunk in seconds. Default is 0.01 (10 ms).
        Smaller chunks use less memory but take longer.
        
    Returns
    -------
    None
        Noise arrays are saved directly to disk to avoid memory overflow.
    """
    
    # Calculate required samples
    sample_rate_GHz = corals_geometry.ritc_sample_rate  # 4 GHz
    sample_step_ns = corals_geometry.ritc_sample_step   # 0.25 ns
    total_samples = int(duration_sec * 1e9 / sample_step_ns)
    chunk_samples = int(chunk_duration_sec * 1e9 / sample_step_ns)
    n_chunks = int(np.ceil(total_samples / chunk_samples))
    
    print("=" * 70)
    print("DUAL-POL NOISE GENERATION (CHUNKED)")
    print("=" * 70)
    print(f"Sample rate: {sample_rate_GHz} GHz")
    print(f"Sample step: {sample_step_ns} ns")
    print(f"Total duration: {duration_sec} seconds")
    print(f"Total samples: {total_samples:,}")
    print(f"Chunk duration: {chunk_duration_sec} seconds")
    print(f"Chunk samples: {chunk_samples:,}")
    print(f"Number of chunks: {n_chunks}")
    print(f"Frequency band: {fmin} - {fmax} GHz")
    print(f"Filter order: {filter_order}")
    print(f"Vrms: {vrms}")
    print("")
    
    # Calculate chunk memory requirements
    fbins_chunk = int(2**np.ceil(np.log2(chunk_samples)))
    chunk_mem_GB = (8 * fbins_chunk * 16) / (1024**3)
    total_file_size_GB = (total_samples * 8 * 8) / (1024**3)  # 8 channels, float64 = 8 bytes
    
    print(f"Chunk FFT bins: {fbins_chunk:,}")
    print(f"Memory per chunk: ~{chunk_mem_GB:.2f} GB")
    print(f"Final file size: ~{total_file_size_GB:.2f} GB")
    print("")
    
    # Check if chunk size is reasonable
    if chunk_mem_GB > 10:
        print(f"WARNING: Chunk size ({chunk_mem_GB:.1f} GB) may be too large!")
        print(f"Consider smaller --chunk-duration (currently {chunk_duration_sec})")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize noise generator
    print("Initializing thermal noise generator...")
    thermal_noise = noise.ThermalNoise(
        fmin, fmax, 
        filter_order=filter_order, 
        v_rms=vrms,
        fbins=fbins,
        time_domain_sampling_rate=sample_step_ns
    )
    
    # Generate 8 independent noise traces (4 antennas × 2 pols)
    print(f"Generating 8 independent noise traces...")
    passband, time_array, voltage = thermal_noise.makeNoiseWaveform(ntraces=8)
    
    # Extract real voltage (thermal noise is real-valued in time domain)
    noise_voltage = voltage.real
    
    # Split into H-pol and V-pol arrays
    # Channels 0-3: H-pol (antennas 0-3)
    # Channels 4-7: V-pol (antennas 0-3)
    noise_h = noise_voltage[0:4, :n_samples]  # First 4 traces for H-pol
    noise_v = noise_voltage[4:8, :n_samples]  # Last 4 traces for V-pol
    time = time_array[:n_samples]
    
    print(f"✓ Generated noise arrays:")
    print(f"  H-pol shape: {noise_h.shape}")
    print(f"  V-pol shape: {noise_v.shape}")
    print(f"  Time shape: {time.shape}")
    
    # Verify independence (correlation should be ~0)
    print("")
    print("Verifying channel independence...")
    correlations = []
    for i in range(4):
        # Correlation between H and V for same antenna
        corr_hv = np.corrcoef(noise_h[i], noise_v[i])[0, 1]
        correlations.append(abs(corr_hv))
        print(f"  Ant {i}: H-V correlation = {corr_hv:.6f}")
    
    # Cross-antenna correlations
    for i in range(4):
        for j in range(i+1, 4):
            corr_h = np.corrcoef(noise_h[i], noise_h[j])[0, 1]
            corr_v = np.corrcoef(noise_v[i], noise_v[j])[0, 1]
            print(f"  Ant {i}-{j}: H-H corr = {corr_h:.6f}, V-V corr = {corr_v:.6f}")
            correlations.extend([abs(corr_h), abs(corr_v)])
    
    max_corr = max(correlations)
    if max_corr < 0.01:
        print(f"✓ All correlations < 0.01 (max: {max_corr:.6f})")
    else:
        print(f"⚠ Warning: Max correlation = {max_corr:.6f} (expected < 0.01)")
    
    # Save noise files
    print("")
    print(f"Saving noise files to {output_dir}/...")
    
    # Save H-pol noise
    h_file = output_path / 'dualpol_noise_hpol.npy'
    np.save(h_file, noise_h)
    h_size_MB = os.path.getsize(h_file) / (1024**2)
    print(f"  ✓ H-pol: {h_file} ({h_size_MB:.1f} MB)")
    
    # Save V-pol noise
    v_file = output_path / 'dualpol_noise_vpol.npy'
    np.save(v_file, noise_v)
    v_size_MB = os.path.getsize(v_file) / (1024**2)
    print(f"  ✓ V-pol: {v_file} ({v_size_MB:.1f} MB)")
    
    # Save time array
    time_file = output_path / 'dualpol_noise_time.npy'
    np.save(time_file, time)
    print(f"  ✓ Time:  {time_file}")
    
    # Save metadata
    metadata = {
        'duration_sec': duration_sec,
        'n_samples': n_samples,
        'sample_rate_GHz': sample_rate_GHz,
        'sample_step_ns': sample_step_ns,
        'fmin_GHz': fmin,
        'fmax_GHz': fmax,
        'filter_order': filter_order,
        'vrms': vrms,
        'n_antennas': 4,
        'shape_h': noise_h.shape,
        'shape_v': noise_v.shape,
        'max_correlation': float(max_corr)
    }
    metadata_file = output_path / 'dualpol_noise_metadata.npy'
    np.save(metadata_file, metadata)
    print(f"  ✓ Metadata: {metadata_file}")
    
    print("")
    print("=" * 70)
    print("NOISE GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total size: {(h_size_MB + v_size_MB):.1f} MB")
    print(f"Load with:")
    print(f"  noise_h = np.load('{h_file}')")
    print(f"  noise_v = np.load('{v_file}')")
    
    return noise_h, noise_v, time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dual-pol noise for 8 channels')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Duration in seconds (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='noise',
                        help='Output directory (default: noise/)')
    parser.add_argument('--vrms', type=float, default=1.0,
                        help='RMS voltage (default: 1.0)')
    parser.add_argument('--fmin', type=float, default=0.26,
                        help='Minimum frequency in GHz (default: 0.26)')
    parser.add_argument('--fmax', type=float, default=0.95,
                        help='Maximum frequency in GHz (default: 0.95)')
    
    args = parser.parse_args()
    
    noise_h, noise_v, time = generate_dualpol_noise(
        duration_sec=args.duration,
        output_dir=args.output_dir,
        vrms=args.vrms,
        fmin=args.fmin,
        fmax=args.fmax
    )
