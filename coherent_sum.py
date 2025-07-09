import numpy
import myplot
import matplotlib.pyplot as plt
import tools.CoRaLs_geometry as aso_geometry
import tools.constants as constants
import tools.filters as filters
import payload_signal as payload
import math
import os

def coherentSum(waveforms, timebase, delays, downsample=False, ringmask=[1,1]):
    '''
    mask = ring mask [B, T]
    '''
    
    #decimates to specified sample rate
    decimate_factor = int(aso_geometry.ritc_sample_step/((timebase[1]-timebase[0])))

    #downsample to ritc sampling, if input is upsampled:
    if downsample == True:
        coh_sum = numpy.zeros(int(len(waveforms[0,0]) / decimate_factor)) # choice to cast as int, so 333.333 becomes 333
        
        for i in range(waveforms.shape[0]):
            for j in range(waveforms.shape[1]):
                if ringmask[j]:
                    _wave = waveforms[i,j][::decimate_factor]
                    _delay = -int(numpy.round(delays[i,j]/(timebase[1]-timebase[0]*decimate_factor)))
                    coh_sum = coh_sum + numpy.roll(_wave[:len(coh_sum)], _delay)

        timebase = timebase[::decimate_factor]
        timebase = timebase[:len(coh_sum)]

    #use input sampling:
    else:
        coh_sum = numpy.zeros(len(waveforms[0,0]))
        for i in range(waveforms.shape[0]):
            for j in range(waveforms.shape[1]):
                try:
                    if len(ringmask) == 0 or j >= len(ringmask) or not ringmask[j]:
                        continue
                except Exception:
                    continue
                _wave = waveforms[i,j]
                _delay = -int(numpy.round(delays[i,j]/(timebase[1]-timebase[0])))
                coh_sum = coh_sum + numpy.roll(_wave, _delay)

        #coh_sum = coh_sum[::decimate_factor]
        #timebase = timebase[::decimate_factor]

    return coh_sum, timebase
def GimmeInfo(waveforms):
    print("len of payload single waveforms[0,0] is: {}".format(int(len(waveforms[0,0]))))
    print("waveforms.shape[0] is: {}".format(waveforms.shape[0]))
    print("waveforms.shape[1] is: {}".format(waveforms.shape[1]))

def powerSum(coh_sum, window=32, step=16):
    '''
    calculate power summed over a length defined by 'window', overlapping at intervals defined by 'step'
    '''
    num_frames = int(math.floor((len(coh_sum)-window) / step))
   # print(num_frames)
    #print(window)
    coh_sum_squared = (coh_sum * coh_sum).astype(numpy.int64)
   # print(coh_sum_squared)
    #print(coh_sum_squared.strides[0]*step)
   #print(coh_sum_squared.strides[0])
    coh_sum_windowed = numpy.lib.stride_tricks.as_strided(coh_sum_squared, (num_frames, window),
                                                          (int(coh_sum_squared.strides[0]*step), coh_sum_squared.strides[0]))
    power = numpy.sum(coh_sum_windowed, axis=1)

    return power.astype(numpy.float64)/window, num_frames

                
if __name__=='__main__':
    phi_scan_width = 10
    el_scan_width = 5
    #phi_values = [phi for phi in range(-phi_scan_width, phi_scan_width + 1, 2)]
    #theta_values = [theta for theta in range(-el_scan_width, el_scan_width + 1, 2)]
    phi_values = [0]
    theta_values = [-45]
    angular_res = 1
    lowpass = filters.Shannon_Whitaker(fs=3e9, plot=False)
    eplane = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
    trigger_sectors_phi=[1,2,3,4,5,6,7,8]
    # Define global scan bounds
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff= 2)

    global_phiscan_start = min(phi_values) - phi_scan_width
    global_phiscan_stop = max(phi_values) + phi_scan_width
    global_elscan_start = min(theta_values) - el_scan_width
    global_elscan_stop = max(theta_values) + el_scan_width

    global_phiscan_range = numpy.arange(global_phiscan_start, global_phiscan_stop, angular_res)
    global_elscan_range = numpy.arange(global_elscan_start, global_elscan_stop, angular_res)
    global_phi_grid, global_el_grid = numpy.meshgrid(global_phiscan_range, global_elscan_range)
    summed_heatmap = numpy.zeros_like(global_phi_grid, dtype=float)
    summed_count = numpy.zeros_like(global_phi_grid, dtype=float)

    for phi in phi_values:
        ringmask = [1,1,0,0,0,0,0,1]
        for theta in theta_values:
            #payload.gimmePlotsImpulse(impulse)
            payload.getPayloadWaveforms(phi, theta, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, plot=True)
            print ('timestep of input pulse [ns]:', impulse.dt)
            delays = payload.getRemappedDelays(phi, theta, trigger_sectors_phi)
            
            #getSinglePayloadWaveform(22.5, -25, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, noise=None, plot=True)
            #GimmeInfo(waveforms)
            
            #scan phi, 45 degrees:
            phiscan_start = phi - phi_scan_width
            phiscan_stop = phi + phi_scan_width
            elscan_start = theta - el_scan_width
            elscan_stop = theta + el_scan_width

            # Create 2D histogram heatmap for phi/theta scan
            phiscan_range = numpy.arange(phiscan_start, phiscan_stop, angular_res)
            elscan_range = numpy.arange(elscan_start, elscan_stop, angular_res)
            phi_grid, el_grid = numpy.meshgrid(phiscan_range, elscan_range)
        
            # Compute power for each (phi, theta) pair
            heatmap = numpy.zeros_like(phi_grid, dtype=float)
            
            # Generate waveforms for fixed source direction
            # Fix delays for beam direction
            #delays = payload.getRemappedDelays(phi, theta, trigger_sectors_phi)

            for i, elscan in enumerate(elscan_range):
                for j, phiscan in enumerate(phiscan_range):
                    # Generate waveforms for each source direction
                    waveforms, timebase, _ = payload.getPayloadWaveforms(phiscan, elscan, trigger_sectors_phi, impulse, (eplane, hplane), downsample=False, snr=20, plot=False)
                    coh_sum, _ = coherentSum(waveforms, timebase, delays, False, ringmask)
                    power, _ = powerSum(coh_sum)
                    heatmap[i, j] = numpy.max(power)  # or numpy.sum(power) depending on what you want
            # Normalize to dB
            heatmap_db = 10 * numpy.log10(heatmap / numpy.max(heatmap))
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.pcolormesh(phi_grid, el_grid, heatmap_db, shading='auto', cmap='viridis')
            ax.set_xlabel('Phi [degrees]')
            ax.set_ylabel('Theta [degrees]')
            ax.set_title(f"Coherent Sum Power Heatmap, Beam at $\\phi$ = {phi}, $\\theta$ = {theta}")
            plt.colorbar(c, ax=ax, label='Power [dB]')
            fig.tight_layout()
            filename = f"plots/plot_phi_{phi}_theta_{theta}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            # Find the indices in the global grid where this local heatmap should be placed
            phi_offset = phiscan_start - global_phiscan_start
            el_offset = elscan_start - global_elscan_start
            heatmap_rows, heatmap_cols = heatmap.shape

            # Add current beam's heatmap to the sum at the correct location
            row_start = max(0, int(round((elscan_start - global_elscan_start) / angular_res)))
            col_start = max(0, int(round((phiscan_start - global_phiscan_start) / angular_res)))
            row_end = min(row_start + heatmap.shape[0], summed_heatmap.shape[0])
            col_end = min(col_start + heatmap.shape[1], summed_heatmap.shape[1])
            local_row_end = row_end - row_start
            local_col_end = col_end - col_start
            # Before adding to summed_heatmap, check shapes
            target_shape = summed_heatmap[row_start:row_end, col_start:col_end].shape
            source_shape = heatmap[:local_row_end, :local_col_end].shape
            if target_shape != source_shape:
                print(f"Shape mismatch: target {target_shape}, source {source_shape}")
                # Optionally, adjust local_row_end/local_col_end to min of both shapes
                min_rows = min(target_shape[0], source_shape[0])
                min_cols = min(target_shape[1], source_shape[1])
                summed_heatmap[row_start:row_start+min_rows, col_start:col_start+min_cols] += heatmap[:min_rows, :min_cols]
                summed_count[row_start:row_start+min_rows, col_start:col_start+min_cols] += (heatmap[:min_rows, :min_cols] > 0).astype(float)
            else:
                summed_heatmap[row_start:row_end, col_start:col_end] += heatmap[:local_row_end, :local_col_end]
                summed_count[row_start:row_end, col_start:col_end] += (heatmap[:local_row_end, :local_col_end] > 0).astype(float)
            # Check if the summed_heatmap or summed_count arrays are empty and raise an error if so
            if summed_heatmap.size == 0 or summed_count.size == 0:
                raise ValueError("summed_heatmap or summed_count array is empty. Check input data and loop logic.")
            # After all phi/theta, save the big plot
            with numpy.errstate(divide='ignore', invalid='ignore'):
                avg_heatmap = numpy.where(summed_count > 0, summed_heatmap / summed_count, 0)
                avg_heatmap_db = 10 * numpy.log10(avg_heatmap / numpy.max(avg_heatmap))
                avg_heatmap = numpy.where(summed_count>0, summed_heatmap / summed_count, 0)
                amax = avg_heatmap.max()
                if amax <= 0:
                    print("Warning: all-zero averaged heatmap across all beams")
                    avg_heatmap_db = numpy.full_like(avg_heatmap, -numpy.inf)
                else:
                    avg_heatmap_db = 10 * numpy.log10(avg_heatmap / amax)
            fig, ax = plt.subplots(figsize=(10, 8))
            c = ax.pcolormesh(global_phi_grid, global_el_grid, avg_heatmap_db, shading='auto', cmap='viridis')
            ax.set_xlabel('Phi [degrees]')
            ax.set_ylabel('Theta [degrees]')
            ax.set_title('Summed Coherent Sum Power Heatmap (All Beams)')
            plt.colorbar(c, ax=ax, label='Power [dB]')
            fig.tight_layout()
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/summed_beam_heatmap.png", bbox_inches='tight')
            plt.close()
