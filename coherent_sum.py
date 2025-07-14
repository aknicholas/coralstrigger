import numpy
import myplot
import matplotlib.pyplot as plt
import tools.CoRaLs_geometry as aso_geometry
import tools.constants as constants
import tools.filters as filters
import payload_signal as payload
import math
import noise
import os

def coherentSum(waveforms, timebase, delays, downsample=False, ringmask=[1,1]):
    '''
    mask = ring mask [B, T]
    '''
    
    #decimates to specified sample rate
    decimate_factor = int(aso_geometry.ritc_sample_step/((timebase[1]-timebase[0])))
    #print('decimate factor = ', decimate_factor)
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
    #print(len(timebase), len(waveforms[0][0]))
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

    phi_start = 0
    el_start = -45
    phi_scan_width = 10
    el_scan_width = 10
    phi_values = [phi_start]
    theta_values = [el_start]
    
    # Parameters for hex grid
    hex_spacing = 3

    el_spacing = hex_spacing**2 - hex_spacing/2
    # Generate phi_values and theta_values for a triangular grid (hexagonal tiling)
    num_rings = 1  # Number of rings around the center (adjust as needed)
    phi_values = []
    theta_values = []

    for q in range(-num_rings, num_rings + 1):
        for r in range(-num_rings, num_rings + 1):
            s = -q - r
            if abs(s) > num_rings:
                continue
            # Convert axial hex coordinates to 2D grid coordinates
            phi = phi_start + hex_spacing * (q + r/2)
            theta = el_start + hex_spacing * (math.sqrt(3)/2) * r
            phi_values.append(phi)
            theta_values.append(theta)
    print("Hex grid phi_values:", phi_values)
    print("Hex grid theta_values:", theta_values)
    #phi_values = [0.00, 1.55, -1.55, 2.12, 3.18, 4.91, 6.00, 2.09, -2.12, -3.18, -4.91, -6.00, -2.09, -1.33, 8.27, -2.25, -4.72, 9.81, 11.91, 9.28, 0.01, -6.08, 2.99, 1.33, -8.27, 2.25, 4.72, -9.81, -11.91, -9.28, -0.01, 6.08, -2.99, 4.09, 7.31, 14.53, 12.38, -5.53, -7.66, -3.46, -6.73, 1.32, 4.24, 2.60, 13.75, 16.01, 19.60, -8.63, 6.51, 1.96, -9.45, -4.09, -7.31, -14.53, -12.38, 5.53, 7.66, 3.46, 6.73, -1.32, -4.24, -2.60, -13.75, -16.01, -19.60, 8.63, -6.51, -1.96, 9.45, 15.68, 19.48, 16.84, 15.03, -10.20, 9.20, 20.85, 15.07, 15.46, 13.54, 5.86, 10.42, 8.71, 17.04, 20.95, 19.01, 22.88, 25.04, 22.15, 25.18, 11.17, -2.61, -14.82, -16.01, -12.81, -0.49, -11.94, -15.35, -15.68, -19.48, -16.84, -15.03, 10.20, -9.20, -20.85, -15.07, -15.46, -13.54, -5.86, -10.42, -8.71, -17.04, -20.95, -19.01, -22.88, -25.04, -22.15, -25.18, -11.17, 2.61, 14.82, 16.01, 12.81, 0.49, 11.94, 15.35]
    #theta_values = [-45.00, -42.69, -42.69, -47.82, -45.79, -38.30, -44.42, -38.60, -47.82, -45.79, -38.30, -44.42, -38.60, -51.96, -38.15, -35.94, -35.76, -44.06, -46.09, -39.59, -34.12, -41.78, -33.39, -51.96, -38.15, -35.94, -35.76, -44.06, -46.09, -39.59, -34.12, -41.78, -33.39, -55.11, -51.34, -36.17, -31.60, -47.42, -31.54, -30.51, -49.87, -56.94, -57.01, -27.83, -50.41, -43.59, -43.76, -46.79, -26.80, -25.07, -49.62, -55.11, -51.34, -36.17, -31.60, -47.42, -31.54, -30.51, -49.87, -56.94, -57.01, -27.83, -50.41, -43.59, -43.76, -46.79, -26.80, -25.07, -49.62, -31.03, -38.94, -34.31, -32.40, -53.05, -57.94, -36.41, -25.77, -28.20, -24.47, -64.32, -22.10, -24.93, -53.95, -48.05, -47.39, -45.30, -42.52, -41.01, -44.99, -26.07, -61.17, -40.60, -32.35, -24.74, -63.51, -53.98, -47.48, -31.03, -38.94, -34.31, -32.40, -53.05, -57.94, -36.41, -25.77, -28.20, -24.47, -64.32, -22.10, -24.93, -53.95, -48.05, -47.39, -45.30, -42.52, -41.01, -44.99, -26.07, -61.17, -40.60, -32.35, -24.74, -63.51, -53.98, -47.48]
    #phi_values = [phi for phi in range(phi_start-phi_scan_width,phi_start + phi_scan_width + 1, 2)]
    #theta_values = [theta for theta in range(el_start-el_scan_width,el_start +  el_scan_width + 1, 2)]
    angular_res = 1

    lowpass = filters.Shannon_Whitaker(fs=3e9, plot=False)
    eplane = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
    trigger_sectors_phi=[1,2,3,4,5,6,7,8]
    # Define global scan bounds
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff= 2)
    #Introduce noise
    thermal_noise = noise.ThermalNoise(0.28, .95, filter_order=(10,10), v_rms=1.0, 
    fbins=len(impulse.voltage), 
    time_domain_sampling_rate=impulse.dt)
    noise = thermal_noise.makeNoiseWaveform(ntraces=len(trigger_sectors_phi) * 2)

    global_phiscan_start = min(phi_values) - phi_scan_width 
    global_phiscan_stop = max(phi_values) + phi_scan_width 
    global_elscan_start = min(theta_values) - el_scan_width 
    global_elscan_stop = max(theta_values) + el_scan_width

    global_phiscan_range = numpy.arange(global_phiscan_start, global_phiscan_stop, angular_res)
    global_elscan_range = numpy.arange(global_elscan_start, global_elscan_stop, angular_res)
    global_phi_grid, global_el_grid = numpy.meshgrid(global_phiscan_range, global_elscan_range)
    summed_heatmap = numpy.zeros_like(global_phi_grid, dtype=float)
    summed_count = numpy.zeros_like(global_phi_grid, dtype=float)

    '''
    # Plot the coherent sum and power for the central beam direction (phi_start, el_start)
    central_phi = phi_start
    central_theta = el_start 
    central_ringmask = [1,1,0,0,0,0,0,1]
    central_delays = payload.getRemappedDelays(central_phi, central_theta, trigger_sectors_phi)
    central_waveforms, central_timebase, _ = payload.getPayloadWaveforms(
        central_phi, central_theta, trigger_sectors_phi, impulse, (eplane, hplane), downsample=True, snr=20, plot=False
    )
    coh_sum, timebase_coh_sum = coherentSum(central_waveforms, central_timebase, central_delays, False, central_ringmask)

    power, frames = powerSum(coh_sum,window=128, step=32)
    power_db = 10 * numpy.log10(power / numpy.max(power))
    if numpy.max(power_db) > -3:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(timebase_coh_sum, coh_sum, ':o', c='black', ms=2)
        ax[0].legend(['Coherent Sum'])
        ax[0].set_ylabel('3-ant coherent sum')

        ax[1].plot(timebase_coh_sum[:frames*16:16]+timebase_coh_sum[8], power, '--', c='black')
        ax[1].legend(['Power'])
        ax[1].set_xlabel('Time [ns]')
        ax[1].set_ylabel('Power [arb]')
        ax[1].set_ylim([-1, 1000])

        plt.tight_layout()
        plt.savefig("plots/central_beam_coherent_sum.png", bbox_inches='tight')
        plt.close()

    #Off center lobe coherent sum
    offset_phi = phi_start + 15
    offset_theta = el_start + 16
    offset_ringmask = [1,1,0,0,0,0,0,1]
    offset_waveforms, offset_timebase, _ = payload.getPayloadWaveforms(
        offset_phi, offset_theta, trigger_sectors_phi, impulse, (eplane, hplane), downsample=True, snr=20, plot=False
    )

    coh_sum, timebase_coh_sum = coherentSum(offset_waveforms, offset_timebase, central_delays, False, offset_ringmask)

    power, frames = powerSum(coh_sum,window=128, step=32)
    power_db = 10 * numpy.log10(power / numpy.max(power))
    if numpy.max(power_db) > -3:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(timebase_coh_sum, coh_sum, ':o', c='black', ms=2)
        ax[0].legend(['Coherent Sum'])
        ax[0].set_ylabel('3-ant coherent sum')

        ax[1].plot(timebase_coh_sum[:frames*16:16]+timebase_coh_sum[8], power, '--', c='black')
        ax[1].legend(['Power'])
        ax[1].set_xlabel('Time [ns]')
        ax[1].set_ylabel('Power [arb]')
        ax[1].set_ylim([-1, 1000])

        plt.tight_layout()
        plt.savefig("plots/offset_beam_coherent_sum.png", bbox_inches='tight')
        plt.close()
    '''

    # Iterate only over the hexagonal grid points (phi, theta) pairs
    for phi, theta in zip(phi_values, theta_values):
        ringmask = [1,1,0,0,0,0,0,1]
        payload.getPayloadWaveforms(phi, theta, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, plot=True)
        print ('timestep of input pulse [ns]:', impulse.dt)
        delays = payload.getRemappedDelays(phi, theta, trigger_sectors_phi)
        
        phiscan_start = phi - phi_scan_width
        phiscan_stop = phi + phi_scan_width 
        elscan_start = theta - el_scan_width
        elscan_stop = theta + el_scan_width 

        phiscan_start = numpy.round(phiscan_start / angular_res) * angular_res
        phiscan_stop = numpy.round(phiscan_stop / angular_res) * angular_res
        elscan_start = numpy.round(elscan_start / angular_res) * angular_res
        elscan_stop = numpy.round(elscan_stop / angular_res) * angular_res

        phiscan_range = numpy.arange(phiscan_start, phiscan_stop, angular_res)
        elscan_range = numpy.arange(elscan_start, elscan_stop, angular_res)
        phi_grid, el_grid = numpy.meshgrid(phiscan_range, elscan_range)

        heatmap = numpy.zeros_like(phi_grid, dtype=float)
        
        for i, elscan in enumerate(elscan_range):
            for j, phiscan in enumerate(phiscan_range):
                waveforms, timebase, _ = payload.getPayloadWaveforms(phiscan, elscan, trigger_sectors_phi, impulse, (eplane, hplane), downsample=True, snr=5, plot=False)
                coh_sum, _ = coherentSum(waveforms, timebase, delays, False, ringmask)
                power, _ = powerSum(coh_sum,window=128, step=32)
                print(phi, theta)
                heatmap[i, j] = numpy.max(power)
        # Normalize to dB
        heatmap_db = 10 * numpy.log10(heatmap / numpy.max(heatmap))
        if numpy.max(heatmap_db) > -3:
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.pcolormesh(phi_grid, el_grid, heatmap_db, shading='auto', cmap='viridis')
            ax.set_xlabel('Phi [degrees]')
            ax.set_ylabel('Theta [degrees]')
            ax.set_title(f"Coherent Sum Power Heatmap, Beam at $\\phi$ = {int(phi)}, $\\theta$ = {int(theta)}")
            plt.colorbar(c, ax=ax, label='Power [dB]')
            fig.tight_layout()
            filename = f"plots/plot_phi_{phi}_theta_{theta}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        
        phi_offset = phiscan_start - global_phiscan_start
        el_offset = elscan_start - global_elscan_start
        heatmap_rows, heatmap_cols = heatmap.shape

        row_start = max(0, int(round((elscan_start - global_elscan_start) / angular_res)))
        col_start = max(0, int(round((phiscan_start - global_phiscan_start) / angular_res)))
        row_end = min(row_start + heatmap.shape[0], summed_heatmap.shape[0])
        col_end = min(col_start + heatmap.shape[1], summed_heatmap.shape[1])
        local_row_end = row_end - row_start
        local_col_end = col_end - col_start
        target_shape = summed_heatmap[row_start:row_end, col_start:col_end].shape
        source_shape = heatmap[:local_row_end, :local_col_end].shape
        if target_shape != source_shape:
            print(f"Shape mismatch: target {target_shape}, source {source_shape}")
            min_rows = min(target_shape[0], source_shape[0])
            min_cols = min(target_shape[1], source_shape[1])
            summed_heatmap[row_start:row_start+min_rows, col_start:col_start+min_cols] += heatmap[:min_rows, :min_cols]
            summed_count[row_start:row_start+min_rows, col_start:col_start+min_cols] += (heatmap[:min_rows, :min_cols] > 0).astype(float)
        else:
            summed_heatmap[row_start:row_end, col_start:col_end] += heatmap[:local_row_end, :local_col_end]
            summed_count[row_start:row_end, col_start:col_end] += (heatmap[:local_row_end, :local_col_end] > 0).astype(float)
        if summed_heatmap.size == 0 or summed_count.size == 0:
            raise ValueError("summed_heatmap or summed_count array is empty. Check input data and loop logic.")
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
        if numpy.max(avg_heatmap_db) > -3:
            fig, ax = plt.subplots(figsize=(10, 8))
            c = ax.pcolormesh(global_phi_grid, global_el_grid, avg_heatmap_db, shading='auto', cmap='viridis')
            ax.set_xlabel('Phi [degrees]')
            ax.set_ylabel('Theta [degrees]')
            ax.set_title('Summed Coherent Sum Power Heatmap (All Beams)')
            plt.colorbar(c, ax=ax, label='Power [dB]')
            # --- Add -3 dB contour line ---
            plot_contour = True
            contour = None
            if plot_contour:
                contour = ax.contour(
                    global_phi_grid, global_el_grid, avg_heatmap_db,
                    levels=[-3], colors='red', linewidths=2
                )
                ax.clabel(contour, fmt='-3 dB', colors='red', fontsize=12)
            # --- Plot beam locations ---
            ax.scatter(phi_values, theta_values, c='black', s=80, marker='x', label='Beam Centers')
            ax.legend()
            fig.tight_layout()
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/summed_beam_heatmap.png", bbox_inches='tight')
            plt.close()
