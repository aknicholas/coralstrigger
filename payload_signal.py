import numpy
import tools.waveform as waveform
import tools.delays as delays
import tools.CoRaLs_geometry as aso_geometry
from scipy import interpolate
from scipy.signal import lfilter, butter, cheby1
import matplotlib.pyplot as plt

#def h_plane_vpol_beam_pattern():
    
ring_map = {
    'B'  : 0,
    'T'  : 1,}
ring_map_inv = {v: k for k, v in ring_map.copy().items()}

def loadImpulse(filename='impulse/corals_impulse_sci.txt'):

    dat=numpy.loadtxt(filename)
    impulse=waveform.Waveform(dat[:,1], time=dat[:,0])
    print(impulse.gimmeInfo(filename))
    return impulse

def prepImpulse(impulse, upsample=10, filter=True, highpass_cutoff=0.28, lowpass_cutoff=1.10 ):
    '''
    upsample, center, and filter impulse
    highpass_cutoff [GHz]
    '''
    impulse.zeropad(4096)
    #impulse.takeWindow([300, 1024+200])
    impulse.fft()
    #impulse.upsampleFreqDomain(upsample)
    impulse.time = impulse.time-impulse.time[0]
    
    if filter:
        #highpass
        filtercoeff = cheby1(4, rp=0.5, Wn=highpass_cutoff/impulse.freq[-1], btype='highpass')
        impulse = waveform.Waveform(lfilter(filtercoeff[0], filtercoeff[1], impulse.voltage), 
                                    time=impulse.time)
        impulse.fft()
        '''
        #lowpass
        filtercoeff = cheby1(4, rp=0.5, Wn=lowpass_cutoff/impulse.freq[-1], btype='lowpass')
        impulse = waveform.Waveform(lfilter(filtercoeff[0], filtercoeff[1], impulse.voltage), 
                                    time=impulse.time)
        impulse.fft()
        '''
    #set Vpp = 1
    impulse.voltage = impulse.voltage / (numpy.max(impulse.voltage) - numpy.min(impulse.voltage))

    #start = numpy.argmax(impulse.voltage)-3200
    #impulse.takeWindow([start, start+10000])
    #impulse.time = impulse.time-impulse.time[0]

    
    return impulse

#beamPattern interpolation from datafiles or function. For corals right now this is function not data.
# we should break this into 2 sections for E_plane and H_plane of antennas and then we can do beam pattern for them based on the plane and both phi and el for each plane...
# we use h_plane and e_plane as the same resp, just h_plane is rotated by 90 for resp with respect to el I guess...

# This function returns a E_plane beamPattern tuple that has both a el and phi resp (2D spline)
# don't really need a spline since its all a function currently, but eventually need a spline for real resp data included...
# This function returns a H_plane beamPattern tuple that has both a el and phi resp

def beamPattern(plot=False, which_plane='E', which_pol='V', selected_az=None, selected_el=None):
    '''
    CoRaLS proxy beam pattern
    '''
    num_az = 361
    az = numpy.linspace(-180, 180, num_az)
    num_el = 181
    el = numpy.linspace(-90, 90, num_el)
    resp = 1 * numpy.ones((num_el, num_az))
    el_i = 0
    az_i = 0
    while el_i < len(el):
        az_i = 0
        while az_i < len(az):
            resp[el_i, az_i] = 10 * numpy.log10(
                resp[el_i, az_i] * (numpy.cos(numpy.radians(el[el_i])) ** (6)) * numpy.cos(numpy.radians((az[az_i]) / (2.0))) ** (10)
            )
            az_i += 1
        el_i += 1

    if which_plane == 'E':
        if which_pol == 'V':
            plane_interp = interpolate.interp1d(el, resp[:, int(num_az / 2)], kind='cubic')
        elif which_pol == 'H':
            plane_interp = interpolate.interp1d(az, resp[int(num_el / 2), :], kind='cubic')
    elif which_plane == 'E1':
        plane_interp = interpolate.interp1d(el, resp[:, int(num_az / 2)], kind='cubic')
    elif which_plane == 'H':
        if which_pol == 'V':
            plane_interp = interpolate.interp1d(az, resp[int(num_el / 2), :], kind='cubic')
        elif which_pol == 'H':
            plane_interp = interpolate.interp1d(el, resp[:, int(num_az / 2)], kind='cubic')
    elif which_plane == 'H1':
        plane_interp = interpolate.interp1d(el, resp[:, int(num_az / 2)], kind='cubic')

    if plot:
        az_i = 0
        while az_i < num_az:
            plt.plot(el, resp[:, az_i])
            if az_i == int(num_az / 2):
                plt.plot(el, resp[:, az_i], '--', label=str(which_plane))
            az_i += 10

        # Highlight the selected azimuth and elevation if provided
        if selected_az is not None:
            plt.axvline(x=selected_az, color='red', linestyle=':', label=f'Selected az={selected_az}°')
        if selected_el is not None:
            plt.axvline(x=selected_el, color='blue', linestyle=':', label=f'Selected el={selected_el}°')

        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xlabel('off-boresight angle [deg.]')
        plt.ylabel('amplitude [dB]')
        plt.xlim([-50, 50])
        plt.ylim([-10, 1])
        # Add annotation box
        if selected_az is not None or selected_el is not None:
            txt = ""
            if selected_az is not None:
                txt += f"Azimuth: {selected_az}°\n"
            if selected_el is not None:
                txt += f"Elevation: {selected_el}°"
            plt.gca().text(0.05, 0.95, txt, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle="round", fc="w", alpha=0.7))
        plt.show()
    return plane_interp
def dBtoVoltsAtten(db_value):
    '''
    does what it says
    '''
    atten_fraction = 10**(db_value/20)
    return atten_fraction

def getRemappedDelays(phi, el, trigger_sectors):
    '''
    converts delays from delays.getAllDelays(), from a dict to a numpy array
    '''
    delay = delays.getAllDelays([phi], [el], phi_sectors=trigger_sectors) #gets delays at all antennas
    delays_remapped = numpy.zeros((len(trigger_sectors), 4))

    for i in delay[0]['delays']:

        delays_remapped[i[0]-numpy.min(trigger_sectors),ring_map[i[1]]] = delay[0]['delays'][i]

    return delays_remapped


def getPayloadDelays(phi,el,trigger_sectors):
    delay = delays.getAllDelays([phi], [el], phi_sectors=trigger_sectors) #gets delays at all antennas     
    print(type(delay))
    print(delay)
    return delay_all

def getSinglePayloadWaveform(phi, el, trigger_sectors, impulse, beam_pattern, snr=1, noise=None, plot=False, downsample=False):
    delay = delays.getAllDelays([phi], [el], phi_sectors=trigger_sectors) #gets delays at all antennas     
    #print(type(delay))
    print("phi = " , delay[0]["phi"], " theta = ",   delay[0]["theta"])
    #construct trigger_waves to keep the wfs that passed the trigger
    trigger_waves=numpy.zeros((len(trigger_sectors), 4, len(impulse.voltage)))
    #print("trigger waves.shape is: {}".format(trigger_waves.shape))

    # the trigger waves isn't going to have the right shape now, because the code below expects each number index to be either a top or bottom, whereas corals doesn't have that kind of geometry

    #I dont know what multiplier does here yet
    multiplier=[]
    
    #not yet optimized for speed
    for delay_i in delay[0]['delays']:
        # calculate the phi and el and correct it to be within the interpolation range [(-90,90) or (-180,180)]
        phi_interp = phi - aso_geometry.phi_ant[delay_i[0]-1]
        # Wrap phi_interp to [-180, 180]
        if phi_interp > 180:
            phi_interp -= 360
        elif phi_interp < -180:
            phi_interp += 360
        # this line below: trigger_waves elements are calculated based on beam_pattern which is a tuple of eplane,hplane interp functions. beam_pattern[1] is hplane and [0] is eplane
        # hplane is evaluated with phi positions of antennas, and eplane is evaluated at el of antennas. The arguments to these are actually the "off-boresight" angle
        # so the argument is an angle that is the difference of incoming wave and the antenna's angle tilt in phi and theta 
        # the numpy.roll function will move the elements forward or backward along the axis, effectively forcing the delay to be taken into account for the impulse.
        # roll the impulse.voltage which is scaled by 2*snr with the delay: int(numpy.round(delay[0]['delays'][delay_i] / impulse.dt))
        # we also multiply that impulse.voltage that was rolled by the beam_pattern, b/c the antenna direction and the angle of the impulse wave should be taken into acct
        # 
        #trigger waves is calculated with 2 voltage fractions multiplied in, one from eplane and one from hplane. Trigger wave must be on both?
        delay_i[0]-numpy.min(trigger_sectors)
        trigger_waves[delay_i[0]-numpy.min(trigger_sectors),ring_map[delay_i[1]]] = \
            numpy.roll(impulse.voltage * 2 * snr, int(numpy.round(delay[0]['delays'][delay_i] / impulse.dt)))
        print(impulse.time)
        #there is a multiplier here that is hplane[dphi(phi)] and eplane[del(el)].
        #multiplier.append(dBtoVoltsAtten(beam_pattern[1](phi-aso_geometry.phi_ant[delay_i[0]-1])) * \
        #    dBtoVoltsAtten(beam_pattern[0](el -aso_geometry.theta_ant[0])))
        
        #add in the noise if its included.
        if noise is not None:
            print("noise shape " + str(noise.shape))
            print("length of delay_i in payload signal is: {}".format(len(delay_i)))
            print("length of ring map in payload signal is: {}".format(len(ring_map)))
            trigger_waves[delay_i[0]-numpy.min(trigger_sectors),ring_map[delay_i[1]]] += \
                                noise[(delay_i[0]-numpy.min(trigger_sectors))*len(ring_map) + ring_map[delay_i[1]]]

    '''
    #numpyfied waveform generation. Factor of 2 multipler since impulse Vpp putatively normalized to = 1.0
    trigger_waves2 = numpy.roll(numpy.tile(impulse.voltage,len(trigger_sectors)*len(ring_map)).reshape((len(trigger_sectors)*len(ring_map), len(impulse.voltage))) 
                                * 2 * snr, (numpy.round(delay2.flatten() / impulse.dt)).astype(numpy.int)).reshape((len(trigger_sectors), len(ring_map), len(impulse.voltage)))  
    '''
        
    if downsample:
        trigger_waves, impulse.time = downsamplePayload(impulse.time, trigger_waves)

    print(impulse.time)

    if plot:
        fig, ax = plt.subplots(len(ring_map), len(trigger_sectors))
        for i in range(len(trigger_sectors)):
            for j in range(len(ring_map)):
                if j != 0:
                    ax[len(ring_map)-j-1,i].set_xticklabels([])
                if i != 0:
                    ax[len(ring_map)-j-1,i].set_yticklabels([])

                ax[len(ring_map)-j-1,i].plot(impulse.time, trigger_waves[i,j], label=str(i)+ring_map_inv[j], c='black', lw=1, alpha=0.7)
                #ax[len(ring_map)-j-1,i].plot(impulse.time, trigger_waves2[i,j],  c='black', lw=1, alpha=0.7)

                ax[len(ring_map)-j-1,i].legend(loc='upper right')
                #ax[len(ring_map)-j-1,i].set_ylim([-snr-1,snr+1])

        plt.suptitle('phi = '+str(phi)+'deg.  theta = '+str(el)+'deg.', fontsize=20)
        #plt.tight_layout()
        plt.show()
    
    return trigger_waves, impulse.time, multiplier

def getPayloadWaveforms(phi, el, trigger_sectors, impulse, beam_pattern, snr=1, noise=None, plot=False, downsample=False):
    '''
    return waveforms for a single phi, el
    waveforms is 3d (N,M,P) array, where N=number trigger_sectors, M=number of rings, P=size of impulse

    typically, impulse is still upsampled here in order to align waveforms to a good 
    approximation of the true phi, el
    '''
    delay = delays.getAllDelays([phi], [el], phi_sectors=trigger_sectors) #gets delays at all antennas
    #delay2 = getRemappedDelays([phi], [el], trigger_sectors) #gets delays at all antennas
    #print(type(delay))
    #print(delay)
    #construct trigger_waves to keep the wfs that passed the trigger
    print ('impulse voltage length: ', len(impulse.voltage))
    trigger_waves=numpy.zeros((len(trigger_sectors), 4, len(impulse.voltage)))
    print ('impulse time length: ', len(impulse.time))
    #print("trigger waves.shape is: {}".format(trigger_waves.shape))

    #I dont know what multiplier does here yet
    multiplier=[]
    
    #not yet optimized for speed
    for i in delay[0]['delays']:
        #print("i[1] for delay is: {}".format(i[1]))
        #print("ring map i[1] is: {}".format(ring_map[i[1]]))
        #print("i[0] is {}".format(i[0]))
        #print("numpy.min(trigger sectors) is: {}".format(numpy.min(trigger_sectors)))
        #,ring_map[i[1]]
        #print("argument to noise: {}".format((i[0]-numpy.min(trigger_sectors))*len(ring_map) + ring_map[i[1]]))

        # calculate the phi and el and correct it to be within the interpolation range [(-90,90) or (-180,180)]
        phi_interp = phi - aso_geometry.phi_ant[i[0]-1]
        # Wrap phi_interp to [-180, 180]
        if phi_interp > 180:
            phi_interp -= 360
        elif phi_interp < -180:
            phi_interp += 360
        #print('geometry adjusted phi for antenna located at ' + str(i[1]) + str(i[0]) + ' = ' + str(phi_interp))
       
        # this line below: trigger_waves elements are calculated based on beam_pattern which is a tuple of eplane,hplane interp functions. beam_pattern[1] is hplane and [0] is eplane
        # hplane is evaluated with phi positions of antennas, and eplane is evaluated at el of antennas. The arguments to these are actually the "off-boresight" angle
        # so the argument is an angle that is the difference of incoming wave and the antenna's angle tilt in phi and theta 
        # the numpy.roll function will move the elements forward or backward along the axis, effectively forcing the delay to be taken into account for the impulse.
        # roll the impulse.voltage which is scaled by 2*snr with the delay: int(numpy.round(delay[0]['delays'][i] / impulse.dt))
        # we also multiply that impulse.voltage that was rolled by the beam_pattern, b/c the antenna direction and the angle of the impulse wave should be taken into acct
        # 
        #trigger waves is calculated with 2 voltage fractions multiplied in, one from eplane and one from hplane. Trigger wave must be on both?
        #print("roll argument as integer for this trigger-wave-delay: {}".format(int(numpy.round(delay[0]['delays'][i] / impulse.dt))))
        trigger_waves[i[0]-numpy.min(trigger_sectors),ring_map[i[1]]] = \
            numpy.roll(impulse.voltage * 2 * snr, int(numpy.round(delay[0]['delays'][i] / impulse.dt))) * \
            dBtoVoltsAtten(beam_pattern[1](phi_interp)) * \
            dBtoVoltsAtten(beam_pattern[0](el - aso_geometry.theta_ant[i[0]-1]))
        #print('Theta ' + str(aso_geometry.theta_ant[i[0]-1]))
        #print('Attenuation factor is ' + str(dBtoVoltsAtten(beam_pattern[1](phi_interp)) * \
        #    dBtoVoltsAtten(beam_pattern[0](el - aso_geometry.theta_ant[i[0]-1]))))
        #there is a multiplier here that is hplane[dphi(phi)] and eplane[del(el)].
        multiplier.append(dBtoVoltsAtten(beam_pattern[1](phi_interp)) * \
            dBtoVoltsAtten(beam_pattern[0](el - aso_geometry.theta_ant[i[0]-1])))
        

        #add in the noise if its included.
        if noise is not None:
            print("noise shape " + str(noise.shape))
            print("length of i in payload signal is: {}".format(len(i)))
            print("length of ring map in payload signal is: {}".format(len(ring_map)))
            trigger_waves[i[0]-numpy.min(trigger_sectors),ring_map[i[1]]] += \
                                noise[(i[0]-numpy.min(trigger_sectors))*len(ring_map) + ring_map[i[1]]]

    '''
    #numpyfied waveform generation. Factor of 2 multipler since impulse Vpp putatively normalized to = 1.0
    trigger_waves2 = numpy.roll(numpy.tile(impulse.voltage,len(trigger_sectors)*len(ring_map)).reshape((len(trigger_sectors)*len(ring_map), len(impulse.voltage))) 
                                * 2 * snr, (numpy.round(delay2.flatten() / impulse.dt)).astype(numpy.int)).reshape((len(trigger_sectors), len(ring_map), len(impulse.voltage)))  
    '''
    print(phi, el)
    #print ('impulse time length: ', len(impulse.time))
    # Downsample if requested
    if downsample:
        print ('impulse time length: ', len(impulse.time))
        trigger_waves, new_time = downsamplePayload(impulse.time, trigger_waves)
        #impulse.time = new_time  # for any later code that reads impulse.time
        new_dt = new_time[1] - new_time[0]
        print(len(trigger_waves[0][0]))
        # Debug print of delays in original vs down‑sampled indices
        print("Delay sample counts (original vs downsampled):")
        for ant_key, raw_delay in delay[0]['delays'].items():
            orig_samples = int(round(raw_delay / impulse.dt))
            down_samples = int(round(raw_delay / new_dt))
            print(f"  Ant {ant_key}: orig {orig_samples}, down {down_samples}")

        # Replace the impulse time with downsampled time
        timebase = new_time
    else:
        timebase = impulse.time
    # Make sure we use the (possibly downsampled) time vector

    #n_samples = trigger_waves.shape[2]  # should now equal len(timebase)
    #print("timebase len:", len(timebase), "waveform len:", n_samples)
    
    if plot:
        used_keys = list(delay[0]['delays'].keys())
        n_used = len(used_keys)

        # Collect all traces for consistent y‑limits
        all_voltages = [
            trigger_waves[key[0] - numpy.min(trigger_sectors),
                          ring_map[key[1]]]
            for key in used_keys
        ]
        global_vmin = min(v.min() for v in all_voltages)
        global_vmax = max(v.max() for v in all_voltages)

        # Create subplots
        fig, axes = plt.subplots(n_used, 1,
                                 figsize=(6, 3 * n_used),
                                 sharex=True)
        if n_used == 1:
            axes = [axes]

        # Plot each antenna’s downsampled waveform
        for ax, key in zip(axes, used_keys):
            sector_idx = key[0] - numpy.min(trigger_sectors)
            ring_idx   = ring_map[key[1]]

            ax.plot(timebase,
                    trigger_waves[sector_idx, ring_idx],
                    color='black', lw=1, alpha=0.7)
            ax.set_ylim(global_vmin, global_vmax)
            ax.set_title(f"Sector {key[0]} Ring {key[1]}")

        axes[-1].set_xlabel("Time [ns]")
        fig.suptitle(f"φ = {phi}°, θ = {el}°", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        plt.savefig("plots/debug.png")

    return trigger_waves, timebase, multiplier

def downsamplePayload(time, trigger_waves):

    decimate_factor = int(aso_geometry.ritc_sample_step/((time[1]-time[0])))
    print('decimate factor in downsamplePayload = ' ,decimate_factor)
    print('time length pre decimation ', len(time))
    trigger_waves = trigger_waves[:,:,::decimate_factor]
    time = time[::decimate_factor]
    print('time length post decimation ', len(time))
    print('trigger_waves length post decimation ', len(trigger_waves[0][0]))
    return trigger_waves, time                                  

def gimmePlotsOriginal(impulse):
            impulse.fft()
            plt.figure(1)
            plt.plot(impulse.time_zeroed, impulse.voltage, '-.', ms=2)
            plt.title("time domain?") 
            plt.figure(2)
            plt.plot(impulse.freq, numpy.abs(impulse.ampl), '-.', ms=2)
            impulse.upsampleFreqDomain(2)
            plt.figure(1)
            plt.plot(impulse.time_zeroed, impulse.voltage, '--', ms=1)
            #plt.xlim([0,60])
            plt.figure(2)
            plt.plot(impulse.freq, numpy.abs(impulse.ampl), '--', ms=1)   
            #plt.xlim([0,3])
            plt.title("frequency?") 
            #plt.figure(4)
            #plt.plot(impulse.time, numpy.correlate(impulse.voltage, impulse.voltage, "same"))
            plt.show()


def gimmePlots(impulse,impulse2):
            impulse.fft()
            impulse2.fft()
            plt.figure(1)
            plt.plot(impulse.time_zeroed, impulse.voltage, '-.', ms=2)
            plt.title("time domain?") 
            plt.figure(2)
            plt.plot(impulse.freq, numpy.abs(impulse.ampl), '-.', ms=2)
            #impulse.upsampleFreqDomain(2)
            plt.figure(1)
            plt.plot(impulse2.time, impulse2.voltage, '--', ms=1)
            #plt.xlim([0,60])
            plt.figure(2)
            plt.plot(impulse2.freq, numpy.abs(impulse2.ampl), '--', ms=1)   
            #plt.xlim([0,3])
            plt.title("frequency?") 
            #plt.figure(4)
            #plt.plot(impulse.time, numpy.correlate(impulse.voltage, impulse.voltage, "same"))
            plt.show()

def gimmePlotsImpulse(impulse):
    impulse.fft()
    plt.figure(1)
    plt.plot(impulse.time_zeroed, impulse.voltage, '-.', ms=2)
    plt.title("Time Domain")
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [V]")

    plt.figure(2)
    impulse.fft()  # Ensure FFT is up to date
    abs_fft = numpy.abs(impulse.ampl)
    eps = 1e-20

    y_vals = 20 * numpy.log10((abs_fft + eps) / (numpy.max(abs_fft) + eps)) + 3
    # Find the maximum frequency where y > -80 dB
    valid_indices = numpy.where(y_vals > -80)[0]
    if valid_indices.size > 0:
        max_freq = impulse.freq[valid_indices[-1]] 
    else:
        max_freq = 2  # fallback if all values are below -80 dB

    plt.plot(impulse.freq, y_vals, '-.', ms=2)
    plt.title("Frequency Domain")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Amplitude [dB]")
    plt.xlim([0, max_freq])
    plt.ylim([-80, 5])
    plt.show()

def gimmePlots2Impulses(impulse,impulse2):
            impulse.fft()
            impulse2.fft()
            fig, ax1 = plt.subplots()
            ax1.plot(impulse.time_zeroed, impulse.voltage, '-.', ms=2,color='red')
            fig.suptitle("time domain?") 
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.plot(impulse2.time, impulse2.voltage, '--', ms=1,color='blue')
            #ax1.set_ylim([-0.5,0.5])
            #ax2.set_ylim([-100,100])
            figB, ax1B = plt.subplots()
            ax1B.plot(impulse.freq, numpy.abs(impulse.ampl), '-.', ms=2,color='red')
            ax2B = ax1B.twinx()  # instantiate a second axes that shares the same x-axis
            ax2B.plot(impulse2.freq, numpy.abs(impulse2.ampl), '--', ms=1,color='blue')   
            #plt.xlim([0,3])
            figB.suptitle("frequency?") 
            #plt.figure(4)
            #plt.plot(impulse.time, numpy.correlate(impulse.voltage, impulse.voltage, "same"))
            plt.show()


def loadPlaneWave(freq_i=100*10**6):
           # testing sample rate in waveform init
    N_samples=4096
    #freq_i=100*10**6 # 100 MHz
    omega=2*numpy.pi*freq_i
    #now omega*t is the argument in the sine wave, so we can get real_times from this
    period=1/freq_i
    print("period: {}".format(period))
    # how many periods do you measure for, a random number maybe?
    rng = numpy.random.default_rng()
    num_periods=1+(rng.random()*3)
    print("plotting {} seconds of wave ".format(period*num_periods))

    # need sampling rate now...
    sampling_period= 3*10**(-10)# lets do 500MHz or 50Mega samples / sec
    times=numpy.linspace(0,N_samples*sampling_period,N_samples)
    #print(times)
    volty=numpy.sin(omega*times)
    real_times=numpy.linspace(0,N_samples*sampling_period,int(100*N_samples))
    #print(real_times[-1])
    real_volty=numpy.sin(omega*real_times)
    reaL_siney=waveform.Waveform(real_volty,real_times*10**9) # what is the sampling rate now? if we had 4096 samples in 2 periods of a 100MHz sine wave then 

    #real_times=numpy.linspace(0,period,N_samples)
    #siney=waveform.Waveform(volty, sampling_rate=(sampling_period)) # what is the sampling rate now? if we had 4096 samples in 2 periods of a 100MHz sine wave then 
    siney=waveform.Waveform(volty, times*10**9) # what is the sampling rate now? if we had 4096 samples in 2 periods of a 100MHz sine wave then 

    #plt.scatter(siney.time,siney.voltage, label='waveform')
    plt.scatter(reaL_siney.time,reaL_siney.voltage, label='real wave')
    plt.scatter(siney.time,siney.voltage, label='sampled wave')

    #plt.scatter(siney.time,siney.voltage, label='sampled wave')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('Time [ns]')
    plt.ylabel('Voltage')
    #plt.xlim([0,period*10**9])
    plt.title("waveform for sine wave")
    plt.show()
    return siney

if __name__=="__main__":
    
    import noise
    # for corals, the beamPatterns get more complex, so we need to break apart to multiple calls
    eplane = beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane = beamPattern(plot=False,which_plane='H',which_pol='V')
    #load new corals lpda impulse
    impulse = loadImpulse('impulse/corals_impulse.txt')
    impulse = prepImpulse(impulse)
    print("length of file is: {} ".format(impulse.n))
    trigger_sectors_phi=[1,2,3,4,5,6,7,8]
    thermal_noise = noise.ThermalNoise(0.28, .95, filter_order=(10,10), v_rms=1.0, 
                                       fbins=len(impulse.voltage), 
                                       time_domain_sampling_rate=impulse.dt)

    noise = thermal_noise.makeNoiseWaveform(ntraces=len(trigger_sectors_phi) * len(ring_map))
    #plot a boresight SNR of 5
    #waveform is incoming at phi and el, so maybe better if beam pattern is 2D now?
    #this worked in some way
    #getPayloadWaveforms(22.5, -25, [1,2,3,4], impulse, (eplane, hplane), snr=5, noise=numpy.real(noise[2]), plot=True)
    #try more channels?
    #phi sectors
    #plot needs to make phi sectors use Corals geometry class, not use just ring_map where it assumes there is a top and bottom antenna at every phi...
    #trigger_sectors_phi=[1,2,3,4]

    #print(len(trigger_sectors_phi))
    
    #getSinglePayloadWaveform(22.5, -25, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, plot=True)
    print("BEGIN getPayloadWaveforms")
    #getPayloadWaveforms(22.5, -25, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, noise=numpy.real(noise[2]), plot=True)
    #try without noise and just plane wave as impulse
    
    getPayloadWaveforms(0, -45, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, plot=True, downsample=True)
    print(impulse.time)
    #getPayloadWaveforms(45, -45, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, noise=numpy.real(noise[2]), plot=True)

    #I think I need to include just the plane wave (i.e. no noise added and maybe not even impulse/ impulse response) to see how the delay and all that works for a "trigger_wave" in the getPayload function
    #the trigger_wave here will nto really be anything about a trigger but a check to see if the delays and all that are working correctly...
      
    #getPayloadWaveforms(22.5, -25, [1,2,3,4], impulse, (eplane, hplane), snr=5,  plot=True)


