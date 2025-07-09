#UWB planewaves->geometric delays
#EJO 11/2016
#   ......cleaned up 12/2017

import myplot  #specific for running on UC midway cluster
import matplotlib.pyplot as plt
import numpy as np
from . constants import *
from . import CoRaLs_geometry as anita

def delay(phi, theta):
    '''
    theta: elevation angle 
    '''
    x_planewave = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    y_planewave = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    z_planewave = np.sin(np.radians(theta))
    delays =  -1.0 * ( (anita.x_ant * x_planewave) \
                       + (anita.y_ant * y_planewave) \
                       + (anita.z_ant * z_planewave) ) / c_light
    #print(delays)
    return delays - np.min(delays)

def scanDelays(phi, theta):
    '''
    phi, theta:  numpy arrays of *equal* length
    '''
    t_delays=[]
    for i in range(len(phi)):
        t_delays.append(delay(phi[i], theta[i]))

    return np.array(t_delays)

def getDelays(phi, theta, phi_sectors=range(1,anita.num_phi_sectors), verbose=True):
    '''
    specify phi and theta values (scalars)
    and list of phi sectors of interest (default is all 16)
    print relative delays to terminal if verbose=True
    '''
    t_delay = delay(phi, theta)
    useful = {}
    useful_delays=[]
    for i in range(len(t_delay)):
        for j in phi_sectors:
            if anita.phisector[i] == j:
                useful[str(anita.phisector[i]) + anita.loc[i]] = t_delay[i]
                useful_delays.append(t_delay[i])

    useful_keys= sorted(useful.keys())
    #dump the info in an organized fashion:
    print ('plane wave direction: phi =', phi, 'deg; theta =', theta, 'deg')
    for i in useful_keys:
        #useful[i]-max(useful_delays)
        if verbose:
            #print i, '{0:.1f}'.format(useful[i]-max(useful_delays)), 'ns'
            print (i, '{0:.3f}'.format(useful[i]), 'ns')
    return useful

def getAllDelays(phi, theta, phi_sectors=range(1,anita.num_phi_sectors)):
    '''
    basically same as getDelays, but phi and theta are now numpy arrays,
    and a dictionary of delays is created for each (phi, theta) combination
    '''
    t_delays = scanDelays(phi, theta)
    #print(type(t_delays))
    #print(t_delays.shape)
    data_dict={}
    delays_only=[]
    for k in range(len(t_delays)):
        data_dict[k]={}
        data_dict[k]['phi']=phi[k]
        data_dict[k]['theta']=theta[k]
        data_dict[k]['delays'] = {}

        for i in range(len(t_delays[0])):
            for j in phi_sectors:
                if anita.phisector[i] == j:
                    data_dict[k]['delays'][anita.phisector[i], anita.loc[i]] = t_delays[k][i]
                    #data_dict[k]['delays'][str(anita.phisector[i]) + anita.loc[i]] = t_delays[k][i]

    #[eventually use json to dump to file]
    return data_dict

def plotDelayDictEvent(delay_dict, event):
    '''
    event is an integer
    '''
    if event in delay_dict:
        print ('------------')
        print ('wave theta:', delay_dict[event]['theta'])
        print ('wave phi:  ', delay_dict[event]['phi'])

        phi_sectors=delay_dict[event]['delays'].keys()
        plt.figure()
        for i in phi_sectors:
            if i[1]=='B':
                plt.plot(delay_dict[event]['delays'][i[0],i[1]],int(i[0]), 'o', ms=4, color='blue')
            #elif i[1]=='M':
            #    plt.plot(delay_dict[event]['delays'][i[0],i[1]],int(i[0]), 'o', ms=4, color='green')
            elif i[1]=='T':
                plt.plot(delay_dict[event]['delays'][i[0],i[1]],int(i[0]), 'o', ms=4, color='red')
        plt.grid()
        str_to_plot="Theta: " +str(delay_dict[event]['theta'])+" , Phi: "+str(delay_dict[event]['phi'])
        #print(str_to_plot)
        #plt.text(delay_dict[event]['theta'])
        plt.xlabel('pulse arrival time [ns]')
        plt.ylabel('CoRaLS phi sector no.')
        plt.title(str_to_plot)
        plt.tight_layout()
        
    else:
        print ('event specified is not in dataset')

def makeDelayElevationPlot(phi=0.0, phi_sector=1, plot=True):
    '''
    generate relative antenna-pair delays vs incoming plane wave elevation angle 
    for specified phi-sector and plane-wave azimuthal direction
    '''
    thetas = np.arange(-65, 41, 2)
    phis = np.ones(len(thetas)) * phi
    phis2 = np.ones(len(thetas)) * (phi+30.0)
    phis3 = np.ones(len(thetas)) * (phi+330.0)

    t_delays = scanDelays(phis, thetas)
    t_delays2 = scanDelays(phis2, thetas)
    t_delays3 = scanDelays(phis3, thetas)
    phi_sector = int(phi_sector)

    if phi_sector < 1 or phi_sector > anita.num_phi_sectors:
        print ('no such phi-sector')
        return

    if plot:
        fig=plt.figure()
        #plt.plot(thetas, t_delays[:,phi_sector+antennas_per_ring-1] - t_delays[:,phi_sector+antennas_per_ring*2-1], label='mid-bot')
        #plt.plot(thetas, t_delays[:,phi_sector-1] - t_delays[:,phi_sector+antennas_per_ring*2-1], label='top-bot')
        #plt.plot(thetas, t_delays[:,phi_sector-1] - t_delays[:,phi_sector], label='top-bottom')
        plt.plot(thetas, t_delays[:,phi_sector-1] - t_delays[:,phi_sector], label='phi='+str(phi))
        plt.plot(thetas, t_delays2[:,phi_sector-1] - t_delays2[:,phi_sector], label='phi='+str(phi+30.0))
        plt.plot(thetas, t_delays3[:,phi_sector-1] - t_delays3[:,phi_sector], label='phi='+str(phi-30.0))
        plt.legend()
        plt.xlabel('Elevation Angle [deg]')
        plt.ylabel('Delay [ns]')
        plt.title('Delay times: Top-Bottom')
        plt.grid()
        plt.tight_layout()
        #plt.show()
        return fig
    else:
        return np.array((thetas, t_delays[:,phi_sector+antennas_per_ring-1] - t_delays[:,phi_sector+antennas_per_ring*2-1], t_delays[:,phi_sector-1] - t_delays[:,phi_sector+antennas_per_ring*2-1],
                         t_delays[:,phi_sector-1] - t_delays[:,phi_sector+antennas_per_ring-1]))
    
if __name__=='__main__':

    # example usage:
    print(f'anita: {anita.loc!s}')
    print(f'anita: {anita.phisector!s}')
    ## getDelays function:
    phi = 45 #11.25
    theta = -45
    #print(delay(phi,theta))
    phi_sectors_of_interest = [1,2,8] #range(1,anita.num_phi_sectors) this will be a top, bottom, and top again..
    getDelays(phi, theta, phi_sectors_of_interest, verbose=True)

    # Apply decimation and loop over phi/theta grid
    decimation_ps = 250  # 250 ps = 0.25 ns
    decimation_ns = decimation_ps * 1e-3  # convert to ns

    phi_range = np.arange(-60, 61, 1)
    theta_range = np.arange(-75, -14, 2)

    sample_dt_ns = 0.25  # <-- Set this to your actual sample spacing in ns

    # Track minimum angle change (delta_phi, delta_theta) that causes at least 1 sample change in any antenna
    min_phi_step = None
    min_theta_step = None

    # We'll use the first phi/theta as a reference
    reference_phi = phi_range[0]
    reference_theta = theta_range[0]
    reference_delays = delay(reference_phi, reference_theta)
    reference_selected_delays = []
    for i in range(len(reference_delays)):
        if anita.phisector[i] in phi_sectors_of_interest:
            reference_selected_delays.append(reference_delays[i])
    reference_delays_decimated = np.round(np.array(reference_selected_delays) / decimation_ns) * decimation_ns
    reference_delays_samples = reference_delays_decimated / sample_dt_ns

    # Search for minimum phi step
    for dphi in np.arange(0.01, 5.0, 0.01):
        test_delays = delay(reference_phi + dphi, reference_theta)
        test_selected_delays = []
        for i in range(len(test_delays)):
            if anita.phisector[i] in phi_sectors_of_interest:
                test_selected_delays.append(test_delays[i])
        test_delays_decimated = np.round(np.array(test_selected_delays) / decimation_ns) * decimation_ns
        test_delays_samples = test_delays_decimated / sample_dt_ns
        if np.any(np.abs(test_delays_samples - reference_delays_samples) >= 1):
            min_phi_step = dphi
            break

    # Search for minimum theta step
    for dtheta in np.arange(0.01, 5.0, 0.01):
        test_delays = delay(reference_phi, reference_theta + dtheta)
        test_selected_delays = []
        for i in range(len(test_delays)):
            if anita.phisector[i] in phi_sectors_of_interest:
                test_selected_delays.append(test_delays[i])
        test_delays_decimated = np.round(np.array(test_selected_delays) / decimation_ns) * decimation_ns
        test_delays_samples = test_delays_decimated / sample_dt_ns
        if np.any(np.abs(test_delays_samples - reference_delays_samples) >= 1):
            min_theta_step = dtheta
            break

    print(f"Minimum phi step for 1 sample change: {min_phi_step} deg")
    print(f"Minimum theta step for 1 sample change: {min_theta_step} deg")

    for phi_val in phi_range:
        for theta_val in theta_range:
            delays = delay(phi_val, theta_val)
            # Only keep delays for selected phi sectors of interest
            selected_delays = []
            selected_labels = []
            for i in range(len(delays)):
                if anita.phisector[i] in phi_sectors_of_interest:
                    selected_delays.append(delays[i])
                    selected_labels.append((anita.phisector[i], anita.loc[i]))
            # Quantize delays to 250 ps steps
            delays_decimated = np.round(np.array(selected_delays) / decimation_ns) * decimation_ns
            delays_samples = delays_decimated / sample_dt_ns
            print(f"phi={phi_val}, theta={theta_val} " +
                  ", ".join([f"{label[0]}{label[1]}, {d}" for label, d in zip(selected_labels, delays_samples)]))

    ## getAllDelays function:
    phi = np.array([45,0.0])
    theta = np.array([-45,0.0])
    phi_sectors_of_interest = [1,2,8] #range(1,8)
    data_dict = getAllDelays(phi, theta, phi_sectors_of_interest)

    ## read DelayDict (read in output of getAllDelays)
    plotDelayDictEvent(data_dict, 0)
    
    ##make del-el plot
    #makeDelayElevationPlot()

    plt.show()
    
