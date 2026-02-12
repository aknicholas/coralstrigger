import numpy
import matplotlib.pyplot as plt
import tools.CoRaLs_geometry as aso_geometry
import tools.constants as constants
import matplotlib.gridspec as gridspec
import tools.waveform as waveform
import tools.filters as filters
import payload_signal as payload
import math
import noise
import os
import coherent_sum as sum
from scipy import ndimage
from skimage.measure import label, regionprops
import pickle
import json
import os
import sys
import time
import argparse
from tools import delays as delays_tools

# Configuration - set which colors to process
PROCESS_COLORS = None               # Process all colors
#PROCESS_COLORS = None             # Process all colors
#EXCLUDE_COLORS = ['blue']         # Process all except these colors
#EXCLUDE_COLORS = ['teal']
EXCLUDE_COLORS = None

def _coord_key(phi, theta, prec=6):
    return (round(float(phi), prec), round(float(theta), prec))

def apply_frontend_filter(waveforms, timebase, lp_gain):
    """
    In-place apply frequency-domain gain (lp_gain) to each channel waveform.
    waveforms shape: (n_antennas, n_samples) or (n_phi, n_rings, n_samples)
    """
    import tools.waveform as waveform
    
    # Handle both 2D and 3D shapes
    if waveforms.ndim == 2:
        n_ant, _ = waveforms.shape
        indices = [(i,) for i in range(n_ant)]
    elif waveforms.ndim == 3:
        n_phi, n_rings, _ = waveforms.shape
        indices = [(i, j) for i in range(n_phi) for j in range(n_rings)]
    else:
        raise ValueError(f"Expected 2D or 3D waveforms, got shape {waveforms.shape}")
    
    for idx in indices:
        wf_obj = waveform.Waveform(waveforms[idx], timebase)
        wf_obj.fft()
        fft_len = len(wf_obj.ampl)
        if len(lp_gain) != fft_len:
            gain = numpy.interp(
                numpy.linspace(0, 1, fft_len),
                numpy.linspace(0, 1, len(lp_gain)),
                lp_gain
            )
        else:
            gain = lp_gain
        wf_obj.ampl *= gain
        wf_obj.voltage = numpy.fft.irfft(wf_obj.ampl, n=wf_obj.n)
        waveforms[idx] = wf_obj.voltage
    return waveforms

def compute_beam_pattern(phi_c, th_c,
                         span_phi, span_theta, res_deg,
                         window, step,
                         impulse, eplane, hplane,
                         trigger_sectors_phi, lp_gain,
                         ringmask=(1,1,1,1),
                         steering_mode='vary_source',
                         delay_quantization='discrete'):
    """
    Compute beam pattern (coherent sum sliding-window peak power) over angular grid.

    steering_mode:
      'vary_source'  : (Correct beam pattern) fixed steering delays at (phi_c,th_c), regenerate source per grid angle.
      'vary_steer'   : Fixed source at center, vary steering (shows quantization artifacts).

    delay_quantization:
      'discrete'  : quantize delays to hardware step (aso_geometry.ritc_sample_step)
      'continuous': use raw (float) delays

    Returns: dict {
        'PHI','TH','heatmap','heatmap_db','delays_used','mode','quantization'
    }
    """
    ringmask = list(ringmask)
    phiscan = numpy.arange(phi_c - span_phi, phi_c + span_phi + 1e-9, res_deg)
    thscan  = numpy.arange(th_c - span_theta, th_c + span_theta + 1e-9, res_deg)
    PHI, TH = numpy.meshgrid(phiscan, thscan)
    heatmap = numpy.zeros_like(PHI, dtype=float)

    # Steering delays for center
    delays_center = payload.getRemappedDelays(phi_c, th_c, trigger_sectors_phi)
    if delay_quantization == 'discrete':
        decim_ps = int(round(float(aso_geometry.ritc_sample_step) * 1000.0))
        delays_center_q = delays_tools.quantize_delays_ns(delays_center, decimation_ps=decim_ps)
    else:
        delays_center_q = delays_center

    if steering_mode == 'vary_steer':
        # Build fixed source at center once
        wave_center, timebase, _ = payload.getPayloadWaveforms(
            phi_c, th_c, impulse, (eplane, hplane),
            antennas=trigger_sectors_phi, downsample=True, snr=5, plot=False)
        wave_center = apply_frontend_filter(wave_center, timebase, lp_gain)

    for i, th in enumerate(thscan):
        for j, ph in enumerate(phiscan):
            if steering_mode == 'vary_source':
                # Source varies, steering fixed
                waveforms, timebase, _ = payload.getPayloadWaveforms(
                    ph, th, impulse, (eplane, hplane),
                    antennas=trigger_sectors_phi, downsample=True, snr=5, plot=False)
                waveforms = apply_frontend_filter(waveforms, timebase, lp_gain)
                coh_sum, _ = sum.coherentSum(waveforms, timebase, delays_center_q, False, ringmask)
            else:
                # Source fixed, steering varies
                delays_steer = payload.getRemappedDelays(ph, th, trigger_sectors_phi)
                if delay_quantization == 'discrete':
                    decim_ps = int(round(float(aso_geometry.ritc_sample_step) * 1000.0))
                    delays_steer = delays_tools.quantize_delays_ns(delays_steer, decimation_ps=decim_ps)
                coh_sum, _ = sum.coherentSum(wave_center, timebase, delays_steer, False, ringmask)

            power, _ = sum.powerSum(coh_sum, window=window, step=step)
            heatmap[i, j] = numpy.max(power)

    eps = 1e-12
    heatmap_db = 10.0 * numpy.log10(numpy.maximum(heatmap, eps) / numpy.max(heatmap))
    return {
        'PHI': PHI, 'TH': TH,
        'heatmap': heatmap,
        'heatmap_db': heatmap_db,
        'delays_used': delays_center_q,
        'mode': steering_mode,
        'quantization': delay_quantization
    }

def generate_quadrant_grid(phi_center=90, theta_center=0, half_span=45, step=1, quadrant='UR'):
    """
    Generate integer grid for one quadrant relative to (phi_center, theta_center).
    Quadrants: UR, UL, LR, LL
    """
    phis = []
    thetas = []
    if quadrant == 'UR':
        phi_range = range(phi_center, phi_center + half_span, step)
        theta_range = range(theta_center, theta_center + half_span, step)
    elif quadrant == 'UL':
        phi_range = range(phi_center - half_span + 1, phi_center + 1, step)
        theta_range = range(theta_center, theta_center + half_span, step)
    elif quadrant == 'LR':
        phi_range = range(phi_center, phi_center + half_span, step)
        theta_range = range(theta_center - half_span + 1, theta_center + 1, step)
    elif quadrant == 'LL':
        phi_range = range(phi_center - half_span + 1, phi_center + 1, step)
        theta_range = range(theta_center - half_span + 1, theta_center + 1, step)
    else:
        raise ValueError("quadrant must be one of UR, UL, LR, LL")
    for p in phi_range:
        for t in theta_range:
            phis.append(p)
            thetas.append(t)
    return phis, thetas

def build_full_square_from_selected(selected_info, phi_center=90, theta_center=0):
    """
    Take selected beams (assumed in UR quadrant: phi>=center, theta>=0) and mirror across:
      theta=0   -> lower (negative theta)
      phi=phi_center -> left
      both      -> lower-left
    Avoid duplicates along axes (theta=0 or phi=phi_center).
    Returns dict with keys: 'original','mirror_theta','mirror_phi','mirror_both'
    """
    original = [(inf['phi'], inf['theta']) for inf in selected_info]
    out = {'original': [], 'mirror_theta': [], 'mirror_phi': [], 'mirror_both': []}
    seen = set()

    for (phi, theta) in original:
        # original
        k = _coord_key(phi, theta)
        if k not in seen:
            seen.add(k)
            out['original'].append((phi, theta))
        # mirror across theta=0 (only if theta>0)
        if theta > 0:
            mt = (phi, -theta)
            k = _coord_key(*mt)
            if k not in seen:
                seen.add(k)
                out['mirror_theta'].append(mt)
        # mirror across phi=phi_center (only if phi>phi_center)
        if phi > phi_center:
            mp = (2*phi_center - phi, theta)
            k = _coord_key(*mp)
            if k not in seen:
                seen.add(k)
                out['mirror_phi'].append(mp)
        # mirror across both
        if theta > 0 and phi > phi_center:
            mb = (2*phi_center - phi, -theta)
            k = _coord_key(*mb)
            if k not in seen:
                seen.add(k)
                out['mirror_both'].append(mb)
    return out

def flatten_mirror_dict(mdict):
    """Flatten dict from build_full_square_from_selected into single ordered list."""
    ordered = []
    for key in ['original','mirror_theta','mirror_phi','mirror_both']:
        ordered.extend(mdict[key])
    return ordered

def dedupe_new_coordinates(existing_info, candidate_coords):
    """
    Remove coordinates already present in existing_info (by exact phi,theta).
    Returns list of unique new coords.
    """
    existing = {_coord_key(inf['phi'], inf['theta']) for inf in existing_info}
    unique_new = []
    added = set()
    for (phi, theta) in candidate_coords:
        k = _coord_key(phi, theta)
        if k not in existing and k not in added:
            added.add(k)
            unique_new.append((phi, theta))
    return unique_new

def compute_new_from_coordinate_list(coord_list, global_phi_values, global_theta_values):
    """
    Append new coordinates to global phi/theta arrays and compute beams.
    Returns (new_contours, new_info, new_indices_appended)
    """
    start_len = len(global_phi_values)
    for (p,t) in coord_list:
        global_phi_values.append(p)
        global_theta_values.append(t)
    new_indices = list(range(start_len, len(global_phi_values)))
    new_contours, new_info = calculate_beams(global_phi_values, global_theta_values, new_indices)
    return new_contours, new_info, new_indices
def save_contours(all_contours, beam_info, filename="beam_contours_data.pkl", format='pickle'):
    """
    Save contour data to file in pickle or JSON format.
    
    Parameters:
    - format: 'pickle' (default, binary, fastest) or 'json' (human-readable)
    """
    # Auto-detect format from filename if not .pkl
    if not filename.endswith('.pkl') and format == 'pickle':
        if filename.endswith('.json'):
            format = 'json'
    
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        contour_data = {
            'contours': [contour.tolist() for contour in all_contours],
            'beam_info': beam_info,
            'timestamp': time.time(),
            'n_beams': len(all_contours)
        }
        with open(filepath, 'w') as f:
            json.dump(contour_data, f, indent=2)
        print(f"Saved {len(all_contours)} contours to {filepath} (JSON format)")
    else:
        # Pickle format (default)
        contour_data = {
            'contours': all_contours,
            'beam_info': beam_info,
            'timestamp': time.time(),
            'settings_used': beam_settings
        }
        with open(filepath, 'wb') as f:
            pickle.dump(contour_data, f)
        print(f"Saved {len(all_contours)} contours to {filepath} (pickle format)")
    
    return filepath

def load_contours(filename="beam_contours_data.pkl", format=None):
    """
    Load contour data from file (pickle or JSON format).
    
    Parameters:
    - format: 'pickle', 'json', or None (auto-detect from extension)
    """
    filepath = os.path.join("plots", filename)
    
    if not os.path.exists(filepath):
        print(f"No saved contour data found at {filepath}")
        return None, None
    
    # Auto-detect format from filename if not specified
    if format is None:
        format = 'json' if filename.endswith('.json') else 'pickle'
    
    try:
        if format == 'json':
            with open(filepath, 'r') as f:
                contour_data = json.load(f)
            # Convert lists back to numpy arrays
            all_contours = [numpy.array(contour) for contour in contour_data['contours']]
        else:
            with open(filepath, 'rb') as f:
                contour_data = pickle.load(f)
            all_contours = contour_data['contours']
        
        beam_info = contour_data['beam_info']
        
        print(f"Loaded {len(all_contours)} contours from {filepath}")
        if 'timestamp' in contour_data:
            timestamp = time.ctime(contour_data['timestamp'])
            print(f"  Data created: {timestamp}")
        
        return all_contours, beam_info
    
    except Exception as e:
        print(f"Error loading contour data: {e}")
        return None, None




def should_process_beam(phi, theta, beam_idx):
    """Check if beam should be processed based on color filters"""
    _, _, _, _, _, _, beam_color = get_beam_sensitivity(phi, theta, beam_idx)
    
    # Check inclusion filter
    if PROCESS_COLORS is not None and beam_color not in PROCESS_COLORS:
        return False, beam_color
        
    # Check exclusion filter  
    if EXCLUDE_COLORS is not None and beam_color in EXCLUDE_COLORS:
        return False, beam_color
        
    return True, beam_color

# Manual sensitivity and scan width settings for each beam
def get_beam_sensitivity(phi, theta, beam_idx):
    if beam_idx in beam_settings:
        settings = beam_settings[beam_idx]
        penalty = 0  # Convert to penalty
        return 1.0, -penalty, settings['phi_start'], settings['phi_stop'], settings['el_start'], settings['el_stop'], settings['color']
    else:
        return 1.0, 0.0, -90, 90, -110, -70, 'gray'  # Default scan boundaries with default color

def load_filtered_contours(filename="beam_contours_data.pkl", beam_indices=None, colors=None):
    """
    Load contours with optional filtering by beam indices or colors.
    
    Parameters:
    - filename: File to load from
    - beam_indices: List of beam indices to include (None = all)
    - colors: List of colors to include (None = all)
    
    Returns:
    - filtered_contours, filtered_info (or None, None on error)
    """
    all_contours, beam_info = load_contours(filename)
    
    if all_contours is None:
        return None, None
    
    # If no filters specified, return everything
    if beam_indices is None and colors is None:
        return all_contours, beam_info
    
    # Apply filters
    filtered_contours = []
    filtered_info = []
    
    for i, info in enumerate(beam_info):
        include = True
        
        # Check beam index filter
        if beam_indices is not None and info['beam_idx'] not in beam_indices:
            include = False
        
        # Check color filter
        if colors is not None and info['color'] not in colors:
            include = False
        
        if include:
            filtered_contours.append(all_contours[i])
            filtered_info.append(info)
    
    if beam_indices is not None:
        print(f"Loaded {len(filtered_contours)} out of {len(beam_indices)} requested beams")
    elif colors is not None:
        print(f"Loaded {len(filtered_contours)} beams with colors {colors}")
    
    return filtered_contours, filtered_info

def list_saved_contours():
    """List all saved contour files"""
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        print("No plots directory found")
        return
    
    contour_files = [f for f in os.listdir(plots_dir) if f.endswith(('.pkl', '.json')) and 'contour' in f]
    
    if not contour_files:
        print("No saved contour files found")
        return
    
    print("Available contour files:")
    for filename in sorted(contour_files):
        filepath = os.path.join(plots_dir, filename)
        size = os.path.getsize(filepath)
        mtime = os.path.getmtime(filepath)
        print(f"  {filename} ({size} bytes, modified: {time.ctime(mtime)})")
def load_or_calculate_beams(phi_values, theta_values):
    """Load existing beams and identify which ones need to be calculated"""
    
    # Try to load existing data
    existing_contours, existing_info = load_contours()
    
    if existing_contours is None:
        # No saved data, calculate all beams
        print("No saved data found, will calculate all beams")
        return [], [], list(range(len(phi_values)))
    
    # Check which beams already exist
    existing_beam_positions = set()
    for info in existing_info:
        existing_beam_positions.add((info['phi'], info['theta']))
    
    # Find beams that need to be calculated
    beams_to_calculate = []
    for beam_idx, (phi, theta) in enumerate(zip(phi_values, theta_values)):
        if (phi, theta) not in existing_beam_positions:
            beams_to_calculate.append(beam_idx)
    
    print(f"Found {len(existing_contours)} existing beams")
    print(f"Need to calculate {len(beams_to_calculate)} new beams: {beams_to_calculate}")
    
    return existing_contours, existing_info, beams_to_calculate

def merge_and_save_contours(existing_contours, existing_info, new_contours, new_info):
    """Merge existing and new contours, then save"""
    
    # Combine the lists
    all_contours = existing_contours + new_contours
    all_info = existing_info + new_info
    
    # Save the merged data
    save_contours(all_contours, all_info)
    save_contours(all_contours, all_info, format='json')
    
    print(f"Saved {len(all_contours)} total beams ({len(existing_contours)} existing + {len(new_contours)} new)")
    
    return all_contours, all_info

def interactive_beam_selection(all_contours, beam_info):
    """
    Interactive tool to manually select/deselect beams with scrollable checkboxes
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    import matplotlib.patches as patches
    
    fig = plt.figure(figsize=(20, 10))
    
    # Create subplots with custom positioning
    ax_plot = plt.subplot2grid((1, 4), (0, 0), colspan=3)  # Plot takes 3/4 of width
    ax_scroll = plt.subplot2grid((1, 4), (0, 3))           # Checkboxes take 1/4
    
    # Plot all beams with reduced alpha for visibility
    beam_artists = []
    beam_labels = []
    beam_text_artists = []  # ADD THIS to track text labels
    
    for i, (contour, info) in enumerate(zip(all_contours, beam_info)):
        color = info['color']
        artist = ax_plot.fill(contour[:, 0], contour[:, 1], alpha=0.2, color=color, 
                             label=f"B{info['beam_idx']+1}")
        beam_artists.append(artist[0])
        beam_labels.append(f"B{info['beam_idx']+1}")
        
        # ADD BEAM NAME LABELS TO THE PLOT
        text_artist = ax_plot.text(info['phi'], info['theta'], f"{info['beam_idx']}", 
                                  fontsize=6, ha='center', va='center', weight='bold',
                                  color='black',
                                  bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                                           alpha=0.8, edgecolor='black'))
        beam_text_artists.append(text_artist)  # Track the text labels
    
    ax_plot.set_title(f"All Beams ({len(all_contours)} total) - Use scroll wheel in checkbox area")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_xlabel('Phi [degrees]')
    ax_plot.set_ylabel('Theta [degrees]')
    
    # Scrollable checkbox implementation
    class ScrollableCheckButtons:
        def __init__(self, ax, labels, actives, max_visible=30):
            self.ax = ax
            self.labels = labels
            self.actives = actives[:]
            self.max_visible = max_visible
            self.scroll_pos = 0
            self.checkboxes = None
            
            # Clear the axis
            self.ax.clear()
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis('off')
            
            # Add scroll instructions
            self.ax.text(0.5, 0.98, 'Scroll wheel to navigate\nClick to toggle beams', 
                        ha='center', va='top', transform=self.ax.transAxes,
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
            
            self.update_checkboxes()
            
        def update_checkboxes(self):
            # Clear previous checkboxes
            if self.checkboxes is not None:
                self.checkboxes.disconnect_events()
            
            # Calculate visible range
            start_idx = max(0, self.scroll_pos)
            end_idx = min(len(self.labels), start_idx + self.max_visible)
            
            if start_idx >= end_idx:
                return
                
            visible_labels = self.labels[start_idx:end_idx]
            visible_actives = self.actives[start_idx:end_idx]
            
            # Create checkbox area (leave space at top for instructions)
            checkbox_height = 0.9  # Use 90% of height for checkboxes
            checkbox_top = 0.9     # Start at 90% from bottom
            
            # Calculate position for checkboxes
            rect_height = checkbox_height / len(visible_labels) if visible_labels else 0.1
            
            # Create new checkboxes
            checkbox_ax = plt.axes([0.76, 0.1, 0.2, checkbox_height])  # Positioned on right side
            self.checkboxes = CheckButtons(checkbox_ax, visible_labels, visible_actives)
            
            # Connect callback
            self.checkboxes.on_clicked(self.on_checkbox_clicked)
            
            # Add scroll position indicator
            total_pages = (len(self.labels) + self.max_visible - 1) // self.max_visible
            current_page = (self.scroll_pos // self.max_visible) + 1
            self.ax.text(0.5, 0.02, f'Page {current_page}/{total_pages} (showing {start_idx+1}-{end_idx})', 
                        ha='center', va='bottom', transform=self.ax.transAxes,
                        fontsize=9)
            
        def on_checkbox_clicked(self, label):
            # Find the actual index in the full list
            try:
                visible_idx = [self.labels[i] for i in range(self.scroll_pos, 
                              min(len(self.labels), self.scroll_pos + self.max_visible))].index(label)
                actual_idx = self.scroll_pos + visible_idx
                
                # Toggle the beam visibility
                self.actives[actual_idx] = not self.actives[actual_idx]
                beam_artists[actual_idx].set_visible(self.actives[actual_idx])
                beam_text_artists[actual_idx].set_visible(self.actives[actual_idx])  # TOGGLE TEXT TOO
                
                # Update alpha based on selection
                if self.actives[actual_idx]:
                    beam_artists[actual_idx].set_alpha(0.4)
                    beam_artists[actual_idx].set_edgecolor('red')
                    beam_artists[actual_idx].set_linewidth(1.5)
                    # Make text more prominent when selected
                    beam_text_artists[actual_idx].set_bbox(dict(boxstyle="round,pad=0.1", 
                                                               facecolor='yellow', alpha=0.9, 
                                                               edgecolor='red'))
                else:
                    beam_artists[actual_idx].set_alpha(0.1)
                    beam_artists[actual_idx].set_edgecolor(beam_info[actual_idx]['color'])
                    beam_artists[actual_idx].set_linewidth(0.5)
                    # Reset text appearance when deselected
                    beam_text_artists[actual_idx].set_bbox(dict(boxstyle="round,pad=0.1", 
                                                               facecolor='white', alpha=0.8, 
                                                               edgecolor='black'))
                
                plt.draw()
                
                # Update counter
                selected_count = sum(self.actives)
                ax_plot.set_title(f"Beams: {selected_count}/{len(all_contours)} selected - Use scroll wheel in checkbox area")
                
            except ValueError:
                pass
                
        def scroll(self, direction):
            old_pos = self.scroll_pos
            
            if direction > 0:  # Scroll up
                self.scroll_pos = max(0, self.scroll_pos - 5)
            else:  # Scroll down
                max_scroll = max(0, len(self.labels) - self.max_visible)
                self.scroll_pos = min(max_scroll, self.scroll_pos + 5)
            
            if self.scroll_pos != old_pos:
                self.update_checkboxes()
                plt.draw()
    
    # Create scrollable checkboxes
    scrollable_checks = ScrollableCheckButtons(ax_scroll, beam_labels, [True] * len(beam_labels))
    
    # Scroll wheel event handler
    def on_scroll(event):
        if event.inaxes == ax_scroll or event.inaxes == scrollable_checks.checkboxes.ax:
            scrollable_checks.scroll(event.step)
    
    # Connect scroll event
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Key event handler for additional navigation
    def on_key(event):
        if event.key == 'up':
            scrollable_checks.scroll(1)
        elif event.key == 'down':
            scrollable_checks.scroll(-1)
        elif event.key == 'home':
            scrollable_checks.scroll_pos = 0
            scrollable_checks.update_checkboxes()
            plt.draw()
        elif event.key == 'end':
            max_scroll = max(0, len(beam_labels) - scrollable_checks.max_visible)
            scrollable_checks.scroll_pos = max_scroll
            scrollable_checks.update_checkboxes()
            plt.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("Controls:")
    print("- Scroll wheel: Navigate through beam list")
    print("- Click checkboxes: Toggle beam visibility")
    print("- Arrow keys: Navigate up/down")
    print("- Home/End keys: Go to start/end of list")
    print("- Close window when finished selecting")
    
    plt.tight_layout()
    plt.show()
    
    # Return selected beams
    selected_indices = [i for i, active in enumerate(scrollable_checks.actives) if active]
    selected_contours = [all_contours[i] for i in selected_indices]
    selected_info = [beam_info[i] for i in selected_indices]
    
    return selected_contours, selected_info, selected_indices

def calculate_beams(phi_values, theta_values, beam_indices_to_calculate):
    """Calculate only the specified beam indices (discrete hardware delays)."""
    angular_res = 1  # (kept as in existing version)
    lp_freq, lp_mult, _ = filters.Shannon_Whitaker(fs=4e9, plot=False)
    lp_gain = 10**(lp_mult/20)
    eplane  = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane  = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
    trigger_sectors_phi = [0, 1, 2, 3]  # 0-indexed antenna list
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff=2)

    span_phi = CALC_CFG['beam_span_phi']
    span_theta = CALC_CFG['beam_span_theta']
    res_deg = CALC_CFG['beam_res']
    radial_limit = CALC_CFG['radial_limit']

    new_contours = []
    new_info = []

    for beam_idx in beam_indices_to_calculate:
        phi = phi_values[beam_idx]; theta = theta_values[beam_idx]

        # Optional radial filter (center at 90,0)
        if radial_limit is not None:
            dr = ((phi - 90.0)**2 + (theta - 0.0)**2)**0.5
            if dr > radial_limit:
                continue

        ok, beam_color = should_process_beam(phi, theta, beam_idx)
        if not ok:
            continue
        print(f'Processing beam {beam_idx+1}: φ={phi:.2f} θ={theta:.2f}')

        pattern = compute_beam_pattern(
            phi, theta,
            span_phi=span_phi, span_theta=span_theta,
            res_deg=res_deg,
            window=256, step=128,
            impulse=impulse, eplane=eplane, hplane=hplane,
            trigger_sectors_phi=trigger_sectors_phi, lp_gain=lp_gain,
            ringmask=(1,1,1,1),
            steering_mode='vary_source',
            delay_quantization='discrete'
        )

        PHI = pattern['PHI']; TH = pattern['TH']; heatmap_db = pattern['heatmap_db']

        if PHI.shape[0] < 2 or PHI.shape[1] < 2:
            print("  Span too small for contour (need >=2x2). Increase --beam-span-phi/theta.")
            continue


        threshold = -3.0
        print(f"  Beam {beam_idx+1}: threshold={threshold:.1f} dB for main-lobe mask")

        if heatmap_db.size == 0:
            print("  Empty heatmap (scan widths 0). Skipping contour.")
            continue

        main_lobe_mask = heatmap_db > threshold
        if not numpy.any(main_lobe_mask):
            print("  No samples above threshold; skipping.")
            continue

        # Label & keep region containing max
        labeled = label(main_lobe_mask)
        max_idx = numpy.unravel_index(numpy.argmax(heatmap_db), heatmap_db.shape)
        main_label = labeled[max_idx]
        core = (labeled == main_label)
        from scipy.ndimage import binary_opening, binary_closing
        core = binary_closing(core, structure=numpy.ones((3,3)))
        core = binary_opening(core, structure=numpy.ones((2,2)))

        core_heatmap = numpy.where(core, heatmap_db, -numpy.inf)
        contour_level = -1.0
        try:
            tmp_fig, tmp_ax = plt.subplots()
            cs = tmp_ax.contour(PHI, TH, core_heatmap, levels=[contour_level])
            plt.close(tmp_fig)
            if cs.collections:
                paths = cs.collections[0].get_paths()
                if paths:
                    # pick path closest to beam center if multiple
                    if len(paths) > 1:
                        best = None; best_dist = 1e9
                        for pth in paths:
                            verts = pth.vertices
                            if len(verts)==0: continue
                            d = numpy.mean(numpy.hypot(verts[:,0]-phi, verts[:,1]-theta))
                            if d < best_dist:
                                best_dist = d; best = pth
                        contour_path = best
                    else:
                        contour_path = paths[0]
                    if contour_path is not None:
                        contour_coords = contour_path.vertices
                        new_contours.append(contour_coords)
                        new_info.append({'phi': phi, 'theta': theta,
                                         'beam_idx': beam_idx, 'color': beam_color})
                        print(f"  -1 dB contour points: {len(contour_coords)}")
                else:
                    print("  No contour paths found.")
            else:
                print("  No contour collection produced.")
        except Exception as e:
            print(f"  Contour extraction error: {e}")

    return new_contours, new_info


# ============================================================================
# SPRINT 6: BEAM OPTIMIZATION FOR DUAL-POL
# ============================================================================

def generate_beam_candidates(theta_min=-90, theta_max=-45, phi_min=0, phi_max=90, 
                            resolution=5):
    """
    Generate systematic grid of candidate beam directions.
    
    Uses actual sky coordinates - no coordinate transformations.
    
    Parameters:
    -----------
    theta_min, theta_max : float
        Elevation range (degrees). -90 = nadir, 0 = horizon
    phi_min, phi_max : float
        Azimuth range (degrees)
    resolution : float
        Angular spacing (degrees)
    
    Returns:
    --------
    candidates : list of (phi, theta) tuples
    """
    phi_vals = numpy.arange(phi_min, phi_max + 0.5*resolution, resolution)
    theta_vals = numpy.arange(theta_min, theta_max + 0.5*resolution, resolution)
    
    candidates = []
    for phi in phi_vals:
        for theta in theta_vals:
            candidates.append((float(phi), float(theta)))
    
    print(f"Generated {len(candidates)} candidate beams:")
    print(f"  Theta: [{theta_min}°, {theta_max}°] in {resolution}° steps")
    print(f"  Phi: [{phi_min}°, {phi_max}°] in {resolution}° steps")
    print(f"  Grid: {len(theta_vals)} × {len(phi_vals)} = {len(candidates)} beams")
    
    return candidates


def compute_beam_pattern_dualpol(phi_c, th_c, span_phi=45, span_theta=45, res_deg=5,
                                 window=160, step=40, impulse=None, 
                                 beam_patterns_h=None, beam_patterns_v=None,
                                 trigger_sectors_phi=None, psi=45.0,
                                 ringmask=(1,1,1,1), output_polarization='total',
                                 return_all_pols=False):
    """
    
    Uses actual waveforms and coherent_sum.py
    
    Parameters:
    -----------
    phi_c, th_c : float
        Center direction for beam steering
    span_phi, span_theta : float
        Angular span to scan around each beam (degrees)
    res_deg : float
        Angular resolution (degrees)
    window, step : int
        Power sum window parameters
    impulse : Waveform
        Impulse response
    beam_patterns_h, beam_patterns_v : tuples
        (eplane, hplane) for each polarization
    trigger_sectors_phi : list
        Antenna indices
    psi : float
        Polarization angle (degrees)
    ringmask : tuple
        Antenna mask
    output_polarization : str
        'lhcp', 'rhcp', or 'total' (default: 'total')
    return_all_pols : bool
        If True, compute and return all three polarizations (LHCP, RHCP, Total)
        regardless of output_polarization setting
    
    Returns:
    --------
    dict with PHI, TH, heatmap, heatmap_db, delays_used
    If return_all_pols=True, also includes heatmap_lhcp, heatmap_rhcp, heatmap_total
    and corresponding _db versions
    """
    if trigger_sectors_phi is None:
        trigger_sectors_phi = [0, 1, 2, 3]
    
    ringmask = list(ringmask)
    # Define scan range
    # For visualization (return_all_pols=True), use symmetric ranges around beam center
    # For optimization, clip to beam pattern data range [0, 90] in phi
    if return_all_pols:
        # Visualization mode: symmetric scan, let payload functions handle wrapping
        phi_min = phi_c - span_phi
        phi_max = phi_c + span_phi
    else:
        # Optimization mode: clip to beam pattern data range
        phi_min = max(0, phi_c - span_phi)
        phi_max = min(90, phi_c + span_phi)
    
    th_min = max(-90, th_c - span_theta)
    th_max = min(90, th_c + span_theta)
    
    phiscan = numpy.arange(phi_min, phi_max + 1e-9, res_deg)
    thscan = numpy.arange(th_min, th_max + 1e-9, res_deg)
    PHI, TH = numpy.meshgrid(phiscan, thscan)
    
    # Initialize heatmaps
    if return_all_pols:
        heatmap_lhcp = numpy.zeros_like(PHI, dtype=float)
        heatmap_rhcp = numpy.zeros_like(PHI, dtype=float)
        heatmap_total = numpy.zeros_like(PHI, dtype=float)
    else:
        heatmap = numpy.zeros_like(PHI, dtype=float)
    
    # Steering delays for beam center
    delays_center = payload.getRemappedDelays(phi_c, th_c, trigger_sectors_phi)
    delays_center_q = numpy.round(delays_center / aso_geometry.ritc_sample_step) * aso_geometry.ritc_sample_step
    
    print(f"Computing dual-pol beam pattern at (φ={phi_c}°, θ={th_c}°)")
    print(f"  Scan range: φ=[{phi_min}°, {phi_max}°], θ=[{th_min}°, {th_max}°]")
    print(f"  Resolution: {res_deg}° ({len(thscan)}×{len(phiscan)} = {len(thscan)*len(phiscan)} points)")
    
    for i, th in enumerate(thscan):
        print(f"  Row {i+1}/{len(thscan)}: θ={th:.1f}°...", end='', flush=True)
        for j, ph in enumerate(phiscan):
            # Generate dual-pol waveforms for this sky direction (returns 8-channel array)
            waveforms, timebase, _ = payload.getPayloadWaveforms_dualpol(
                ph, th, impulse, 
                beam_patterns_h, beam_patterns_v,
                antennas=trigger_sectors_phi,
                snr=1e10,  # No noise
                psi=psi
            )
            
            # Split into H-pol (channels 0-3) and V-pol (channels 4-7)
            waveforms_h = waveforms[:4]
            waveforms_v = waveforms[4:]
            
            # Apply coherent beamforming (converts to LHCP/RHCP)
            # Use hardware-correct order: digitize → filter → sum → circular
            lhcp_wf, rhcp_wf, tb = sum.coherentSum_dualpol(
                waveforms_h, waveforms_v, timebase, delays_center_q,
                downsample=False, channel_mask=ringmask, output='circular',
                apply_filter=True, digitize_first=True
            )
            
            # Compute sliding-window power
            if return_all_pols:
                # Compute all three polarizations
                power_lhcp, _ = sum.powerSum(lhcp_wf, window=window, step=step)
                power_rhcp, _ = sum.powerSum(rhcp_wf, window=window, step=step)
                heatmap_lhcp[i, j] = numpy.max(power_lhcp)
                heatmap_rhcp[i, j] = numpy.max(power_rhcp)
                heatmap_total[i, j] = numpy.max(power_lhcp + power_rhcp)
            else:
                # Compute requested polarization only
                if output_polarization == 'lhcp':
                    power, _ = sum.powerSum(lhcp_wf, window=window, step=step)
                elif output_polarization == 'rhcp':
                    power, _ = sum.powerSum(rhcp_wf, window=window, step=step)
                else:  # 'total'
                    power_lhcp, _ = sum.powerSum(lhcp_wf, window=window, step=step)
                    power_rhcp, _ = sum.powerSum(rhcp_wf, window=window, step=step)
                    power = power_lhcp + power_rhcp
                
                heatmap[i, j] = numpy.max(power)
        
        if return_all_pols:
            print(f" done (LHCP max={numpy.max(heatmap_lhcp[i, :]):.2e}, RHCP max={numpy.max(heatmap_rhcp[i, :]):.2e})")
        else:
            print(f" done (max={numpy.max(heatmap[i, :]):.2e})")
    
    # Convert to dB
    eps = 1e-12
    if return_all_pols:
        heatmap_lhcp_db = 10.0 * numpy.log10(numpy.maximum(heatmap_lhcp, eps) / numpy.max(heatmap_total))
        heatmap_rhcp_db = 10.0 * numpy.log10(numpy.maximum(heatmap_rhcp, eps) / numpy.max(heatmap_total))
        heatmap_total_db = 10.0 * numpy.log10(numpy.maximum(heatmap_total, eps) / numpy.max(heatmap_total))
        
        return {
            'PHI': PHI, 'TH': TH,
            'heatmap_lhcp': heatmap_lhcp,
            'heatmap_rhcp': heatmap_rhcp,
            'heatmap_total': heatmap_total,
            'heatmap_lhcp_db': heatmap_lhcp_db,
            'heatmap_rhcp_db': heatmap_rhcp_db,
            'heatmap_total_db': heatmap_total_db,
            'delays_used': delays_center_q,
            'beam_phi': phi_c,
            'beam_theta': th_c,
            'psi': psi
        }
    else:
        heatmap_db = 10.0 * numpy.log10(numpy.maximum(heatmap, eps) / numpy.max(heatmap))
        
        return {
            'PHI': PHI, 'TH': TH,
            'heatmap': heatmap,
            'heatmap_db': heatmap_db,
            'delays_used': delays_center_q,
            'beam_phi': phi_c,
            'beam_theta': th_c,
            'psi': psi,
            'output_pol': output_polarization
        }


def compute_coverage_mask(beam_result, threshold_db=-3.0):
    """
    Determine which sky directions are covered by this beam.
    
    Parameters:
    -----------
    beam_result : dict
        Output from compute_beam_pattern_dualpol()
    threshold_db : float
        Coverage threshold in dB (default: -3 dB)
    
    Returns:
    --------
    coverage_mask : boolean array
        True where beam exceeds threshold
    """
    coverage_mask = beam_result['heatmap_db'] >= threshold_db
    coverage_fraction = numpy.sum(coverage_mask) / coverage_mask.size
    
    print(f"  Beam at (φ={beam_result['beam_phi']}°, θ={beam_result['beam_theta']}°): "
          f"{coverage_fraction*100:.1f}% coverage at {threshold_db} dB")
    
    return coverage_mask


def compute_all_coverage(beam_candidates, span_phi=45, span_theta=45, resolution=5,
                        window=160, step=40, threshold_db=-3.0, psi=45.0,
                        phi_range=(-90, 90), theta_range=(-90, 90),
                        save_progress=True):
    """
    Compute coverage masks for all candidate beams using full dual-pol processing.
    
    This is computationally expensive - uses actual waveforms and beamforming.
    All beams are interpolated onto a common sky grid for consistent coverage analysis.
    
    Parameters:
    -----------
    beam_candidates : list of (phi, theta)
        Candidate beam directions
    span_phi, span_theta : float
        How far to scan around each beam (degrees)
    resolution : float
        Angular resolution (degrees)
    window, step : int
        Power sum parameters
    threshold_db : float
        Coverage threshold (dB)
    psi : float
        Polarization angle (degrees)
    phi_range, theta_range : tuple
        (min, max) for common sky grid
    save_progress : bool
        Save after each beam (default: True)
    
    Returns:
    --------
    beam_results : list of dicts
        Beam pattern data for each candidate
    coverage_masks : list of boolean arrays
        Coverage mask for each candidate (on common grid)
    sky_grid : tuple (PHI, TH)
        Common sky grid for all beams
    """
    from scipy.interpolate import RegularGridInterpolator
    
    print("="*70)
    print("COMPUTING BEAM COVERAGE (DUAL-POL TIME-DOMAIN)")
    print("="*70)
    print(f"Number of beams: {len(beam_candidates)}")
    print(f"Scan span: ±{span_phi}° phi, ±{span_theta}° theta")
    print(f"Resolution: {resolution}°")
    print(f"Threshold: {threshold_db} dB")
    print(f"Polarization: ψ={psi}°")
    print(f"Common sky grid: φ={phi_range}, θ={theta_range}")
    print("="*70)
    
    # Create common sky grid
    phi_common = numpy.arange(phi_range[0], phi_range[1] + 1e-9, resolution)
    theta_common = numpy.arange(theta_range[0], theta_range[1] + 1e-9, resolution)
    PHI_common, TH_common = numpy.meshgrid(phi_common, theta_common)
    sky_grid = (PHI_common, TH_common)
    
    # Load beam patterns and impulse once
    eplane_h = payload.beamPattern(plot=False, which_plane='E', which_pol='H')
    hplane_h = payload.beamPattern(plot=False, which_plane='H', which_pol='H')
    eplane_v = payload.beamPattern(plot=False, which_plane='E', which_pol='V')
    hplane_v = payload.beamPattern(plot=False, which_plane='H', which_pol='V')
    beam_patterns_h = (eplane_h, hplane_h)
    beam_patterns_v = (eplane_v, hplane_v)
    
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse)
    
    trigger_sectors_phi = [0, 1, 2, 3]
    ringmask = (1, 1, 1, 1)
    
    beam_results = []
    coverage_masks = []
    
    for idx, (phi_c, th_c) in enumerate(beam_candidates):
        print(f"\n[{idx+1}/{len(beam_candidates)}] Computing beam at (φ={phi_c}°, θ={th_c}°)")
        
        result = compute_beam_pattern_dualpol(
            phi_c, th_c, span_phi, span_theta, resolution,
            window, step, impulse, beam_patterns_h, beam_patterns_v,
            trigger_sectors_phi, psi=psi, ringmask=ringmask
        )
        
        # Interpolate onto common grid
        phi_local = result['PHI'][0, :]  # First row
        theta_local = result['TH'][:, 0]  # First column
        heatmap_db_local = result['heatmap_db']
        
        # Create interpolator
        interp = RegularGridInterpolator(
            (theta_local, phi_local), heatmap_db_local,
            method='linear', bounds_error=False, fill_value=-numpy.inf
        )
        
        # Interpolate to common grid
        points = numpy.stack([TH_common.ravel(), PHI_common.ravel()], axis=-1)
        heatmap_db_common = interp(points).reshape(PHI_common.shape)
        
        # Compute coverage mask on common grid
        mask = heatmap_db_common >= threshold_db
        coverage_fraction = numpy.sum(mask) / mask.size
        
        print(f"  Beam at (φ={phi_c}°, θ={th_c}°): "
              f"{coverage_fraction*100:.1f}% coverage at {threshold_db} dB")
        
        beam_results.append(result)
        coverage_masks.append(mask)
        
        # Save progress
        if save_progress:
            os.makedirs('plots', exist_ok=True)
            save_data = {
                'beam_results': beam_results,
                'coverage_masks': coverage_masks,
                'sky_grid': sky_grid,
                'candidates': beam_candidates[:idx+1],
                'threshold_db': threshold_db,
                'psi': psi
            }
            numpy.save('plots/beam_coverage_progress.npy', save_data)
    
    print("\n" + "="*70)
    print("COVERAGE COMPUTATION COMPLETE")
    print("="*70)
    
    return beam_results, coverage_masks, sky_grid


def select_optimal_beams(coverage_masks, beam_candidates, target_coverage=0.95):
    """
    Select minimum set of beams to achieve target sky coverage.
    
    Uses greedy algorithm: repeatedly select beam that covers most uncovered sky.
    
    Parameters:
    -----------
    coverage_masks : list of boolean arrays
        Coverage mask for each candidate beam
    beam_candidates : list of (phi, theta)
        Candidate beam directions
    target_coverage : float
        Desired fraction of sky coverage (default: 0.95 = 95%)
    
    Returns:
    --------
    selected_indices : list of int
        Indices of selected beams
    selected_beams : list of (phi, theta)
        Directions of selected beams
    coverage_history : list of float
        Coverage fraction after each beam added
    """
    print("\n" + "="*70)
    print("BEAM SELECTION OPTIMIZATION")
    print("="*70)
    print(f"Target coverage: {target_coverage*100:.1f}%")
    print(f"Total candidates: {len(beam_candidates)}")
    
    n_beams = len(coverage_masks)
    total_pixels = coverage_masks[0].size
    
    # Initialize: no coverage
    total_coverage = numpy.zeros_like(coverage_masks[0], dtype=bool)
    remaining_beams = set(range(n_beams))
    selected_indices = []
    coverage_history = []
    
    while True:
        current_coverage_fraction = numpy.sum(total_coverage) / total_pixels
        coverage_history.append(current_coverage_fraction)
        
        print(f"\nIteration {len(selected_indices)+1}:")
        print(f"  Current coverage: {current_coverage_fraction*100:.2f}%")
        
        if current_coverage_fraction >= target_coverage:
            print(f"  ✓ Target coverage achieved!")
            break
        
        if len(remaining_beams) == 0:
            print(f"  ⚠ No more beams available (coverage: {current_coverage_fraction*100:.2f}%)")
            break
        
        # Find beam that covers most new sky
        best_beam = None
        best_new_coverage = 0
        
        for beam_idx in remaining_beams:
            new_coverage = numpy.sum(coverage_masks[beam_idx] & ~total_coverage)
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_beam = beam_idx
        
        # Check if any beam adds coverage
        if best_beam is None:
            print(f"  ⚠ No remaining beams add coverage (stuck at {current_coverage_fraction*100:.2f}%)")
            break
        
        # Add best beam
        selected_indices.append(best_beam)
        total_coverage |= coverage_masks[best_beam]
        remaining_beams.remove(best_beam)
        
        phi, theta = beam_candidates[best_beam]
        new_fraction = best_new_coverage / total_pixels
        print(f"  Selected beam {best_beam}: (φ={phi}°, θ={theta}°)")
        print(f"  New coverage: {new_fraction*100:.2f}% ({best_new_coverage} pixels)")
        print(f"  Total beams: {len(selected_indices)}")
    
    selected_beams = [beam_candidates[i] for i in selected_indices]
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Selected {len(selected_beams)} beams for {coverage_history[-1]*100:.2f}% coverage")
    print("\nSelected beam directions:")
    for i, (phi, theta) in enumerate(selected_beams):
        print(f"  {i+1}. φ={phi:6.1f}°, θ={theta:6.1f}°")
    
    return selected_indices, selected_beams, coverage_history


def plot_coverage_analysis(beam_results, coverage_masks, selected_indices, 
                           sky_grid, coverage_history, threshold_db=-3.0,
                           filename='beam_coverage_optimization.png'):
    """
    Visualize beam coverage optimization results.
    
    Creates multi-panel plot showing:
    - Individual beam patterns
    - Combined coverage map
    - Coverage vs number of beams
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    PHI, TH = sky_grid
    n_selected = len(selected_indices)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Combined coverage map
    ax_combined = fig.add_subplot(gs[0, :])
    total_coverage = numpy.zeros_like(coverage_masks[0], dtype=int)
    for idx in selected_indices:
        total_coverage += coverage_masks[idx].astype(int)
        im = ax_combined.contourf(PHI, TH, total_coverage, 
                             levels=numpy.arange(0, numpy.max(total_coverage)+2),
                             cmap='viridis')
    plt.colorbar(im, ax=ax_combined, label='Number of beams covering')
    
    # Mark beam centers
    for i, idx in enumerate(selected_indices):
        phi_b = beam_results[idx]['beam_phi']
        th_b = beam_results[idx]['beam_theta']
        ax_combined.plot(phi_b, th_b, 'r*', markersize=10, markeredgecolor='white', markeredgewidth=1)
        ax_combined.text(phi_b, th_b, f'{i+1}', color='white', fontsize=8, 
                        ha='center', va='center', fontweight='bold')
    
    ax_combined.set_xlabel('φ (azimuth) [deg]', fontsize=12)
    ax_combined.set_ylabel('θ (elevation) [deg]', fontsize=12)
    ax_combined.set_title(f'Combined Coverage: {n_selected} Beams ({threshold_db} dB threshold)', 
                         fontsize=13, fontweight='bold')
    ax_combined.grid(True, alpha=0.3)
    
    # Coverage vs number of beams
    ax_history = fig.add_subplot(gs[1, :])
    ax_history.plot(range(1, len(coverage_history)+1), 
                   numpy.array(coverage_history) * 100, 'o-', linewidth=2)
    ax_history.axhline(95, color='red', linestyle='--', label='95% target')
    ax_history.set_xlabel('Number of Beams', fontsize=12)
    ax_history.set_ylabel('Sky Coverage [%]', fontsize=12)
    ax_history.set_title('Coverage vs Beams', fontsize=13, fontweight='bold')
    ax_history.grid(True, alpha=0.3)
    ax_history.legend()
    ax_history.set_ylim(0, 105)
    
    plt.suptitle('Dual-Pol Beam Coverage Optimization (Time-Domain)', 
                fontsize=15, fontweight='bold')
    
    os.makedirs('plots', exist_ok=True)
    filepath = os.path.join('plots', filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filepath}")
    plt.close()
    
    return fig


def plot_beam_pattern_dualpol(result, output_path='plots/beam_pattern_dualpol.png',
                              contour_db=-3.0):
    """
    Plot traditional beam pattern heatmaps for dual-pol (LHCP/RHCP/Total).
    
    Creates a 3-panel figure showing oval-shaped beam profiles like the old
    coherent_sum.py single-pol visualizations.
    
    Parameters:
    -----------
    result : dict
        Output from compute_beam_pattern_dualpol() with return_all_pols=True
        Must contain: PHI, TH, heatmap_lhcp_db, heatmap_rhcp_db, heatmap_total_db
    output_path : str
        Where to save the figure
    contour_db : float
        Contour level in dB (default: -3.0 for half-power beamwidth)
    """
    import matplotlib.pyplot as plt
    
    PHI = result['PHI']
    TH = result['TH']
    lhcp_db = result['heatmap_lhcp_db']
    rhcp_db = result['heatmap_rhcp_db']
    total_db = result['heatmap_total_db']
    
    phi_c = result['beam_phi']
    th_c = result['beam_theta']
    psi = result['psi']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common colormap settings
    vmin, vmax = -30, 0
    cmap = 'viridis'
    
    # Panel 1: LHCP
    ax = axes[0]
    pcm = ax.pcolormesh(PHI, TH, lhcp_db, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    cs = ax.contour(PHI, TH, lhcp_db, levels=[contour_db], colors='red', linewidths=2)
    ax.plot(phi_c, th_c, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    ax.set_xlabel('Azimuth φ (deg)', fontsize=12)
    ax.set_ylabel('Elevation θ (deg)', fontsize=12)
    ax.set_title(f'LHCP Beam Pattern\nψ={psi}°', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(pcm, ax=ax, label='Power (dB)')
    
    # Panel 2: RHCP
    ax = axes[1]
    pcm = ax.pcolormesh(PHI, TH, rhcp_db, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    cs = ax.contour(PHI, TH, rhcp_db, levels=[contour_db], colors='red', linewidths=2)
    ax.plot(phi_c, th_c, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    ax.set_xlabel('Azimuth φ (deg)', fontsize=12)
    ax.set_ylabel('Elevation θ (deg)', fontsize=12)
    ax.set_title(f'RHCP Beam Pattern\nψ={psi}°', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(pcm, ax=ax, label='Power (dB)')
    
    # Panel 3: Total
    ax = axes[2]
    pcm = ax.pcolormesh(PHI, TH, total_db, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    cs = ax.contour(PHI, TH, total_db, levels=[contour_db], colors='red', linewidths=2)
    ax.plot(phi_c, th_c, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    ax.set_xlabel('Azimuth φ (deg)', fontsize=12)
    ax.set_ylabel('Elevation θ (deg)', fontsize=12)
    ax.set_title(f'Total (LHCP+RHCP) Beam Pattern\nψ={psi}°', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(pcm, ax=ax, label='Power (dB)')
    
    # Overall title
    fig.suptitle(f'Dual-Pol Beam Pattern: φ={phi_c}°, θ={th_c}° | Contour: {contour_db} dB',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBeam pattern visualization saved to {output_path}")


if __name__=='__main__':
    beam_settings = {}

    # ---------------- Argument Parsing (replaces old sys.argv checks) ----------------
    parser = argparse.ArgumentParser(description="Beam coverage / pruning workflow")
    parser.add_argument('--load', action='store_true', help='Incrementally load existing beams and compute missing')
    parser.add_argument('--force', action='store_true', help='Force recalculation of (current phase) beams')
    parser.add_argument('--clean', action='store_true', help='Delete saved beam contour files before running')
    parser.add_argument('--quadrant-only', action='store_true',
                        help='Phase 1: compute & prune only upper-right (UR) quadrant (φ>=90, θ>=0)')
    parser.add_argument('--mirror-from', type=str,
                        help='Phase 2: path to previously pruned UR selection .pkl (e.g. plots/selected_beam_contours.pkl) to mirror to full square')
    parser.add_argument('--half-span', type=int, default=45,
                        help='Half span (deg) around (90,0) defining ± span square (default 45)')
    parser.add_argument('--step', type=int, default=1,
                        help='Grid step (deg) for generating quadrant points (default 1)')
    parser.add_argument('--load-file', type=str,
                        help='Load a contours file (.pkl or .json) and skip computation')
    parser.add_argument('--select', action='store_true',
                        help='After --load-file, open interactive selector and save a new pruned set')
    parser.add_argument('--beam-span-phi', type=float, default=3.0,
                        help='Half-span in phi (deg) for per-beam pattern evaluation')
    parser.add_argument('--beam-span-theta', type=float, default=3.0,
                        help='Half-span in theta (deg) for per-beam pattern evaluation')
    parser.add_argument('--beam-res', type=float, default=0.25,
                        help='Angular resolution (deg) inside each beam pattern map')
    parser.add_argument('--radius', type=float, default=None,
                        help='Optional radial limit (deg) from (90,0) to keep beam centers (quadrant mode)')
   
    # --- Single-beam visualization (dual-pol by default) ---
    parser.add_argument('--single', action='store_true',
                        help='Single-beam mode: visualize beam pattern with LHCP/RHCP/Total heatmaps')
    parser.add_argument('--single-pol', action='store_true',
                        help='Use legacy single-pol mode (V-pol only) instead of dual-pol')
    parser.add_argument('--phi', type=float, default=0.0, help='Beam center phi (deg)')
    parser.add_argument('--theta', type=float, default=0.0, help='Beam center theta (deg)')
    parser.add_argument('--span-phi', type=float, default=30.0, help='+/- span in phi (deg) around center')
    parser.add_argument('--span-theta', type=float, default=30.0, help='+/- span in theta (deg) around center')
    parser.add_argument('--resolution', type=float, default=1.0, help='Angular resolution (deg)')
    parser.add_argument('--window', type=int, default=256, help='PowerSum window (samples)')
    parser.add_argument('--pstep', type=int, default=128, help='PowerSum step (samples)')
    parser.add_argument('--contour-db', type=float, default=-3.0, help='Contour level in dB (default: -3.0)')
    parser.add_argument('--single-mode', choices=['continuous','discrete','both'], default='discrete',
                        help='Delay model for single-pol mode; "both" shows side-by-side (single-pol only)')
    parser.add_argument('--single-steering', choices=['vary_source','vary_steer'], default='vary_source',
                        help='vary_source=fixed steering (beam pattern); vary_steer=fixed source, scan steering (single-pol only)')

    
    # --- Sprint 6: Dual-pol beam optimization ---
    parser.add_argument('--optimize-beams', action='store_true',
                        help='Sprint 6: Run dual-pol beam optimization workflow')
    parser.add_argument('--theta-min', type=float, default=-90,
                        help='Minimum elevation angle for candidates (deg, default: -90 = nadir)')
    parser.add_argument('--theta-max', type=float, default=-45,
                        help='Maximum elevation angle for candidates (deg, default: -45)')
    parser.add_argument('--phi-min', type=float, default=0,
                        help='Minimum azimuth angle for candidates (deg, default: 0)')
    parser.add_argument('--phi-max', type=float, default=90,
                        help='Maximum azimuth angle for candidates (deg, default: 90)')
    parser.add_argument('--beam-resolution', type=float, default=10,
                        help='Angular spacing between candidate beams (deg, default: 10)')
    parser.add_argument('--scan-span-phi', type=float, default=20,
                        help='Scan span around each beam center in phi (deg, default: 20)')
    parser.add_argument('--scan-span-theta', type=float, default=20,
                        help='Scan span around each beam center in theta (deg, default: 20)')
    parser.add_argument('--scan-resolution', type=float, default=5,
                        help='Angular resolution for beam pattern computation (deg, default: 5)')
    parser.add_argument('--threshold-db', type=float, default=-3.0,
                        help='Coverage threshold in dB (default: -3.0)')
    parser.add_argument('--target-coverage', type=float, default=0.95,
                        help='Target sky coverage fraction (default: 0.95 = 95%%)')
    parser.add_argument('--psi', type=float, default=45.0,
                        help='Polarization angle in degrees (0=H-pol, 45=balanced, 90=V-pol, default: 45)')
    parser.add_argument('--output-pol', choices=['lhcp', 'rhcp', 'total'], default='total',
                        help='Output polarization for optimization (default: total)')

    args = parser.parse_args()
    
    # ================ SINGLE-BEAM VISUALIZATION MODE ================
    if args.single:
        phi_c = float(args.phi); th_c = float(args.theta)
        dphi = float(args.span_phi); dth = float(args.span_theta)
        angular_res = float(args.resolution)
        window = int(args.window); step = int(args.pstep)

        if args.single_pol:
            # Legacy single-pol mode (V-pol only)
            print("\n" + "="*70)
            print("SINGLE-BEAM SINGLE-POL VISUALIZATION (V-POL)")
            print("="*70)
            print(f"Beam center: φ={phi_c}°, θ={th_c}°")
            print(f"Scan span: φ=±{dphi}°, θ=±{dth}°")
            print(f"Resolution: {angular_res}°")
            print(f"Mode: {args.single_mode}, Steering: {args.single_steering}")
            print(f"Window/Step: {window}/{step} samples")
            print("="*70 + "\n")

            os.makedirs("plots", exist_ok=True)
            lp_freq, lp_mult, _ = filters.Shannon_Whitaker(fs=4e9, plot=False)
            lp_gain = 10**(lp_mult/20)
            eplane  = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
            hplane  = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
            trigger_sectors_phi = [0, 1, 2, 3]  # 0-indexed antenna list
            impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
            impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff=2)

            result_sets = []
            if args.single_mode in ('continuous','both'):
                result_sets.append(
                    compute_beam_pattern(phi_c, th_c, dphi, dth, angular_res,
                                         window, step,
                                         impulse, eplane, hplane,
                                         trigger_sectors_phi, lp_gain,
                                         steering_mode=args.single_steering,
                                         delay_quantization='continuous')
                )
            if args.single_mode in ('discrete','both'):
                result_sets.append(
                    compute_beam_pattern(phi_c, th_c, dphi, dth, angular_res,
                                         window, step,
                                         impulse, eplane, hplane,
                                         trigger_sectors_phi, lp_gain,
                                         steering_mode=args.single_steering,
                                         delay_quantization='discrete')
                )

            # Plot
            ncols = len(result_sets)
            fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 6), sharex=True, sharey=True)
            if ncols == 1:
                axes = [axes]

            for ax, res_dict in zip(axes, result_sets):
                PHI = res_dict['PHI']; TH = res_dict['TH']; Z = res_dict['heatmap_db']
                title = f"{res_dict['mode']} / {res_dict['quantization']}"
                pcm = ax.pcolormesh(PHI, TH, Z, shading='auto', cmap='viridis',
                                    vmin=args.contour_db-6, vmax=0.0)
                fig.colorbar(pcm, ax=ax, label='Coherent power [dB rel. max]')
                cs = ax.contour(PHI, TH, Z, levels=[args.contour_db], colors='red', linewidths=1.5)
                ax.clabel(cs, fmt={args.contour_db: f'{args.contour_db:.1f} dB'}, inline=True, fontsize=8)
                ax.set_xlabel('Phi [deg]')
                ax.set_ylabel('Theta [deg]')
                ax.set_title(title)
                ax.grid(alpha=0.25)

            fig.suptitle(f'Single Beam @ (φ={phi_c:.2f}°, θ={th_c:.2f}°) window={window} step={step}')
            plt.tight_layout()
            out_png = f"plots/single_beam_{phi_c:.2f}_{th_c:.2f}_{args.single_mode}_{args.single_steering}.png".replace('.','p')
            plt.savefig(out_png, dpi=150)
            print(f"Saved: {out_png}")
            plt.show()
            sys.exit(0)
        
        # Dual-pol mode (default): LHCP/RHCP visualization
        print("\n" + "="*70)
        print("SINGLE-BEAM DUAL-POL VISUALIZATION")
        print("="*70)
        print(f"Beam center: φ={phi_c}°, θ={th_c}°")
        print(f"Scan span: φ=±{dphi}°, θ=±{dth}°")
        print(f"Resolution: {angular_res}°")
        print(f"Polarization angle: ψ={args.psi}°")
        print(f"Window/Step: {window}/{step} samples")
        print(f"Contour level: {args.contour_db} dB")
        print("="*70 + "\n")
        
        # Load impulse and beam patterns
        print("Loading impulse and beam patterns...")
        impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
        impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff=2)
        
        eplane_h = payload.beamPattern(plot=False, which_plane='E', which_pol='H')
        hplane_h = payload.beamPattern(plot=False, which_plane='H', which_pol='H')
        eplane_v = payload.beamPattern(plot=False, which_plane='E', which_pol='V')
        hplane_v = payload.beamPattern(plot=False, which_plane='H', which_pol='V')
        beam_patterns_h = (eplane_h, hplane_h)
        beam_patterns_v = (eplane_v, hplane_v)
        print("  Loaded successfully.")
        
        # Compute dual-pol beam pattern
        print(f"\nComputing dual-pol beam pattern...")
        result = compute_beam_pattern_dualpol(
            phi_c=phi_c,
            th_c=th_c,
            span_phi=dphi,
            span_theta=dth,
            res_deg=angular_res,
            window=window,
            step=step,
            impulse=impulse,
            beam_patterns_h=beam_patterns_h,
            beam_patterns_v=beam_patterns_v,
            trigger_sectors_phi=[0, 1, 2, 3],
            psi=args.psi,
            ringmask=(1, 1, 1, 1),
            return_all_pols=True
        )
        
        # Generate visualization
        print(f"\nGenerating visualization...")
        os.makedirs('plots', exist_ok=True)
        output_filename = f'beam_pattern_dualpol_phi{phi_c:.0f}_theta{th_c:.0f}_psi{args.psi:.0f}.png'
        output_path = os.path.join('plots', output_filename)
        plot_beam_pattern_dualpol(
            result,
            output_path=output_path,
            contour_db=args.contour_db
        )
        
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)
        sys.exit(0)

    # ---------------- Sprint 6: Beam Optimization Mode ----------------
    if args.optimize_beams:
        print("\n" + "="*70)
        print("SPRINT 6: DUAL-POL BEAM OPTIMIZATION")
        print("="*70)
        
        # Generate candidate beams
        candidates = generate_beam_candidates(
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            resolution=args.beam_resolution
        )
        
        # Compute coverage for all candidates
        beam_results, coverage_masks, sky_grid = compute_all_coverage(
            candidates,
            span_phi=args.scan_span_phi,
            span_theta=args.scan_span_theta,
            resolution=args.scan_resolution,
            window=args.window,
            step=args.pstep,
            threshold_db=args.threshold_db,
            psi=args.psi,
            phi_range=(args.phi_min, args.phi_max),
            theta_range=(args.theta_min, args.theta_max),
            save_progress=True
        )
        
        # Select optimal beam set
        selected_indices, selected_beams, coverage_history = select_optimal_beams(
            coverage_masks,
            candidates,
            target_coverage=args.target_coverage
        )
        
        # Plot results
        plot_coverage_analysis(
            beam_results,
            coverage_masks,
            selected_indices,
            sky_grid,
            coverage_history,
            threshold_db=args.threshold_db,
            filename='beam_coverage_optimization.png'
        )
        
        # Save optimal beam set
        os.makedirs('plots', exist_ok=True)
        results_file = 'plots/optimal_beams_sprint6.npy'
        numpy.save(results_file, {
            'selected_beams': selected_beams,
            'selected_indices': selected_indices,
            'coverage_history': coverage_history,
            'candidates': candidates,
            'beam_results': beam_results,
            'coverage_masks': coverage_masks,
            'sky_grid': sky_grid,
            'threshold_db': args.threshold_db,
            'psi': args.psi,
            'output_pol': args.output_pol,
            'target_coverage': args.target_coverage
        })
        print(f"\nSaved optimal beam set to: {results_file}")
        
        sys.exit(0)

    beam_span_phi = args.beam_span_phi
    beam_span_theta = args.beam_span_theta
    beam_res = args.beam_res
    radial_limit = args.radius
    CALC_CFG = {
        'beam_span_phi': beam_span_phi,
        'beam_span_theta': beam_span_theta,
        'beam_res': beam_res,
        'radial_limit': radial_limit
    }

    # ---------------- Cleaning ----------------
    if args.clean:
        print("Cleaning saved data...")
        for filename in [
            "beam_contours_data.pkl",
            "beam_contours_data.json",
            "selected_beam_contours.pkl",
            "selected_beam_contours.json",
            "full_square_beams.pkl",
            "full_square_beams.json"
        ]:
            fp = os.path.join("plots", filename)
            if os.path.exists(fp):
                os.remove(fp)
                print("  Deleted", fp)

    # ---------------- Phase 2: Mirror mode ----------------
    if args.mirror_from:
        print(f"[Mirror Phase] Loading selected UR file: {args.mirror_from}")
        sel_contours, sel_info = load_contours(args.mirror_from)
        if sel_contours is None:
            print("Could not load the supplied --mirror-from file. Exiting.")
            sys.exit(1)

        # Use ONLY the selected beams as the UR base (prevent repopulating with full dense grid)
        existing_contours = sel_contours[:]          # clone
        existing_info = sel_info[:]                  # clone

        # Build mirrored coordinate sets
        mirror_dict = build_full_square_from_selected(sel_info, phi_center=90, theta_center=0)
        print("Mirror category counts:")
        for k, v in mirror_dict.items():
            print(f"  {k}: {len(v)}")

        # Flatten to single list of candidate coordinates
        mirrored_coords = flatten_mirror_dict(mirror_dict)


        # Reconstruct current phi/theta arrays from existing info
        phi_values = [inf['phi'] for inf in existing_info]
        theta_values = [inf['theta'] for inf in existing_info]

        # Filter out coordinates already present
        new_coords = dedupe_new_coordinates(existing_info, mirrored_coords)
        print(f"New mirrored coordinates to compute: {len(new_coords)}")

        if new_coords:
            new_contours, new_info, _ = compute_new_from_coordinate_list(new_coords, phi_values, theta_values)
            all_contours, beam_info = merge_and_save_contours(existing_contours, existing_info, new_contours, new_info)
        else:
            all_contours, beam_info = existing_contours, existing_info

        # Manual pruning over full square
        if all_contours:
            print("Starting interactive pruning on full mirrored square...")
            full_selected_contours, full_selected_info, _ = interactive_beam_selection(all_contours, beam_info)
            save_contours(full_selected_contours, full_selected_info, filename="full_square_beams.pkl")
            save_contours(full_selected_contours, full_selected_info, filename="full_square_beams.json")
            print(f"Final pruned full-square beam count: {len(full_selected_contours)}")

            # Optional coverage area
            total_area = 0.0
            for contour in full_selected_contours:
                x = contour[:,0]; y = contour[:,1]
                area = 0.5 * abs(sum(x[i]*y[(i+1)%len(x)] - x[(i+1)%len(x)]*y[i] for i in range(len(x))))
                total_area += area
            print(f"Approx total coverage area (sum of polygons): {total_area:.2f} deg^2")

        sys.exit(0)

    # ---------------- Phase 1: Quadrant-only or standard mode ----------------
    phi_values = []
    theta_values = []

    if args.quadrant_only:
        # Generate ONLY upper-right quadrant relative to (90,0)
        ur_phi, ur_theta = generate_quadrant_grid(90, 0, args.half_span, args.step, quadrant='UR')
        phi_values.extend(ur_phi)
        theta_values.extend(ur_theta)
        print(f"UR quadrant beams generated: {len(phi_values)}")
    else:
        # Original full (currently single quadrant) range you had
        for phi in range(90, 135):
            for theta in range(0, 45):
                phi_values.append(phi)
                theta_values.append(theta)
        print(f"Standard grid beams: {len(phi_values)}")

    # ---------------- Calculation workflow (same logic adapted) ----------------
    if args.force:
        print("Force recalculating ALL beams in current phase...")
        all_contours, beam_info = calculate_beams(phi_values, theta_values, list(range(len(phi_values))))
        if all_contours:
            save_contours(all_contours, beam_info)
            save_contours(all_contours, beam_info, format='json')

    else:
        if args.load:
            print("Incremental mode with existing data...")
        existing_contours, existing_info, beams_to_calculate = load_or_calculate_beams(phi_values, theta_values)

        if beams_to_calculate:
            print(f"Calculating {len(beams_to_calculate)} new beams...")
            new_contours, new_info = calculate_beams(phi_values, theta_values, beams_to_calculate)
            all_contours, beam_info = merge_and_save_contours(existing_contours, existing_info, new_contours, new_info)
        else:
            print("All beams already present.")
            all_contours, beam_info = existing_contours, existing_info

    # ---------------- Plot + optional pruning of this phase (UR or standard) ----------------
    if all_contours:
        print("Plotting and starting interactive pruning (phase result)...")
        save_contours(all_contours, beam_info)
        save_contours(all_contours, beam_info, format='json')  # Also save as JSON
        fig, ax = plt.subplots(figsize=(16, 10))
        
        for i, (contour, info) in enumerate(zip(all_contours, beam_info)):
            beam_color = info['color']
            import matplotlib.colors as mcolors
            fill_color = mcolors.to_rgba(beam_color, alpha=0.3)
            
            # Fill and contour keep their beam colors
            ax.fill(contour[:, 0], contour[:, 1], color=fill_color, alpha=0.3)
            beam_label = f"Beam {info['beam_idx']+1} (φ={info['phi']:.1f}°, θ={info['theta']:.1f}°)"
            ax.plot(contour[:, 0], contour[:, 1], color=beam_color, linewidth=2, label=beam_label)
            
            # ADD THESE LINES FOR SMALL LABELS:
            # Small beam number label at beam center
            ax.text(info['phi'], info['theta'], f"{info['beam_idx']}", 
                    fontsize=6, ha='center', va='center', weight='bold',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='black'))

        ax.set_xlabel('Phi [degrees]')
        ax.set_ylabel('Theta [degrees]')
        ax.set_title('Main Lobe -1 dB Contours for All Beams')
        ax.grid(True, alpha=0.3)
        
        # Legend with black text
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        legend.get_frame().set_facecolor('white')  # White background for legend
        legend.get_frame().set_alpha(0.9)          # Slight transparency
        legend.get_frame().set_edgecolor('black')  # Black border for legend box
        
        plt.tight_layout()
        plt.savefig("plots/beam_contours_minus1dB.png", bbox_inches='tight', dpi=150)
        plt.show()

        even_contours = []
        even_info = []
        even_indices = []
        for i, info in enumerate(beam_info):
            if info['beam_idx'] % 2 == 0:  # Even beam indices only
                even_contours.append(all_contours[i])
                even_info.append(info)
                even_indices.append(i)
        
        print(f"Filtered to {len(even_contours)} even-indexed beams out of {len(all_contours)} total beams")
    
        # NOW use the interactive selection tool
        print(f"\nStarting interactive beam selection with {len(all_contours)} beams...")
        selected_contours, selected_info, selected_indices = interactive_beam_selection(even_contours, even_info)
        
        print(f"Interactive selection complete!")
        print(f"Selected {len(selected_contours)} out of {len(all_contours)} beams")
        print(f"Selected beam indices: {selected_indices}")
        
        # Plot the selected beams
        if len(selected_contours) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, (contour, info) in enumerate(zip(selected_contours, selected_info)):
                beam_color = info['color']
                import matplotlib.colors as mcolors
                fill_color = mcolors.to_rgba(beam_color, alpha=0.3)
                
                ax.fill(contour[:, 0], contour[:, 1], color=fill_color, alpha=0.3)
                beam_label = f"Beam {info['beam_idx']+1} (φ={info['phi']:.1f}°, θ={info['theta']:.1f}°)"
                ax.plot(contour[:, 0], contour[:, 1], color=beam_color, linewidth=2, label=beam_label)
            
            ax.set_xlabel('Phi [degrees]')
            ax.set_ylabel('Theta [degrees]')
            ax.set_title(f'Selected Beam Coverage ({len(selected_contours)} beams)')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            plt.tight_layout()
            plt.savefig("plots/selected_beam_contours.png", bbox_inches='tight', dpi=150)
            plt.show()
            
            # Save the selected beams
            save_contours(selected_contours, selected_info, filename="selected_beam_contours.pkl")
            save_contours(selected_contours, selected_info, filename="selected_beam_contours.json")
            
            # Calculate total coverage area
            total_area = 0
            for contour in selected_contours:
                x, y = contour[:, 0], contour[:, 1]
                area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
                total_area += area
            
            print(f"Total coverage area of selected beams: {total_area:.1f} deg²")
        else:
            print("No beams selected!")
        plt.close()
        print(f"\nFound -1 dB contours for {len(all_contours)}/{len(phi_values)} beams")
        for i, (contour, info) in enumerate(zip(all_contours, beam_info)):
            x, y = contour[:, 0], contour[:, 1]
            area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
            print(f"  Beam {info['beam_idx']+1}: Coverage area ≈ {area:.1f} deg²")
        pass
    else:
        print("No -1 dB contours found for any beams!")
