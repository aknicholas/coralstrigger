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

# Configuration - set which colors to process
PROCESS_COLORS = None               # Process all colors
#PROCESS_COLORS = None             # Process all colors
#EXCLUDE_COLORS = ['blue']         # Process all except these colors
#EXCLUDE_COLORS = ['teal']
EXCLUDE_COLORS = None

def _coord_key(phi, theta, prec=6):
    return (round(float(phi), prec), round(float(theta), prec))


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
def save_contours(all_contours, beam_info, filename="beam_contours_data.pkl"):
    """Save contour data to file"""
    contour_data = {
        'contours': all_contours,
        'beam_info': beam_info,
        'timestamp': time.time(),
        'settings_used': beam_settings
    }
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(contour_data, f)
    
    print(f"Saved contour data to {filepath}")
    return filepath

def load_contours(filename="beam_contours_data.pkl"):
    """Load contour data from file"""
    filepath = os.path.join("plots", filename)
    
    if not os.path.exists(filepath):
        print(f"No saved contour data found at {filepath}")
        return None, None
    
    try:
        with open(filepath, 'rb') as f:
            contour_data = pickle.load(f)
        
        all_contours = contour_data['contours']
        beam_info = contour_data['beam_info']
        
        print(f"Loaded {len(all_contours)} contours from {filepath}")
        if 'timestamp' in contour_data:
            import time
            timestamp = time.ctime(contour_data['timestamp'])
            print(f"  Data created: {timestamp}")
        
        return all_contours, beam_info
    
    except Exception as e:
        print(f"Error loading contour data: {e}")
        return None, None

def save_contours_json(all_contours, beam_info, filename="beam_contours_data.json"):
    """Save contour data to JSON (human-readable format)"""
    # Convert numpy arrays to lists for JSON serialization
    contour_data = {
        'contours': [contour.tolist() for contour in all_contours],
        'beam_info': beam_info,
        'timestamp': time.time(),
        'n_beams': len(all_contours)
    }
    
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    
    with open(filepath, 'w') as f:
        json.dump(contour_data, f, indent=2)
    
    print(f"Saved contour data to {filepath} (JSON format)")
    return filepath

def load_contours_json(filename="beam_contours_data.json"):
    """Load contour data from JSON file"""
    filepath = os.path.join("plots", filename)
    
    if not os.path.exists(filepath):
        print(f"No saved contour data found at {filepath}")
        return None, None
    
    try:
        with open(filepath, 'r') as f:
            contour_data = json.load(f)
        
        # Convert lists back to numpy arrays
        all_contours = [numpy.array(contour) for contour in contour_data['contours']]
        beam_info = contour_data['beam_info']
        
        print(f"Loaded {len(all_contours)} contours from {filepath}")
        if 'timestamp' in contour_data:
            import time
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

def load_specific_beams(beam_indices, filename="beam_contours_data.pkl"):
    """Load only specific beams from saved data"""
    all_contours, beam_info = load_contours(filename)
    
    if all_contours is None:
        return None, None
    
    # Filter to only requested beams
    filtered_contours = []
    filtered_info = []
    
    for i, info in enumerate(beam_info):
        if info['beam_idx'] in beam_indices:
            filtered_contours.append(all_contours[i])
            filtered_info.append(info)
    
    print(f"Loaded {len(filtered_contours)} out of {len(beam_indices)} requested beams")
    return filtered_contours, filtered_info

def load_beams_by_color(colors, filename="beam_contours_data.pkl"):
    """Load only beams with specific colors"""
    all_contours, beam_info = load_contours(filename)
    
    if all_contours is None:
        return None, None
    
    # Filter by color
    filtered_contours = []
    filtered_info = []
    
    for i, info in enumerate(beam_info):
        if info['color'] in colors:
            filtered_contours.append(all_contours[i])
            filtered_info.append(info)
    
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
    save_contours_json(all_contours, all_info)
    
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
    """Calculate only the specified beam indices"""
    
    angular_res = 0.4

    lp_freq, lp_mult, _ = filters.Shannon_Whitaker(fs=4e9, plot=False)
    lp_gain = 10**(lp_mult/20)
    eplane  = payload.beamPattern(plot=False,which_plane='E',which_pol='V')
    hplane = payload.beamPattern(plot=False,which_plane='H',which_pol='V')
    trigger_sectors_phi = [1,2,3,4]
    impulse = payload.loadImpulse('impulse/corals_impulse_sci.txt')
    impulse = payload.prepImpulse(impulse, highpass_cutoff=0.15, lowpass_cutoff=2)

    # Simple fixed scan width
    phi_scan_width = 0
    el_scan_width = 0

    filtering = True
    new_contours = []
    new_info = []
    
    for beam_idx in beam_indices_to_calculate:
        phi = phi_values[beam_idx]
        theta = theta_values[beam_idx]
        
        should_process, beam_color = should_process_beam(phi, theta, beam_idx)
        if not should_process:
            print(f"Skipping beam {beam_idx+1} (color: {beam_color})")
            continue

        print(f'Processing beam {beam_idx+1}/{len(phi_values)}: phi={phi:.1f}, theta={theta:.1f} (color: {beam_color})')
        
        ringmask = [1,1,1,1]
        payload.getPayloadWaveforms(phi, theta, trigger_sectors_phi, impulse, (eplane, hplane), snr=5, plot=False)
        delays = payload.getRemappedDelays(phi, theta, trigger_sectors_phi)
        
        # Simple fixed beam settings
        beam_settings[beam_idx] = {
            'threshold': -3.0, 
            'contour': -1, 
            'phi_start': phi - phi_scan_width, 
            'phi_stop': phi + phi_scan_width, 
            'el_start': theta - el_scan_width, 
            'el_stop': theta + el_scan_width, 
            'color': beam_color
        }

        # Simple scan boundaries
        phiscan_start = phi - phi_scan_width
        phiscan_stop = phi + phi_scan_width
        elscan_start = theta - el_scan_width
        elscan_stop = theta + el_scan_width
        
        print(f"  Using scan boundaries: phi [{phiscan_start}° to {phiscan_stop}°], el [{elscan_start}° to {elscan_stop}°]")

        phiscan_range = numpy.arange(phiscan_start, phiscan_stop, angular_res)
        elscan_range = numpy.arange(elscan_start, elscan_stop, angular_res)
        phi_grid, el_grid = numpy.meshgrid(phiscan_range, elscan_range)
        heatmap = numpy.zeros_like(phi_grid, dtype=float)
                
        for i, elscan in enumerate(elscan_range):
            for j, phiscan in enumerate(phiscan_range):
                waveforms, timebase, _ = payload.getPayloadWaveforms(phiscan, elscan, trigger_sectors_phi, impulse, (eplane, hplane), downsample=True, snr=5, plot=False)

                if filtering:
                    n_waveforms = waveforms.shape[0]
                    n_rings = waveforms.shape[1]
                    for wf_i in range(n_waveforms):
                        for ring_i in range(n_rings):
                            wf = waveforms[wf_i, ring_i]
                            wf_obj = waveform.Waveform(wf, timebase)
                            wf_obj.fft()
                            fft_len = len(wf_obj.ampl)
                            if len(lp_gain) != fft_len:
                                lp_gain_interp = numpy.interp(
                                    numpy.linspace(0, 1, fft_len),
                                    numpy.linspace(0, 1, len(lp_gain)),
                                    lp_gain
                                )
                            else:
                                lp_gain_interp = lp_gain
                            wf_obj.ampl = wf_obj.ampl * lp_gain_interp
                            wf_obj.voltage = numpy.fft.irfft(wf_obj.ampl, n=wf_obj.n)
                            waveforms[wf_i, ring_i] = wf_obj.voltage
                            
                coh_sum, _ = sum.coherentSum(waveforms, timebase, delays, False, ringmask)
                power, _ = sum.powerSum(coh_sum, window=256, step=128)
                heatmap[i, j] = numpy.max(power)
        
        # Simple processing without elevation penalty
        min_heatmap_value = 1e-12
        heatmap_safe = numpy.where(heatmap == 0, min_heatmap_value, heatmap)
        heatmap_db = 10 * numpy.log10(heatmap_safe / numpy.max(heatmap_safe))
        
        # Fixed threshold
        threshold = -3.0
        print(f"  Beam {beam_idx+1} at (φ={phi:.1f}°, θ={theta:.1f}°): using threshold={threshold:.1f} dB")
        
        main_lobe_mask = heatmap_db > threshold
        labeled_regions = label(main_lobe_mask)
        contour_level = -1.0  # Fixed contour level

        if labeled_regions.max() > 0:
            max_idx = numpy.unravel_index(numpy.argmax(heatmap_db), heatmap_db.shape)
            main_lobe_label = labeled_regions[max_idx]
            main_lobe_only = labeled_regions == main_lobe_label
            from scipy.ndimage import binary_opening, binary_closing
            main_lobe_cleaned = binary_closing(main_lobe_only, structure=numpy.ones((3,3)))
            main_lobe_cleaned = binary_opening(main_lobe_cleaned, structure=numpy.ones((2,2)))
            main_lobe_heatmap = numpy.where(main_lobe_cleaned, heatmap_db, -numpy.inf)
            
            try:
                fig_temp, ax_temp = plt.subplots()
                contour_set = ax_temp.contour(phi_grid, el_grid, main_lobe_heatmap, levels=[contour_level])
                plt.close(fig_temp)
                print(f"  Using {contour_level:.1f} dB contour level")
                
                if len(contour_set.collections) > 0:
                    all_paths = contour_set.collections[0].get_paths()
                    if len(all_paths) > 0:
                        if len(all_paths) > 1:
                            # Pick contour closest to beam center
                            best_path = None
                            min_distance = float('inf')
                            for path in all_paths:
                                vertices = path.vertices
                                if len(vertices) > 0:
                                    distances = numpy.sqrt((vertices[:, 0] - phi)**2 + (vertices[:, 1] - theta)**2)
                                    avg_distance = numpy.mean(distances)
                                    if avg_distance < min_distance:
                                        min_distance = avg_distance
                                        best_path = path
                            contour_path = best_path
                        else:
                            contour_path = all_paths[0]
                        
                        if contour_path is not None:
                            contour_coords = contour_path.vertices
                            new_contours.append(contour_coords)
                            new_info.append({'phi': phi, 'theta': theta, 'beam_idx': beam_idx, 'color': beam_color})
                            print(f"  Found -1 dB contour with {len(contour_coords)} points")
                    else:
                        print(f"  No -1 dB contour found for beam at phi={phi:.1f}, theta={theta:.1f}")
            except Exception as e:
                print(f"  Error extracting contour: {e}")
    
    return new_contours, new_info
# Usage:
# list_saved_contours()

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
    args = parser.parse_args()

    use_saved_data = args.load
    force_recalculate = args.force

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
    
    
    if hasattr(args, 'load_file') and args.load_file:
        load_path = args.load_file
        print(f"[Load-Only] Loading contours from: {load_path}")
        if load_path.endswith('.json'):
            all_contours, beam_info = load_contours_json(load_path)
        else:
            all_contours, beam_info = load_contours(load_path)

        if all_contours is None:
            print("Failed to load contours file.")
            sys.exit(1)

        print(f"Loaded {len(all_contours)} beams from {load_path}")

        # Quick plot
        fig, ax = plt.subplots(figsize=(12,8))
        for contour, info in zip(all_contours, beam_info):
            ax.plot(contour[:,0], contour[:,1], color=info.get('color','gray'), lw=0.8)
            ax.fill(contour[:,0], contour[:,1], color=info.get('color','gray'), alpha=0.18)
            ax.text(info['phi'], info['theta'], f"{info['beam_idx']}", fontsize=5,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, lw=0.2))
        ax.axvline(90, color='k', lw=0.6, ls='--')
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('Phi (deg)')
        ax.set_ylabel('Theta (deg)')
        ax.set_title(f"Loaded Beam Map ({len(all_contours)} beams)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/loaded_beams.png", bbox_inches='tight', dpi=150)
        plt.show()

        if getattr(args, 'select', False):
            print("Starting interactive selection on loaded set...")
            sel_contours, sel_info, sel_idx = interactive_beam_selection(all_contours, beam_info)
            save_contours(sel_contours, sel_info, filename="reselected_beams.pkl")
            save_contours_json(sel_contours, sel_info, filename="reselected_beams.json")
            print(f"Reselected {len(sel_contours)} beams (indices in loaded set: {sel_idx})")

        sys.exit(0)
    # ---------------- Phase 2: Mirror mode ----------------
    if args.mirror_from:
        print(f"[Mirror Phase] Loading selected UR file: {args.mirror_from}")
        sel_contours, sel_info = load_contours(args.mirror_from)
        if sel_contours is None:
            print("Could not load the supplied --mirror-from file. Exiting.")
            sys.exit(1)

        # Build mirrored coordinate sets
        mirror_dict = build_full_square_from_selected(sel_info, phi_center=90, theta_center=0)
        print("Mirror category counts:")
        for k, v in mirror_dict.items():
            print(f"  {k}: {len(v)}")

        # Flatten to single list of candidate coordinates
        mirrored_coords = flatten_mirror_dict(mirror_dict)

        # Load any existing master set
        existing_contours, existing_info = load_contours()
        if existing_contours is None:
            existing_contours, existing_info = [], []

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
            save_contours_json(full_selected_contours, full_selected_info, filename="full_square_beams.json")
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
    if force_recalculate:
        print("Force recalculating ALL beams in current phase...")
        all_contours, beam_info = calculate_beams(phi_values, theta_values, list(range(len(phi_values))))
        if all_contours:
            save_contours(all_contours, beam_info)
            save_contours_json(all_contours, beam_info)

    else:
        if use_saved_data:
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
        save_contours_json(all_contours, beam_info)  # Also save as JSON
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
            save_contours_json(selected_contours, selected_info, filename="selected_beam_contours.json")
            
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
