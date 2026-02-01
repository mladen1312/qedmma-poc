#!/usr/bin/env python3
"""
QEDMMA PoC - Real-time Radar Display
ASCII and Matplotlib visualization

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved
"""

import numpy as np
import sys
import time
from collections import deque

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Using ASCII display.")

#=============================================================================
# ASCII Display (No Dependencies)
#=============================================================================

class ASCIIDisplay:
    """Simple ASCII radar display for terminals"""
    
    def __init__(self, width=80, height=20, max_range_km=100):
        self.width = width
        self.height = height
        self.max_range_km = max_range_km
        self.history = deque(maxlen=height)
        
    def clear(self):
        """Clear screen"""
        print("\033[2J\033[H", end="")
    
    def draw_header(self, cpi_num, detections):
        """Draw header with status"""
        print("╔" + "═" * (self.width-2) + "╗")
        title = f" QEDMMA PoC - CPI #{cpi_num} | Detections: {len(detections)} "
        padding = self.width - 2 - len(title)
        print("║" + title + " " * padding + "║")
        print("╠" + "═" * (self.width-2) + "╣")
    
    def draw_range_profile(self, range_profile, detections):
        """Draw range profile as ASCII bar chart"""
        # Normalize profile
        max_val = np.max(range_profile) if np.max(range_profile) > 0 else 1
        normalized = range_profile / max_val
        
        # Downsample to fit width
        bins_per_char = len(range_profile) // (self.width - 10)
        if bins_per_char < 1:
            bins_per_char = 1
        
        # Get detection bins
        det_bins = set(d['bin'] for d in detections)
        
        # Draw scale
        print("║ Range  │" + "─" * (self.width - 11) + "║")
        
        # Draw profile
        for row in range(self.height // 2 - 2, -1, -1):
            threshold = (row + 0.5) / (self.height // 2)
            line = "║ {:5.0f}km│".format(self.max_range_km * row / (self.height // 2))
            
            for i in range(0, len(range_profile), bins_per_char):
                chunk = normalized[i:i+bins_per_char]
                val = np.max(chunk) if len(chunk) > 0 else 0
                
                # Check for detection in this chunk
                is_detection = any(b in det_bins for b in range(i, min(i+bins_per_char, len(range_profile))))
                
                if val >= threshold:
                    if is_detection:
                        line += "█"  # Detection
                    else:
                        line += "▓"  # Signal
                elif val >= threshold - 0.2:
                    line += "░"
                else:
                    line += " "
            
            line = line[:self.width-1] + "║"
            print(line)
        
        print("╠" + "═" * (self.width-2) + "╣")
    
    def draw_detections(self, detections, range_resolution):
        """Draw detection list"""
        print("║ DETECTIONS:" + " " * (self.width - 14) + "║")
        
        if len(detections) == 0:
            print("║   (none)" + " " * (self.width - 11) + "║")
        else:
            for det in detections[:5]:  # Top 5
                range_km = det['bin'] * range_resolution / 1000
                line = f"║   Bin {det['bin']:3d} | Range: {range_km:6.1f} km | SNR: {det['snr_db']:5.1f} dB"
                line = line + " " * (self.width - 1 - len(line)) + "║"
                print(line)
        
        print("╚" + "═" * (self.width-2) + "╝")
    
    def update(self, cpi_num, range_profile, detections, range_resolution):
        """Full display update"""
        self.clear()
        self.draw_header(cpi_num, detections)
        self.draw_range_profile(range_profile, detections)
        self.draw_detections(detections, range_resolution)
        sys.stdout.flush()

#=============================================================================
# Matplotlib Display
#=============================================================================

class MatplotlibDisplay:
    """Matplotlib-based radar display"""
    
    def __init__(self, num_bins=512, max_range_km=100, history_len=50):
        self.num_bins = num_bins
        self.max_range_km = max_range_km
        self.history_len = history_len
        
        # Create figure
        self.fig, (self.ax_profile, self.ax_waterfall) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]}
        )
        
        self.fig.suptitle('QEDMMA PoC - Real-time Radar Display', fontsize=14)
        
        # Range axis
        self.range_axis = np.linspace(0, max_range_km, num_bins)
        
        # Initialize range profile plot
        self.profile_line, = self.ax_profile.plot(
            self.range_axis, np.zeros(num_bins), 'g-', linewidth=1
        )
        self.threshold_line, = self.ax_profile.plot(
            self.range_axis, np.zeros(num_bins), 'r--', linewidth=0.5, alpha=0.7
        )
        self.ax_profile.set_xlabel('Range (km)')
        self.ax_profile.set_ylabel('Magnitude (dB)')
        self.ax_profile.set_xlim(0, max_range_km)
        self.ax_profile.set_ylim(-40, 60)
        self.ax_profile.grid(True, alpha=0.3)
        self.ax_profile.set_title('Range Profile')
        
        # Detection markers
        self.det_scatter = self.ax_profile.scatter([], [], c='red', s=100, marker='v')
        
        # Initialize waterfall
        self.waterfall_data = np.zeros((history_len, num_bins))
        self.waterfall_img = self.ax_waterfall.imshow(
            self.waterfall_data, aspect='auto', cmap='viridis',
            extent=[0, max_range_km, history_len, 0],
            vmin=-30, vmax=30
        )
        self.ax_waterfall.set_xlabel('Range (km)')
        self.ax_waterfall.set_ylabel('Time (CPIs)')
        self.ax_waterfall.set_title('Range-Time Waterfall')
        
        # Colorbar
        self.cbar = self.fig.colorbar(self.waterfall_img, ax=self.ax_waterfall, label='dB')
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def update(self, range_profile, detections, threshold=None):
        """Update display with new data"""
        # Convert to dB
        profile_db = 20 * np.log10(range_profile + 1e-10)
        noise_floor = np.median(profile_db)
        profile_db_normalized = profile_db - noise_floor
        
        # Update range profile
        self.profile_line.set_ydata(profile_db_normalized)
        
        # Update threshold
        if threshold is not None:
            threshold_db = 20 * np.log10(threshold + 1e-10) - noise_floor
            self.threshold_line.set_ydata(threshold_db)
        
        # Update detections
        if len(detections) > 0:
            det_ranges = [d['bin'] * self.max_range_km / self.num_bins for d in detections]
            det_mags = [d['snr_db'] for d in detections]
            self.det_scatter.set_offsets(np.column_stack([det_ranges, det_mags]))
        else:
            self.det_scatter.set_offsets(np.empty((0, 2)))
        
        # Update waterfall
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0, :] = profile_db_normalized[:self.num_bins]
        self.waterfall_img.set_data(self.waterfall_data)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close display"""
        plt.close(self.fig)

#=============================================================================
# Display Factory
#=============================================================================

def create_display(display_type='auto', **kwargs):
    """
    Create appropriate display
    
    Args:
        display_type: 'ascii', 'matplotlib', or 'auto'
        **kwargs: Display-specific arguments
    """
    if display_type == 'auto':
        if MATPLOTLIB_AVAILABLE and sys.stdout.isatty():
            display_type = 'matplotlib'
        else:
            display_type = 'ascii'
    
    if display_type == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        return MatplotlibDisplay(**kwargs)
    else:
        return ASCIIDisplay(**kwargs)

#=============================================================================
# Demo
#=============================================================================

def demo():
    """Demo the ASCII display"""
    print("QEDMMA PoC Display Demo")
    print("=" * 40)
    
    display = ASCIIDisplay(width=80, height=20, max_range_km=100)
    
    for cpi in range(10):
        # Fake data
        profile = np.random.exponential(1, 512)
        profile[50] = 50  # Target 1
        profile[200] = 30  # Target 2
        profile[350] = 20  # Target 3
        
        detections = [
            {'bin': 50, 'snr_db': 35.2},
            {'bin': 200, 'snr_db': 28.1},
            {'bin': 350, 'snr_db': 22.5},
        ]
        
        display.update(cpi, profile, detections, range_resolution=195)
        time.sleep(0.5)

if __name__ == "__main__":
    demo()
