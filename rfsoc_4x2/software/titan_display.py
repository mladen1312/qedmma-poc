#!/usr/bin/env python3
"""
TITAN Radar - Real-Time Display System
Based on POC radar_display.py

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Features:
    - A-Scope (Range Profile)
    - B-Scope (Range vs Time)
    - PPI (Plan Position Indicator)
    - Range-Doppler Map
    - Track Display
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import time
import threading

# Try to import display libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Display] Matplotlib not available")

try:
    from ipywidgets import interact, IntSlider, FloatSlider
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


#=============================================================================
# Display Configuration
#=============================================================================

@dataclass
class DisplayConfig:
    """Display system configuration"""
    
    # Window sizes
    ascope_history: int = 100           # A-scope traces to average
    bscope_history: int = 256           # B-scope time samples
    rdmap_update_rate: float = 2.0      # Hz
    
    # Range settings
    max_range_km: float = 500.0
    range_rings_km: List[float] = None  # Default: [100, 200, 300, 400, 500]
    
    # Color settings
    colormap: str = 'viridis'
    detection_color: str = 'red'
    track_color: str = 'yellow'
    
    # Display options
    show_grid: bool = True
    show_detections: bool = True
    show_tracks: bool = True
    dark_mode: bool = True
    
    def __post_init__(self):
        if self.range_rings_km is None:
            self.range_rings_km = [100, 200, 300, 400, 500]


#=============================================================================
# Detection & Track Data Structures
#=============================================================================

@dataclass
class DisplayDetection:
    """Detection for display"""
    range_km: float
    velocity_mps: float
    azimuth_deg: float
    snr_db: float
    timestamp: float


@dataclass
class DisplayTrack:
    """Track for display"""
    track_id: int
    range_km: float
    velocity_mps: float
    azimuth_deg: float
    elevation_deg: float = 0.0
    history: List[Tuple[float, float, float]] = None  # (time, range, azimuth)
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


#=============================================================================
# A-Scope Display (Range Profile)
#=============================================================================

class AScopeDisplay:
    """
    A-Scope Display - Range Profile Visualization
    
    Shows amplitude vs range (classic oscilloscope view)
    """
    
    def __init__(self, config: DisplayConfig, max_range_bins: int = 512):
        self.config = config
        self.max_range_bins = max_range_bins
        
        self.range_profile = np.zeros(max_range_bins)
        self.history = deque(maxlen=config.ascope_history)
        
        self.fig = None
        self.ax = None
        self.line = None
        self.threshold_line = None
        self.detection_markers = None
    
    def setup_plot(self, range_axis_km: np.ndarray):
        """Setup matplotlib figure"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('dark_background' if self.config.dark_mode else 'default')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.set_xlim(0, range_axis_km[-1])
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('Range (km)')
        self.ax.set_ylabel('Amplitude (dB)')
        self.ax.set_title('A-Scope - Range Profile')
        
        if self.config.show_grid:
            self.ax.grid(True, alpha=0.3)
        
        # Initialize plot elements
        self.line, = self.ax.plot(range_axis_km, np.zeros(len(range_axis_km)), 
                                   'g-', linewidth=1)
        self.threshold_line, = self.ax.plot(range_axis_km, np.zeros(len(range_axis_km)),
                                            'r--', linewidth=0.5, alpha=0.5)
        self.detection_markers, = self.ax.plot([], [], 'ro', markersize=8)
        
        plt.tight_layout()
        
        return self.fig
    
    def update(self, range_profile: np.ndarray, threshold: np.ndarray = None,
               detections: List[DisplayDetection] = None):
        """Update A-scope display"""
        # Store in history
        self.history.append(range_profile)
        
        # Average for smoother display
        self.range_profile = np.mean(list(self.history), axis=0)
        
        # Convert to dB
        profile_db = 20 * np.log10(self.range_profile + 1e-10)
        profile_db = np.clip(profile_db, 0, 100)
        
        if self.line is not None:
            self.line.set_ydata(profile_db)
            
            if threshold is not None:
                thresh_db = 20 * np.log10(threshold + 1e-10)
                self.threshold_line.set_ydata(np.clip(thresh_db, 0, 100))
            
            if detections and self.config.show_detections:
                det_ranges = [d.range_km for d in detections]
                det_snrs = [d.snr_db for d in detections]
                self.detection_markers.set_data(det_ranges, det_snrs)
        
        return profile_db


#=============================================================================
# B-Scope Display (Range vs Time)
#=============================================================================

class BScopeDisplay:
    """
    B-Scope Display - Range-Time Indicator (RTI)
    
    Shows range vs time (waterfall display)
    """
    
    def __init__(self, config: DisplayConfig, max_range_bins: int = 512):
        self.config = config
        self.max_range_bins = max_range_bins
        
        self.data = np.zeros((config.bscope_history, max_range_bins))
        self.time_idx = 0
        
        self.fig = None
        self.ax = None
        self.img = None
    
    def setup_plot(self, range_axis_km: np.ndarray):
        """Setup matplotlib figure"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('dark_background' if self.config.dark_mode else 'default')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        self.img = self.ax.imshow(
            self.data,
            aspect='auto',
            origin='lower',
            extent=[0, range_axis_km[-1], 0, self.config.bscope_history],
            cmap=self.config.colormap,
            vmin=0, vmax=60
        )
        
        self.ax.set_xlabel('Range (km)')
        self.ax.set_ylabel('Time (sweeps)')
        self.ax.set_title('B-Scope - Range-Time Indicator')
        
        plt.colorbar(self.img, ax=self.ax, label='Amplitude (dB)')
        plt.tight_layout()
        
        return self.fig
    
    def update(self, range_profile: np.ndarray):
        """Update B-scope with new range profile"""
        # Convert to dB
        profile_db = 20 * np.log10(range_profile + 1e-10)
        profile_db = np.clip(profile_db, 0, 60)
        
        # Scroll data
        self.data = np.roll(self.data, 1, axis=0)
        self.data[0, :] = profile_db
        
        if self.img is not None:
            self.img.set_data(self.data)
        
        self.time_idx += 1


#=============================================================================
# Range-Doppler Display
#=============================================================================

class RDMapDisplay:
    """
    Range-Doppler Map Display
    
    Shows range vs Doppler velocity
    """
    
    def __init__(self, config: DisplayConfig, 
                 num_range_bins: int = 512, 
                 num_doppler_bins: int = 256):
        self.config = config
        self.num_range_bins = num_range_bins
        self.num_doppler_bins = num_doppler_bins
        
        self.rdmap = np.zeros((num_doppler_bins, num_range_bins))
        
        self.fig = None
        self.ax = None
        self.img = None
        self.det_scatter = None
    
    def setup_plot(self, range_axis_km: np.ndarray, velocity_axis_mps: np.ndarray):
        """Setup matplotlib figure"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('dark_background' if self.config.dark_mode else 'default')
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        self.img = self.ax.imshow(
            self.rdmap,
            aspect='auto',
            origin='lower',
            extent=[0, range_axis_km[-1], velocity_axis_mps[0], velocity_axis_mps[-1]],
            cmap=self.config.colormap,
            vmin=0, vmax=50
        )
        
        self.ax.set_xlabel('Range (km)')
        self.ax.set_ylabel('Velocity (m/s)')
        self.ax.set_title('Range-Doppler Map')
        
        # Detection markers
        self.det_scatter = self.ax.scatter([], [], c='red', s=50, marker='o',
                                            edgecolors='white', linewidth=1)
        
        plt.colorbar(self.img, ax=self.ax, label='Amplitude (dB)')
        
        if self.config.show_grid:
            self.ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        return self.fig
    
    def update(self, rdmap: np.ndarray, detections: List[DisplayDetection] = None):
        """Update Range-Doppler display"""
        # Convert to dB
        rdmap_db = 20 * np.log10(rdmap + 1e-10)
        rdmap_db = np.clip(rdmap_db, 0, 50)
        
        self.rdmap = rdmap_db
        
        if self.img is not None:
            self.img.set_data(rdmap_db)
            
            if detections and self.config.show_detections:
                ranges = [d.range_km for d in detections]
                velocities = [d.velocity_mps for d in detections]
                self.det_scatter.set_offsets(np.c_[ranges, velocities])


#=============================================================================
# PPI Display (Plan Position Indicator)
#=============================================================================

class PPIDisplay:
    """
    PPI Display - Plan Position Indicator
    
    Shows targets in polar coordinates (classic radar display)
    """
    
    def __init__(self, config: DisplayConfig, max_range_km: float = 500):
        self.config = config
        self.max_range_km = max_range_km
        
        self.fig = None
        self.ax = None
        self.detection_scatter = None
        self.track_lines = {}
    
    def setup_plot(self):
        """Setup matplotlib figure"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        plt.style.use('dark_background' if self.config.dark_mode else 'default')
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        # Configure polar plot
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_ylim(0, self.max_range_km)
        self.ax.set_title('PPI - Plan Position Indicator', pad=20)
        
        # Range rings
        for ring_km in self.config.range_rings_km:
            if ring_km <= self.max_range_km:
                circle = Circle((0, 0), ring_km, transform=self.ax.transData._b,
                               fill=False, color='gray', alpha=0.3)
                # Add range ring label
                self.ax.annotate(f'{ring_km}km', xy=(np.pi/4, ring_km), 
                                color='gray', fontsize=8)
        
        # Detection markers
        self.detection_scatter = self.ax.scatter([], [], c=self.config.detection_color,
                                                  s=100, marker='o', zorder=5)
        
        # Track history will be added dynamically
        
        plt.tight_layout()
        
        return self.fig
    
    def update(self, detections: List[DisplayDetection] = None,
               tracks: List[DisplayTrack] = None):
        """Update PPI display"""
        if self.detection_scatter is None:
            return
        
        # Update detections
        if detections and self.config.show_detections:
            azimuths = [np.deg2rad(d.azimuth_deg) for d in detections]
            ranges = [d.range_km for d in detections]
            self.detection_scatter.set_offsets(np.c_[azimuths, ranges])
        else:
            self.detection_scatter.set_offsets(np.empty((0, 2)))
        
        # Update tracks
        if tracks and self.config.show_tracks:
            for track in tracks:
                if track.track_id not in self.track_lines:
                    # Create new track line
                    line, = self.ax.plot([], [], '-', color=self.config.track_color,
                                         linewidth=2, alpha=0.7)
                    marker, = self.ax.plot([], [], 'o', color=self.config.track_color,
                                           markersize=10)
                    self.track_lines[track.track_id] = (line, marker)
                
                # Update track line
                if track.history:
                    times, ranges, azimuths = zip(*track.history[-20:])  # Last 20 points
                    azimuths_rad = [np.deg2rad(a) for a in azimuths]
                    line, marker = self.track_lines[track.track_id]
                    line.set_data(azimuths_rad, ranges)
                    marker.set_data([np.deg2rad(track.azimuth_deg)], [track.range_km])


#=============================================================================
# Combined Radar Display
#=============================================================================

class TITANDisplay:
    """
    Complete TITAN Radar Display System
    
    Combines all display modes in a single interface.
    """
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()
        
        self.ascope = None
        self.bscope = None
        self.rdmap_display = None
        self.ppi = None
        
        self.detections: List[DisplayDetection] = []
        self.tracks: List[DisplayTrack] = []
        
        self.range_axis_km = None
        self.velocity_axis_mps = None
        
        self.update_count = 0
        self.last_update_time = 0
    
    def initialize(self, 
                   num_range_bins: int,
                   num_doppler_bins: int,
                   range_resolution_m: float,
                   velocity_resolution_mps: float):
        """
        Initialize display system
        
        Args:
            num_range_bins: Number of range bins
            num_doppler_bins: Number of Doppler bins
            range_resolution_m: Range resolution in meters
            velocity_resolution_mps: Velocity resolution in m/s
        """
        # Create range axis
        self.range_axis_km = np.arange(num_range_bins) * range_resolution_m / 1000
        
        # Create velocity axis
        doppler_center = num_doppler_bins // 2
        self.velocity_axis_mps = (np.arange(num_doppler_bins) - doppler_center) * velocity_resolution_mps
        
        # Initialize displays
        self.ascope = AScopeDisplay(self.config, num_range_bins)
        self.bscope = BScopeDisplay(self.config, num_range_bins)
        self.rdmap_display = RDMapDisplay(self.config, num_range_bins, num_doppler_bins)
        self.ppi = PPIDisplay(self.config, max_range_km=self.range_axis_km[-1])
        
        print(f"[Display] Initialized:")
        print(f"  Range: 0 - {self.range_axis_km[-1]:.1f} km")
        print(f"  Velocity: {self.velocity_axis_mps[0]:.1f} - {self.velocity_axis_mps[-1]:.1f} m/s")
    
    def setup_all_plots(self):
        """Setup all plot figures"""
        if not MATPLOTLIB_AVAILABLE:
            print("[Display] Matplotlib not available")
            return
        
        figs = []
        
        if self.ascope:
            figs.append(('A-Scope', self.ascope.setup_plot(self.range_axis_km)))
        
        if self.bscope:
            figs.append(('B-Scope', self.bscope.setup_plot(self.range_axis_km)))
        
        if self.rdmap_display:
            figs.append(('RD-Map', self.rdmap_display.setup_plot(
                self.range_axis_km, self.velocity_axis_mps)))
        
        if self.ppi:
            figs.append(('PPI', self.ppi.setup_plot()))
        
        return figs
    
    def update(self,
               range_profile: np.ndarray = None,
               rdmap: np.ndarray = None,
               detections: List = None,
               tracks: List = None):
        """
        Update all displays
        
        Args:
            range_profile: 1D range profile
            rdmap: 2D Range-Doppler map
            detections: List of Detection objects
            tracks: List of Track objects
        """
        # Convert detections to display format
        if detections:
            self.detections = [
                DisplayDetection(
                    range_km=d.range_m / 1000,
                    velocity_mps=d.velocity_mps,
                    azimuth_deg=getattr(d, 'azimuth_deg', 0),
                    snr_db=d.snr_db,
                    timestamp=time.time()
                )
                for d in detections
            ]
        
        # Update A-scope
        if range_profile is not None and self.ascope:
            self.ascope.update(range_profile, detections=self.detections)
        
        # Update B-scope
        if range_profile is not None and self.bscope:
            self.bscope.update(range_profile)
        
        # Update RD map
        if rdmap is not None and self.rdmap_display:
            self.rdmap_display.update(rdmap, self.detections)
        
        # Update PPI
        if self.ppi:
            self.ppi.update(self.detections, tracks)
        
        self.update_count += 1
        self.last_update_time = time.time()
    
    def refresh(self):
        """Refresh all displays"""
        if MATPLOTLIB_AVAILABLE:
            plt.pause(0.001)


#=============================================================================
# Demo
#=============================================================================

def demo():
    """Demonstrate display system"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib required for display demo")
        return
    
    print("\n" + "=" * 60)
    print("TITAN RADAR DISPLAY DEMO")
    print("=" * 60)
    
    # Configuration
    num_range_bins = 512
    num_doppler_bins = 128
    range_resolution_m = 150  # 150m resolution
    velocity_resolution_mps = 2  # 2 m/s resolution
    
    # Initialize display
    config = DisplayConfig(dark_mode=True)
    display = TITANDisplay(config)
    display.initialize(
        num_range_bins, num_doppler_bins,
        range_resolution_m, velocity_resolution_mps
    )
    
    # Setup plots
    display.setup_all_plots()
    
    # Simulate data updates
    print("\n[Demo] Running display update loop (press Ctrl+C to stop)...")
    
    try:
        for i in range(100):
            # Generate synthetic data
            range_profile = np.random.exponential(0.1, num_range_bins)
            
            # Add some targets
            range_profile[50] += 2.0 * np.random.rand()
            range_profile[150] += 1.0 * np.random.rand()
            range_profile[300] += 0.5 * np.random.rand()
            
            # Generate RD map
            rdmap = np.random.exponential(0.1, (num_doppler_bins, num_range_bins))
            rdmap[64, 50] += 3.0  # Target 1
            rdmap[80, 150] += 2.0  # Target 2 (moving away)
            rdmap[40, 300] += 1.0  # Target 3 (approaching)
            
            # Create detections
            from titan_signal_processor import Detection
            detections = [
                Detection(range_bin=50, doppler_bin=64, amplitude=3.0, snr_db=25,
                         range_m=50*range_resolution_m, velocity_mps=0),
                Detection(range_bin=150, doppler_bin=80, amplitude=2.0, snr_db=20,
                         range_m=150*range_resolution_m, velocity_mps=32),
                Detection(range_bin=300, doppler_bin=40, amplitude=1.0, snr_db=15,
                         range_m=300*range_resolution_m, velocity_mps=-48),
            ]
            
            # Update display
            display.update(
                range_profile=range_profile,
                rdmap=rdmap,
                detections=detections
            )
            display.refresh()
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[Demo] Stopped")
    
    plt.show()


if __name__ == "__main__":
    demo()
