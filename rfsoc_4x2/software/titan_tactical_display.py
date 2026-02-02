#!/usr/bin/env python3
"""
TITAN VHF Anti-Stealth Radar - Advanced Tactical Display
Real-Time PPI + Range-Doppler Visualization

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Based on Gemini Factory Spec UI template, extended for:
- Multi-target tracking with track history
- Range-Doppler map visualization
- Doppler filter bank display
- Real-time detection overlay
- Google Earth KML export
- Threat classification display

Requirements:
    pip install PyQt5 pyqtgraph numpy scipy

Usage:
    python3 titan_tactical_display.py                    # Simulation mode
    python3 titan_tactical_display.py --mode hardware   # Real hardware
    python3 titan_tactical_display.py --kml output.kml  # Enable KML export
"""

import sys
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QGridLayout, QGroupBox, QTabWidget,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QStatusBar, QProgressBar, QCheckBox
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette

# PyQtGraph for fast plotting
import pyqtgraph as pg
from pyqtgraph import ColorMap

# Configure PyQtGraph for radar theme
pg.setConfigOption('background', (0, 0, 0))        # Black background
pg.setConfigOption('foreground', (0, 255, 0))      # Green foreground
pg.setConfigOptions(antialias=True)


#===============================================================================
# Constants
#===============================================================================

C = 299792458.0  # Speed of light (m/s)
RADAR_GREEN = (0, 255, 0)
RADAR_AMBER = (255, 191, 0)
RADAR_RED = (255, 0, 0)
RADAR_CYAN = (0, 255, 255)


#===============================================================================
# Data Models
#===============================================================================

class ThreatLevel(Enum):
    """Target threat classification"""
    UNKNOWN = "UNKNOWN"
    FRIENDLY = "FRIENDLY"
    NEUTRAL = "NEUTRAL"
    HOSTILE = "HOSTILE"
    STEALTH = "STEALTH"


@dataclass
class Track:
    """Single radar track (target)"""
    track_id: int
    range_m: float
    azimuth_deg: float
    velocity_mps: float
    rcs_m2: float
    snr_db: float
    threat: ThreatLevel = ThreatLevel.UNKNOWN
    
    # Cartesian position
    x: float = 0.0
    y: float = 0.0
    
    # Track history
    history_x: List[float] = field(default_factory=list)
    history_y: List[float] = field(default_factory=list)
    history_time: List[float] = field(default_factory=list)
    
    # Track quality
    age_seconds: float = 0.0
    updates: int = 0
    coasted: bool = False
    
    def __post_init__(self):
        # Convert polar to Cartesian
        self.x = self.range_m * np.cos(np.radians(self.azimuth_deg))
        self.y = self.range_m * np.sin(np.radians(self.azimuth_deg))
    
    def update(self, range_m: float, azimuth_deg: float, velocity_mps: float, 
               snr_db: float, timestamp: float):
        """Update track with new measurement"""
        self.range_m = range_m
        self.azimuth_deg = azimuth_deg
        self.velocity_mps = velocity_mps
        self.snr_db = snr_db
        
        # Update Cartesian
        self.x = range_m * np.cos(np.radians(azimuth_deg))
        self.y = range_m * np.sin(np.radians(azimuth_deg))
        
        # Add to history
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        self.history_time.append(timestamp)
        
        # Limit history length
        max_history = 100
        if len(self.history_x) > max_history:
            self.history_x = self.history_x[-max_history:]
            self.history_y = self.history_y[-max_history:]
            self.history_time = self.history_time[-max_history:]
        
        self.updates += 1
        self.coasted = False
    
    @property
    def speed_kmh(self) -> float:
        return abs(self.velocity_mps) * 3.6
    
    @property
    def mach_number(self) -> float:
        return abs(self.velocity_mps) / 340.0


#===============================================================================
# Radar Simulation (for demo without hardware)
#===============================================================================

class RadarSimulator:
    """Simulates radar targets for UI testing"""
    
    def __init__(self, num_targets: int = 3):
        self.num_targets = num_targets
        self.targets = []
        self.time = 0
        
        # Initialize random targets
        for i in range(num_targets):
            target = {
                'id': i + 1,
                'range': np.random.uniform(20000, 80000),  # 20-80 km
                'azimuth': np.random.uniform(0, 360),
                'velocity': np.random.uniform(100, 400),   # 100-400 m/s
                'rcs': np.random.choice([0.01, 0.1, 1, 5, 10, 50]),  # Various RCS
                'orbit_rate': np.random.uniform(0.5, 2) * np.random.choice([-1, 1]),
                'radial_osc': np.random.uniform(0, 0.3),
            }
            self.targets.append(target)
    
    def update(self, dt: float = 0.05) -> List[Dict]:
        """Generate simulated radar detections"""
        self.time += dt
        detections = []
        
        for t in self.targets:
            # Update target motion
            t['azimuth'] += t['orbit_rate'] * dt
            t['azimuth'] %= 360
            
            # Radial oscillation
            range_var = 5000 * np.sin(self.time * t['radial_osc'])
            
            # Add noise
            range_noise = np.random.normal(0, 100)
            az_noise = np.random.normal(0, 0.5)
            vel_noise = np.random.normal(0, 5)
            
            # Calculate SNR based on range and RCS
            # SNR ∝ RCS / R^4
            snr = 10 * np.log10(t['rcs']) - 40 * np.log10(t['range'] / 50000) + 30
            snr += np.random.normal(0, 2)
            
            detection = {
                'track_id': t['id'],
                'range_m': t['range'] + range_var + range_noise,
                'azimuth_deg': t['azimuth'] + az_noise,
                'velocity_mps': t['velocity'] * np.cos(np.radians(t['azimuth'])) + vel_noise,
                'rcs_m2': t['rcs'],
                'snr_db': snr,
                'timestamp': self.time,
            }
            detections.append(detection)
        
        return detections
    
    def generate_range_doppler_map(self, num_range: int = 512, 
                                   num_doppler: int = 256) -> np.ndarray:
        """Generate simulated Range-Doppler map"""
        # Noise floor
        rd_map = np.random.exponential(1, (num_doppler, num_range)) * 0.1
        
        # Add targets as peaks
        for t in self.targets:
            # Range bin
            range_bin = int((t['range'] / 100000) * num_range)
            range_bin = max(0, min(num_range - 1, range_bin))
            
            # Doppler bin (velocity to Doppler)
            doppler = 2 * t['velocity'] * np.cos(np.radians(t['azimuth'])) / 1.93  # λ = 1.93m
            doppler_bin = int((doppler + 500) / 1000 * num_doppler)
            doppler_bin = max(0, min(num_doppler - 1, doppler_bin))
            
            # Add Gaussian peak
            for dr in range(-5, 6):
                for dd in range(-3, 4):
                    r_idx = range_bin + dr
                    d_idx = doppler_bin + dd
                    if 0 <= r_idx < num_range and 0 <= d_idx < num_doppler:
                        amplitude = t['rcs'] * np.exp(-(dr**2 / 4 + dd**2 / 2))
                        rd_map[d_idx, r_idx] += amplitude
        
        return rd_map


#===============================================================================
# PPI Display Widget
#===============================================================================

class PPIDisplay(pg.PlotWidget):
    """Plan Position Indicator (PPI) - Classic radar circular display"""
    
    def __init__(self, max_range_km: float = 100):
        super().__init__()
        
        self.max_range_km = max_range_km
        self.max_range_m = max_range_km * 1000
        
        # Configure plot
        self.setTitle("PPI TACTICAL SCOPE", color='g', size='12pt')
        self.setAspectLocked(True)
        self.setRange(xRange=[-self.max_range_m, self.max_range_m],
                     yRange=[-self.max_range_m, self.max_range_m])
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel('bottom', 'East-West', units='m')
        self.setLabel('left', 'North-South', units='m')
        
        # Radar center marker
        self.radar_center = self.plot([0], [0], symbol='+', symbolSize=25, 
                                      symbolBrush='r', symbolPen='r')
        
        # Range rings
        self.draw_range_rings()
        
        # Azimuth lines
        self.draw_azimuth_lines()
        
        # Track items
        self.track_items = {}  # track_id -> trail plot
        
        # Detection scatter
        self.detection_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
        self.addItem(self.detection_scatter)
        
        # Sweep line
        self.sweep_angle = 0
        self.sweep_line = self.plot([0, 0], [0, 0], pen=pg.mkPen('g', width=2))
    
    def draw_range_rings(self):
        """Draw concentric range rings"""
        ring_distances = [25, 50, 75, 100]  # km
        
        for r_km in ring_distances:
            if r_km <= self.max_range_km:
                r_m = r_km * 1000
                
                # Draw circle using QGraphicsEllipseItem
                circle = pg.QtWidgets.QGraphicsEllipseItem(-r_m, -r_m, r_m*2, r_m*2)
                circle.setPen(pg.mkPen((0, 255, 0, 80), width=1))
                self.addItem(circle)
                
                # Range label
                text = pg.TextItem(f"{r_km} km", color=(0, 255, 0, 150))
                text.setPos(r_m * 0.7, r_m * 0.7)
                self.addItem(text)
    
    def draw_azimuth_lines(self):
        """Draw azimuth reference lines every 30°"""
        for az in range(0, 360, 30):
            rad = np.radians(az)
            x = [0, self.max_range_m * np.sin(rad)]
            y = [0, self.max_range_m * np.cos(rad)]
            
            self.plot(x, y, pen=pg.mkPen((0, 255, 0, 50), width=1))
            
            # Cardinal labels
            if az == 0:
                label = "N"
            elif az == 90:
                label = "E"
            elif az == 180:
                label = "S"
            elif az == 270:
                label = "W"
            else:
                label = f"{az}°"
            
            text = pg.TextItem(label, color=(0, 255, 0, 100))
            text.setPos(self.max_range_m * 0.95 * np.sin(rad),
                       self.max_range_m * 0.95 * np.cos(rad))
            self.addItem(text)
    
    def update_sweep(self, angle_deg: float):
        """Update radar sweep line"""
        self.sweep_angle = angle_deg
        rad = np.radians(angle_deg)
        self.sweep_line.setData([0, self.max_range_m * np.sin(rad)],
                                [0, self.max_range_m * np.cos(rad)])
    
    def update_tracks(self, tracks: List[Track]):
        """Update all tracks on display"""
        # Collect data for scatter plot
        x_data = []
        y_data = []
        colors = []
        
        for track in tracks:
            x_data.append(track.x)
            y_data.append(track.y)
            
            # Color based on threat
            if track.threat == ThreatLevel.HOSTILE:
                colors.append(RADAR_RED)
            elif track.threat == ThreatLevel.STEALTH:
                colors.append(RADAR_AMBER)
            elif track.threat == ThreatLevel.FRIENDLY:
                colors.append(RADAR_CYAN)
            else:
                colors.append(RADAR_GREEN)
            
            # Update or create track trail
            track_id = track.track_id
            if track_id not in self.track_items:
                # Create new trail
                trail = self.plot(pen=pg.mkPen(RADAR_GREEN + (150,), width=2, style=Qt.DashLine))
                self.track_items[track_id] = trail
            
            # Update trail
            if len(track.history_x) > 1:
                self.track_items[track_id].setData(track.history_x, track.history_y)
        
        # Update scatter
        if x_data:
            brushes = [pg.mkBrush(*c) for c in colors]
            self.detection_scatter.setData(x_data, y_data, brush=brushes)
        else:
            self.detection_scatter.clear()


#===============================================================================
# Range-Doppler Map Display
#===============================================================================

class RangeDopplerDisplay(pg.PlotWidget):
    """Range-Doppler map visualization with CFAR overlay"""
    
    def __init__(self, num_range: int = 512, num_doppler: int = 256,
                 max_range_km: float = 100, max_velocity_mps: float = 500):
        super().__init__()
        
        self.num_range = num_range
        self.num_doppler = num_doppler
        self.max_range_km = max_range_km
        self.max_velocity_mps = max_velocity_mps
        
        # Configure plot
        self.setTitle("RANGE-DOPPLER MAP", color='g', size='12pt')
        self.setLabel('bottom', 'Range', units='km')
        self.setLabel('left', 'Velocity', units='m/s')
        
        # Create image item for R-D map
        self.img = pg.ImageItem()
        self.addItem(self.img)
        
        # Create colormap (jet-like for radar)
        colors = [
            (0, 0, 0),       # Black
            (0, 0, 128),     # Dark blue
            (0, 128, 255),   # Light blue
            (0, 255, 0),     # Green
            (255, 255, 0),   # Yellow
            (255, 128, 0),   # Orange
            (255, 0, 0),     # Red
            (255, 255, 255), # White
        ]
        cmap = pg.ColorMap(np.linspace(0, 1, len(colors)), 
                          [pg.mkColor(*c) for c in colors])
        self.img.setLookupTable(cmap.getLookupTable())
        
        # Set axes
        self.img.setRect(0, -max_velocity_mps, max_range_km, 2 * max_velocity_mps)
        
        # Detection overlay
        self.detection_scatter = pg.ScatterPlotItem(size=10, 
                                                     pen=pg.mkPen('w', width=2),
                                                     brush=pg.mkBrush(None))
        self.addItem(self.detection_scatter)
    
    def update_map(self, rd_map: np.ndarray, detections: List[Tuple[float, float]] = None):
        """Update Range-Doppler map display"""
        # Convert to dB
        rd_db = 10 * np.log10(rd_map + 1e-10)
        rd_db = np.clip(rd_db, -10, 40)
        
        # Normalize
        rd_norm = (rd_db + 10) / 50  # Map -10..40 dB to 0..1
        
        self.img.setImage(rd_norm.T)
        
        # Update detections overlay
        if detections:
            ranges = [d[0] / 1000 for d in detections]  # Convert to km
            velocities = [d[1] for d in detections]
            self.detection_scatter.setData(ranges, velocities)


#===============================================================================
# Track Table Widget
#===============================================================================

class TrackTable(QTableWidget):
    """Table showing all active tracks"""
    
    def __init__(self):
        super().__init__()
        
        # Configure table
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels([
            'ID', 'Range (km)', 'Azimuth (°)', 'Velocity (m/s)', 
            'RCS (m²)', 'SNR (dB)', 'Threat', 'Age (s)'
        ])
        
        # Style
        self.setStyleSheet("""
            QTableWidget {
                background-color: #000000;
                color: #00FF00;
                gridline-color: #004400;
                font-family: 'Courier New';
            }
            QHeaderView::section {
                background-color: #002200;
                color: #00FF00;
                font-weight: bold;
            }
        """)
        
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
    
    def update_tracks(self, tracks: List[Track]):
        """Update table with current tracks"""
        self.setRowCount(len(tracks))
        
        for row, track in enumerate(tracks):
            # ID
            self.setItem(row, 0, QTableWidgetItem(f"T{track.track_id:03d}"))
            
            # Range
            self.setItem(row, 1, QTableWidgetItem(f"{track.range_m/1000:.1f}"))
            
            # Azimuth
            self.setItem(row, 2, QTableWidgetItem(f"{track.azimuth_deg:.1f}"))
            
            # Velocity
            vel_item = QTableWidgetItem(f"{track.velocity_mps:.0f}")
            if abs(track.velocity_mps) > 300:  # Supersonic
                vel_item.setForeground(QColor(255, 191, 0))
            self.setItem(row, 3, vel_item)
            
            # RCS
            rcs_item = QTableWidgetItem(f"{track.rcs_m2:.2f}")
            if track.rcs_m2 < 1.0:  # Low RCS = potential stealth
                rcs_item.setForeground(QColor(255, 0, 0))
            self.setItem(row, 4, rcs_item)
            
            # SNR
            snr_item = QTableWidgetItem(f"{track.snr_db:.1f}")
            if track.snr_db < 15:
                snr_item.setForeground(QColor(255, 191, 0))
            self.setItem(row, 5, snr_item)
            
            # Threat
            threat_item = QTableWidgetItem(track.threat.value)
            if track.threat == ThreatLevel.HOSTILE:
                threat_item.setForeground(QColor(255, 0, 0))
            elif track.threat == ThreatLevel.STEALTH:
                threat_item.setForeground(QColor(255, 191, 0))
            self.setItem(row, 6, threat_item)
            
            # Age
            self.setItem(row, 7, QTableWidgetItem(f"{track.age_seconds:.1f}"))


#===============================================================================
# System Status Panel
#===============================================================================

class StatusPanel(QGroupBox):
    """System status display panel"""
    
    def __init__(self):
        super().__init__("SYSTEM STATUS")
        
        self.setStyleSheet("""
            QGroupBox {
                color: #00FF00;
                font-weight: bold;
                border: 1px solid #004400;
            }
            QLabel {
                color: #00FF00;
                font-family: 'Courier New';
            }
        """)
        
        layout = QGridLayout()
        
        # Status indicators
        self.labels = {}
        
        indicators = [
            ('MODE', 'SIMULATION'),
            ('PRBS', 'PRBS-15'),
            ('PROC GAIN', '45.2 dB'),
            ('TX POWER', '60 W'),
            ('FREQUENCY', '155 MHz'),
            ('BANDWIDTH', '100 MHz'),
            ('RANGE RES', '1.5 m'),
            ('VEL RES', '9.7 m/s'),
            ('TRACKS', '0'),
            ('UPTIME', '00:00:00'),
        ]
        
        for i, (name, default) in enumerate(indicators):
            row = i // 2
            col = (i % 2) * 2
            
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet("color: #008800;")
            layout.addWidget(name_label, row, col)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #00FF00; font-weight: bold;")
            layout.addWidget(value_label, row, col + 1)
            
            self.labels[name] = value_label
        
        self.setLayout(layout)
    
    def update_status(self, key: str, value: str):
        """Update a status indicator"""
        if key in self.labels:
            self.labels[key].setText(value)


#===============================================================================
# Main TITAN Tactical Display
#===============================================================================

class TITANTacticalDisplay(QMainWindow):
    """
    TITAN VHF Anti-Stealth Radar - Main Tactical Display
    
    Features:
    - Real-time PPI (Plan Position Indicator)
    - Range-Doppler map visualization
    - Multi-target tracking with history
    - Threat classification
    - KML export for Google Earth
    """
    
    def __init__(self, mode: str = 'simulation'):
        super().__init__()
        
        self.mode = mode
        self.tracks: Dict[int, Track] = {}
        self.start_time = time.time()
        self.frame_count = 0
        
        # Initialize simulator (or hardware interface)
        if mode == 'simulation':
            self.simulator = RadarSimulator(num_targets=4)
        else:
            self.simulator = None
            # TODO: Initialize hardware interface
            # from titan_rfsoc_driver import TITANRFSoC
            # self.hardware = TITANRFSoC()
        
        self.setup_ui()
        self.setup_timers()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle(f"TITAN v2.0 - VHF Anti-Stealth Radar | Mode: {self.mode.upper()}")
        self.resize(1600, 1000)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QGroupBox {
                color: #00FF00;
                border: 1px solid #004400;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left panel - PPI and controls
        left_panel = QVBoxLayout()
        
        # PPI Display
        ppi_group = QGroupBox("PPI TACTICAL SCOPE")
        ppi_layout = QVBoxLayout(ppi_group)
        self.ppi = PPIDisplay(max_range_km=100)
        ppi_layout.addWidget(self.ppi)
        left_panel.addWidget(ppi_group, stretch=3)
        
        # Status Panel
        self.status_panel = StatusPanel()
        left_panel.addWidget(self.status_panel, stretch=1)
        
        # Right panel - R-D map and tracks
        right_panel = QVBoxLayout()
        
        # Tabs for different views
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #004400;
                background-color: #000000;
            }
            QTabBar::tab {
                background-color: #002200;
                color: #00FF00;
                padding: 8px 20px;
            }
            QTabBar::tab:selected {
                background-color: #004400;
            }
        """)
        
        # Range-Doppler tab
        rd_widget = QWidget()
        rd_layout = QVBoxLayout(rd_widget)
        self.rd_display = RangeDopplerDisplay()
        rd_layout.addWidget(self.rd_display)
        tabs.addTab(rd_widget, "Range-Doppler")
        
        # Doppler Filter Bank tab
        dfb_widget = QWidget()
        dfb_layout = QVBoxLayout(dfb_widget)
        self.dfb_display = pg.PlotWidget(title="DOPPLER FILTER BANK OUTPUT")
        self.dfb_display.setLabel('bottom', 'Doppler Bin')
        self.dfb_display.setLabel('left', 'Amplitude (dB)')
        self.dfb_curve = self.dfb_display.plot(pen=pg.mkPen('g', width=2))
        dfb_layout.addWidget(self.dfb_display)
        tabs.addTab(dfb_widget, "Doppler Bank")
        
        # Range Profile tab
        rp_widget = QWidget()
        rp_layout = QVBoxLayout(rp_widget)
        self.rp_display = pg.PlotWidget(title="RANGE PROFILE (CORRELATION OUTPUT)")
        self.rp_display.setLabel('bottom', 'Range', units='km')
        self.rp_display.setLabel('left', 'Amplitude (dB)')
        self.rp_curve = self.rp_display.plot(pen=pg.mkPen('g', width=2))
        self.rp_threshold = self.rp_display.plot(pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        rp_layout.addWidget(self.rp_display)
        tabs.addTab(rp_widget, "Range Profile")
        
        right_panel.addWidget(tabs, stretch=2)
        
        # Track Table
        track_group = QGroupBox("ACTIVE TRACKS")
        track_layout = QVBoxLayout(track_group)
        self.track_table = TrackTable()
        track_layout.addWidget(self.track_table)
        right_panel.addWidget(track_group, stretch=1)
        
        # Add panels to main layout
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 800])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("color: #00FF00; background-color: #001100;")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("SYSTEM ONLINE | WAITING FOR TARGETS...")
    
    def setup_timers(self):
        """Setup update timers"""
        # Main update timer (20 Hz)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)  # 50 ms = 20 Hz
        
        # Slow update timer (1 Hz) for status
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
    
    def update_display(self):
        """Main display update loop"""
        self.frame_count += 1
        current_time = time.time() - self.start_time
        
        # Get radar data (simulation or hardware)
        if self.simulator:
            detections = self.simulator.update()
            rd_map = self.simulator.generate_range_doppler_map()
        else:
            # TODO: Get from hardware
            detections = []
            rd_map = np.zeros((256, 512))
        
        # Update tracks
        for det in detections:
            track_id = det['track_id']
            
            if track_id in self.tracks:
                # Update existing track
                self.tracks[track_id].update(
                    det['range_m'], det['azimuth_deg'], det['velocity_mps'],
                    det['snr_db'], current_time
                )
            else:
                # Create new track
                track = Track(
                    track_id=track_id,
                    range_m=det['range_m'],
                    azimuth_deg=det['azimuth_deg'],
                    velocity_mps=det['velocity_mps'],
                    rcs_m2=det['rcs_m2'],
                    snr_db=det['snr_db'],
                )
                # Classify threat
                track.threat = self.classify_threat(track)
                track.history_x.append(track.x)
                track.history_y.append(track.y)
                track.history_time.append(current_time)
                self.tracks[track_id] = track
        
        # Update track ages
        for track in self.tracks.values():
            if track.history_time:
                track.age_seconds = current_time - track.history_time[0]
        
        # Update displays
        track_list = list(self.tracks.values())
        
        # PPI
        self.ppi.update_tracks(track_list)
        self.ppi.update_sweep((current_time * 30) % 360)
        
        # Range-Doppler
        det_points = [(t.range_m, t.velocity_mps) for t in track_list]
        self.rd_display.update_map(rd_map, det_points)
        
        # Range profile (sum across Doppler)
        range_profile = np.max(rd_map, axis=0)
        range_axis = np.linspace(0, 100, len(range_profile))
        range_db = 10 * np.log10(range_profile + 1e-10)
        self.rp_curve.setData(range_axis, range_db)
        self.rp_threshold.setData(range_axis, np.ones_like(range_axis) * 15)
        
        # Doppler profile (sum across range)
        doppler_profile = np.max(rd_map, axis=1)
        doppler_db = 10 * np.log10(doppler_profile + 1e-10)
        self.dfb_curve.setData(doppler_db)
        
        # Track table
        self.track_table.update_tracks(track_list)
        
        # Status bar
        fps = self.frame_count / max(current_time, 0.001)
        self.statusBar.showMessage(
            f"TRACKS: {len(track_list)} | "
            f"TIME: {current_time:.1f}s | "
            f"FPS: {fps:.1f} | "
            f"MODE: {'SIM' if self.simulator else 'HW'}"
        )
    
    def update_status(self):
        """Update status panel (1 Hz)"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        self.status_panel.update_status('UPTIME', f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.status_panel.update_status('TRACKS', str(len(self.tracks)))
        self.status_panel.update_status('MODE', self.mode.upper())
    
    def classify_threat(self, track: Track) -> ThreatLevel:
        """Classify target threat level based on characteristics"""
        # Low RCS + high speed = potential stealth
        if track.rcs_m2 < 1.0 and abs(track.velocity_mps) > 200:
            return ThreatLevel.STEALTH
        
        # Very low RCS
        if track.rcs_m2 < 0.1:
            return ThreatLevel.STEALTH
        
        # High speed approaching
        if track.velocity_mps < -300:  # Negative = approaching
            return ThreatLevel.HOSTILE
        
        # Large slow target
        if track.rcs_m2 > 50 and abs(track.velocity_mps) < 150:
            return ThreatLevel.NEUTRAL
        
        return ThreatLevel.UNKNOWN
    
    def export_kml(self, filename: str, origin_lat: float = 45.815, 
                   origin_lon: float = 15.982):
        """Export current tracks to KML for Google Earth"""
        kml = ['<?xml version="1.0" encoding="UTF-8"?>']
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append(f'<name>TITAN Radar Tracks - {datetime.now().isoformat()}</name>')
        
        for track in self.tracks.values():
            # Convert local X,Y to lat/lon (approximate)
            lat = origin_lat + (track.y / 1000) / 111
            lon = origin_lon + (track.x / 1000) / (111 * np.cos(np.radians(origin_lat)))
            
            kml.append(f'''
            <Placemark>
                <name>Track {track.track_id} - {track.threat.value}</name>
                <description>
                    Range: {track.range_m/1000:.1f} km
                    Velocity: {track.velocity_mps:.0f} m/s
                    RCS: {track.rcs_m2:.2f} m²
                </description>
                <Point>
                    <coordinates>{lon},{lat},5000</coordinates>
                </Point>
            </Placemark>
            ''')
        
        kml.append('</Document></kml>')
        
        with open(filename, 'w') as f:
            f.write('\n'.join(kml))
        
        print(f"KML exported to: {filename}")


#===============================================================================
# Main Entry Point
#===============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TITAN VHF Radar Tactical Display')
    parser.add_argument('--mode', choices=['simulation', 'hardware'], 
                       default='simulation', help='Operating mode')
    parser.add_argument('--kml', type=str, help='Enable KML export to file')
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(0, 0, 0))
    palette.setColor(QPalette.WindowText, QColor(0, 255, 0))
    palette.setColor(QPalette.Base, QColor(0, 17, 0))
    palette.setColor(QPalette.Text, QColor(0, 255, 0))
    app.setPalette(palette)
    
    display = TITANTacticalDisplay(mode=args.mode)
    
    if args.fullscreen:
        display.showFullScreen()
    else:
        display.show()
    
    # KML export timer (if enabled)
    if args.kml:
        kml_timer = QTimer()
        kml_timer.timeout.connect(lambda: display.export_kml(args.kml))
        kml_timer.start(5000)  # Export every 5 seconds
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
