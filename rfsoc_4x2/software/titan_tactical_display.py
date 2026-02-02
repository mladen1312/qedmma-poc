#!/usr/bin/env python3
"""
TITAN VHF Anti-Stealth Radar - Tactical Display System
Real-Time PPI Display, Range-Doppler Map & Multi-Target Tracking

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Based on SVETOVID UI architecture, enhanced for TITAN RFSoC 4x2 platform.

Features:
- Plan Position Indicator (PPI) with range rings
- Range-Doppler Map (waterfall/intensity)
- Multi-target tracking with trail history
- Real-time detection list
- System status panel
- Google Earth KML export
- Configurable themes (Classic Green, Modern Blue, Night Red)

Requirements:
    pip install PyQt5 pyqtgraph numpy scipy

Usage:
    python titan_tactical_display.py                    # Simulation mode
    python titan_tactical_display.py --hardware         # With RFSoC hardware
    python titan_tactical_display.py --theme modern     # Blue theme
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import json
import time

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QSplitter, QStatusBar,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QFrame, QProgressBar
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette

# PyQtGraph for high-performance plotting
import pyqtgraph as pg
from pyqtgraph import ColorMap


#===============================================================================
# Constants & Configuration
#===============================================================================

C = 299792458.0  # Speed of light (m/s)


class Theme(Enum):
    """Display themes"""
    CLASSIC = "classic"    # Green on black (traditional radar)
    MODERN = "modern"      # Blue tones
    NIGHT = "night"        # Red for night vision
    HIGH_CONTRAST = "high_contrast"


THEMES = {
    Theme.CLASSIC: {
        'background': '#000000',
        'foreground': '#00FF00',
        'grid': '#004400',
        'target': '#FF0000',
        'track': '#00FF00',
        'ring': '#00AA00',
        'text': '#00FF00',
        'alert': '#FFFF00',
    },
    Theme.MODERN: {
        'background': '#0a1628',
        'foreground': '#00BFFF',
        'grid': '#1a3a5c',
        'target': '#FF6B6B',
        'track': '#4ECDC4',
        'ring': '#2980B9',
        'text': '#E0E0E0',
        'alert': '#FFD93D',
    },
    Theme.NIGHT: {
        'background': '#1a0000',
        'foreground': '#FF4444',
        'grid': '#330000',
        'target': '#FFFF00',
        'track': '#FF6666',
        'ring': '#660000',
        'text': '#FF8888',
        'alert': '#FFFFFF',
    },
    Theme.HIGH_CONTRAST: {
        'background': '#000000',
        'foreground': '#FFFFFF',
        'grid': '#333333',
        'target': '#FF0000',
        'track': '#00FF00',
        'ring': '#666666',
        'text': '#FFFFFF',
        'alert': '#FFFF00',
    },
}


@dataclass
class TargetTrack:
    """Single target track"""
    track_id: int
    x: float = 0.0
    y: float = 0.0
    range_m: float = 0.0
    azimuth_deg: float = 0.0
    velocity_mps: float = 0.0
    heading_deg: float = 0.0
    snr_db: float = 0.0
    rcs_m2: float = 0.0
    classification: str = "UNKNOWN"
    threat_level: str = "LOW"
    history_x: List[float] = field(default_factory=list)
    history_y: List[float] = field(default_factory=list)
    last_update: float = 0.0
    age_seconds: float = 0.0
    
    def update_position(self, x: float, y: float, max_history: int = 50):
        """Update track position and history"""
        self.x = x
        self.y = y
        self.range_m = np.sqrt(x**2 + y**2)
        self.azimuth_deg = np.degrees(np.arctan2(y, x)) % 360
        
        self.history_x.append(x)
        self.history_y.append(y)
        
        if len(self.history_x) > max_history:
            self.history_x.pop(0)
            self.history_y.pop(0)
        
        self.last_update = time.time()


@dataclass
class RadarConfig:
    """Radar display configuration"""
    max_range_m: float = 150000  # 150 km
    range_rings: List[float] = field(default_factory=lambda: [25000, 50000, 75000, 100000, 125000, 150000])
    azimuth_lines: int = 12  # Every 30°
    update_rate_hz: float = 20.0
    trail_length: int = 50
    doppler_bins: int = 256
    range_bins: int = 512
    theme: Theme = Theme.CLASSIC


#===============================================================================
# Range-Doppler Map Widget
#===============================================================================

class RangeDopplerWidget(pg.PlotWidget):
    """Real-time Range-Doppler intensity map"""
    
    def __init__(self, config: RadarConfig, parent=None):
        super().__init__(parent)
        self.config = config
        
        self.setTitle("RANGE-DOPPLER MAP")
        self.setLabel('left', 'Doppler Bin')
        self.setLabel('bottom', 'Range (km)')
        
        # Create image item for intensity display
        self.img = pg.ImageItem()
        self.addItem(self.img)
        
        # Color map (radar intensity)
        colors = [
            (0, 0, 0),
            (0, 0, 128),
            (0, 128, 0),
            (128, 128, 0),
            (255, 0, 0),
            (255, 255, 255)
        ]
        cmap = ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.img.setLookupTable(cmap.getLookupTable())
        
        # Initialize empty data
        self.rd_data = np.zeros((config.doppler_bins, config.range_bins))
        self.update_display(self.rd_data)
    
    def update_display(self, rd_map: np.ndarray):
        """Update Range-Doppler map display"""
        # Convert to dB scale
        rd_db = 20 * np.log10(np.abs(rd_map) + 1e-10)
        
        # Normalize to 0-255
        rd_min = np.percentile(rd_db, 5)
        rd_max = np.percentile(rd_db, 99)
        rd_norm = np.clip((rd_db - rd_min) / (rd_max - rd_min + 1e-10), 0, 1)
        
        self.img.setImage(rd_norm.T, autoLevels=False)
        
        # Set axis scaling
        self.img.setRect(0, 0, self.config.max_range_m/1000, self.config.doppler_bins)


#===============================================================================
# PPI (Plan Position Indicator) Widget
#===============================================================================

class PPIWidget(pg.PlotWidget):
    """Traditional circular PPI radar display"""
    
    def __init__(self, config: RadarConfig, theme_colors: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.colors = theme_colors
        
        self.setTitle("PPI TRACKING SCOPE")
        self.setAspectLocked(True)
        
        max_r = config.max_range_m / 1000  # km
        self.setRange(xRange=[-max_r, max_r], yRange=[-max_r, max_r])
        self.setLabel('left', 'North-South (km)')
        self.setLabel('bottom', 'East-West (km)')
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Draw range rings
        self._draw_range_rings()
        
        # Draw azimuth lines
        self._draw_azimuth_lines()
        
        # Radar center marker
        self.radar_center = self.plot(
            [0], [0], 
            symbol='+', 
            symbolSize=20, 
            symbolBrush=self.colors['target']
        )
        
        # Target scatter plot
        self.target_scatter = pg.ScatterPlotItem(
            size=15, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(self.colors['target'])
        )
        self.addItem(self.target_scatter)
        
        # Track trails (one per track)
        self.track_plots = {}
        
        # Sweep line (rotating radar beam simulation)
        self.sweep_angle = 0
        self.sweep_line = self.plot(
            [0, 0], [0, 0],
            pen=pg.mkPen(self.colors['foreground'], width=2)
        )
    
    def _draw_range_rings(self):
        """Draw concentric range rings"""
        for r_m in self.config.range_rings:
            r_km = r_m / 1000
            
            # Create circle
            theta = np.linspace(0, 2*np.pi, 100)
            x = r_km * np.cos(theta)
            y = r_km * np.sin(theta)
            
            self.plot(x, y, pen=pg.mkPen(self.colors['ring'], width=1))
            
            # Range label
            text = pg.TextItem(f"{int(r_km)} km", color=self.colors['text'])
            text.setPos(r_km * 0.7, r_km * 0.7)
            self.addItem(text)
    
    def _draw_azimuth_lines(self):
        """Draw azimuth reference lines"""
        max_r = self.config.max_range_m / 1000
        
        for i in range(self.config.azimuth_lines):
            angle = i * 360 / self.config.azimuth_lines
            angle_rad = np.radians(angle)
            
            x = [0, max_r * np.cos(angle_rad)]
            y = [0, max_r * np.sin(angle_rad)]
            
            self.plot(x, y, pen=pg.mkPen(self.colors['grid'], width=1, style=Qt.DashLine))
            
            # Cardinal direction labels
            if angle == 0:
                label = "E"
            elif angle == 90:
                label = "N"
            elif angle == 180:
                label = "W"
            elif angle == 270:
                label = "S"
            else:
                label = f"{int(angle)}°"
            
            text = pg.TextItem(label, color=self.colors['text'])
            text.setPos(max_r * 1.05 * np.cos(angle_rad), max_r * 1.05 * np.sin(angle_rad))
            self.addItem(text)
    
    def update_sweep(self, angle_deg: float):
        """Update rotating sweep line"""
        max_r = self.config.max_range_m / 1000
        angle_rad = np.radians(angle_deg)
        
        self.sweep_line.setData(
            [0, max_r * np.cos(angle_rad)],
            [0, max_r * np.sin(angle_rad)]
        )
    
    def update_targets(self, tracks: Dict[int, TargetTrack]):
        """Update target positions and trails"""
        # Collect all current target positions
        x_list = []
        y_list = []
        
        for track_id, track in tracks.items():
            x_km = track.x / 1000
            y_km = track.y / 1000
            x_list.append(x_km)
            y_list.append(y_km)
            
            # Update or create trail plot
            if track_id not in self.track_plots:
                self.track_plots[track_id] = self.plot(
                    pen=pg.mkPen(self.colors['track'], width=2, style=Qt.DashLine)
                )
            
            # Update trail
            history_x_km = [x/1000 for x in track.history_x]
            history_y_km = [y/1000 for y in track.history_y]
            self.track_plots[track_id].setData(history_x_km, history_y_km)
        
        # Update scatter plot with current positions
        self.target_scatter.setData(x_list, y_list)
        
        # Remove stale track plots
        stale_ids = set(self.track_plots.keys()) - set(tracks.keys())
        for track_id in stale_ids:
            self.removeItem(self.track_plots[track_id])
            del self.track_plots[track_id]


#===============================================================================
# Detection Table Widget
#===============================================================================

class DetectionTableWidget(QTableWidget):
    """Table showing all current detections"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setColumnCount(9)
        self.setHorizontalHeaderLabels([
            "ID", "Range (km)", "Azimuth (°)", "Velocity (m/s)",
            "SNR (dB)", "RCS (m²)", "Class", "Threat", "Age (s)"
        ])
        
        # Style
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
    
    def update_tracks(self, tracks: Dict[int, TargetTrack]):
        """Update table with current tracks"""
        self.setRowCount(len(tracks))
        
        for row, (track_id, track) in enumerate(sorted(tracks.items())):
            self.setItem(row, 0, QTableWidgetItem(f"T{track_id:03d}"))
            self.setItem(row, 1, QTableWidgetItem(f"{track.range_m/1000:.1f}"))
            self.setItem(row, 2, QTableWidgetItem(f"{track.azimuth_deg:.1f}"))
            self.setItem(row, 3, QTableWidgetItem(f"{track.velocity_mps:.1f}"))
            self.setItem(row, 4, QTableWidgetItem(f"{track.snr_db:.1f}"))
            self.setItem(row, 5, QTableWidgetItem(f"{track.rcs_m2:.1f}"))
            self.setItem(row, 6, QTableWidgetItem(track.classification))
            self.setItem(row, 7, QTableWidgetItem(track.threat_level))
            self.setItem(row, 8, QTableWidgetItem(f"{track.age_seconds:.1f}"))
            
            # Color coding by threat level
            if track.threat_level == "HIGH":
                for col in range(9):
                    self.item(row, col).setBackground(QColor(128, 0, 0))
            elif track.threat_level == "MEDIUM":
                for col in range(9):
                    self.item(row, col).setBackground(QColor(128, 128, 0))


#===============================================================================
# System Status Widget
#===============================================================================

class SystemStatusWidget(QGroupBox):
    """System status and health indicators"""
    
    def __init__(self, parent=None):
        super().__init__("SYSTEM STATUS", parent)
        
        layout = QGridLayout()
        self.setLayout(layout)
        
        # Status indicators
        self.labels = {}
        
        indicators = [
            ("Mode", "SIMULATION"),
            ("PRBS", "PRBS-15"),
            ("TX Power", "0 W"),
            ("RX Gain", "0 dB"),
            ("Proc. Gain", "0 dB"),
            ("Detections", "0"),
            ("CPU Load", "0%"),
            ("Temp", "0°C"),
        ]
        
        for i, (name, default) in enumerate(indicators):
            row = i // 2
            col = (i % 2) * 2
            
            label = QLabel(f"{name}:")
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, row, col)
            
            value = QLabel(default)
            value.setStyleSheet("font-family: monospace;")
            layout.addWidget(value, row, col + 1)
            
            self.labels[name] = value
    
    def update_status(self, status: dict):
        """Update status display"""
        for key, value in status.items():
            if key in self.labels:
                self.labels[key].setText(str(value))


#===============================================================================
# Main Tactical Display Window
#===============================================================================

class TITANTacticalDisplay(QMainWindow):
    """
    TITAN Radar Tactical Display
    
    Main window integrating:
    - PPI scope (circular display)
    - Range-Doppler map
    - Detection table
    - System status
    - Controls
    """
    
    def __init__(self, config: RadarConfig = None, hardware_mode: bool = False):
        super().__init__()
        
        self.config = config or RadarConfig()
        self.hardware_mode = hardware_mode
        self.theme_colors = THEMES[self.config.theme]
        
        # Track storage
        self.tracks: Dict[int, TargetTrack] = {}
        self.next_track_id = 1
        
        # Simulation state
        self.sim_time = 0
        self.sweep_angle = 0
        
        # Setup UI
        self._setup_window()
        self._setup_menu()
        self._setup_central_widget()
        self._setup_status_bar()
        self._apply_theme()
        
        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(int(1000 / self.config.update_rate_hz))
        
        # Hardware connection (if enabled)
        self.processor = None
        if hardware_mode:
            self._connect_hardware()
    
    def _setup_window(self):
        """Configure main window"""
        self.setWindowTitle(f"TITAN v2.0 - VHF Anti-Stealth Radar Tactical Display")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 800)
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Export KML...", self._export_kml)
        file_menu.addAction("Export Track Log...", self._export_tracks)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("Classic Theme", lambda: self._set_theme(Theme.CLASSIC))
        view_menu.addAction("Modern Theme", lambda: self._set_theme(Theme.MODERN))
        view_menu.addAction("Night Theme", lambda: self._set_theme(Theme.NIGHT))
        
        # Radar menu
        radar_menu = menubar.addMenu("&Radar")
        radar_menu.addAction("Start Transmission", self._start_tx)
        radar_menu.addAction("Stop Transmission", self._stop_tx)
        radar_menu.addSeparator()
        radar_menu.addAction("Calibrate", self._calibrate)
    
    def _setup_central_widget(self):
        """Setup central widget with all displays"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        
        # Left panel: PPI + Range-Doppler
        left_splitter = QSplitter(Qt.Vertical)
        
        # PPI Display
        self.ppi = PPIWidget(self.config, self.theme_colors)
        left_splitter.addWidget(self.ppi)
        
        # Range-Doppler Map
        self.rd_map = RangeDopplerWidget(self.config)
        left_splitter.addWidget(self.rd_map)
        
        left_splitter.setSizes([600, 300])
        main_layout.addWidget(left_splitter, stretch=3)
        
        # Right panel: Controls + Table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # System Status
        self.status_widget = SystemStatusWidget()
        right_layout.addWidget(self.status_widget)
        
        # Control Panel
        control_group = QGroupBox("CONTROLS")
        control_layout = QGridLayout()
        control_group.setLayout(control_layout)
        
        # PRBS Order
        control_layout.addWidget(QLabel("PRBS Order:"), 0, 0)
        self.prbs_combo = QComboBox()
        self.prbs_combo.addItems(["PRBS-10", "PRBS-11", "PRBS-15", "PRBS-18", "PRBS-20"])
        self.prbs_combo.setCurrentText("PRBS-15")
        control_layout.addWidget(self.prbs_combo, 0, 1)
        
        # TX Power
        control_layout.addWidget(QLabel("TX Power (W):"), 1, 0)
        self.tx_power_spin = QSpinBox()
        self.tx_power_spin.setRange(0, 1000)
        self.tx_power_spin.setValue(100)
        control_layout.addWidget(self.tx_power_spin, 1, 1)
        
        # Detection Threshold
        control_layout.addWidget(QLabel("CFAR Threshold:"), 2, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(6, 30)
        self.threshold_spin.setValue(13)
        self.threshold_spin.setSuffix(" dB")
        control_layout.addWidget(self.threshold_spin, 2, 1)
        
        # Buttons
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setStyleSheet("background-color: #006400; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self._toggle_radar)
        control_layout.addWidget(self.start_btn, 3, 0, 1, 2)
        
        self.sim_btn = QPushButton("🎯 ADD TARGET")
        self.sim_btn.clicked.connect(self._add_simulated_target)
        control_layout.addWidget(self.sim_btn, 4, 0, 1, 2)
        
        right_layout.addWidget(control_group)
        
        # Detection Table
        table_group = QGroupBox("TRACK TABLE")
        table_layout = QVBoxLayout()
        table_group.setLayout(table_layout)
        
        self.track_table = DetectionTableWidget()
        table_layout.addWidget(self.track_table)
        
        right_layout.addWidget(table_group, stretch=1)
        
        main_layout.addWidget(right_panel, stretch=1)
    
    def _setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.time_label = QLabel()
        self.status_bar.addPermanentWidget(self.time_label)
        
        self.status_bar.showMessage("SYSTEM ONLINE | MODE: SIMULATION")
    
    def _apply_theme(self):
        """Apply current theme colors"""
        colors = self.theme_colors
        
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {colors['background']};
            }}
            QWidget {{
                background-color: {colors['background']};
                color: {colors['text']};
            }}
            QGroupBox {{
                border: 1px solid {colors['foreground']};
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                color: {colors['foreground']};
            }}
            QTableWidget {{
                gridline-color: {colors['grid']};
            }}
            QHeaderView::section {{
                background-color: {colors['grid']};
                color: {colors['text']};
            }}
            QPushButton {{
                border: 1px solid {colors['foreground']};
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {colors['grid']};
            }}
        """)
        
        # Configure PyQtGraph
        pg.setConfigOption('background', colors['background'])
        pg.setConfigOption('foreground', colors['foreground'])
    
    def _set_theme(self, theme: Theme):
        """Change display theme"""
        self.config.theme = theme
        self.theme_colors = THEMES[theme]
        self._apply_theme()
    
    def _update_loop(self):
        """Main update loop - called by timer"""
        self.sim_time += 1 / self.config.update_rate_hz
        
        # Update sweep angle
        self.sweep_angle = (self.sweep_angle + 2) % 360
        self.ppi.update_sweep(self.sweep_angle)
        
        # Get radar data (simulation or hardware)
        if self.hardware_mode and self.processor:
            self._process_hardware_data()
        else:
            self._update_simulation()
        
        # Update displays
        self.ppi.update_targets(self.tracks)
        self.track_table.update_tracks(self.tracks)
        
        # Update status
        status = {
            "Mode": "HARDWARE" if self.hardware_mode else "SIMULATION",
            "PRBS": self.prbs_combo.currentText(),
            "TX Power": f"{self.tx_power_spin.value()} W",
            "Detections": len(self.tracks),
            "CPU Load": f"{np.random.randint(10, 30)}%",
            "Temp": f"{np.random.randint(40, 55)}°C",
        }
        self.status_widget.update_status(status)
        
        # Update time
        self.time_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        # Update Range-Doppler map (simulated)
        self._update_rd_map()
    
    def _update_simulation(self):
        """Update simulation targets"""
        current_time = time.time()
        
        for track_id, track in list(self.tracks.items()):
            # Move target
            if hasattr(track, 'sim_velocity') and hasattr(track, 'sim_heading'):
                vx = track.sim_velocity * np.cos(np.radians(track.sim_heading))
                vy = track.sim_velocity * np.sin(np.radians(track.sim_heading))
                
                new_x = track.x + vx / self.config.update_rate_hz
                new_y = track.y + vy / self.config.update_rate_hz
                
                # Add noise
                new_x += np.random.normal(0, 50)
                new_y += np.random.normal(0, 50)
                
                track.update_position(new_x, new_y, self.config.trail_length)
                track.velocity_mps = track.sim_velocity
                track.heading_deg = track.sim_heading
            
            # Update age
            track.age_seconds = current_time - track.last_update
            
            # Remove old tracks
            if track.age_seconds > 30:
                del self.tracks[track_id]
    
    def _update_rd_map(self):
        """Update Range-Doppler map display"""
        # Generate simulated R-D map
        rd_map = np.random.randn(self.config.doppler_bins, self.config.range_bins) * 0.1
        
        # Add targets as bright spots
        for track in self.tracks.values():
            range_bin = int(track.range_m / self.config.max_range_m * self.config.range_bins)
            doppler_bin = self.config.doppler_bins // 2 + int(track.velocity_mps / 10)
            
            if 0 <= range_bin < self.config.range_bins and 0 <= doppler_bin < self.config.doppler_bins:
                # Create target blob
                for dr in range(-3, 4):
                    for dd in range(-3, 4):
                        r = range_bin + dr
                        d = doppler_bin + dd
                        if 0 <= r < self.config.range_bins and 0 <= d < self.config.doppler_bins:
                            intensity = track.snr_db / 30 * np.exp(-(dr**2 + dd**2) / 4)
                            rd_map[d, r] += intensity
        
        self.rd_map.update_display(rd_map)
    
    def _add_simulated_target(self):
        """Add a simulated target for testing"""
        track = TargetTrack(track_id=self.next_track_id)
        self.next_track_id += 1
        
        # Random initial position (50-100 km range)
        range_m = np.random.uniform(50000, 100000)
        azimuth_rad = np.random.uniform(0, 2 * np.pi)
        
        x = range_m * np.cos(azimuth_rad)
        y = range_m * np.sin(azimuth_rad)
        
        track.update_position(x, y)
        
        # Simulation parameters
        track.sim_velocity = np.random.uniform(100, 400)  # m/s
        track.sim_heading = np.random.uniform(0, 360)
        
        # Detection parameters
        track.snr_db = np.random.uniform(15, 35)
        track.rcs_m2 = np.random.uniform(1, 20)
        track.classification = np.random.choice(["AIRCRAFT", "UAV", "HELICOPTER", "UNKNOWN"])
        track.threat_level = np.random.choice(["LOW", "MEDIUM", "HIGH"])
        
        self.tracks[track.track_id] = track
        
        self.status_bar.showMessage(
            f"TARGET ADDED: T{track.track_id:03d} at {track.range_m/1000:.1f} km, {track.azimuth_deg:.1f}°"
        )
    
    def _toggle_radar(self):
        """Toggle radar operation"""
        if self.start_btn.text() == "▶ START":
            self.start_btn.setText("■ STOP")
            self.start_btn.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;")
            self.status_bar.showMessage("RADAR ACTIVE | TRANSMITTING")
        else:
            self.start_btn.setText("▶ START")
            self.start_btn.setStyleSheet("background-color: #006400; color: white; font-weight: bold;")
            self.status_bar.showMessage("RADAR STANDBY")
    
    def _connect_hardware(self):
        """Connect to TITAN RFSoC hardware"""
        try:
            from titan_signal_processor import TITANProcessor, TITANConfig
            
            config = TITANConfig(
                prbs_order=15,
                num_range_bins=self.config.range_bins,
                num_doppler_bins=self.config.doppler_bins,
            )
            self.processor = TITANProcessor(config)
            self.status_bar.showMessage("HARDWARE CONNECTED | TITAN ONLINE")
            
        except ImportError as e:
            QMessageBox.warning(
                self, "Hardware Error",
                f"Could not connect to TITAN hardware:\n{e}\n\nRunning in simulation mode."
            )
            self.hardware_mode = False
    
    def _process_hardware_data(self):
        """Process data from TITAN hardware"""
        # TODO: Integrate with titan_rfsoc_driver
        pass
    
    def _start_tx(self):
        """Start radar transmission"""
        self.status_bar.showMessage("TRANSMISSION STARTED")
    
    def _stop_tx(self):
        """Stop radar transmission"""
        self.status_bar.showMessage("TRANSMISSION STOPPED")
    
    def _calibrate(self):
        """Run calibration routine"""
        QMessageBox.information(self, "Calibration", "Calibration routine not implemented.")
    
    def _export_kml(self):
        """Export tracks to Google Earth KML format"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export KML", "", "KML Files (*.kml)"
        )
        
        if filename:
            self._write_kml(filename)
            self.status_bar.showMessage(f"Exported to {filename}")
    
    def _write_kml(self, filename: str):
        """Write tracks to KML file"""
        # Reference location (radar position) - would be configurable
        ref_lat = 45.8150  # Zagreb
        ref_lon = 15.9819
        
        kml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        kml += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        kml += '<Document>\n'
        kml += f'  <name>TITAN Radar Tracks - {datetime.now().isoformat()}</name>\n'
        
        for track_id, track in self.tracks.items():
            # Convert local coordinates to lat/lon (simplified)
            # 1 degree ≈ 111 km
            lat = ref_lat + track.y / 111000
            lon = ref_lon + track.x / (111000 * np.cos(np.radians(ref_lat)))
            
            kml += f'''
  <Placemark>
    <name>T{track_id:03d} - {track.classification}</name>
    <description>
      Range: {track.range_m/1000:.1f} km
      Velocity: {track.velocity_mps:.0f} m/s
      SNR: {track.snr_db:.1f} dB
    </description>
    <Point>
      <coordinates>{lon},{lat},1000</coordinates>
    </Point>
  </Placemark>
'''
        
        kml += '</Document>\n</kml>'
        
        with open(filename, 'w') as f:
            f.write(kml)
    
    def _export_tracks(self):
        """Export track log to CSV"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Tracks", "", "CSV Files (*.csv)"
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write("track_id,range_m,azimuth_deg,velocity_mps,snr_db,rcs_m2,classification,threat\n")
                for track_id, track in self.tracks.items():
                    f.write(f"{track_id},{track.range_m:.1f},{track.azimuth_deg:.1f},"
                           f"{track.velocity_mps:.1f},{track.snr_db:.1f},{track.rcs_m2:.1f},"
                           f"{track.classification},{track.threat_level}\n")
            
            self.status_bar.showMessage(f"Exported to {filename}")


#===============================================================================
# Main Entry Point
#===============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TITAN Tactical Display")
    parser.add_argument("--hardware", action="store_true", help="Connect to RFSoC hardware")
    parser.add_argument("--theme", choices=["classic", "modern", "night"], default="classic",
                       help="Display theme")
    parser.add_argument("--range", type=float, default=150, help="Max range in km")
    
    args = parser.parse_args()
    
    # Configure
    config = RadarConfig(
        max_range_m=args.range * 1000,
        theme=Theme(args.theme),
    )
    
    # Start application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 10, 10))
    palette.setColor(QPalette.WindowText, QColor(0, 255, 0))
    app.setPalette(palette)
    
    # Create and show main window
    window = TITANTacticalDisplay(config, hardware_mode=args.hardware)
    window.show()
    
    # Add initial simulated targets
    for _ in range(3):
        window._add_simulated_target()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
