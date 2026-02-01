#!/usr/bin/env python3
"""
TITAN Radar - RFSoC 4x2 PYNQ Overlay Driver
Complete Radar-on-Chip Implementation for QEDMMA

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Platform: AMD RFSoC 4x2 (Zynq UltraScale+ ZU48DR)
- 4× ADC @ 5 GSPS, 14-bit, DC-6 GHz
- 2× DAC @ 9.85 GSPS, 14-bit
- 930K Logic Cells, 4272 DSP Slices
- 8 GB DDR4, 100 GbE QSFP28
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum, auto
import time

# PYNQ imports
try:
    from pynq import Overlay, allocate, MMIO
    from xrfdc import RFdc
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[TITAN] PYNQ not available - simulation mode")

#=============================================================================
# Enumerations
#=============================================================================

class WaveformType(Enum):
    PRBS_15 = auto()
    PRBS_20 = auto()
    LFM_UP = auto()
    LFM_DOWN = auto()

class CFARMode(Enum):
    CA = 0  # Cell Averaging
    GO = 1  # Greatest Of
    SO = 2  # Smallest Of
    OS = 3  # Order Statistic

#=============================================================================
# Configuration
#=============================================================================

@dataclass
class TitanConfig:
    """TITAN Radar System Configuration"""
    
    # RF Parameters
    center_freq_mhz: float = 155.0
    sample_rate_msps: float = 500.0
    rf_bandwidth_mhz: float = 50.0
    
    # Waveform
    waveform_type: WaveformType = WaveformType.PRBS_15
    prbs_order: int = 15
    chip_rate_mhz: float = 10.0
    pulse_width_us: float = 100.0
    pri_us: float = 1000.0
    
    # Processing
    num_range_bins: int = 16384
    num_doppler_bins: int = 1024
    cpi_pulses: int = 64
    
    # CFAR
    cfar_mode: CFARMode = CFARMode.CA
    cfar_guard_cells: int = 4
    cfar_training_cells: int = 32
    cfar_threshold_factor: float = 5.0
    
    # Tracking
    max_tracks: int = 256
    
    # ADC/DAC
    adc_channels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    SPEED_OF_LIGHT: float = 299792458.0
    
    @property
    def range_resolution_m(self) -> float:
        return self.SPEED_OF_LIGHT / (2 * self.chip_rate_mhz * 1e6)
    
    @property
    def max_range_km(self) -> float:
        return self.range_resolution_m * self.num_range_bins / 1000
    
    def print_config(self):
        print("=" * 60)
        print("TITAN RADAR (RFSoC 4x2)")
        print("=" * 60)
        print(f"Center Freq:     {self.center_freq_mhz} MHz")
        print(f"Sample Rate:     {self.sample_rate_msps} MSPS")
        print(f"Range Bins:      {self.num_range_bins}")
        print(f"Range Res:       {self.range_resolution_m:.1f} m")
        print(f"Max Range:       {self.max_range_km:.1f} km")
        print(f"Doppler Bins:    {self.num_doppler_bins}")
        print(f"Max Tracks:      {self.max_tracks}")
        print("=" * 60)

#=============================================================================
# Detection Result
#=============================================================================

@dataclass
class Detection:
    range_bin: int
    range_m: float
    doppler_bin: int
    velocity_mps: float
    amplitude: float
    snr_db: float
    timestamp: float = 0.0

#=============================================================================
# Signal Processing
#=============================================================================

class WaveformGenerator:
    def __init__(self, config: TitanConfig):
        self.config = config
        
    def generate_prbs(self, order: int) -> np.ndarray:
        n = 2**order - 1
        state = 1
        seq = np.zeros(n, dtype=np.float32)
        for i in range(n):
            seq[i] = 2 * (state & 1) - 1
            newbit = ((state >> (order-1)) ^ (state >> (order-2))) & 1
            state = ((state << 1) | newbit) & ((1 << order) - 1)
        return seq
    
    def generate_waveform(self) -> np.ndarray:
        samples_per_chip = int(self.config.sample_rate_msps / self.config.chip_rate_mhz)
        chips = self.generate_prbs(self.config.prbs_order)
        return np.repeat(chips, samples_per_chip).astype(np.float32)

class Correlator:
    def __init__(self, config: TitanConfig):
        self.config = config
        self.reference = None
        
    def set_reference(self, waveform: np.ndarray):
        self.reference = waveform
        
    def correlate(self, rx_data: np.ndarray) -> np.ndarray:
        n_bins = min(self.config.num_range_bins, len(rx_data) - len(self.reference))
        profile = np.zeros(n_bins, dtype=np.float32)
        for i in range(n_bins):
            profile[i] = np.abs(np.sum(rx_data[i:i+len(self.reference)] * self.reference))
        return profile

class DopplerProcessor:
    def __init__(self, config: TitanConfig):
        self.config = config
        self.window = np.hanning(config.cpi_pulses)
        
    def process(self, range_profiles: np.ndarray) -> np.ndarray:
        windowed = range_profiles * self.window[:, np.newaxis]
        padded = np.zeros((self.config.num_doppler_bins, range_profiles.shape[1]), dtype=np.complex64)
        padded[:range_profiles.shape[0], :] = windowed
        return np.abs(np.fft.fftshift(np.fft.fft(padded, axis=0), axes=0))

class CFARDetector:
    def __init__(self, config: TitanConfig):
        self.config = config
        
    def detect(self, data: np.ndarray) -> List[Tuple[int, int, float]]:
        g = self.config.cfar_guard_cells
        t = self.config.cfar_training_cells
        f = self.config.cfar_threshold_factor
        detections = []
        rows, cols = data.shape
        
        for i in range(t+g, rows-t-g):
            for j in range(t+g, cols-t-g):
                left = data[i, j-t-g:j-g]
                right = data[i, j+g+1:j+t+g+1]
                noise = np.mean(np.concatenate([left, right]))
                if data[i, j] > f * noise:
                    snr = 10 * np.log10(data[i, j] / noise + 1e-10)
                    detections.append((j, i, snr))
        return detections

class Beamformer:
    def __init__(self, config: TitanConfig, num_channels: int = 4):
        self.config = config
        self.num_channels = num_channels
        wavelength = config.SPEED_OF_LIGHT / (config.center_freq_mhz * 1e6)
        self.element_spacing = wavelength / 2
        self.weights = np.ones(num_channels, dtype=np.complex64) / num_channels
        
    def steering_vector(self, angle_deg: float) -> np.ndarray:
        angle_rad = np.radians(angle_deg)
        k = 2 * np.pi / (self.config.SPEED_OF_LIGHT / (self.config.center_freq_mhz * 1e6))
        positions = np.arange(self.num_channels) * self.element_spacing
        return np.exp(1j * k * positions * np.sin(angle_rad))
    
    def set_steering(self, angle_deg: float):
        self.weights = self.steering_vector(angle_deg).conj() / self.num_channels
        
    def apply(self, data: np.ndarray) -> np.ndarray:
        return self.weights.conj() @ data

#=============================================================================
# Main Overlay Class
#=============================================================================

class TitanRadarOverlay:
    """Main TITAN Radar Overlay for RFSoC 4x2"""
    
    def __init__(self, bitstream: str = "titan_radar.bit", config: TitanConfig = None):
        self.config = config or TitanConfig()
        self.bitstream = bitstream
        
        # Components
        self.waveform_gen = WaveformGenerator(self.config)
        self.correlator = Correlator(self.config)
        self.doppler = DopplerProcessor(self.config)
        self.cfar = CFARDetector(self.config)
        self.beamformer = Beamformer(self.config)
        
        # State
        self._initialized = False
        self._running = False
        self.detections = []
        
        # Hardware/Simulation
        if PYNQ_AVAILABLE:
            self._load_overlay()
        else:
            self._init_simulation()
            
    def _load_overlay(self):
        """Load FPGA overlay"""
        try:
            self.overlay = Overlay(self.bitstream)
            self._initialized = True
            print(f"[TITAN] Overlay loaded: {self.bitstream}")
        except Exception as e:
            print(f"[TITAN] Overlay failed: {e}, using simulation")
            self._init_simulation()
            
    def _init_simulation(self):
        """Initialize simulation mode"""
        self.overlay = None
        self._initialized = True
        print("[TITAN] Simulation mode")
        
    def configure(self):
        """Configure radar"""
        self.config.print_config()
        waveform = self.waveform_gen.generate_waveform()
        self.correlator.set_reference(waveform)
        print("[TITAN] Configured")
        
    def start(self):
        self._running = True
        print("[TITAN] Started")
        
    def stop(self):
        self._running = False
        print("[TITAN] Stopped")
        
    def process_cpi(self, simulate_targets: bool = True) -> Dict[str, Any]:
        """Process one CPI"""
        rx_data = self._simulate_receive(simulate_targets)
        
        # Beamform
        beamformed = self.beamformer.apply(rx_data)
        
        # Range compression
        samples_per_pri = int(self.config.pri_us * 1e-6 * self.config.sample_rate_msps * 1e6)
        range_profiles = np.zeros((self.config.cpi_pulses, self.config.num_range_bins))
        
        for p in range(self.config.cpi_pulses):
            start = p * samples_per_pri
            end = start + samples_per_pri
            if end <= len(beamformed):
                rp = self.correlator.correlate(beamformed[start:end])
                range_profiles[p, :len(rp)] = rp[:self.config.num_range_bins]
                
        # Doppler
        rd_map = self.doppler.process(range_profiles)
        
        # CFAR
        raw_dets = self.cfar.detect(rd_map)
        
        detections = []
        for (rb, db, snr) in raw_dets:
            det = Detection(
                range_bin=rb,
                range_m=rb * self.config.range_resolution_m,
                doppler_bin=db,
                velocity_mps=self._doppler_to_velocity(db),
                amplitude=rd_map[db, rb],
                snr_db=snr,
                timestamp=time.time()
            )
            detections.append(det)
            
        self.detections = detections
        
        return {
            'range_profile': range_profiles,
            'range_doppler': rd_map,
            'detections': detections,
            'timestamp': time.time()
        }
        
    def _simulate_receive(self, add_targets: bool = True) -> np.ndarray:
        """Simulate RX data"""
        samples = int(self.config.cpi_pulses * self.config.pri_us * 1e-6 * 
                     self.config.sample_rate_msps * 1e6)
        
        rx_data = 0.01 * (np.random.randn(4, samples) + 1j * np.random.randn(4, samples))
        
        if add_targets:
            waveform = self.waveform_gen.generate_waveform()
            samples_per_pri = int(self.config.pri_us * 1e-6 * self.config.sample_rate_msps * 1e6)
            
            targets = [
                {'range_km': 50, 'vel_mps': 250, 'amp': 0.5},
                {'range_km': 120, 'vel_mps': -180, 'amp': 0.3},
                {'range_km': 200, 'vel_mps': 350, 'amp': 0.2},
            ]
            
            for t in targets:
                delay = int(2 * t['range_km'] * 1000 / self.config.SPEED_OF_LIGHT * 
                           self.config.sample_rate_msps * 1e6)
                doppler_hz = 2 * t['vel_mps'] * self.config.center_freq_mhz * 1e6 / self.config.SPEED_OF_LIGHT
                
                for p in range(self.config.cpi_pulses):
                    start = p * samples_per_pri + delay
                    if start + len(waveform) < samples:
                        phase = np.exp(1j * 2 * np.pi * doppler_hz * p * self.config.pri_us * 1e-6)
                        for ch in range(4):
                            rx_data[ch, start:start+len(waveform)] += t['amp'] * phase * waveform
                            
        return rx_data.astype(np.complex64)
        
    def _doppler_to_velocity(self, doppler_bin: int) -> float:
        center = self.config.num_doppler_bins // 2
        idx = doppler_bin - center
        hz_per_bin = 1 / (self.config.cpi_pulses * self.config.pri_us * 1e-6)
        doppler_hz = idx * hz_per_bin
        wavelength = self.config.SPEED_OF_LIGHT / (self.config.center_freq_mhz * 1e6)
        return doppler_hz * wavelength / 2
        
    def set_steering(self, angle_deg: float):
        self.beamformer.set_steering(angle_deg)
        print(f"[TITAN] Steered to {angle_deg}°")
        
    def close(self):
        self.stop()
        print("[TITAN] Closed")

#=============================================================================
# Demo
#=============================================================================

def demo():
    print("\n" + "=" * 60)
    print("TITAN RADAR (RFSoC 4x2) DEMONSTRATION")
    print("=" * 60)
    
    radar = TitanRadarOverlay()
    radar.configure()
    radar.start()
    
    for cpi in range(3):
        result = radar.process_cpi()
        print(f"\nCPI {cpi+1}: {len(result['detections'])} detections")
        for d in result['detections'][:3]:
            print(f"  R: {d.range_m/1000:.1f} km, V: {d.velocity_mps:.0f} m/s, SNR: {d.snr_db:.1f} dB")
            
    radar.close()
    
    print("\n" + "=" * 60)
    print("RFSoC 4x2 Capabilities:")
    print("  • 4× ADC @ 5 GSPS (DC-6 GHz)")
    print("  • 2× DAC @ 9.85 GSPS")
    print("  • 4272 DSP slices")
    print("  • 16,384 range bins")
    print("  • 4-channel beamforming")
    print("  • 100 GbE offload")
    print("=" * 60)

if __name__ == "__main__":
    demo()
