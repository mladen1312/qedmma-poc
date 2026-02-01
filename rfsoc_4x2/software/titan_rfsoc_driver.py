#!/usr/bin/env python3
"""
TITAN Radar - RFSoC 4x2 Platform Driver
Integrates POC Algorithms with FPGA Acceleration

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

This driver provides:
    - FPGA overlay management
    - RF Data Converter configuration
    - DMA data transfer
    - Hybrid processing (FPGA + CPU)
    - Real-time display interface

Based on POC algorithms:
    - Zero-DSP correlation
    - PRBS waveform generation
    - CA-CFAR detection
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import IntEnum
import time
import threading
from queue import Queue

# PYNQ imports (available on RFSoC)
try:
    from pynq import Overlay, allocate
    from pynq.lib import AxiGPIO
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[TITAN] PYNQ not available - running in simulation mode")

# Import our signal processor
from titan_signal_processor import (
    TITANConfig, TITANProcessor, Detection,
    generate_prbs, prbs_to_bpsk, cfar_ca_2d
)


#=============================================================================
# RFSoC Configuration
#=============================================================================

@dataclass
class RFSoCConfig:
    """RFSoC 4x2 Hardware Configuration"""
    
    # ADC Configuration (4 channels available)
    adc_sample_rate: float = 4.9152e9     # 4.9152 GSPS
    adc_center_freq: float = 155e6        # 155 MHz VHF
    adc_decimation: int = 8               # Decimation factor
    num_adc_channels: int = 4             # 4-channel receive array
    
    # DAC Configuration (2 channels available)
    dac_sample_rate: float = 9.8304e9     # 9.8304 GSPS
    dac_center_freq: float = 155e6        # 155 MHz VHF
    dac_interpolation: int = 8            # Interpolation factor
    
    # DMA Configuration
    dma_buffer_size: int = 2**20          # 1M samples per buffer
    num_dma_buffers: int = 4              # Ping-pong + processing
    
    # Processing Configuration
    use_fpga_correlator: bool = True      # Use FPGA or CPU
    use_fpga_cfar: bool = True            # Use FPGA CFAR
    
    @property
    def effective_adc_rate(self) -> float:
        return self.adc_sample_rate / self.adc_decimation
    
    @property
    def effective_dac_rate(self) -> float:
        return self.dac_sample_rate / self.dac_interpolation


#=============================================================================
# FPGA IP Register Maps
#=============================================================================

class CorrelatorRegs:
    """Zero-DSP Correlator IP Register Map"""
    CTRL = 0x00           # Control register
    STATUS = 0x04         # Status register
    NUM_SAMPLES = 0x08    # Number of samples to process
    NUM_LANES = 0x0C      # Number of correlation lanes
    PRBS_ORDER = 0x10     # PRBS order (15 or 20)
    PRBS_SEED = 0x14      # PRBS initial seed
    THRESHOLD = 0x18      # Detection threshold (fixed-point)
    DET_COUNT = 0x1C      # Number of detections
    
    # Control bits
    CTRL_START = 0x01
    CTRL_STOP = 0x02
    CTRL_RESET = 0x04
    
    # Status bits
    STATUS_BUSY = 0x01
    STATUS_DONE = 0x02
    STATUS_OVERFLOW = 0x04


class CFARRegs:
    """VI-CFAR IP Register Map"""
    CTRL = 0x00
    NUM_RANGE = 0x04
    NUM_DOPPLER = 0x08
    GUARD_CELLS = 0x0C
    REF_CELLS = 0x10
    ALPHA = 0x14          # Threshold multiplier (Q16.16)
    FORCE_MODE = 0x18     # 0=CA, 1=GO, 2=SO, 4=VI-auto
    DET_COUNT = 0x1C
    VI_STATS = 0x20


class WaveformRegs:
    """Waveform Generator IP Register Map"""
    CTRL = 0x00
    WAVEFORM_TYPE = 0x04  # 0=PRBS, 1=LFM, 2=CW
    PRBS_ORDER = 0x08
    CHIP_RATE = 0x0C
    NUM_SAMPLES = 0x10
    AMPLITUDE = 0x14      # Q1.15 format


#=============================================================================
# RFSoC 4x2 Driver
#=============================================================================

class TITANRFSoC:
    """
    TITAN Radar RFSoC 4x2 Driver
    
    Manages:
    - FPGA overlay loading
    - RF Data Converter setup
    - DMA transfers
    - Signal processing (FPGA + CPU hybrid)
    - Real-time operation
    """
    
    def __init__(
        self,
        bitstream_path: str = "titan_radar.bit",
        config: Optional[TITANConfig] = None,
        rfsoc_config: Optional[RFSoCConfig] = None
    ):
        """
        Initialize RFSoC driver
        
        Args:
            bitstream_path: Path to FPGA bitstream
            config: Signal processing configuration
            rfsoc_config: Hardware configuration
        """
        self.bitstream_path = bitstream_path
        self.config = config or TITANConfig()
        self.rfsoc_config = rfsoc_config or RFSoCConfig()
        
        self.overlay = None
        self.processor = None
        
        # DMA buffers
        self.tx_buffer = None
        self.rx_buffers = []
        
        # Processing state
        self.is_running = False
        self.cpi_count = 0
        
        # Callbacks
        self.detection_callback: Optional[Callable] = None
        self.rdmap_callback: Optional[Callable] = None
        
        # Initialize signal processor (CPU-side)
        self.processor = TITANProcessor(self.config)
        
        print("[TITAN-RFSoC] Driver initialized")
        print(f"  Bitstream: {bitstream_path}")
        print(f"  FPGA Correlator: {self.rfsoc_config.use_fpga_correlator}")
        print(f"  FPGA CFAR: {self.rfsoc_config.use_fpga_cfar}")
    
    def load_overlay(self) -> bool:
        """
        Load FPGA bitstream
        
        Returns:
            True if successful
        """
        if not PYNQ_AVAILABLE:
            print("[TITAN-RFSoC] Simulation mode - no overlay loaded")
            return True
        
        try:
            print(f"[TITAN-RFSoC] Loading overlay: {self.bitstream_path}")
            self.overlay = Overlay(self.bitstream_path)
            
            # Get IP references
            self.correlator = self.overlay.correlator
            self.cfar = self.overlay.vi_cfar_detector
            self.waveform_gen = self.overlay.waveform_gen
            self.rf_dc = self.overlay.rf_data_converter
            self.dma_tx = self.overlay.dma_tx
            self.dma_rx = [
                self.overlay.dma_rx_0,
                self.overlay.dma_rx_1,
                self.overlay.dma_rx_2,
                self.overlay.dma_rx_3,
            ]
            
            print("[TITAN-RFSoC] Overlay loaded successfully!")
            print(f"  IPs: {list(self.overlay.ip_dict.keys())}")
            
            return True
            
        except Exception as e:
            print(f"[TITAN-RFSoC] Overlay load failed: {e}")
            return False
    
    def configure_rf(self) -> bool:
        """
        Configure RF Data Converter
        
        Returns:
            True if successful
        """
        if not PYNQ_AVAILABLE or self.overlay is None:
            print("[TITAN-RFSoC] Simulation mode - RF config skipped")
            return True
        
        cfg = self.rfsoc_config
        
        try:
            print("[TITAN-RFSoC] Configuring RF Data Converter...")
            
            # Configure ADCs (4 channels for receive array)
            for ch in range(cfg.num_adc_channels):
                # Set NCO frequency for digital downconversion
                nco_freq = cfg.adc_center_freq / 1e6  # MHz
                self.rf_dc.adc_tiles[ch // 2].blocks[ch % 2].NyquistZone = 1
                self.rf_dc.adc_tiles[ch // 2].blocks[ch % 2].MixerSettings = {
                    'CoarseMixFreq': 'COARSE_MIX_OFF',
                    'EventSource': 'TILE',
                    'FineMixerScale': 'FINE_MIXER_SCALE_1P0',
                    'Freq': nco_freq,
                    'MixerMode': 'C2R',
                    'MixerType': 'FINE',
                    'PhaseOffset': 0.0
                }
                
                # Set decimation
                self.rf_dc.adc_tiles[ch // 2].blocks[ch % 2].DecimationFactor = cfg.adc_decimation
            
            # Configure DACs (2 channels for transmit)
            for ch in range(2):
                nco_freq = cfg.dac_center_freq / 1e6
                self.rf_dc.dac_tiles[ch].blocks[0].NyquistZone = 1
                self.rf_dc.dac_tiles[ch].blocks[0].MixerSettings = {
                    'CoarseMixFreq': 'COARSE_MIX_OFF',
                    'EventSource': 'TILE',
                    'FineMixerScale': 'FINE_MIXER_SCALE_1P0',
                    'Freq': nco_freq,
                    'MixerMode': 'R2C',
                    'MixerType': 'FINE',
                    'PhaseOffset': 0.0
                }
                
                # Set interpolation
                self.rf_dc.dac_tiles[ch].blocks[0].InterpolationFactor = cfg.dac_interpolation
            
            print(f"[TITAN-RFSoC] RF configured:")
            print(f"  ADC: {cfg.effective_adc_rate/1e6:.2f} MSPS, {cfg.adc_center_freq/1e6:.1f} MHz NCO")
            print(f"  DAC: {cfg.effective_dac_rate/1e6:.2f} MSPS, {cfg.dac_center_freq/1e6:.1f} MHz NCO")
            
            return True
            
        except Exception as e:
            print(f"[TITAN-RFSoC] RF config failed: {e}")
            return False
    
    def allocate_buffers(self) -> bool:
        """
        Allocate DMA buffers
        
        Returns:
            True if successful
        """
        cfg = self.rfsoc_config
        
        if PYNQ_AVAILABLE:
            try:
                # TX buffer (waveform)
                self.tx_buffer = allocate(
                    shape=(self.config.cpi_samples,),
                    dtype=np.complex64
                )
                
                # RX buffers (one per ADC channel)
                self.rx_buffers = []
                for _ in range(cfg.num_adc_channels):
                    buf = allocate(
                        shape=(self.config.cpi_samples,),
                        dtype=np.complex64
                    )
                    self.rx_buffers.append(buf)
                
                print(f"[TITAN-RFSoC] Allocated {cfg.num_adc_channels + 1} DMA buffers")
                
            except Exception as e:
                print(f"[TITAN-RFSoC] Buffer allocation failed: {e}")
                return False
        else:
            # Simulation mode - use numpy arrays
            self.tx_buffer = np.zeros(self.config.cpi_samples, dtype=np.complex64)
            self.rx_buffers = [
                np.zeros(self.config.cpi_samples, dtype=np.complex64)
                for _ in range(cfg.num_adc_channels)
            ]
            print("[TITAN-RFSoC] Simulation buffers allocated")
        
        return True
    
    def configure_waveform_generator(self) -> bool:
        """
        Configure FPGA waveform generator with PRBS
        
        Returns:
            True if successful
        """
        if not PYNQ_AVAILABLE or self.overlay is None:
            # Generate waveform in software
            prbs_bits = generate_prbs(self.config.prbs_order, self.config.cpi_samples)
            waveform = prbs_to_bpsk(prbs_bits).astype(np.complex64) * 0.8
            np.copyto(self.tx_buffer, waveform)
            print("[TITAN-RFSoC] Software waveform generated")
            return True
        
        try:
            # Configure FPGA waveform generator
            self.waveform_gen.write(WaveformRegs.WAVEFORM_TYPE, 0)  # PRBS
            self.waveform_gen.write(WaveformRegs.PRBS_ORDER, self.config.prbs_order)
            self.waveform_gen.write(WaveformRegs.NUM_SAMPLES, self.config.cpi_samples)
            self.waveform_gen.write(WaveformRegs.AMPLITUDE, int(0.8 * 32767))  # Q1.15
            
            # Calculate chip rate divider
            chip_rate_div = int(self.rfsoc_config.effective_dac_rate / self.config.chip_rate_hz)
            self.waveform_gen.write(WaveformRegs.CHIP_RATE, chip_rate_div)
            
            print(f"[TITAN-RFSoC] FPGA waveform generator configured:")
            print(f"  PRBS-{self.config.prbs_order}, {self.config.chip_rate_hz/1e6:.1f} Mchip/s")
            
            return True
            
        except Exception as e:
            print(f"[TITAN-RFSoC] Waveform config failed: {e}")
            return False
    
    def configure_correlator(self) -> bool:
        """
        Configure FPGA correlator IP
        
        Returns:
            True if successful
        """
        if not PYNQ_AVAILABLE or self.overlay is None:
            print("[TITAN-RFSoC] Using CPU correlator")
            return True
        
        try:
            # Configure correlator
            self.correlator.write(CorrelatorRegs.NUM_SAMPLES, self.config.cpi_samples)
            self.correlator.write(CorrelatorRegs.NUM_LANES, self.config.num_range_bins)
            self.correlator.write(CorrelatorRegs.PRBS_ORDER, self.config.prbs_order)
            self.correlator.write(CorrelatorRegs.PRBS_SEED, 0x7FFF)  # All ones
            
            # Reset
            self.correlator.write(CorrelatorRegs.CTRL, CorrelatorRegs.CTRL_RESET)
            time.sleep(0.001)
            self.correlator.write(CorrelatorRegs.CTRL, 0)
            
            print("[TITAN-RFSoC] FPGA correlator configured")
            return True
            
        except Exception as e:
            print(f"[TITAN-RFSoC] Correlator config failed: {e}")
            return False
    
    def configure_cfar(self) -> bool:
        """
        Configure FPGA VI-CFAR detector
        
        Returns:
            True if successful
        """
        if not PYNQ_AVAILABLE or self.overlay is None:
            print("[TITAN-RFSoC] Using CPU CFAR")
            return True
        
        try:
            # Configure CFAR
            self.cfar.write(CFARRegs.NUM_RANGE, self.config.num_range_bins)
            self.cfar.write(CFARRegs.NUM_DOPPLER, self.config.num_doppler_bins)
            self.cfar.write(CFARRegs.GUARD_CELLS, self.config.cfar_guard_cells)
            self.cfar.write(CFARRegs.REF_CELLS, self.config.cfar_ref_cells)
            
            # Calculate alpha for Pfa
            N = 2 * self.config.cfar_ref_cells
            alpha = N * (self.config.cfar_pfa ** (-1/N) - 1)
            alpha_fixed = int(alpha * 65536)  # Q16.16
            self.cfar.write(CFARRegs.ALPHA, alpha_fixed)
            
            # Set VI-auto mode
            self.cfar.write(CFARRegs.FORCE_MODE, 4)  # VI-auto
            
            print(f"[TITAN-RFSoC] FPGA CFAR configured (VI-auto, α={alpha:.3f})")
            return True
            
        except Exception as e:
            print(f"[TITAN-RFSoC] CFAR config failed: {e}")
            return False
    
    def initialize(self) -> bool:
        """
        Complete system initialization
        
        Returns:
            True if all steps successful
        """
        print("\n" + "=" * 60)
        print("TITAN RFSoC 4x2 INITIALIZATION")
        print("=" * 60)
        
        steps = [
            ("Load overlay", self.load_overlay),
            ("Configure RF", self.configure_rf),
            ("Allocate buffers", self.allocate_buffers),
            ("Configure waveform", self.configure_waveform_generator),
            ("Configure correlator", self.configure_correlator),
            ("Configure CFAR", self.configure_cfar),
        ]
        
        for name, func in steps:
            print(f"\n[{name}]")
            if not func():
                print(f"[TITAN-RFSoC] Initialization failed at: {name}")
                return False
        
        print("\n" + "=" * 60)
        print("INITIALIZATION COMPLETE")
        print("=" * 60)
        
        return True
    
    def start_tx(self):
        """Start continuous transmit"""
        if PYNQ_AVAILABLE and self.overlay is not None:
            # Start DMA TX (cyclic)
            self.dma_tx.sendchannel.transfer(self.tx_buffer)
            
            # Enable waveform generator
            self.waveform_gen.write(WaveformRegs.CTRL, 0x01)
            
        print("[TITAN-RFSoC] TX started")
    
    def stop_tx(self):
        """Stop transmit"""
        if PYNQ_AVAILABLE and self.overlay is not None:
            self.waveform_gen.write(WaveformRegs.CTRL, 0x00)
            
        print("[TITAN-RFSoC] TX stopped")
    
    def capture_cpi(self) -> List[np.ndarray]:
        """
        Capture one CPI from all ADC channels
        
        Returns:
            List of complex sample arrays (one per channel)
        """
        if PYNQ_AVAILABLE and self.overlay is not None:
            # Start DMA transfers for all channels
            for i, dma in enumerate(self.dma_rx[:self.rfsoc_config.num_adc_channels]):
                dma.recvchannel.transfer(self.rx_buffers[i])
            
            # Wait for completion
            for dma in self.dma_rx[:self.rfsoc_config.num_adc_channels]:
                dma.recvchannel.wait()
            
            return [np.array(buf) for buf in self.rx_buffers]
        else:
            # Simulation mode - generate synthetic data
            return self._simulate_receive()
    
    def _simulate_receive(self) -> List[np.ndarray]:
        """Generate simulated receive data"""
        n_samples = self.config.cpi_samples
        channels = []
        
        # Reference waveform
        prbs_bpsk = prbs_to_bpsk(generate_prbs(self.config.prbs_order, n_samples))
        
        # Simulate targets (different phase per channel for beamforming)
        targets = [
            (100, 1.0, 0.0),    # delay, amplitude, doppler
            (300, 0.5, 50.0),
            (500, 0.2, -30.0),
        ]
        
        for ch in range(self.rfsoc_config.num_adc_channels):
            rx = np.zeros(n_samples, dtype=np.complex64)
            
            for delay, amp, doppler in targets:
                # Add phase offset per channel (simulating array)
                phase_offset = ch * np.pi / 4  # λ/2 spacing, 45° per element
                
                n_copy = min(len(prbs_bpsk), n_samples - delay)
                if n_copy > 0:
                    t = np.arange(n_copy) / self.config.sample_rate_hz
                    doppler_phase = np.exp(2j * np.pi * doppler * t)
                    array_phase = np.exp(1j * phase_offset)
                    rx[delay:delay+n_copy] += amp * prbs_bpsk[:n_copy] * doppler_phase * array_phase
            
            # Add noise
            noise = 0.1 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
            rx += noise.astype(np.complex64)
            
            channels.append(rx)
        
        return channels
    
    def process_cpi_fpga(self, rx_channels: List[np.ndarray]) -> np.ndarray:
        """
        Process CPI using FPGA correlator
        
        Args:
            rx_channels: List of receive channel data
            
        Returns:
            Range profile (after beamforming)
        """
        if PYNQ_AVAILABLE and self.overlay is not None and self.rfsoc_config.use_fpga_correlator:
            # TODO: Implement FPGA beamforming + correlation pipeline
            # For now, use CPU as fallback
            pass
        
        # CPU fallback: Simple beamforming + correlation
        # Sum channels (uniform weighting)
        beamformed = np.sum(rx_channels, axis=0) / len(rx_channels)
        
        # Correlate
        range_profile = self.processor.correlate_cpi(beamformed, mode='fft')
        
        return range_profile
    
    def run_processing_loop(
        self,
        num_cpis: int = 1000,
        callback: Optional[Callable] = None
    ):
        """
        Run continuous processing loop
        
        Args:
            num_cpis: Number of CPIs to process
            callback: Function called with (cpi_idx, rdmap, detections)
        """
        print(f"\n[TITAN-RFSoC] Starting processing loop ({num_cpis} CPIs)...")
        
        self.is_running = True
        self.start_tx()
        
        try:
            range_profiles = []
            
            for cpi_idx in range(num_cpis):
                if not self.is_running:
                    break
                
                # Capture
                rx_channels = self.capture_cpi()
                
                # Process
                range_profile = self.process_cpi_fpga(rx_channels)
                range_profiles.append(range_profile)
                
                # Accumulate for Doppler processing
                if len(range_profiles) >= self.config.num_doppler_bins:
                    # Generate Range-Doppler map
                    rp_array = np.array(range_profiles[-self.config.num_doppler_bins:])
                    
                    # Doppler FFT
                    from titan_signal_processor import generate_doppler_fft
                    rdmap = generate_doppler_fft(rp_array, window='hanning')
                    
                    # Detect
                    detections = self.processor.detect_2d(rdmap)
                    
                    # Callback
                    if callback:
                        callback(cpi_idx, rdmap, detections)
                    elif detections:
                        print(f"[CPI {cpi_idx}] {len(detections)} detections")
                        for det in detections[:3]:
                            print(f"    R={det.range_m/1000:.1f}km, "
                                  f"V={det.velocity_mps:.0f}m/s, "
                                  f"SNR={det.snr_db:.1f}dB")
                
                self.cpi_count += 1
                
                # Progress
                if (cpi_idx + 1) % 100 == 0:
                    print(f"[TITAN-RFSoC] Processed {cpi_idx + 1}/{num_cpis} CPIs")
        
        finally:
            self.stop_tx()
            self.is_running = False
        
        print(f"[TITAN-RFSoC] Processing complete: {self.cpi_count} CPIs")
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        self.stop_tx()


#=============================================================================
# Demo / Test
#=============================================================================

def demo():
    """Demonstrate RFSoC driver"""
    print("\n" + "=" * 70)
    print("TITAN RFSoC 4x2 DRIVER DEMO")
    print("=" * 70)
    
    # Create configuration
    config = TITANConfig(
        prbs_order=15,
        num_range_bins=512,
        num_doppler_bins=64,
        cpi_samples=32768,
    )
    config.print_config()
    
    rfsoc_config = RFSoCConfig(
        num_adc_channels=4,
        use_fpga_correlator=False,  # Use CPU for demo
        use_fpga_cfar=False,
    )
    
    # Initialize driver
    driver = TITANRFSoC(
        bitstream_path="titan_radar.bit",
        config=config,
        rfsoc_config=rfsoc_config
    )
    
    if driver.initialize():
        # Define callback
        def on_detection(cpi_idx, rdmap, detections):
            if detections:
                print(f"\n[CPI {cpi_idx}] DETECTIONS:")
                for det in detections[:5]:
                    print(f"  Range: {det.range_m/1000:.2f} km, "
                          f"Velocity: {det.velocity_mps:+.1f} m/s, "
                          f"SNR: {det.snr_db:.1f} dB")
        
        # Run processing
        driver.run_processing_loop(num_cpis=200, callback=on_detection)
    
    print("\n[Demo] Complete!")


if __name__ == "__main__":
    demo()
