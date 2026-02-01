#!/usr/bin/env python3
"""
ZEUS Radar - PYNQ Framework for Kria KV260
Rapid Prototyping Interface for QEDMMA Radar

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Platform: AMD Kria KV260 Vision AI Kit
FPGA: Zynq UltraScale+ (256K logic cells, 1248 DSP)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import time

# PYNQ imports (available on KV260 with Ubuntu)
try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[ZEUS] PYNQ not available - simulation mode")

#=============================================================================
# Configuration
#=============================================================================

@dataclass
class ZeusConfig:
    """ZEUS Radar Configuration"""
    CENTER_FREQ: float = 155e6      # 155 MHz VHF
    SAMPLE_RATE: float = 10e6       # 10 MSPS
    PRBS_ORDER: int = 15            # PRBS-15
    NUM_RANGE_BINS: int = 4096      # 4× more than Pluto!
    CPI_LENGTH: int = 32768
    CFAR_GUARD_CELLS: int = 4
    CFAR_TRAINING_CELLS: int = 16
    CFAR_THRESHOLD: float = 5.0
    MAX_TRACKS: int = 64
    SPEED_OF_LIGHT: float = 299792458.0
    
    @property
    def range_resolution(self) -> float:
        return self.SPEED_OF_LIGHT / (2 * 1e6)  # ~150 m
    
    @property
    def max_range(self) -> float:
        return self.range_resolution * self.NUM_RANGE_BINS  # ~614 km

#=============================================================================
# ZEUS Overlay
#=============================================================================

class ZeusOverlay:
    """PYNQ Overlay wrapper for ZEUS radar"""
    
    def __init__(self, bitstream: str = "zeus_radar.bit"):
        self.config = ZeusConfig()
        
        if PYNQ_AVAILABLE:
            self.overlay = Overlay(bitstream)
            self._init_buffers()
        else:
            self.overlay = None
            print("[ZEUS] Simulation mode active")
    
    def _init_buffers(self):
        self.rx_buffer = allocate(shape=(self.config.CPI_LENGTH,), dtype=np.complex64)
        self.range_profile = allocate(shape=(self.config.NUM_RANGE_BINS,), dtype=np.float32)
    
    def process_cpi(self, rx_data: Optional[np.ndarray] = None) -> dict:
        """Process one CPI"""
        if rx_data is None:
            rx_data = self._gen_test_signal()
        
        # Generate PRBS
        prbs = self._gen_prbs(self.config.PRBS_ORDER)
        
        # Correlate
        rp = np.abs(np.correlate(rx_data[:8192], prbs[:1024], mode='full'))
        rp = rp[:self.config.NUM_RANGE_BINS]
        
        # CFAR
        detections = self._cfar(rp)
        
        return {'range_profile': rp, 'detections': detections, 'timestamp': time.time()}
    
    def _gen_prbs(self, order: int) -> np.ndarray:
        n = 2**order - 1
        state = 1
        prbs = np.zeros(n, dtype=np.float32)
        for i in range(n):
            prbs[i] = 2 * (state & 1) - 1
            newbit = ((state >> 14) ^ (state >> 13)) & 1
            state = ((state << 1) | newbit) & ((1 << order) - 1)
        return prbs
    
    def _gen_test_signal(self) -> np.ndarray:
        n = self.config.CPI_LENGTH
        signal = 0.1 * (np.random.randn(n) + 1j * np.random.randn(n))
        prbs = self._gen_prbs(self.config.PRBS_ORDER)
        # Targets at 75km, 180km, 300km
        for delay, amp in [(500, 10.0), (1200, 5.0), (2000, 3.0)]:
            if delay < n - len(prbs):
                signal[delay:delay+len(prbs)] += amp * prbs
        return signal.astype(np.complex64)
    
    def _cfar(self, rp: np.ndarray) -> List[dict]:
        g, t, f = self.config.CFAR_GUARD_CELLS, self.config.CFAR_TRAINING_CELLS, self.config.CFAR_THRESHOLD
        dets = []
        for i in range(t+g, len(rp)-t-g):
            noise = np.mean(np.concatenate([rp[i-g-t:i-g], rp[i+g+1:i+g+t+1]]))
            if rp[i] > f * noise:
                dets.append({
                    'bin': i,
                    'range': i * self.config.range_resolution,
                    'snr_db': 10 * np.log10(rp[i] / noise + 1e-10)
                })
        return dets

#=============================================================================
# Main
#=============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ZEUS RADAR (Kria KV260) - Demo")
    print("=" * 60)
    
    zeus = ZeusOverlay()
    
    for frame in range(3):
        result = zeus.process_cpi()
        print(f"\nFrame {frame+1}: {len(result['detections'])} detections")
        for d in result['detections'][:3]:
            print(f"  Range: {d['range']/1000:.1f} km, SNR: {d['snr_db']:.1f} dB")
    
    print("\n" + "=" * 60)
    print("KV260 Advantages:")
    print("  • 1248 DSP slices (vs 80 on Pluto)")
    print("  • 4096 range bins (vs 512)")
    print("  • 4GB DDR4 memory")
    print("  • Quad ARM A53 + Dual R5F")
    print("  • 1.4 TOPS AI accelerator")
    print("=" * 60)
