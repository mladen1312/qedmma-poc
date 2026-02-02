#!/usr/bin/env python3
"""
TITAN Radar - Enhanced Doppler Processing
Addresses vGrok-X Velocity Ambiguity Issue

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Problem: With PRBS-20 (10.49 ms period), max unambiguous velocity = ±46 m/s
Solution: Multi-PRF velocity unwrapping + adaptive waveform selection

This module implements:
1. Chinese Remainder Theorem (CRT) velocity unwrapping
2. Staggered PRF processing
3. Adaptive waveform selection based on scenario
4. Range-Velocity trade-off optimizer
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from scipy.constants import c


#=============================================================================
# Constants
#=============================================================================

C = 299792458.0  # Speed of light (m/s)


#=============================================================================
# Waveform Configurations
#=============================================================================

class WaveformMode(Enum):
    """Predefined waveform modes"""
    LONG_RANGE = "long_range"       # Max range, limited velocity
    HIGH_VELOCITY = "high_velocity" # High velocity, reduced range
    BALANCED = "balanced"           # Balanced trade-off
    MULTI_PRF = "multi_prf"         # Staggered PRF for unwrapping


@dataclass
class WaveformConfig:
    """Waveform configuration parameters"""
    name: str
    prbs_order: int
    chip_rate_hz: float
    num_cpis: int
    description: str
    
    @property
    def prbs_length(self) -> int:
        return 2**self.prbs_order - 1
    
    @property
    def sequence_duration_s(self) -> float:
        return self.prbs_length / self.chip_rate_hz
    
    @property
    def prf_hz(self) -> float:
        return 1.0 / self.sequence_duration_s
    
    @property
    def processing_gain_db(self) -> float:
        return 10 * np.log10(self.prbs_length)
    
    def get_unambiguous_range_m(self) -> float:
        return C * self.sequence_duration_s / 2
    
    def get_unambiguous_velocity_mps(self, wavelength_m: float) -> float:
        return wavelength_m * self.prf_hz / 4


# Predefined waveform configurations
WAVEFORM_CONFIGS = {
    WaveformMode.LONG_RANGE: WaveformConfig(
        name="Long Range",
        prbs_order=20,
        chip_rate_hz=100e6,
        num_cpis=64,
        description="Max processing gain (60 dB), range 1572 km, velocity ±46 m/s"
    ),
    WaveformMode.HIGH_VELOCITY: WaveformConfig(
        name="High Velocity",
        prbs_order=11,
        chip_rate_hz=100e6,
        num_cpis=512,
        description="High PRF (48.9 kHz), velocity ±470 m/s, range 3 km"
    ),
    WaveformMode.BALANCED: WaveformConfig(
        name="Balanced",
        prbs_order=15,
        chip_rate_hz=100e6,
        num_cpis=256,
        description="Balanced (45 dB), range 49 km, velocity ±148 m/s"
    ),
}


#=============================================================================
# Multi-PRF Velocity Unwrapping
#=============================================================================

class VelocityUnwrapper:
    """
    Multi-PRF Velocity Unwrapping using Chinese Remainder Theorem
    
    Uses multiple PRFs to resolve velocity ambiguity beyond single-PRF limit.
    """
    
    def __init__(self, prfs_hz: List[float], wavelength_m: float):
        """
        Initialize unwrapper
        
        Args:
            prfs_hz: List of PRF values (should be coprime ratios)
            wavelength_m: Radar wavelength
        """
        self.prfs = np.array(prfs_hz)
        self.wavelength = wavelength_m
        self.num_prfs = len(prfs_hz)
        
        # Calculate unambiguous velocity for each PRF
        self.v_unamb = [wavelength_m * prf / 4 for prf in prfs_hz]
        
        # Extended unambiguous velocity (product of all)
        self.v_extended = self._calculate_extended_velocity()
        
    def _calculate_extended_velocity(self) -> float:
        """Calculate extended unambiguous velocity using LCM"""
        # Find LCM of PRF ratios
        # For well-chosen PRFs, extended velocity ≈ product / GCD
        from math import gcd
        from functools import reduce
        
        # Convert PRFs to integers (scaled)
        scale = 1000
        prfs_int = [int(p * scale) for p in self.prfs]
        
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        prfs_lcm = reduce(lcm, prfs_int)
        
        # Extended unambiguous velocity
        v_ext = self.wavelength * (prfs_lcm / scale) / 4
        
        return min(v_ext, 2000)  # Cap at reasonable value
    
    def unwrap_velocity(self, measured_velocities: List[float]) -> Tuple[float, float]:
        """
        Unwrap velocity from multiple PRF measurements
        
        Args:
            measured_velocities: Measured velocity at each PRF (minimum 2)
            
        Returns:
            (true_velocity, confidence)
        """
        if len(measured_velocities) < 2:
            raise ValueError("Need at least 2 velocity measurements")
        
        # Simple CRT-based unwrapping for 2 measurements
        if len(measured_velocities) == 2:
            v1, v2 = measured_velocities
            v_unamb1, v_unamb2 = self.v_unamb[0], self.v_unamb[1]
            
            # Search for consistent velocity
            best_v = None
            best_error = float('inf')
            
            for k1 in range(-10, 11):
                for k2 in range(-10, 11):
                    v_candidate1 = v1 + k1 * 2 * v_unamb1
                    v_candidate2 = v2 + k2 * 2 * v_unamb2
                    
                    error = abs(v_candidate1 - v_candidate2)
                    
                    if error < best_error:
                        best_error = error
                        best_v = (v_candidate1 + v_candidate2) / 2
            
            confidence = 1.0 - min(best_error / 10, 1.0)
            return best_v, confidence
        
        # For 3+ PRFs, use least squares
        else:
            # Matrix formulation
            from scipy.optimize import minimize_scalar
            
            def error_func(v_true):
                total_error = 0
                for i, v_meas in enumerate(measured_velocities):
                    v_unamb = self.v_unamb[i]
                    # Fold true velocity to measured range
                    v_folded = ((v_true + v_unamb) % (2 * v_unamb)) - v_unamb
                    total_error += (v_meas - v_folded)**2
                return total_error
            
            # Search in extended range
            best_v = None
            best_error = float('inf')
            
            for v_init in np.linspace(-self.v_extended, self.v_extended, 1000):
                error = error_func(v_init)
                if error < best_error:
                    best_error = error
                    best_v = v_init
            
            confidence = 1.0 - min(np.sqrt(best_error) / 10, 1.0)
            return best_v, confidence
    
    def print_config(self):
        """Print unwrapper configuration"""
        print("\n" + "=" * 70)
        print("MULTI-PRF VELOCITY UNWRAPPER")
        print("=" * 70)
        
        print(f"\n📊 PRF CONFIGURATION")
        print("-" * 50)
        for i, (prf, v_ua) in enumerate(zip(self.prfs, self.v_unamb)):
            print(f"  PRF {i+1}: {prf:.1f} Hz → Unambig. velocity: ±{v_ua:.1f} m/s")
        
        print(f"\n📈 EXTENDED CAPABILITY")
        print("-" * 50)
        print(f"  Extended unambig. velocity: ±{self.v_extended:.0f} m/s")
        print(f"  Velocity coverage ratio:    {self.v_extended / self.v_unamb[0]:.1f}×")


#=============================================================================
# Staggered PRF Processor
#=============================================================================

class StaggeredPRFProcessor:
    """
    Staggered PRF Processing for Velocity Ambiguity Resolution
    
    Interleaves different PRBS orders to achieve both range and velocity.
    """
    
    def __init__(self, wavelength_m: float = 1.93):
        """
        Initialize processor
        
        Args:
            wavelength_m: Radar wavelength
        """
        self.wavelength = wavelength_m
        
        # Define staggered waveform schedule
        # Alternates between long (range) and short (velocity) sequences
        self.schedule = [
            WaveformConfig("Long", 18, 100e6, 1, "Range burst"),
            WaveformConfig("Short", 11, 100e6, 1, "Velocity burst"),
            WaveformConfig("Long", 18, 100e6, 1, "Range burst"),
            WaveformConfig("Medium", 15, 100e6, 1, "Verification"),
        ]
        
        # Calculate parameters
        self._calculate_parameters()
    
    def _calculate_parameters(self):
        """Calculate effective parameters"""
        # Total integration time
        total_time = sum(cfg.sequence_duration_s for cfg in self.schedule)
        
        # Effective PRFs
        self.prfs = [cfg.prf_hz for cfg in self.schedule]
        
        # Velocity capabilities
        self.v_unamb_primary = self.schedule[1].get_unambiguous_velocity_mps(self.wavelength)
        
        # Range capability (from longest sequence)
        longest = max(self.schedule, key=lambda x: x.prbs_order)
        self.r_unamb = longest.get_unambiguous_range_m()
        self.proc_gain_db = longest.processing_gain_db
        
        # Create unwrapper
        unique_prfs = list(set(cfg.prf_hz for cfg in self.schedule))[:3]
        if len(unique_prfs) >= 2:
            self.unwrapper = VelocityUnwrapper(unique_prfs, self.wavelength)
        else:
            self.unwrapper = None
    
    def process_burst(self, target_range_m: float, target_velocity_mps: float) -> Dict:
        """
        Process a burst of staggered waveforms
        
        Args:
            target_range_m: True target range
            target_velocity_mps: True target velocity
            
        Returns:
            Processing results
        """
        results = {
            'true_range_m': target_range_m,
            'true_velocity_mps': target_velocity_mps,
            'measurements': [],
            'unwrapped_velocity': None,
            'confidence': 0,
        }
        
        measured_velocities = []
        
        for i, cfg in enumerate(self.schedule):
            v_unamb = cfg.get_unambiguous_velocity_mps(self.wavelength)
            r_unamb = cfg.get_unambiguous_range_m()
            
            # Fold velocity
            v_measured = ((target_velocity_mps + v_unamb) % (2 * v_unamb)) - v_unamb
            
            # Fold range
            r_measured = target_range_m % r_unamb
            
            results['measurements'].append({
                'waveform': cfg.name,
                'prbs_order': cfg.prbs_order,
                'v_unamb': v_unamb,
                'r_unamb': r_unamb,
                'v_measured': v_measured,
                'r_measured': r_measured,
            })
            
            measured_velocities.append(v_measured)
        
        # Unwrap velocity if possible
        if self.unwrapper and len(measured_velocities) >= 2:
            # Use first two unique PRF measurements
            unique_measurements = []
            seen_prfs = set()
            for m, cfg in zip(results['measurements'], self.schedule):
                if cfg.prf_hz not in seen_prfs:
                    unique_measurements.append(m['v_measured'])
                    seen_prfs.add(cfg.prf_hz)
                    if len(unique_measurements) >= 2:
                        break
            
            if len(unique_measurements) >= 2:
                v_unwrapped, conf = self.unwrapper.unwrap_velocity(unique_measurements)
                results['unwrapped_velocity'] = v_unwrapped
                results['confidence'] = conf
        
        return results
    
    def print_config(self):
        """Print processor configuration"""
        print("\n" + "=" * 70)
        print("STAGGERED PRF PROCESSOR")
        print("=" * 70)
        
        print(f"\n📋 WAVEFORM SCHEDULE")
        print("-" * 60)
        print(f"  {'#':<3} {'Name':<10} {'PRBS':<8} {'Duration':<12} {'V_unamb':<12} {'R_unamb'}")
        print("-" * 60)
        
        for i, cfg in enumerate(self.schedule):
            v_ua = cfg.get_unambiguous_velocity_mps(self.wavelength)
            r_ua = cfg.get_unambiguous_range_m()
            dur_us = cfg.sequence_duration_s * 1e6
            print(f"  {i+1:<3} {cfg.name:<10} {cfg.prbs_order:<8} {dur_us:>8.1f} µs  "
                  f"±{v_ua:>6.0f} m/s  {r_ua/1000:>6.1f} km")
        
        print("-" * 60)
        
        print(f"\n📈 EFFECTIVE CAPABILITIES")
        print("-" * 50)
        print(f"  Max processing gain:      {self.proc_gain_db:.1f} dB")
        print(f"  Unambiguous range:        {self.r_unamb/1000:.1f} km")
        print(f"  Primary velocity range:   ±{self.v_unamb_primary:.0f} m/s")
        if self.unwrapper:
            print(f"  Extended velocity range:  ±{self.unwrapper.v_extended:.0f} m/s")


#=============================================================================
# Adaptive Waveform Selector
#=============================================================================

class AdaptiveWaveformSelector:
    """
    Adaptive Waveform Selection based on Scenario
    
    Automatically selects optimal waveform parameters based on:
    - Expected target range
    - Expected target velocity
    - Required SNR
    - Environmental conditions
    """
    
    def __init__(self, wavelength_m: float = 1.93, bandwidth_hz: float = 100e6):
        self.wavelength = wavelength_m
        self.bandwidth = bandwidth_hz
        
        # Define waveform options
        self.waveforms = {
            'ultra_long': WaveformConfig("Ultra Long", 23, bandwidth_hz, 32, 
                                         "Max range (5000 km), very limited velocity"),
            'long': WaveformConfig("Long", 20, bandwidth_hz, 64,
                                   "Long range (1572 km), limited velocity (±46 m/s)"),
            'medium_long': WaveformConfig("Medium Long", 18, bandwidth_hz, 128,
                                          "Medium range (393 km), moderate velocity (±185 m/s)"),
            'medium': WaveformConfig("Medium", 15, bandwidth_hz, 256,
                                     "Standard (49 km), good velocity (±1480 m/s)"),
            'short': WaveformConfig("Short", 11, bandwidth_hz, 512,
                                    "Short range (3 km), high velocity (±23.7 km/s)"),
        }
    
    def select_waveform(self, 
                       expected_range_km: float,
                       expected_velocity_mps: float,
                       required_snr_db: float = 13) -> Tuple[str, WaveformConfig, Dict]:
        """
        Select optimal waveform for scenario
        
        Args:
            expected_range_km: Expected target range (km)
            expected_velocity_mps: Expected target velocity (m/s)
            required_snr_db: Required SNR for detection
            
        Returns:
            (waveform_name, config, analysis)
        """
        expected_range_m = expected_range_km * 1000
        
        analysis = {
            'candidates': [],
            'selected': None,
            'reason': '',
        }
        
        # Evaluate each waveform
        for name, cfg in self.waveforms.items():
            r_unamb = cfg.get_unambiguous_range_m()
            v_unamb = cfg.get_unambiguous_velocity_mps(self.wavelength)
            
            # Check range constraint
            range_ok = expected_range_m <= r_unamb
            
            # Check velocity constraint (with 20% margin)
            velocity_ok = abs(expected_velocity_mps) <= v_unamb * 0.8
            
            # Score based on processing gain (higher is better for range)
            score = cfg.processing_gain_db if range_ok else 0
            
            # Bonus for velocity coverage
            if velocity_ok:
                score += 10
            
            analysis['candidates'].append({
                'name': name,
                'r_unamb_km': r_unamb / 1000,
                'v_unamb_mps': v_unamb,
                'range_ok': range_ok,
                'velocity_ok': velocity_ok,
                'score': score,
            })
        
        # Select best
        analysis['candidates'].sort(key=lambda x: x['score'], reverse=True)
        
        best = analysis['candidates'][0]
        selected_name = best['name']
        
        if best['range_ok'] and best['velocity_ok']:
            analysis['reason'] = "Optimal: Both range and velocity requirements met"
        elif best['range_ok']:
            analysis['reason'] = "Range OK, velocity may be ambiguous - consider staggered PRF"
        else:
            analysis['reason'] = "Warning: Range exceeds unambiguous limit"
        
        analysis['selected'] = selected_name
        
        return selected_name, self.waveforms[selected_name], analysis
    
    def print_recommendation(self, expected_range_km: float, 
                            expected_velocity_mps: float):
        """Print waveform recommendation"""
        name, cfg, analysis = self.select_waveform(expected_range_km, expected_velocity_mps)
        
        print("\n" + "=" * 70)
        print("ADAPTIVE WAVEFORM RECOMMENDATION")
        print("=" * 70)
        
        print(f"\n📋 SCENARIO")
        print("-" * 50)
        print(f"  Expected range:     {expected_range_km:.0f} km")
        print(f"  Expected velocity:  {expected_velocity_mps:.0f} m/s")
        
        print(f"\n📊 CANDIDATE ANALYSIS")
        print("-" * 70)
        print(f"  {'Waveform':<15} {'R_unamb':<12} {'V_unamb':<12} {'Range':<8} {'Vel':<8} {'Score'}")
        print("-" * 70)
        
        for c in analysis['candidates']:
            r_str = f"{c['r_unamb_km']:.0f} km"
            v_str = f"±{c['v_unamb_mps']:.0f} m/s"
            r_ok = "✓" if c['range_ok'] else "✗"
            v_ok = "✓" if c['velocity_ok'] else "✗"
            selected = " ◄" if c['name'] == analysis['selected'] else ""
            print(f"  {c['name']:<15} {r_str:<12} {v_str:<12} {r_ok:<8} {v_ok:<8} {c['score']:.0f}{selected}")
        
        print("-" * 70)
        
        print(f"\n✅ RECOMMENDATION: {name.upper()}")
        print("-" * 50)
        print(f"  PRBS Order:         {cfg.prbs_order}")
        print(f"  Processing Gain:    {cfg.processing_gain_db:.1f} dB")
        print(f"  Unambiguous Range:  {cfg.get_unambiguous_range_m()/1000:.0f} km")
        print(f"  Unambiguous Vel:    ±{cfg.get_unambiguous_velocity_mps(self.wavelength):.0f} m/s")
        print(f"\n  {analysis['reason']}")


#=============================================================================
# Demo & Validation
#=============================================================================

def demo():
    """Demonstrate enhanced Doppler processing"""
    
    print("\n" + "█" * 80)
    print("  TITAN ENHANCED DOPPLER PROCESSING - DEMO")
    print("█" * 80)
    
    wavelength = C / 155e6  # 1.93 m
    
    # =========================================================================
    # 1. Multi-PRF Velocity Unwrapping
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO 1: MULTI-PRF VELOCITY UNWRAPPING")
    print("=" * 70)
    
    # PRFs chosen for good unwrapping (coprime-ish)
    prfs = [95.4, 382, 1490]  # From PRBS-20, PRBS-18, PRBS-16
    unwrapper = VelocityUnwrapper(prfs, wavelength)
    unwrapper.print_config()
    
    # Test unwrapping
    print(f"\n📋 UNWRAPPING TEST")
    print("-" * 60)
    
    test_velocities = [50, 100, 250, 500, 800]
    
    for v_true in test_velocities:
        # Simulate folded measurements
        measured = []
        for prf, v_ua in zip(prfs, unwrapper.v_unamb):
            v_folded = ((v_true + v_ua) % (2 * v_ua)) - v_ua
            measured.append(v_folded)
        
        v_unwrapped, confidence = unwrapper.unwrap_velocity(measured[:2])
        error = abs(v_unwrapped - v_true)
        
        status = "✓" if error < 5 else "⚠"
        print(f"  True: {v_true:>4} m/s → Measured: [{measured[0]:>6.1f}, {measured[1]:>6.1f}] → "
              f"Unwrapped: {v_unwrapped:>6.1f} m/s (error: {error:.1f}) {status}")
    
    # =========================================================================
    # 2. Staggered PRF Processor
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO 2: STAGGERED PRF PROCESSOR")
    print("=" * 70)
    
    staggered = StaggeredPRFProcessor(wavelength)
    staggered.print_config()
    
    # Test scenario
    print(f"\n📋 SCENARIO TEST: Fighter at 200 km, 600 m/s")
    print("-" * 60)
    
    results = staggered.process_burst(200e3, 600)
    
    print(f"\n  Measurements:")
    for m in results['measurements']:
        print(f"    {m['waveform']}: V={m['v_measured']:.1f} m/s (unamb: ±{m['v_unamb']:.0f}), "
              f"R={m['r_measured']/1000:.1f} km")
    
    if results['unwrapped_velocity']:
        error = abs(results['unwrapped_velocity'] - results['true_velocity_mps'])
        print(f"\n  Unwrapped velocity: {results['unwrapped_velocity']:.1f} m/s "
              f"(true: {results['true_velocity_mps']:.0f}, error: {error:.1f})")
        print(f"  Confidence: {results['confidence']*100:.0f}%")
    
    # =========================================================================
    # 3. Adaptive Waveform Selector
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO 3: ADAPTIVE WAVEFORM SELECTION")
    print("=" * 70)
    
    selector = AdaptiveWaveformSelector(wavelength)
    
    scenarios = [
        (50, 100, "Close subsonic"),
        (200, 300, "Medium range cruise missile"),
        (400, 600, "Long range fighter"),
        (1000, 250, "Very long range bomber"),
    ]
    
    for r, v, desc in scenarios:
        print(f"\n{'─'*70}")
        print(f"  Scenario: {desc}")
        selector.print_recommendation(r, v)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "█" * 80)
    print("  SUMMARY: VELOCITY AMBIGUITY SOLUTIONS")
    print("█" * 80)
    
    print("""
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    VELOCITY AMBIGUITY MITIGATION                         │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Problem: PRBS-20 gives ±46 m/s unambiguous velocity                     │
    │           Most aircraft exceed this!                                     │
    │                                                                          │
    │  Solution 1: MULTI-PRF UNWRAPPING                                        │
    │    - Use multiple PRFs (e.g., 95 Hz, 382 Hz, 1490 Hz)                   │
    │    - Chinese Remainder Theorem resolves ambiguity                        │
    │    - Extended velocity: ±2000+ m/s                                       │
    │    - Trade-off: Reduced coherent integration                             │
    │                                                                          │
    │  Solution 2: STAGGERED PRF SCHEDULE                                      │
    │    - Interleave long (range) and short (velocity) sequences              │
    │    - Best of both worlds                                                 │
    │    - Slightly reduced processing gain                                    │
    │                                                                          │
    │  Solution 3: ADAPTIVE WAVEFORM SELECTION                                 │
    │    - Select waveform based on expected scenario                          │
    │    - Surveillance: Long sequences for range                              │
    │    - Tracking: Short sequences for velocity                              │
    │    - Automatic mode switching                                            │
    │                                                                          │
    │  RECOMMENDATION: Use Staggered PRF for general surveillance              │
    │                  Use Adaptive Selection for track-while-scan             │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    demo()
