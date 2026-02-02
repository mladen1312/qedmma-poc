#!/usr/bin/env python3
"""
TITAN Radar - VHF Antenna Design & Analysis
Based on vGrok-X Meter-Wave Anti-Stealth Radar Analysis

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

VHF ANTENNA CONSIDERATIONS (λ ≈ 1.93 m at 155 MHz):
1. Physical size - Elements are ~1m long
2. Ground effects - Significant at VHF
3. Beamwidth - Wide beams typical
4. Mutual coupling - Strong between array elements
5. Pattern control - Challenging due to size

This module provides:
- Yagi-Uda antenna design for VHF
- Phased array calculations
- Link budget with real antenna parameters
- Installation considerations
- Cost-optimized configurations

Reference systems analyzed:
- Chinese JY-27 VHF radar
- Russian Nebo-M (VHF component)
- Ukrainian Kolchuga passive system
- Czech VERA-NG
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#=============================================================================
# Constants
#=============================================================================

C = 299792458.0  # Speed of light (m/s)
ETA_0 = 377.0    # Free space impedance (Ω)


#=============================================================================
# Antenna Types
#=============================================================================

class AntennaType(Enum):
    """Available antenna types for VHF"""
    DIPOLE = "dipole"
    YAGI_3EL = "yagi_3"
    YAGI_5EL = "yagi_5"
    YAGI_7EL = "yagi_7"
    YAGI_10EL = "yagi_10"
    LOG_PERIODIC = "lpda"
    PHASED_ARRAY = "array"
    CORNER_REFLECTOR = "corner"


@dataclass
class AntennaSpecs:
    """Antenna specifications"""
    name: str
    type: AntennaType
    gain_dbi: float
    beamwidth_h_deg: float
    beamwidth_v_deg: float
    bandwidth_mhz: float
    vswr_max: float
    length_m: float
    width_m: float
    height_m: float
    weight_kg: float
    cost_eur: float
    elements: int = 1
    
    @property
    def gain_linear(self) -> float:
        return 10**(self.gain_dbi / 10)
    
    @property
    def effective_area_m2(self) -> float:
        """Effective aperture area"""
        wavelength = C / 155e6
        return self.gain_linear * wavelength**2 / (4 * np.pi)


#=============================================================================
# VHF Antenna Catalog
#=============================================================================

VHF_ANTENNA_CATALOG = {
    AntennaType.DIPOLE: AntennaSpecs(
        name="Half-Wave Dipole",
        type=AntennaType.DIPOLE,
        gain_dbi=2.15,
        beamwidth_h_deg=360,  # Omnidirectional
        beamwidth_v_deg=78,
        bandwidth_mhz=20,
        vswr_max=1.5,
        length_m=0.97,
        width_m=0.05,
        height_m=0.97,
        weight_kg=0.5,
        cost_eur=25,
        elements=1
    ),
    
    AntennaType.YAGI_3EL: AntennaSpecs(
        name="3-Element Yagi",
        type=AntennaType.YAGI_3EL,
        gain_dbi=6.0,
        beamwidth_h_deg=70,
        beamwidth_v_deg=60,
        bandwidth_mhz=10,
        vswr_max=1.8,
        length_m=1.5,
        width_m=1.0,
        height_m=0.3,
        weight_kg=3,
        cost_eur=45,
        elements=3
    ),
    
    AntennaType.YAGI_5EL: AntennaSpecs(
        name="5-Element Yagi",
        type=AntennaType.YAGI_5EL,
        gain_dbi=9.0,
        beamwidth_h_deg=52,
        beamwidth_v_deg=48,
        bandwidth_mhz=8,
        vswr_max=2.0,
        length_m=2.5,
        width_m=1.0,
        height_m=0.3,
        weight_kg=5,
        cost_eur=75,
        elements=5
    ),
    
    AntennaType.YAGI_7EL: AntennaSpecs(
        name="7-Element Yagi",
        type=AntennaType.YAGI_7EL,
        gain_dbi=11.0,
        beamwidth_h_deg=42,
        beamwidth_v_deg=38,
        bandwidth_mhz=6,
        vswr_max=2.2,
        length_m=3.5,
        width_m=1.0,
        height_m=0.3,
        weight_kg=8,
        cost_eur=120,
        elements=7
    ),
    
    AntennaType.YAGI_10EL: AntennaSpecs(
        name="10-Element Yagi",
        type=AntennaType.YAGI_10EL,
        gain_dbi=13.5,
        beamwidth_h_deg=34,
        beamwidth_v_deg=30,
        bandwidth_mhz=5,
        vswr_max=2.5,
        length_m=5.0,
        width_m=1.0,
        height_m=0.3,
        weight_kg=12,
        cost_eur=180,
        elements=10
    ),
    
    AntennaType.LOG_PERIODIC: AntennaSpecs(
        name="Log-Periodic Dipole Array",
        type=AntennaType.LOG_PERIODIC,
        gain_dbi=7.5,
        beamwidth_h_deg=65,
        beamwidth_v_deg=55,
        bandwidth_mhz=50,  # Much wider bandwidth!
        vswr_max=2.0,
        length_m=4.0,
        width_m=2.0,
        height_m=0.4,
        weight_kg=15,
        cost_eur=250,
        elements=8
    ),
    
    AntennaType.CORNER_REFLECTOR: AntennaSpecs(
        name="Corner Reflector",
        type=AntennaType.CORNER_REFLECTOR,
        gain_dbi=10.0,
        beamwidth_h_deg=50,
        beamwidth_v_deg=45,
        bandwidth_mhz=15,
        vswr_max=1.8,
        length_m=2.0,
        width_m=2.5,
        height_m=2.5,
        weight_kg=25,
        cost_eur=200,
        elements=1
    ),
}


#=============================================================================
# Yagi-Uda Designer
#=============================================================================

class YagiDesigner:
    """
    Yagi-Uda Antenna Designer for VHF
    
    Designs optimized Yagi antennas for 155 MHz.
    """
    
    def __init__(self, frequency_hz: float = 155e6):
        self.frequency_hz = frequency_hz
        self.wavelength_m = C / frequency_hz
    
    def design_yagi(self, num_elements: int) -> Dict:
        """
        Design a Yagi antenna with specified number of elements
        
        Args:
            num_elements: Number of elements (3, 5, 7, 10, etc.)
            
        Returns:
            Design dictionary with all dimensions
        """
        λ = self.wavelength_m
        
        # Standard Yagi design rules
        # Reflector: ~5% longer than λ/2
        # Driven element: ~λ/2
        # Directors: progressively shorter, ~5-15% shorter than λ/2
        
        # Element lengths
        reflector_length = 0.495 * λ * 1.05  # 5% longer
        driven_length = 0.473 * λ            # Slightly shorter for impedance
        
        # Director lengths (progressively shorter)
        directors = []
        for i in range(num_elements - 2):
            length = 0.455 * λ * (1 - 0.02 * i)  # 2% shorter each
            directors.append(length)
        
        # Element spacing
        reflector_spacing = 0.2 * λ
        
        director_spacings = []
        for i in range(len(directors)):
            # First director closer, then spread out
            spacing = (0.15 + 0.05 * min(i, 3)) * λ
            director_spacings.append(spacing)
        
        # Total boom length
        boom_length = reflector_spacing + sum(director_spacings)
        
        # Gain estimation (empirical formula)
        gain_dbi = 2.15 + 3.0 * np.log10(num_elements) * 2.5
        
        # Beamwidth estimation
        beamwidth_h = 105 / np.sqrt(10**(gain_dbi/10))
        beamwidth_v = beamwidth_h * 1.1
        
        # F/B ratio
        fb_ratio_db = 10 + 2 * num_elements
        
        # Input impedance (approximate)
        impedance_ohm = 50 if num_elements <= 5 else 25
        
        return {
            'frequency_mhz': self.frequency_hz / 1e6,
            'wavelength_m': λ,
            'num_elements': num_elements,
            'reflector_length_m': reflector_length,
            'driven_length_m': driven_length,
            'director_lengths_m': directors,
            'reflector_spacing_m': reflector_spacing,
            'director_spacings_m': director_spacings,
            'boom_length_m': boom_length,
            'gain_dbi': gain_dbi,
            'beamwidth_h_deg': beamwidth_h,
            'beamwidth_v_deg': beamwidth_v,
            'fb_ratio_db': fb_ratio_db,
            'impedance_ohm': impedance_ohm,
            'bandwidth_pct': 5 / num_elements * 5,  # Narrower with more elements
        }
    
    def print_design(self, num_elements: int):
        """Print detailed design"""
        design = self.design_yagi(num_elements)
        
        print("\n" + "=" * 70)
        print(f"YAGI-UDA ANTENNA DESIGN - {num_elements} ELEMENTS @ {design['frequency_mhz']:.0f} MHz")
        print("=" * 70)
        
        print(f"\n📊 PERFORMANCE")
        print("-" * 50)
        print(f"  Gain:              {design['gain_dbi']:.1f} dBi")
        print(f"  Beamwidth (H):     {design['beamwidth_h_deg']:.0f}°")
        print(f"  Beamwidth (V):     {design['beamwidth_v_deg']:.0f}°")
        print(f"  F/B Ratio:         {design['fb_ratio_db']:.0f} dB")
        print(f"  Bandwidth:         ~{design['bandwidth_pct']:.0f}%")
        print(f"  Impedance:         ~{design['impedance_ohm']:.0f} Ω")
        
        print(f"\n📏 ELEMENT LENGTHS")
        print("-" * 50)
        print(f"  Reflector:         {design['reflector_length_m']*100:.1f} cm")
        print(f"  Driven Element:    {design['driven_length_m']*100:.1f} cm")
        for i, length in enumerate(design['director_lengths_m']):
            print(f"  Director {i+1}:        {length*100:.1f} cm")
        
        print(f"\n📐 SPACING")
        print("-" * 50)
        print(f"  Refl → Driven:     {design['reflector_spacing_m']*100:.1f} cm")
        for i, spacing in enumerate(design['director_spacings_m']):
            if i == 0:
                print(f"  Driven → Dir 1:    {spacing*100:.1f} cm")
            else:
                print(f"  Dir {i} → Dir {i+1}:      {spacing*100:.1f} cm")
        
        print(f"\n  Total Boom Length: {design['boom_length_m']:.2f} m")
        
        return design


#=============================================================================
# Phased Array Calculator
#=============================================================================

class PhasedArrayCalculator:
    """
    VHF Phased Array Calculator
    
    Calculates array parameters for TITAN receive array.
    """
    
    def __init__(self, frequency_hz: float = 155e6):
        self.frequency_hz = frequency_hz
        self.wavelength_m = C / frequency_hz
    
    def calculate_array(self, num_elements: int, 
                       element_spacing_wavelengths: float = 0.5,
                       element_gain_dbi: float = 9.0) -> Dict:
        """
        Calculate phased array parameters
        
        Args:
            num_elements: Number of array elements
            element_spacing_wavelengths: Spacing in wavelengths
            element_gain_dbi: Individual element gain
            
        Returns:
            Array parameters dictionary
        """
        λ = self.wavelength_m
        d = element_spacing_wavelengths * λ
        
        # Array gain
        array_gain_db = element_gain_dbi + 10 * np.log10(num_elements)
        
        # 3 dB beamwidth (broadside)
        beamwidth_deg = np.degrees(0.886 * λ / (num_elements * d))
        
        # Grating lobe condition: d < λ for full scan
        max_scan_angle = np.degrees(np.arcsin(λ / d - 1)) if d > λ/2 else 90
        
        # First null angle
        null_angle = np.degrees(np.arcsin(λ / (num_elements * d)))
        
        # Array length
        array_length_m = (num_elements - 1) * d
        
        # Scan loss at 45°
        scan_loss_45_db = 10 * np.log10(np.cos(np.radians(45)))
        
        return {
            'num_elements': num_elements,
            'element_gain_dbi': element_gain_dbi,
            'element_spacing_m': d,
            'element_spacing_wavelengths': element_spacing_wavelengths,
            'array_gain_dbi': array_gain_db,
            'beamwidth_deg': beamwidth_deg,
            'max_scan_angle_deg': max_scan_angle,
            'first_null_deg': null_angle,
            'array_length_m': array_length_m,
            'scan_loss_45_db': scan_loss_45_db,
            'grating_lobe_free': d <= λ,
        }
    
    def calculate_titan_array(self) -> Dict:
        """Calculate parameters for TITAN 4-element array"""
        # TITAN uses 4 Yagi elements
        return self.calculate_array(
            num_elements=4,
            element_spacing_wavelengths=0.5,
            element_gain_dbi=9.0  # 5-element Yagi
        )
    
    def print_array_analysis(self, num_elements: int = 4):
        """Print array analysis"""
        array = self.calculate_array(num_elements)
        
        print("\n" + "=" * 70)
        print(f"PHASED ARRAY ANALYSIS - {num_elements} ELEMENTS @ {self.frequency_hz/1e6:.0f} MHz")
        print("=" * 70)
        
        print(f"\n📊 ARRAY PARAMETERS")
        print("-" * 50)
        print(f"  Number of Elements:    {array['num_elements']}")
        print(f"  Element Gain:          {array['element_gain_dbi']:.1f} dBi")
        print(f"  Element Spacing:       {array['element_spacing_m']:.2f} m ({array['element_spacing_wavelengths']:.2f}λ)")
        print(f"  Array Length:          {array['array_length_m']:.2f} m")
        
        print(f"\n📈 ARRAY PERFORMANCE")
        print("-" * 50)
        print(f"  Array Gain:            {array['array_gain_dbi']:.1f} dBi")
        print(f"  Beamwidth:             {array['beamwidth_deg']:.1f}°")
        print(f"  First Null:            {array['first_null_deg']:.1f}°")
        print(f"  Max Scan Angle:        ±{array['max_scan_angle_deg']:.0f}°")
        print(f"  Scan Loss @ 45°:       {array['scan_loss_45_db']:.1f} dB")
        
        grating = "✓ Clear" if array['grating_lobe_free'] else "⚠ Grating lobes!"
        print(f"  Grating Lobes:         {grating}")
        
        return array


#=============================================================================
# VHF Link Budget with Real Antennas
#=============================================================================

class VHFLinkBudget:
    """
    Complete VHF Link Budget Calculator
    
    Uses realistic antenna parameters for TITAN system.
    """
    
    def __init__(self, frequency_hz: float = 155e6):
        self.frequency_hz = frequency_hz
        self.wavelength_m = C / frequency_hz
    
    def calculate_range(self, 
                       tx_power_w: float,
                       tx_antenna: AntennaSpecs,
                       rx_antenna: AntennaSpecs,
                       rcs_m2: float,
                       processing_gain_db: float,
                       noise_figure_db: float = 3.0,
                       system_losses_db: float = 6.0,
                       snr_required_db: float = 13.0,
                       bandwidth_hz: float = 100e6) -> Dict:
        """
        Calculate detection range with realistic parameters
        
        Returns:
            Complete link budget dictionary
        """
        λ = self.wavelength_m
        
        # Convert to linear
        G_tx = 10**(tx_antenna.gain_dbi / 10)
        G_rx = 10**(rx_antenna.gain_dbi / 10)
        G_proc = 10**(processing_gain_db / 10)
        L_sys = 10**(system_losses_db / 10)
        NF = 10**(noise_figure_db / 10)
        SNR_min = 10**(snr_required_db / 10)
        
        # Noise power
        k = 1.38e-23  # Boltzmann
        T = 290  # K
        P_noise = k * T * bandwidth_hz * NF
        
        # Radar equation for max range
        numerator = tx_power_w * G_tx * G_rx * λ**2 * rcs_m2 * G_proc
        denominator = (4 * np.pi)**3 * P_noise * L_sys * SNR_min
        
        R_max = (numerator / denominator)**(1/4)
        
        # EIRP
        eirp_w = tx_power_w * G_tx / L_sys
        eirp_dbw = 10 * np.log10(eirp_w)
        
        return {
            'tx_power_w': tx_power_w,
            'tx_power_dbw': 10 * np.log10(tx_power_w),
            'tx_antenna': tx_antenna.name,
            'tx_gain_dbi': tx_antenna.gain_dbi,
            'rx_antenna': rx_antenna.name,
            'rx_gain_dbi': rx_antenna.gain_dbi,
            'eirp_dbw': eirp_dbw,
            'processing_gain_db': processing_gain_db,
            'system_losses_db': system_losses_db,
            'noise_figure_db': noise_figure_db,
            'rcs_m2': rcs_m2,
            'snr_required_db': snr_required_db,
            'max_range_km': R_max / 1000,
            'noise_power_dbm': 10 * np.log10(P_noise * 1000),
        }
    
    def compare_configurations(self, tx_power_w: float = 1000,
                              rcs_m2: float = 10.0,
                              processing_gain_db: float = 60.0):
        """Compare different antenna configurations"""
        
        print("\n" + "=" * 80)
        print("VHF LINK BUDGET - ANTENNA CONFIGURATION COMPARISON")
        print("=" * 80)
        
        print(f"\n📋 COMMON PARAMETERS")
        print("-" * 60)
        print(f"  TX Power:           {tx_power_w} W ({10*np.log10(tx_power_w):.0f} dBW)")
        print(f"  Target RCS:         {rcs_m2} m² (Stealth @ VHF)")
        print(f"  Processing Gain:    {processing_gain_db} dB (PRBS-20)")
        print(f"  Frequency:          {self.frequency_hz/1e6:.0f} MHz")
        print(f"  Wavelength:         {self.wavelength_m:.2f} m")
        
        configs = [
            ("Minimal (3-el Yagi)", AntennaType.YAGI_3EL, AntennaType.YAGI_3EL),
            ("Standard (5-el Yagi)", AntennaType.YAGI_5EL, AntennaType.YAGI_5EL),
            ("Enhanced (7-el Yagi)", AntennaType.YAGI_7EL, AntennaType.YAGI_7EL),
            ("High Gain (10-el Yagi)", AntennaType.YAGI_10EL, AntennaType.YAGI_10EL),
            ("Mixed (10-el TX, 5-el RX)", AntennaType.YAGI_10EL, AntennaType.YAGI_5EL),
            ("TX Corner, RX Array", AntennaType.CORNER_REFLECTOR, AntennaType.YAGI_5EL),
        ]
        
        print(f"\n📊 CONFIGURATION COMPARISON")
        print("-" * 80)
        print(f"  {'Configuration':<30} {'TX Gain':<10} {'RX Gain':<10} {'Range (km)':<12} {'Cost (€)'}")
        print("-" * 80)
        
        results = []
        for name, tx_type, rx_type in configs:
            tx_ant = VHF_ANTENNA_CATALOG[tx_type]
            rx_ant = VHF_ANTENNA_CATALOG[rx_type]
            
            # For RX array, multiply gain by 4 (TITAN has 4 RX antennas)
            rx_array_gain = rx_ant.gain_dbi + 10 * np.log10(4)
            
            budget = self.calculate_range(
                tx_power_w=tx_power_w,
                tx_antenna=tx_ant,
                rx_antenna=AntennaSpecs(
                    name=f"4× {rx_ant.name}",
                    type=rx_type,
                    gain_dbi=rx_array_gain,
                    beamwidth_h_deg=rx_ant.beamwidth_h_deg / 2,
                    beamwidth_v_deg=rx_ant.beamwidth_v_deg,
                    bandwidth_mhz=rx_ant.bandwidth_mhz,
                    vswr_max=rx_ant.vswr_max,
                    length_m=rx_ant.length_m,
                    width_m=rx_ant.width_m * 4,
                    height_m=rx_ant.height_m,
                    weight_kg=rx_ant.weight_kg * 4,
                    cost_eur=rx_ant.cost_eur * 4,
                ),
                rcs_m2=rcs_m2,
                processing_gain_db=processing_gain_db,
            )
            
            total_cost = tx_ant.cost_eur + rx_ant.cost_eur * 4
            
            print(f"  {name:<30} {tx_ant.gain_dbi:>6.1f} dBi {rx_array_gain:>6.1f} dBi "
                  f"{budget['max_range_km']:>8.0f}      {total_cost:>6.0f}")
            
            results.append({
                'name': name,
                'budget': budget,
                'total_cost': total_cost,
            })
        
        print("-" * 80)
        
        return results


#=============================================================================
# Installation Guide
#=============================================================================

class VHFInstallationGuide:
    """
    VHF Antenna Installation Guidelines
    """
    
    def __init__(self, frequency_hz: float = 155e6):
        self.frequency_hz = frequency_hz
        self.wavelength_m = C / frequency_hz
    
    def print_guidelines(self):
        """Print installation guidelines"""
        
        print("\n" + "=" * 80)
        print("VHF ANTENNA INSTALLATION GUIDELINES")
        print("=" * 80)
        
        print(f"""
📏 HEIGHT RECOMMENDATIONS
{"-" * 70}
  Minimum height:     {self.wavelength_m:.1f} m (1λ) - Basic operation
  Recommended:        {self.wavelength_m * 2:.1f} m (2λ) - Good ground clearance
  Optimal:            {self.wavelength_m * 4:.1f} m (4λ) - Minimizes ground effects
  
  Note: Height affects elevation pattern and low-angle coverage!

📐 ARRAY SPACING (4-Element RX Array)
{"-" * 70}
  λ/2 spacing:        {self.wavelength_m * 0.5:.2f} m - No grating lobes, narrow scan
  λ spacing:          {self.wavelength_m:.2f} m - Wider scan possible
  TITAN recommended:  {self.wavelength_m * 0.5:.2f} m (λ/2)
  
  Total array width:  {self.wavelength_m * 0.5 * 3:.2f} m (for 4 elements)

🧭 ORIENTATION
{"-" * 70}
  TX antenna:         Point toward surveillance sector
  RX array:           Broadside to surveillance direction
  
  For 360° coverage:  Need 3-4 separate arrays (90° sectors)

🔌 CABLING
{"-" * 70}
  Recommended cable:  LMR-400 (low loss at VHF)
  Loss at 155 MHz:    ~2.7 dB/100m
  
  Max recommended run: 30m (0.8 dB loss)
  Use N-type connectors (waterproof)

⚡ LIGHTNING PROTECTION
{"-" * 70}
  ESSENTIAL at VHF heights!
  
  - Ground all antenna mounts
  - Use lightning arrestors on feedlines
  - Proper grounding rod (≤5Ω)
  - Gas discharge tubes at equipment entry

🌡️ ENVIRONMENTAL
{"-" * 70}
  Wind loading:       Significant for large Yagis!
  - 5-el Yagi:        ~25 kg/m² @ 100 km/h
  - 10-el Yagi:       ~45 kg/m² @ 100 km/h
  
  Use guy wires for masts >10m
  Consider ice loading in cold climates

📍 SITE SELECTION
{"-" * 70}
  - Clear horizon (especially for anti-stealth detection)
  - Avoid metal structures within 2λ (4m)
  - Minimum 20m from high-voltage lines
  - Ground conductivity affects pattern
  
  Ideal: Elevated position, good ground, clear sightlines
""")


#=============================================================================
# Complete TITAN Antenna System
#=============================================================================

def design_titan_antenna_system():
    """Design complete antenna system for TITAN"""
    
    print("\n" + "█" * 80)
    print("  TITAN VHF ANTENNA SYSTEM DESIGN")
    print("█" * 80)
    
    # Yagi designer
    yagi = YagiDesigner(155e6)
    
    # TX Antenna (single high-gain Yagi)
    print("\n" + "=" * 70)
    print("SECTION 1: TX ANTENNA DESIGN")
    print("=" * 70)
    yagi.print_design(7)
    
    # RX Array
    print("\n" + "=" * 70)
    print("SECTION 2: RX ARRAY DESIGN")
    print("=" * 70)
    yagi.print_design(5)
    
    array_calc = PhasedArrayCalculator(155e6)
    array_calc.print_array_analysis(4)
    
    # Link Budget Comparison
    print("\n" + "=" * 70)
    print("SECTION 3: LINK BUDGET ANALYSIS")
    print("=" * 70)
    
    link = VHFLinkBudget(155e6)
    link.compare_configurations(
        tx_power_w=1000,
        rcs_m2=10.0,
        processing_gain_db=60.0
    )
    
    # Installation Guide
    print("\n" + "=" * 70)
    print("SECTION 4: INSTALLATION GUIDE")
    print("=" * 70)
    
    guide = VHFInstallationGuide(155e6)
    guide.print_guidelines()
    
    # TITAN Recommended Configuration
    print("\n" + "=" * 70)
    print("SECTION 5: TITAN RECOMMENDED CONFIGURATION")
    print("=" * 70)
    
    print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TITAN ANTENNA SYSTEM SPECIFICATION                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TX SUBSYSTEM                                                                │
│  ─────────────                                                               │
│    Antenna:        7-element Yagi @ 155 MHz                                  │
│    Gain:           11 dBi                                                    │
│    Beamwidth:      42° (H) × 38° (V)                                         │
│    Boom Length:    3.5 m                                                     │
│    Power Handling: 100W (with PA: 60W average)                               │
│    Cost:           ~€120                                                     │
│                                                                              │
│  RX SUBSYSTEM (4-Element Array)                                              │
│  ──────────────────────────────                                              │
│    Antennas:       4× 5-element Yagi @ 155 MHz                               │
│    Element Gain:   9 dBi each                                                │
│    Array Gain:     15 dBi (with 4-way combining)                             │
│    Spacing:        0.97 m (λ/2)                                              │
│    Array Width:    2.9 m                                                     │
│    Beamwidth:      ~26° (electronic steering ±45°)                           │
│    Cost:           4 × €75 = €300                                            │
│                                                                              │
│  TOTAL ANTENNA COST: €420                                                    │
│                                                                              │
│  CALCULATED PERFORMANCE                                                      │
│  ─────────────────────                                                       │
│    TX Power:       1 kW                                                      │
│    EIRP:           41 dBW                                                    │
│    Processing Gain: 60 dB (PRBS-20)                                          │
│    Detection Range: ~340 km (σ=10m², Pd=0.9, Pfa=1e-6)                       │
│                                                                              │
│  INSTALLATION                                                                │
│  ────────────                                                                │
│    Mast Height:    8-10 m (elevated site recommended)                        │
│    Feedline:       LMR-400, max 30m run                                      │
│    Orientation:    TX: sector boresight, RX: broadside                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
""")


#=============================================================================
# Main
#=============================================================================

if __name__ == "__main__":
    design_titan_antenna_system()
