#!/usr/bin/env python3
"""
TITAN Radar - Main Application
Production VHF Radar System for RFSoC 4x2

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Usage:
    python3 run_titan.py --mode simulation    # Test without hardware
    python3 run_titan.py --mode loopback      # Self-test with loopback
    python3 run_titan.py --mode radar         # Full radar operation
    python3 run_titan.py --benchmark          # Performance benchmark

Based on POC "Garažni Pobunjenik" algorithms, now production-ready.
"""

import argparse
import sys
import time
import numpy as np
from typing import Optional

# Import TITAN modules
from titan_signal_processor import TITANConfig, TITANProcessor, generate_test_signal
from titan_rfsoc_driver import TITANRFSoC, RFSoCConfig

# Optional display
try:
    from titan_display import TITANDisplay, DisplayConfig
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False


#=============================================================================
# Version Info
#=============================================================================

VERSION = "2.0.0"
CODENAME = "Garažni Pobunjenik Production"


def print_banner():
    """Print TITAN banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ████████╗██╗████████╗ █████╗ ███╗   ██╗    ██████╗  █████╗ ██████╗  █████╗  ██████╗  ║
║      ██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗ ║
║      ██║   ██║   ██║   ███████║██╔██╗ ██║    ██████╔╝███████║██║  ██║███████║██████╔╝ ║
║      ██║   ██║   ██║   ██╔══██║██║╚██╗██║    ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗ ║
║      ██║   ██║   ██║   ██║  ██║██║ ╚████║    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║ ║
║      ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ║
║                                                                               ║
║                    VHF Anti-Stealth Radar System                              ║
║                    RFSoC 4x2 Platform - Production                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"    Version: {VERSION} ({CODENAME})")
    print(f"    Author:  Dr. Mladen Mešter")
    print(f"    (c) 2026 - All Rights Reserved")
    print()


#=============================================================================
# Operating Modes
#=============================================================================

def run_simulation_mode(config: TITANConfig, num_cpis: int = 1000):
    """
    Run in simulation mode (no hardware required)
    
    Tests the signal processing algorithms with synthetic data.
    """
    print("\n" + "=" * 70)
    print("SIMULATION MODE")
    print("=" * 70)
    
    config.print_config()
    
    # Initialize processor
    processor = TITANProcessor(config)
    
    # Define test targets
    targets = [
        (100, 1.0, 0),      # Close, strong, stationary
        (500, 0.5, 100),    # Medium range, moving away
        (1000, 0.2, -50),   # Far, weak, approaching
    ]
    
    print(f"\n[Simulation] Generating {config.num_doppler_bins} CPIs with {len(targets)} targets...")
    print(f"  Targets: {targets}")
    
    # Process CPIs
    start_time = time.time()
    
    for cpi_idx in range(config.num_doppler_bins):
        rx_cpi = generate_test_signal(config, targets)
        processor.process_cpi(rx_cpi, mode='fft')
        
        if (cpi_idx + 1) % 100 == 0:
            print(f"  Processed {cpi_idx + 1}/{config.num_doppler_bins} CPIs")
    
    process_time = time.time() - start_time
    
    # Generate Range-Doppler map
    print("\n[Simulation] Generating Range-Doppler map...")
    rdmap = processor.generate_rdmap()
    
    # Detect targets
    print("[Simulation] Running 2D CFAR detection...")
    detections = processor.detect_2d(rdmap, pfa=config.cfar_pfa)
    
    # Results
    print("\n" + "-" * 70)
    print("SIMULATION RESULTS")
    print("-" * 70)
    print(f"  Processing time:     {process_time:.2f} s")
    print(f"  CPIs per second:     {config.num_doppler_bins / process_time:.1f}")
    print(f"  Total detections:    {len(detections)}")
    print()
    
    if detections:
        print("  Detected targets:")
        for i, det in enumerate(detections[:10]):  # Show first 10
            print(f"    {i+1}. Range: {det.range_m/1000:.2f} km, "
                  f"Velocity: {det.velocity_mps:+.1f} m/s, "
                  f"SNR: {det.snr_db:.1f} dB")
    
    print("-" * 70)
    
    # Benchmark
    print("\n[Simulation] Running benchmark...")
    processor.benchmark(n_cpis=100, mode='fft')
    
    return processor, rdmap, detections


def run_loopback_mode(config: TITANConfig):
    """
    Run loopback self-test
    
    Requires TX connected to RX with attenuator.
    """
    print("\n" + "=" * 70)
    print("LOOPBACK TEST MODE")
    print("=" * 70)
    print()
    print("  ┌──────────────────────────────────────────────────────────────┐")
    print("  │  HARDWARE SETUP:                                            │")
    print("  │                                                              │")
    print("  │    DAC0 ───[30dB ATT]──► ADC0                               │")
    print("  │                                                              │")
    print("  │  Connect DAC channel 0 to ADC channel 0 with 30dB           │")
    print("  │  attenuator for loopback testing.                           │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    
    input("Press Enter when hardware is connected...")
    
    rfsoc_config = RFSoCConfig(
        num_adc_channels=1,
        use_fpga_correlator=False,
        use_fpga_cfar=False,
    )
    
    driver = TITANRFSoC(
        bitstream_path="titan_radar.bit",
        config=config,
        rfsoc_config=rfsoc_config
    )
    
    if not driver.initialize():
        print("[Loopback] Initialization failed!")
        return None
    
    # Run loopback test
    print("\n[Loopback] Starting loopback test...")
    
    driver.start_tx()
    time.sleep(0.5)
    
    # Capture CPIs
    range_profiles = []
    for i in range(10):
        rx_channels = driver.capture_cpi()
        range_profile = driver.processor.correlate_cpi(rx_channels[0], mode='fft')
        range_profiles.append(range_profile)
        print(f"  CPI {i+1}/10")
    
    driver.stop_tx()
    
    # Average
    avg_profile = np.mean(range_profiles, axis=0)
    
    # Analyze
    peak_idx = np.argmax(avg_profile)
    peak_val = avg_profile[peak_idx]
    noise_floor = np.median(avg_profile)
    snr_db = 20 * np.log10(peak_val / noise_floor) if noise_floor > 0 else 0
    
    print("\n" + "-" * 70)
    print("LOOPBACK TEST RESULTS")
    print("-" * 70)
    print(f"  Peak range bin:  {peak_idx}")
    print(f"  Peak amplitude:  {peak_val:.2f}")
    print(f"  Noise floor:     {noise_floor:.4f}")
    print(f"  SNR:             {snr_db:.1f} dB")
    print(f"  Expected gain:   {config.processing_gain_db:.1f} dB")
    print()
    
    if snr_db > config.processing_gain_db * 0.8:
        print("  ✅ TEST PASSED - System working correctly")
    else:
        print("  ⚠️  TEST WARNING - SNR lower than expected")
        print("      Check connections and attenuator")
    
    print("-" * 70)
    
    return avg_profile


def run_radar_mode(config: TITANConfig, num_cpis: int = 10000):
    """
    Run full radar operation
    """
    print("\n" + "=" * 70)
    print("RADAR MODE - OPERATIONAL")
    print("=" * 70)
    
    config.print_config()
    
    rfsoc_config = RFSoCConfig(
        num_adc_channels=4,
        use_fpga_correlator=True,
        use_fpga_cfar=True,
    )
    
    driver = TITANRFSoC(
        bitstream_path="titan_radar.bit",
        config=config,
        rfsoc_config=rfsoc_config
    )
    
    if not driver.initialize():
        print("[Radar] Initialization failed!")
        return
    
    # Initialize display if available
    display = None
    if DISPLAY_AVAILABLE:
        print("[Radar] Initializing display...")
        display_config = DisplayConfig(dark_mode=True)
        display = TITANDisplay(display_config)
        display.initialize(
            config.num_range_bins,
            config.num_doppler_bins,
            config.range_resolution_m,
            config.velocity_resolution_mps
        )
        display.setup_all_plots()
    
    # Detection callback
    def on_detection(cpi_idx, rdmap, detections):
        if detections:
            print(f"\n[CPI {cpi_idx}] === CONTACTS ===")
            for i, det in enumerate(detections[:5]):
                print(f"  Contact {i+1}: "
                      f"R={det.range_m/1000:.1f}km, "
                      f"V={det.velocity_mps:+.0f}m/s, "
                      f"SNR={det.snr_db:.0f}dB")
            
            if display:
                display.update(rdmap=rdmap, detections=detections)
                display.refresh()
    
    print("\n[Radar] Starting operational loop...")
    print("Press Ctrl+C to stop\n")
    
    try:
        driver.run_processing_loop(num_cpis=num_cpis, callback=on_detection)
    except KeyboardInterrupt:
        print("\n[Radar] Stopping...")
    
    driver.stop()
    
    # Print final statistics
    stats = driver.processor.get_statistics()
    print("\n" + "-" * 70)
    print("RADAR SESSION STATISTICS")
    print("-" * 70)
    print(f"  Total CPIs processed: {stats['total_cpis']}")
    print(f"  Total detections:     {stats['total_detections']}")
    print("-" * 70)


def run_benchmark(config: TITANConfig):
    """
    Run performance benchmark
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    config.print_config()
    
    processor = TITANProcessor(config)
    
    # Benchmark FFT correlation
    print("\n[Benchmark] Testing FFT correlation...")
    fft_results = processor.benchmark(n_cpis=200, mode='fft')
    
    # Benchmark zero-DSP if Numba available
    from titan_signal_processor import NUMBA_AVAILABLE
    if NUMBA_AVAILABLE:
        print("\n[Benchmark] Testing Zero-DSP correlation (Numba)...")
        zdsp_results = processor.benchmark(n_cpis=20, mode='zero_dsp')
    
    # Summary
    print("\n" + "-" * 70)
    print("BENCHMARK SUMMARY")
    print("-" * 70)
    print(f"  FFT Correlation:")
    print(f"    Time per CPI:  {fft_results['time_per_cpi_ms']:.2f} ms")
    print(f"    Throughput:    {fft_results['throughput_msps']:.1f} Msps")
    
    if NUMBA_AVAILABLE:
        print(f"\n  Zero-DSP Correlation (Numba):")
        print(f"    Time per CPI:  {zdsp_results['time_per_cpi_ms']:.2f} ms")
        print(f"    Throughput:    {zdsp_results['throughput_msps']:.1f} Msps")
    
    print("-" * 70)


#=============================================================================
# Main Entry Point
#=============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TITAN VHF Radar System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_titan.py --mode simulation        # Test without hardware
  python run_titan.py --mode loopback          # Self-test with loopback
  python run_titan.py --mode radar             # Full radar operation
  python run_titan.py --benchmark              # Performance benchmark
  python run_titan.py --mode simulation --prbs 20  # Use PRBS-20
        """
    )
    
    parser.add_argument('--mode', choices=['simulation', 'loopback', 'radar'],
                       default='simulation', help='Operating mode')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--prbs', type=int, choices=[15, 20, 23], default=15,
                       help='PRBS order (default: 15)')
    parser.add_argument('--range-bins', type=int, default=512,
                       help='Number of range bins (default: 512)')
    parser.add_argument('--doppler-bins', type=int, default=256,
                       help='Number of Doppler bins (default: 256)')
    parser.add_argument('--cpis', type=int, default=1000,
                       help='Number of CPIs to process (default: 1000)')
    parser.add_argument('--bitstream', type=str, default='titan_radar.bit',
                       help='FPGA bitstream path')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress banner')
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Create configuration
    config = TITANConfig(
        prbs_order=args.prbs,
        num_range_bins=args.range_bins,
        num_doppler_bins=args.doppler_bins,
        cpi_samples=32768,
    )
    
    # Run selected mode
    if args.benchmark:
        run_benchmark(config)
    elif args.mode == 'simulation':
        run_simulation_mode(config, num_cpis=args.cpis)
    elif args.mode == 'loopback':
        run_loopback_mode(config)
    elif args.mode == 'radar':
        run_radar_mode(config, num_cpis=args.cpis)
    
    print("\n[TITAN] Session complete.")


if __name__ == "__main__":
    main()
