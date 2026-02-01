#!/usr/bin/env python3
"""
QEDMMA PoC - Main Launcher Script
"Garažni Pobunjenik" v3.4

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Usage:
    python3 run_poc.py                    # Interactive menu
    python3 run_poc.py --test             # Run self-tests
    python3 run_poc.py --sim              # Simulation demo
    python3 run_poc.py --radar            # Live radar mode
"""

import sys
import os
import argparse

# Add directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'software'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'test'))

#=============================================================================
# Banner
#=============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ███████╗██████╗ ███╗   ███╗███╗   ███╗ █████╗                   ║
║    ██╔═══██╗██╔════╝██╔══██╗████╗ ████║████╗ ████║██╔══██╗                  ║
║    ██║   ██║█████╗  ██║  ██║██╔████╔██║██╔████╔██║███████║                  ║
║    ██║▄▄ ██║██╔══╝  ██║  ██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║                  ║
║    ╚██████╔╝███████╗██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║                  ║
║     ╚══▀▀═╝ ╚══════╝╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝                  ║
║                                                                              ║
║            "GARAŽNI POBUNJENIK" - Proof of Concept v3.4                     ║
║                                                                              ║
║    Budget: €495 | Range: 10-100 km | Processing Gain: 45 dB                 ║
║                                                                              ║
║    Author: Dr. Mladen Mešter                                                 ║
║    Copyright (c) 2026 - All Rights Reserved                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

MENU = """
┌──────────────────────────────────────────────────────────────────────────────┐
│                              MAIN MENU                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   [1] Run Self-Test Suite        - Verify correlator & processing gain      │
│   [2] Simulation Demo            - Demo with synthetic targets               │
│   [3] Loopback Test              - Test with PlutoSDR (needs hardware)       │
│   [4] Live Radar Mode            - Full radar operation                      │
│   [5] Correlator Benchmark       - Measure processing speed                  │
│   [6] Display BOM                - Show Bill of Materials                    │
│   [7] System Info                - Show system configuration                 │
│                                                                              │
│   [Q] Quit                                                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""

#=============================================================================
# Menu Functions
#=============================================================================

def run_self_tests():
    """Run complete self-test suite"""
    print("\n[Launching Self-Test Suite...]\n")
    try:
        from loopback_test import run_all_tests
        run_all_tests()
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to continue...")

def run_simulation():
    """Run simulation demo"""
    print("\n[Launching Simulation Demo...]\n")
    try:
        from zero_dsp_correlator import demo
        demo()
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to continue...")

def run_loopback():
    """Run loopback test with hardware"""
    print("\n[Launching Loopback Test...]\n")
    print("⚠️  Requires: PlutoSDR TX → 30dB Attenuator → PlutoSDR RX")
    print()
    try:
        from pluto_radar import run_loopback_test
        run_loopback_test()
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to continue...")

def run_radar():
    """Run live radar mode"""
    print("\n[Launching Live Radar Mode...]\n")
    print("⚠️  Requires: Full hardware setup (PA, LNA, Antennas)")
    print()
    try:
        from pluto_radar import run_radar_mode
        run_radar_mode("monostatic")
    except KeyboardInterrupt:
        print("\nRadar stopped.")
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to continue...")

def run_benchmark():
    """Run correlator benchmark"""
    print("\n[Launching Correlator Benchmark...]\n")
    try:
        from zero_dsp_correlator import ZeroDSPCorrelator
        
        for order in [11, 15, 20]:
            print(f"\n{'='*40}")
            print(f"PRBS-{order} Correlator")
            print('='*40)
            corr = ZeroDSPCorrelator(prbs_order=order, num_lanes=512, mode='fft')
            corr.benchmark(n_iterations=100)
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to continue...")

def show_bom():
    """Display Bill of Materials"""
    bom_path = os.path.join(SCRIPT_DIR, 'hardware', 'BOM_GARAZNI_POBUNJENIK.csv')
    
    print("\n" + "=" * 80)
    print("BILL OF MATERIALS - Garažni Pobunjenik v3.4")
    print("=" * 80 + "\n")
    
    if os.path.exists(bom_path):
        import csv
        with open(bom_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Print header
            print(f"{'#':>2} {'Category':<12} {'Part':<20} {'Qty':>3} {'Price':>8} {'Total':>8}")
            print("-" * 60)
            
            total = 0
            for row in reader:
                if row and not row[0].startswith('#') and row[0]:
                    try:
                        num = row[0]
                        cat = row[1][:12]
                        part = row[2][:20]
                        qty = row[4]
                        price = row[5]
                        tot = row[6]
                        
                        if 'TOTAL' in cat:
                            print("-" * 60)
                            print(f"{'':>2} {'TOTAL':<12} {'':<20} {'':>3} {'':>8} {'€'+tot:>8}")
                        else:
                            print(f"{num:>2} {cat:<12} {part:<20} {qty:>3} {'€'+price:>8} {'€'+tot:>8}")
                            try:
                                total += float(tot)
                            except:
                                pass
                    except:
                        pass
    else:
        print("BOM file not found!")
    
    print("\n" + "=" * 80)
    input("\nPress Enter to continue...")

def show_system_info():
    """Display system configuration"""
    print("\n" + "=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    
    print("\n[Radar Parameters]")
    print(f"  Center Frequency:   155 MHz")
    print(f"  Sample Rate:        4 MSPS")
    print(f"  PRBS Order:         15 (32,767 chips)")
    print(f"  Chip Rate:          1 Mchip/s")
    print(f"  Processing Gain:    45.1 dB")
    print(f"  Range Resolution:   150 m")
    print(f"  Max Range Bins:     512")
    print(f"  Max Range:          76.8 km")
    
    print("\n[Hardware]")
    print(f"  SDR:                ADALM-PLUTO")
    print(f"  PA:                 RA30H1317M (30W)")
    print(f"  LNA:                SPF5189Z (NF 0.6dB)")
    print(f"  Antenna:            5-element Yagi (~10.5 dBi)")
    
    print("\n[Software Dependencies]")
    
    deps = ['numpy', 'scipy', 'matplotlib', 'numba', 'adi']
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"  {dep:<15} ✅ v{ver}")
        except ImportError:
            print(f"  {dep:<15} ❌ Not installed")
    
    print("\n" + "=" * 60)
    input("\nPress Enter to continue...")

#=============================================================================
# Main
#=============================================================================

def interactive_menu():
    """Run interactive menu"""
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print(BANNER)
        print(MENU)
        
        choice = input("Select option: ").strip().upper()
        
        if choice == '1':
            run_self_tests()
        elif choice == '2':
            run_simulation()
        elif choice == '3':
            run_loopback()
        elif choice == '4':
            run_radar()
        elif choice == '5':
            run_benchmark()
        elif choice == '6':
            show_bom()
        elif choice == '7':
            show_system_info()
        elif choice == 'Q':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("Invalid option!")
            input("Press Enter to continue...")

def main():
    parser = argparse.ArgumentParser(description="QEDMMA PoC Launcher")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    parser.add_argument("--radar", action="store_true", help="Run live radar")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()
    
    print(BANNER)
    
    if args.test:
        run_self_tests()
    elif args.sim:
        run_simulation()
    elif args.radar:
        run_radar()
    elif args.bench:
        run_benchmark()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
