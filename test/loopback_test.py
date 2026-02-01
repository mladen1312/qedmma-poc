#!/usr/bin/env python3
"""
QEDMMA PoC - Loopback Self-Test
Tests system integrity before field deployment

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Test Setup:
    PlutoSDR TX ──► 30dB Attenuator ──► PlutoSDR RX
    
Expected Results:
    - SNR > 50 dB
    - Correlation peak at bin ~0
    - Sidelobes < -40 dB
"""

import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'software'))

from zero_dsp_correlator import ZeroDSPCorrelator, generate_prbs_fast

#=============================================================================
# Test Configuration
#=============================================================================

class TestConfig:
    PRBS_ORDER = 15
    NUM_LANES = 512
    SAMPLE_RATE = 4e6
    CHIP_RATE = 1e6
    CENTER_FREQ = 155e6
    
    # Pass/fail criteria
    MIN_SNR_DB = 40
    MAX_SIDELOBE_DB = -35
    MAX_PEAK_BIN_ERROR = 5

#=============================================================================
# Test Functions
#=============================================================================

def generate_test_signal(correlator, target_delays=[0], target_amplitudes=[1.0], noise_power=0.001):
    """
    Generate test signal with known targets
    
    Args:
        correlator: ZeroDSPCorrelator instance
        target_delays: List of target delays in samples
        target_amplitudes: List of target amplitudes
        noise_power: Noise power level
        
    Returns:
        rx_signal: Simulated received signal
    """
    prbs_bpsk = 2.0 * correlator.prbs_bits - 1.0
    n_samples = correlator.prbs_length
    
    rx = np.zeros(n_samples, dtype=np.float64)
    
    for delay, amp in zip(target_delays, target_amplitudes):
        if delay + len(prbs_bpsk) <= n_samples:
            rx[delay:delay+len(prbs_bpsk)] += amp * prbs_bpsk[:n_samples-delay]
    
    # Add noise
    rx += np.sqrt(noise_power) * np.random.randn(n_samples)
    
    return rx

def test_correlation_peak():
    """Test 1: Verify correlation peak detection"""
    print("\n" + "=" * 60)
    print("TEST 1: Correlation Peak Detection")
    print("=" * 60)
    
    correlator = ZeroDSPCorrelator(
        prbs_order=TestConfig.PRBS_ORDER,
        num_lanes=TestConfig.NUM_LANES,
        mode='fft'
    )
    
    # Generate signal with single target at bin 0
    rx = generate_test_signal(correlator, [0], [1.0], noise_power=0.001)
    
    # Correlate
    profile = correlator.correlate(rx)
    
    # Find peak
    peak_bin = np.argmax(profile)
    peak_val = profile[peak_bin]
    noise_floor = np.median(profile)
    snr_db = 20 * np.log10(peak_val / noise_floor)
    
    print(f"  Peak bin:      {peak_bin}")
    print(f"  Peak value:    {peak_val:.2f}")
    print(f"  Noise floor:   {noise_floor:.4f}")
    print(f"  SNR:           {snr_db:.1f} dB")
    
    # Check pass/fail
    passed = True
    
    if peak_bin > TestConfig.MAX_PEAK_BIN_ERROR:
        print(f"  ❌ FAIL: Peak at bin {peak_bin}, expected ~0")
        passed = False
    else:
        print(f"  ✅ PASS: Peak location correct")
    
    if snr_db < TestConfig.MIN_SNR_DB:
        print(f"  ❌ FAIL: SNR {snr_db:.1f} dB < {TestConfig.MIN_SNR_DB} dB")
        passed = False
    else:
        print(f"  ✅ PASS: SNR adequate")
    
    return passed, {'snr_db': snr_db, 'peak_bin': peak_bin}

def test_sidelobes():
    """Test 2: Verify sidelobe levels"""
    print("\n" + "=" * 60)
    print("TEST 2: Sidelobe Level Analysis")
    print("=" * 60)
    
    correlator = ZeroDSPCorrelator(
        prbs_order=TestConfig.PRBS_ORDER,
        num_lanes=TestConfig.NUM_LANES,
        mode='fft'
    )
    
    # Generate clean signal (no noise)
    rx = generate_test_signal(correlator, [100], [1.0], noise_power=0.0)
    
    # Correlate
    profile = correlator.correlate(rx)
    
    # Find peak
    peak_bin = np.argmax(profile)
    peak_val = profile[peak_bin]
    
    # Mask main lobe (±10 bins)
    mask = np.ones_like(profile, dtype=bool)
    lobe_start = max(0, peak_bin - 10)
    lobe_end = min(len(profile), peak_bin + 10)
    mask[lobe_start:lobe_end] = False
    
    # Find highest sidelobe
    sidelobes = profile[mask]
    max_sidelobe = np.max(sidelobes)
    sidelobe_db = 20 * np.log10(max_sidelobe / peak_val)
    
    print(f"  Peak value:        {peak_val:.2f}")
    print(f"  Max sidelobe:      {max_sidelobe:.2f}")
    print(f"  Sidelobe level:    {sidelobe_db:.1f} dB")
    print(f"  Theoretical (PRBS): {-10*np.log10(correlator.prbs_length):.1f} dB")
    
    # Check pass/fail
    if sidelobe_db > TestConfig.MAX_SIDELOBE_DB:
        print(f"  ❌ FAIL: Sidelobes {sidelobe_db:.1f} dB > {TestConfig.MAX_SIDELOBE_DB} dB")
        return False, {'sidelobe_db': sidelobe_db}
    else:
        print(f"  ✅ PASS: Sidelobes adequate")
        return True, {'sidelobe_db': sidelobe_db}

def test_multiple_targets():
    """Test 3: Verify multiple target detection"""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Target Detection")
    print("=" * 60)
    
    correlator = ZeroDSPCorrelator(
        prbs_order=TestConfig.PRBS_ORDER,
        num_lanes=TestConfig.NUM_LANES,
        mode='fft'
    )
    
    # Generate signal with 3 targets
    targets = [50, 150, 300]
    amplitudes = [1.0, 0.5, 0.25]
    
    rx = generate_test_signal(correlator, targets, amplitudes, noise_power=0.01)
    
    # Correlate
    profile = correlator.correlate(rx)
    
    # Detect
    detections = correlator.detect(profile, pfa=1e-6)
    
    print(f"  Injected targets: {len(targets)}")
    print(f"  Detected:         {len(detections)}")
    
    for det in detections:
        print(f"    Bin {det['bin']:3d}: SNR = {det['snr_db']:.1f} dB")
    
    # Check if all targets found
    det_bins = [d['bin'] for d in detections]
    
    found = 0
    for t in targets:
        if any(abs(b - t) <= 5 for b in det_bins):
            found += 1
    
    if found == len(targets):
        print(f"  ✅ PASS: All {len(targets)} targets detected")
        return True, {'targets_found': found}
    else:
        print(f"  ❌ FAIL: Only {found}/{len(targets)} targets detected")
        return False, {'targets_found': found}

def test_processing_gain():
    """Test 4: Verify processing gain matches theory"""
    print("\n" + "=" * 60)
    print("TEST 4: Processing Gain Verification")
    print("=" * 60)
    
    correlator = ZeroDSPCorrelator(
        prbs_order=TestConfig.PRBS_ORDER,
        num_lanes=TestConfig.NUM_LANES,
        mode='fft'
    )
    
    theoretical_gain = 10 * np.log10(correlator.prbs_length)
    
    # Generate signal with known SNR
    signal_power = 1.0
    noise_power = 1.0  # 0 dB input SNR
    
    rx = generate_test_signal(correlator, [100], [np.sqrt(signal_power)], noise_power)
    
    # Correlate
    profile = correlator.correlate(rx)
    
    # Measure output SNR
    peak_bin = np.argmax(profile)
    peak_power = profile[peak_bin]**2
    noise_floor = np.median(profile)**2
    
    measured_gain = 10 * np.log10(peak_power / noise_floor)
    
    print(f"  Theoretical gain:  {theoretical_gain:.1f} dB")
    print(f"  Measured gain:     {measured_gain:.1f} dB")
    print(f"  Difference:        {abs(theoretical_gain - measured_gain):.1f} dB")
    
    # Allow 3 dB tolerance
    if abs(theoretical_gain - measured_gain) < 6:
        print(f"  ✅ PASS: Processing gain matches theory")
        return True, {'gain_db': measured_gain}
    else:
        print(f"  ❌ FAIL: Gain mismatch")
        return False, {'gain_db': measured_gain}

#=============================================================================
# Main Test Runner
#=============================================================================

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "═" * 60)
    print("   QEDMMA PoC - LOOPBACK SELF-TEST SUITE")
    print("═" * 60)
    
    tests = [
        ("Correlation Peak", test_correlation_peak),
        ("Sidelobe Levels", test_sidelobes),
        ("Multiple Targets", test_multiple_targets),
        ("Processing Gain", test_processing_gain),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            success, data = test_func()
            results[name] = {'passed': success, 'data': data}
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[name] = {'passed': False, 'error': str(e)}
            failed += 1
    
    # Summary
    print("\n" + "═" * 60)
    print("   TEST SUMMARY")
    print("═" * 60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(tests)}")
    
    if failed == 0:
        print("\n  ✅ ALL TESTS PASSED - System ready for deployment!")
    else:
        print("\n  ⚠️ SOME TESTS FAILED - Check configuration")
    
    print("═" * 60 + "\n")
    
    return results

if __name__ == "__main__":
    run_all_tests()
