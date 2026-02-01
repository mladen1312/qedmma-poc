/*
 * TITAN Radar - Common HLS Type Definitions
 * 
 * Author: Dr. Mladen Mešter
 * Copyright (c) 2026 - All Rights Reserved
 * 
 * Platform: AMD RFSoC 4x2 (Zynq UltraScale+ ZU48DR)
 */

#ifndef TITAN_TYPES_HPP
#define TITAN_TYPES_HPP

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <complex>

namespace titan {

//=============================================================================
// System Constants
//=============================================================================

constexpr int MAX_RANGE_BINS = 16384;
constexpr int MAX_DOPPLER_BINS = 1024;
constexpr int MAX_TRACKS = 256;
constexpr int NUM_CHANNELS = 4;
constexpr int PRBS_15_LENGTH = 32767;
constexpr int PRBS_20_LENGTH = 1048575;

// FFT Configuration
constexpr int FFT_SIZE = 1024;
constexpr int FFT_RADIX = 4;
constexpr int FFT_STAGES = 5;  // log4(1024) = 5

// CFAR Configuration
constexpr int CFAR_MAX_GUARD = 16;
constexpr int CFAR_MAX_TRAIN = 64;

//=============================================================================
// Basic Data Types
//=============================================================================

// ADC sample (14-bit from RFSoC)
typedef ap_int<16> sample_t;

// DAC sample (14-bit to RFSoC)
typedef ap_int<16> dac_sample_t;

// Complex sample (16-bit I/Q)
typedef ap_int<16> iq_sample_t;

// Fixed-point types for signal processing
typedef ap_fixed<16, 1, AP_RND, AP_SAT> frac16_t;      // Q1.15
typedef ap_fixed<32, 16, AP_RND, AP_SAT> fixed32_t;    // Q16.16
typedef ap_fixed<24, 8, AP_RND, AP_SAT> fixed24_t;     // Q8.16

// Accumulator types
typedef ap_int<32> accum32_t;
typedef ap_int<48> accum48_t;
typedef ap_int<64> accum64_t;

// Magnitude/power types
typedef ap_uint<24> magnitude_t;
typedef ap_uint<32> power_t;

// Index types
typedef ap_uint<16> range_idx_t;
typedef ap_uint<12> doppler_idx_t;
typedef ap_uint<8> track_idx_t;
typedef ap_uint<3> channel_idx_t;

// Angle types (degrees in fixed point)
typedef ap_fixed<16, 9, AP_RND, AP_SAT> angle_t;  // ±180° with 0.01° resolution

// Velocity type (m/s in fixed point)
typedef ap_fixed<24, 12, AP_RND, AP_SAT> velocity_t;  // ±2048 m/s

// Range type (meters in fixed point)
typedef ap_fixed<32, 24, AP_RND, AP_SAT> range_t;  // Up to 16M meters

//=============================================================================
// Complex Types
//=============================================================================

template<typename T>
struct Complex {
    T real;
    T imag;
    
    Complex() : real(0), imag(0) {}
    Complex(T r, T i) : real(r), imag(i) {}
    
    // Conjugate
    Complex conj() const {
        return Complex(real, -imag);
    }
    
    // Magnitude squared (no sqrt)
    auto mag_sq() const {
        return real * real + imag * imag;
    }
};

typedef Complex<sample_t> complex_sample_t;
typedef Complex<frac16_t> complex_frac16_t;
typedef Complex<fixed32_t> complex_fixed32_t;

//=============================================================================
// AXI Stream Structures
//=============================================================================

// Generic AXI-Stream word (128-bit)
struct axis_word_t {
    ap_uint<128> data;
    ap_uint<16> keep;
    ap_uint<1> last;
};

// ADC data word (8 samples × 16 bits = 128 bits)
struct adc_word_t {
    sample_t samples[8];
    ap_uint<1> last;
    
    sample_t& operator[](int i) { return samples[i]; }
    const sample_t& operator[](int i) const { return samples[i]; }
};

// DAC data word
struct dac_word_t {
    dac_sample_t samples[8];
    ap_uint<1> last;
};

// Range profile word (4 magnitudes × 32 bits = 128 bits)
struct range_word_t {
    magnitude_t mag[4];
    range_idx_t start_bin;
    ap_uint<1> last;
};

// Detection structure
struct detection_t {
    range_idx_t range_bin;
    doppler_idx_t doppler_bin;
    magnitude_t amplitude;
    ap_uint<8> snr_db;        // SNR in 0.5 dB steps
    channel_idx_t channel;
    ap_uint<1> valid;
};

// Track structure
struct track_t {
    track_idx_t id;
    range_t range;
    velocity_t velocity;
    velocity_t acceleration;
    angle_t azimuth;
    angle_t azimuth_rate;
    magnitude_t amplitude;
    ap_uint<8> quality;       // Track quality (0-255)
    ap_uint<8> hits;          // Consecutive hits
    ap_uint<8> misses;        // Consecutive misses
    ap_uint<16> age;          // Track age in frames
    ap_uint<2> state;         // 0=free, 1=tentative, 2=confirmed, 3=coasting
    ap_uint<1> valid;
};

//=============================================================================
// Configuration Structures
//=============================================================================

// Waveform generator configuration
struct waveform_config_t {
    ap_uint<3> waveform_type;   // 0=PRBS15, 1=PRBS20, 2=LFM_UP, 3=LFM_DN
    ap_uint<8> prbs_order;      // PRBS polynomial order
    ap_uint<32> chip_rate;      // Chips per second
    ap_uint<32> pulse_length;   // Samples per pulse
    ap_uint<32> pri_samples;    // Samples per PRI
    ap_uint<16> num_pulses;     // Pulses per CPI
    fixed32_t nco_freq;         // NCO frequency (normalized)
};

// Correlator configuration
struct correlator_config_t {
    ap_uint<16> num_range_bins;
    ap_uint<32> ref_length;
    ap_uint<32> threshold;
    ap_uint<1> enable;
};

// CFAR configuration
struct cfar_config_t {
    ap_uint<2> mode;            // 0=CA, 1=GO, 2=SO, 3=OS
    ap_uint<8> guard_cells;
    ap_uint<8> training_cells;
    ap_uint<16> threshold_factor;  // Q8.8 format
    ap_uint<8> os_rank;         // For OS-CFAR
    ap_uint<1> enable;
};

// Beamformer configuration
struct beamformer_config_t {
    angle_t steering_angle;
    angle_t null_angles[3];
    ap_uint<3> num_nulls;
    ap_uint<2> mode;            // 0=conventional, 1=MVDR, 2=nulling
    ap_uint<1> enable;
};

// Tracker configuration
struct tracker_config_t {
    ap_uint<8> max_tracks;
    range_t gate_range;
    velocity_t gate_velocity;
    ap_uint<4> confirm_hits;
    ap_uint<4> delete_misses;
    ap_uint<1> enable;
};

//=============================================================================
// HLS Stream Types
//=============================================================================

typedef hls::stream<adc_word_t> adc_stream_t;
typedef hls::stream<dac_word_t> dac_stream_t;
typedef hls::stream<range_word_t> range_stream_t;
typedef hls::stream<detection_t> detection_stream_t;
typedef hls::stream<track_t> track_stream_t;
typedef hls::stream<axis_word_t> axis_stream_t;

//=============================================================================
// Utility Functions
//=============================================================================

// Fast approximate magnitude: max(|I|,|Q|) + 0.4*min(|I|,|Q|)
template<typename T>
inline auto approx_magnitude(T real, T imag) {
    #pragma HLS INLINE
    auto abs_real = (real >= 0) ? real : (T)(-real);
    auto abs_imag = (imag >= 0) ? imag : (T)(-imag);
    auto max_val = (abs_real > abs_imag) ? abs_real : abs_imag;
    auto min_val = (abs_real <= abs_imag) ? abs_real : abs_imag;
    return max_val + ((min_val * 3) >> 3);  // 3/8 ≈ 0.375
}

// Saturating add
template<typename T>
inline T sat_add(T a, T b) {
    #pragma HLS INLINE
    auto sum = a + b;
    // Overflow check for signed
    if ((a > 0) && (b > 0) && (sum < 0)) {
        return (T)(((1ULL << (T::width - 1)) - 1));  // Max positive
    }
    if ((a < 0) && (b < 0) && (sum > 0)) {
        return (T)(1ULL << (T::width - 1));  // Max negative
    }
    return sum;
}

} // namespace titan

#endif // TITAN_TYPES_HPP
