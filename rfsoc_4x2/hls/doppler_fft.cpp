/*
 * TITAN Radar - Doppler FFT Processor
 * Radix-4 1024-point FFT for Range-Doppler Processing
 * 
 * Author: Dr. Mladen Mešter
 * Copyright (c) 2026 - All Rights Reserved
 * 
 * Platform: AMD RFSoC 4x2
 */

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "common/types.hpp"

using namespace titan;

//=============================================================================
// Constants
//=============================================================================

#define FFT_N 1024
#define FFT_LOG2N 10
#define FFT_LOG4N 5
#define NUM_RANGE_BINS 16384
#define PI 3.14159265358979323846f

//=============================================================================
// Twiddle Factor ROM
//=============================================================================

// Pre-computed twiddle factors for Radix-4 FFT
// W_N^k = exp(-j*2*pi*k/N) = cos(2*pi*k/N) - j*sin(2*pi*k/N)

void init_twiddle_rom(
    frac16_t twiddle_real[FFT_N],
    frac16_t twiddle_imag[FFT_N]
) {
    for (int k = 0; k < FFT_N; k++) {
        float angle = -2.0f * PI * k / FFT_N;
        twiddle_real[k] = frac16_t(cosf(angle));
        twiddle_imag[k] = frac16_t(sinf(angle));
    }
}

//=============================================================================
// Radix-4 Butterfly
//=============================================================================

void radix4_butterfly(
    complex_frac16_t &x0,
    complex_frac16_t &x1,
    complex_frac16_t &x2,
    complex_frac16_t &x3,
    const complex_frac16_t &w1,
    const complex_frac16_t &w2,
    const complex_frac16_t &w3
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    // Multiply by twiddle factors
    complex_frac16_t t1, t2, t3;
    
    // t1 = x1 * w1
    t1.real = x1.real * w1.real - x1.imag * w1.imag;
    t1.imag = x1.real * w1.imag + x1.imag * w1.real;
    
    // t2 = x2 * w2
    t2.real = x2.real * w2.real - x2.imag * w2.imag;
    t2.imag = x2.real * w2.imag + x2.imag * w2.real;
    
    // t3 = x3 * w3
    t3.real = x3.real * w3.real - x3.imag * w3.imag;
    t3.imag = x3.real * w3.imag + x3.imag * w3.real;
    
    // Radix-4 butterfly
    // X[0] = x0 + t1 + t2 + t3
    // X[1] = x0 - j*t1 - t2 + j*t3
    // X[2] = x0 - t1 + t2 - t3
    // X[3] = x0 + j*t1 - t2 - j*t3
    
    complex_frac16_t a = {x0.real + t2.real, x0.imag + t2.imag};
    complex_frac16_t b = {x0.real - t2.real, x0.imag - t2.imag};
    complex_frac16_t c = {t1.real + t3.real, t1.imag + t3.imag};
    complex_frac16_t d = {t1.imag - t3.imag, t3.real - t1.real};  // j*(t1-t3)
    
    x0.real = a.real + c.real;
    x0.imag = a.imag + c.imag;
    
    x1.real = b.real + d.real;
    x1.imag = b.imag + d.imag;
    
    x2.real = a.real - c.real;
    x2.imag = a.imag - c.imag;
    
    x3.real = b.real - d.real;
    x3.imag = b.imag - d.imag;
}

//=============================================================================
// Bit Reversal for Radix-4
//=============================================================================

ap_uint<FFT_LOG2N> bit_reverse(ap_uint<FFT_LOG2N> x) {
    #pragma HLS INLINE
    ap_uint<FFT_LOG2N> result = 0;
    for (int i = 0; i < FFT_LOG2N; i++) {
        #pragma HLS UNROLL
        result[FFT_LOG2N - 1 - i] = x[i];
    }
    return result;
}

//=============================================================================
// Window Function (Hanning)
//=============================================================================

void apply_window(
    complex_frac16_t data[FFT_N],
    const frac16_t window[FFT_N]
) {
    #pragma HLS PIPELINE II=1
    for (int i = 0; i < FFT_N; i++) {
        #pragma HLS UNROLL factor=8
        data[i].real = data[i].real * window[i];
        data[i].imag = data[i].imag * window[i];
    }
}

//=============================================================================
// 1024-point Radix-4 FFT
//=============================================================================

void fft_1024(
    complex_frac16_t x_in[FFT_N],
    complex_frac16_t X_out[FFT_N],
    const frac16_t twiddle_real[FFT_N],
    const frac16_t twiddle_imag[FFT_N]
) {
    #pragma HLS INLINE off
    
    // Working buffer
    complex_frac16_t buffer[FFT_N];
    #pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=4
    
    // Bit-reverse input
    bit_reverse_loop: for (int i = 0; i < FFT_N; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<FFT_LOG2N> rev_idx = bit_reverse(i);
        buffer[rev_idx] = x_in[i];
    }
    
    // FFT stages (5 stages for 1024-point radix-4)
    int stride = 1;
    
    fft_stages: for (int stage = 0; stage < FFT_LOG4N; stage++) {
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        
        int group_size = stride * 4;
        int num_groups = FFT_N / group_size;
        
        groups: for (int g = 0; g < num_groups; g++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=256
            
            butterflies: for (int k = 0; k < stride; k++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=256
                
                int base = g * group_size + k;
                
                // Get butterfly inputs
                complex_frac16_t x0 = buffer[base];
                complex_frac16_t x1 = buffer[base + stride];
                complex_frac16_t x2 = buffer[base + 2*stride];
                complex_frac16_t x3 = buffer[base + 3*stride];
                
                // Twiddle factors
                int tw_idx1 = (k * num_groups) % FFT_N;
                int tw_idx2 = (2 * k * num_groups) % FFT_N;
                int tw_idx3 = (3 * k * num_groups) % FFT_N;
                
                complex_frac16_t w1 = {twiddle_real[tw_idx1], twiddle_imag[tw_idx1]};
                complex_frac16_t w2 = {twiddle_real[tw_idx2], twiddle_imag[tw_idx2]};
                complex_frac16_t w3 = {twiddle_real[tw_idx3], twiddle_imag[tw_idx3]};
                
                // Perform butterfly
                radix4_butterfly(x0, x1, x2, x3, w1, w2, w3);
                
                // Store results
                buffer[base] = x0;
                buffer[base + stride] = x1;
                buffer[base + 2*stride] = x2;
                buffer[base + 3*stride] = x3;
            }
        }
        
        stride *= 4;
    }
    
    // FFT shift (move DC to center)
    output_loop: for (int i = 0; i < FFT_N; i++) {
        #pragma HLS PIPELINE II=1
        int shifted_idx = (i + FFT_N/2) % FFT_N;
        X_out[shifted_idx] = buffer[i];
    }
}

//=============================================================================
// Range-Doppler Processor
//=============================================================================

void doppler_processor(
    // Input: Range profiles for all pulses in CPI
    hls::stream<range_word_t> &s_axis_range,
    
    // Output: Range-Doppler map
    hls::stream<axis_word_t> &m_axis_rdmap,
    
    // Configuration
    ap_uint<16> num_range_bins,
    ap_uint<16> num_pulses,
    ap_uint<1> enable,
    
    // Status
    ap_uint<1> *busy,
    ap_uint<32> *processed_bins
) {
    #pragma HLS INTERFACE axis port=s_axis_range
    #pragma HLS INTERFACE axis port=m_axis_rdmap
    #pragma HLS INTERFACE s_axilite port=num_range_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=num_pulses bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=processed_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    if (!enable) {
        *busy = 0;
        return;
    }
    
    *busy = 1;
    
    // Twiddle factors (could be ROM in real implementation)
    static frac16_t twiddle_real[FFT_N];
    static frac16_t twiddle_imag[FFT_N];
    static frac16_t window[FFT_N];
    #pragma HLS BIND_STORAGE variable=twiddle_real type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=twiddle_imag type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=window type=ROM_1P impl=BRAM
    
    static bool initialized = false;
    if (!initialized) {
        // Initialize twiddle factors
        for (int k = 0; k < FFT_N; k++) {
            float angle = -2.0f * PI * k / FFT_N;
            twiddle_real[k] = frac16_t(hls::cosf(angle));
            twiddle_imag[k] = frac16_t(hls::sinf(angle));
            // Hanning window
            window[k] = frac16_t(0.5f * (1.0f - hls::cosf(2.0f * PI * k / (FFT_N - 1))));
        }
        initialized = true;
    }
    
    // Range profile storage (one column of range-Doppler matrix)
    static complex_frac16_t pulse_data[FFT_N];
    #pragma HLS BIND_STORAGE variable=pulse_data type=RAM_2P impl=BRAM
    
    static complex_frac16_t fft_out[FFT_N];
    #pragma HLS BIND_STORAGE variable=fft_out type=RAM_2P impl=BRAM
    
    ap_uint<32> bin_count = 0;
    
    // Process each range bin
    range_bins: for (int rb = 0; rb < num_range_bins; rb++) {
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=16384
        
        // Read pulse data for this range bin
        read_pulses: for (int p = 0; p < num_pulses && p < FFT_N; p++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=64 max=1024
            
            if (!s_axis_range.empty()) {
                range_word_t word = s_axis_range.read();
                // Extract magnitude and convert to complex (real only)
                pulse_data[p].real = frac16_t(word.mag[rb % 4] >> 8);
                pulse_data[p].imag = frac16_t(0);
            }
        }
        
        // Zero-pad if needed
        for (int p = num_pulses; p < FFT_N; p++) {
            #pragma HLS UNROLL factor=8
            pulse_data[p].real = 0;
            pulse_data[p].imag = 0;
        }
        
        // Apply window
        apply_window(pulse_data, window);
        
        // Perform FFT
        fft_1024(pulse_data, fft_out, twiddle_real, twiddle_imag);
        
        // Output FFT magnitude
        output_doppler: for (int d = 0; d < FFT_N; d += 4) {
            #pragma HLS PIPELINE II=1
            
            axis_word_t out_word;
            for (int i = 0; i < 4; i++) {
                #pragma HLS UNROLL
                auto mag = approx_magnitude(fft_out[d+i].real, fft_out[d+i].imag);
                out_word.data.range(32*(i+1)-1, 32*i) = ap_uint<32>(mag);
            }
            out_word.keep = 0xFFFF;
            out_word.last = (d + 4 >= FFT_N && rb + 1 >= num_range_bins) ? 1 : 0;
            
            m_axis_rdmap.write(out_word);
        }
        
        bin_count++;
    }
    
    *processed_bins = bin_count;
    *busy = 0;
}

//=============================================================================
// Top-Level Wrapper
//=============================================================================

void titan_doppler_fft(
    hls::stream<range_word_t> &s_axis_range,
    hls::stream<axis_word_t> &m_axis_rdmap,
    ap_uint<16> num_range_bins,
    ap_uint<16> num_pulses,
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<32> *processed_bins
) {
    #pragma HLS INTERFACE axis port=s_axis_range
    #pragma HLS INTERFACE axis port=m_axis_rdmap
    #pragma HLS INTERFACE s_axilite port=num_range_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=num_pulses bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=processed_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    doppler_processor(
        s_axis_range,
        m_axis_rdmap,
        num_range_bins,
        num_pulses,
        enable,
        busy,
        processed_bins
    );
}
