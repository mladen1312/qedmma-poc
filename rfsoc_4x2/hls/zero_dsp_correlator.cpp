/*
 * TITAN Radar - Zero-DSP Correlator
 * High-Level Synthesis Implementation for RFSoC 4x2
 * 
 * Author: Dr. Mladen Mešter
 * Copyright (c) 2026 - All Rights Reserved
 * 
 * This correlator uses XOR + popcount for binary correlation,
 * achieving massive parallelism without using DSP slices.
 * Perfect for RFSoC 4x2 with 930K logic cells.
 */

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

// ============================================================================
// Type Definitions
// ============================================================================

// Input sample type (14-bit ADC)
typedef ap_int<16> sample_t;

// Binary representation for XOR correlation
typedef ap_uint<1> bit_t;

// Accumulator type
typedef ap_int<32> accum_t;

// Output magnitude type
typedef ap_uint<24> mag_t;

// Fixed point for thresholds
typedef ap_fixed<16, 8> threshold_t;

// Configuration
#define MAX_RANGE_BINS 16384
#define PRBS_LENGTH 32767
#define PARALLEL_CORRELATORS 64
#define SAMPLES_PER_WORD 8

// ============================================================================
// AXI Stream Interfaces
// ============================================================================

struct axis_sample_t {
    ap_int<128> data;   // 8 x 16-bit samples
    ap_uint<1> last;
};

struct axis_range_t {
    ap_uint<128> data;  // Range profile samples
    ap_uint<1> last;
};

// ============================================================================
// Register Interface
// ============================================================================

struct correlator_config_t {
    ap_uint<16> num_range_bins;
    ap_uint<8> prbs_order;
    ap_uint<32> threshold;
    ap_uint<1> enable;
    ap_uint<1> reset;
};

// ============================================================================
// PRBS Generator
// ============================================================================

class PRBSGenerator {
public:
    ap_uint<32> state;
    ap_uint<8> order;
    
    PRBSGenerator() : state(1), order(15) {}
    
    void set_order(ap_uint<8> new_order) {
        order = new_order;
        state = 1;
    }
    
    void reset() {
        state = 1;
    }
    
    bit_t next() {
        #pragma HLS INLINE
        bit_t output = state & 1;
        
        // Feedback polynomial depends on order
        ap_uint<1> feedback;
        switch(order) {
            case 7:  feedback = (state >> 6) ^ (state >> 5); break;
            case 9:  feedback = (state >> 8) ^ (state >> 4); break;
            case 15: feedback = (state >> 14) ^ (state >> 13); break;
            case 20: feedback = (state >> 19) ^ (state >> 16); break;
            default: feedback = (state >> 14) ^ (state >> 13); break;
        }
        
        state = ((state << 1) | (feedback & 1)) & ((1 << order) - 1);
        return output;
    }
    
    // Generate block of PRBS bits
    void generate_block(bit_t prbs_block[PARALLEL_CORRELATORS]) {
        #pragma HLS INLINE
        for (int i = 0; i < PARALLEL_CORRELATORS; i++) {
            #pragma HLS UNROLL
            prbs_block[i] = next();
        }
    }
};

// ============================================================================
// Zero-DSP Correlator Core
// ============================================================================

void zero_dsp_correlate_block(
    bit_t rx_bits[PARALLEL_CORRELATORS],
    bit_t ref_bits[PARALLEL_CORRELATORS],
    accum_t &correlation
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    // XOR correlation: count matching bits
    ap_uint<PARALLEL_CORRELATORS> xor_result;
    
    for (int i = 0; i < PARALLEL_CORRELATORS; i++) {
        #pragma HLS UNROLL
        // XOR gives 0 for match, 1 for mismatch
        // We want to count matches, so invert
        xor_result[i] = !(rx_bits[i] ^ ref_bits[i]);
    }
    
    // Popcount - count number of 1s (matches)
    // This synthesizes to LUT-based adder tree
    ap_uint<8> match_count = 0;
    for (int i = 0; i < PARALLEL_CORRELATORS; i++) {
        #pragma HLS UNROLL
        match_count += xor_result[i];
    }
    
    // Convert to correlation value
    // matches - mismatches = 2*matches - total
    correlation = 2 * match_count - PARALLEL_CORRELATORS;
}

// ============================================================================
// Sample to Bit Converter
// ============================================================================

void samples_to_bits(
    sample_t samples[SAMPLES_PER_WORD],
    bit_t bits[SAMPLES_PER_WORD]
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    for (int i = 0; i < SAMPLES_PER_WORD; i++) {
        #pragma HLS UNROLL
        // Simple sign-based quantization
        bits[i] = (samples[i] >= 0) ? 1 : 0;
    }
}

// ============================================================================
// Magnitude Calculator (for complex correlation output)
// ============================================================================

mag_t calculate_magnitude(accum_t real, accum_t imag) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    // Approximate magnitude: |z| ≈ max(|Re|,|Im|) + 0.4*min(|Re|,|Im|)
    accum_t abs_real = (real >= 0) ? real : -real;
    accum_t abs_imag = (imag >= 0) ? imag : -imag;
    
    accum_t max_val = (abs_real > abs_imag) ? abs_real : abs_imag;
    accum_t min_val = (abs_real <= abs_imag) ? abs_real : abs_imag;
    
    // 0.4 ≈ 3/8 in fixed point
    accum_t mag = max_val + ((min_val * 3) >> 3);
    
    return (mag_t)(mag & 0xFFFFFF);
}

// ============================================================================
// Main Correlator Function
// ============================================================================

void zero_dsp_correlator(
    // AXI Stream interfaces
    hls::stream<axis_sample_t> &s_axis_rx,
    hls::stream<axis_range_t> &m_axis_range,
    
    // Control registers
    ap_uint<16> num_range_bins,
    ap_uint<8> prbs_order,
    ap_uint<32> threshold,
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<32> *detected_count
) {
    #pragma HLS INTERFACE axis port=s_axis_rx
    #pragma HLS INTERFACE axis port=m_axis_range
    #pragma HLS INTERFACE s_axilite port=num_range_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=prbs_order bundle=control
    #pragma HLS INTERFACE s_axilite port=threshold bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=detected_count bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Local storage
    static PRBSGenerator prbs;
    static accum_t correlation_accum[MAX_RANGE_BINS];
    #pragma HLS BIND_STORAGE variable=correlation_accum type=RAM_2P impl=URAM
    
    static bit_t prbs_ref[PRBS_LENGTH];
    #pragma HLS BIND_STORAGE variable=prbs_ref type=RAM_1P impl=BRAM
    
    if (!enable) {
        *busy = 0;
        return;
    }
    
    *busy = 1;
    *detected_count = 0;
    
    // Initialize PRBS reference
    prbs.set_order(prbs_order);
    init_prbs: for (int i = 0; i < PRBS_LENGTH; i++) {
        #pragma HLS PIPELINE II=1
        prbs_ref[i] = prbs.next();
    }
    
    // Clear accumulators
    clear_accum: for (int i = 0; i < num_range_bins; i++) {
        #pragma HLS PIPELINE II=1
        correlation_accum[i] = 0;
    }
    
    // Process incoming samples
    int sample_count = 0;
    
    process_samples: while (!s_axis_rx.empty()) {
        #pragma HLS PIPELINE II=1
        
        axis_sample_t in_word = s_axis_rx.read();
        
        // Extract samples from AXI word
        sample_t samples[SAMPLES_PER_WORD];
        #pragma HLS ARRAY_PARTITION variable=samples complete
        
        for (int i = 0; i < SAMPLES_PER_WORD; i++) {
            #pragma HLS UNROLL
            samples[i] = in_word.data.range(16*i+15, 16*i);
        }
        
        // Convert to bits
        bit_t rx_bits[SAMPLES_PER_WORD];
        #pragma HLS ARRAY_PARTITION variable=rx_bits complete
        samples_to_bits(samples, rx_bits);
        
        // Correlate against all range bins in parallel
        // Using sliding window correlation
        correlate_bins: for (int bin = 0; bin < num_range_bins; bin += PARALLEL_CORRELATORS) {
            #pragma HLS PIPELINE II=1
            
            // Load reference bits for this range offset
            bit_t ref_block[PARALLEL_CORRELATORS];
            #pragma HLS ARRAY_PARTITION variable=ref_block complete
            
            for (int i = 0; i < PARALLEL_CORRELATORS; i++) {
                #pragma HLS UNROLL
                int ref_idx = (sample_count + bin + i) % PRBS_LENGTH;
                ref_block[i] = prbs_ref[ref_idx];
            }
            
            // Perform block correlation
            accum_t block_corr;
            // Note: This is simplified - real implementation would handle
            // sample-to-reference alignment properly
            for (int i = 0; i < PARALLEL_CORRELATORS && (bin + i) < num_range_bins; i++) {
                #pragma HLS UNROLL
                ap_uint<1> match = !(rx_bits[i % SAMPLES_PER_WORD] ^ ref_block[i]);
                correlation_accum[bin + i] += (match ? 1 : -1);
            }
        }
        
        sample_count += SAMPLES_PER_WORD;
        
        if (in_word.last) break;
    }
    
    // Output range profile
    ap_uint<32> det_count = 0;
    
    output_range: for (int i = 0; i < num_range_bins; i += 4) {
        #pragma HLS PIPELINE II=1
        
        axis_range_t out_word;
        
        for (int j = 0; j < 4; j++) {
            #pragma HLS UNROLL
            
            if (i + j < num_range_bins) {
                // Calculate magnitude
                mag_t mag = calculate_magnitude(correlation_accum[i+j], 0);
                out_word.data.range(32*j+23, 32*j) = mag;
                out_word.data.range(32*j+31, 32*j+24) = 0;
                
                // Count threshold crossings
                if (mag > threshold) {
                    det_count++;
                }
            }
        }
        
        out_word.last = (i + 4 >= num_range_bins) ? 1 : 0;
        m_axis_range.write(out_word);
    }
    
    *detected_count = det_count;
    *busy = 0;
}

// ============================================================================
// CFAR Detector Core
// ============================================================================

void cfar_detector(
    hls::stream<axis_range_t> &s_axis_range,
    hls::stream<axis_range_t> &m_axis_detections,
    
    ap_uint<16> num_bins,
    ap_uint<8> guard_cells,
    ap_uint<8> training_cells,
    ap_uint<16> threshold_factor,  // Q8.8 fixed point
    ap_uint<8> cfar_mode,          // 0=CA, 1=GO, 2=SO, 3=OS
    ap_uint<1> enable,
    ap_uint<32> *detection_count
) {
    #pragma HLS INTERFACE axis port=s_axis_range
    #pragma HLS INTERFACE axis port=m_axis_detections
    #pragma HLS INTERFACE s_axilite port=num_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=guard_cells bundle=control
    #pragma HLS INTERFACE s_axilite port=training_cells bundle=control
    #pragma HLS INTERFACE s_axilite port=threshold_factor bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_mode bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=detection_count bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Store range profile locally
    static mag_t range_profile[MAX_RANGE_BINS];
    #pragma HLS BIND_STORAGE variable=range_profile type=RAM_2P impl=BRAM
    
    if (!enable) {
        return;
    }
    
    // Read range profile
    int bin_idx = 0;
    read_profile: while (!s_axis_range.empty() && bin_idx < num_bins) {
        #pragma HLS PIPELINE II=1
        
        axis_range_t in_word = s_axis_range.read();
        
        for (int i = 0; i < 4 && bin_idx < num_bins; i++) {
            #pragma HLS UNROLL
            range_profile[bin_idx++] = in_word.data.range(32*i+23, 32*i);
        }
        
        if (in_word.last) break;
    }
    
    // Perform CFAR detection
    ap_uint<32> det_cnt = 0;
    int window = guard_cells + training_cells;
    
    cfar_loop: for (int i = window; i < num_bins - window; i++) {
        #pragma HLS PIPELINE II=1
        
        // Calculate noise estimate from training cells
        ap_uint<32> left_sum = 0;
        ap_uint<32> right_sum = 0;
        
        // Left training cells
        for (int j = 0; j < training_cells; j++) {
            #pragma HLS UNROLL factor=8
            left_sum += range_profile[i - guard_cells - training_cells + j];
        }
        
        // Right training cells
        for (int j = 0; j < training_cells; j++) {
            #pragma HLS UNROLL factor=8
            right_sum += range_profile[i + guard_cells + 1 + j];
        }
        
        // Calculate threshold based on CFAR mode
        ap_uint<32> noise_est;
        switch(cfar_mode) {
            case 0: // CA-CFAR: average
                noise_est = (left_sum + right_sum) / (2 * training_cells);
                break;
            case 1: // GO-CFAR: greatest of
                noise_est = (left_sum > right_sum) ? 
                           left_sum / training_cells : right_sum / training_cells;
                break;
            case 2: // SO-CFAR: smallest of
                noise_est = (left_sum < right_sum) ? 
                           left_sum / training_cells : right_sum / training_cells;
                break;
            default:
                noise_est = (left_sum + right_sum) / (2 * training_cells);
        }
        
        // Calculate adaptive threshold (Q8.8 multiplication)
        ap_uint<32> threshold = (noise_est * threshold_factor) >> 8;
        
        // Detection decision
        if (range_profile[i] > threshold) {
            // Output detection
            axis_range_t det;
            det.data.range(15, 0) = i;  // Range bin
            det.data.range(47, 16) = range_profile[i];  // Amplitude
            det.data.range(79, 48) = threshold;  // Threshold (for debug)
            det.data.range(127, 80) = 0;
            det.last = 0;
            
            m_axis_detections.write(det);
            det_cnt++;
        }
    }
    
    // Send end marker
    axis_range_t end_marker;
    end_marker.data = 0xFFFFFFFF;
    end_marker.last = 1;
    m_axis_detections.write(end_marker);
    
    *detection_count = det_cnt;
}

// ============================================================================
// Top-Level Wrapper for Complete Correlator + CFAR
// ============================================================================

void titan_correlator_top(
    // AXI Stream
    hls::stream<axis_sample_t> &s_axis_rx,
    hls::stream<axis_range_t> &m_axis_range,
    hls::stream<axis_range_t> &m_axis_detections,
    
    // Configuration
    ap_uint<16> num_range_bins,
    ap_uint<8> prbs_order,
    ap_uint<32> corr_threshold,
    ap_uint<8> cfar_guard,
    ap_uint<8> cfar_train,
    ap_uint<16> cfar_factor,
    ap_uint<8> cfar_mode,
    
    // Control/Status
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<32> *raw_detections,
    ap_uint<32> *cfar_detections
) {
    #pragma HLS INTERFACE axis port=s_axis_rx
    #pragma HLS INTERFACE axis port=m_axis_range
    #pragma HLS INTERFACE axis port=m_axis_detections
    #pragma HLS INTERFACE s_axilite port=num_range_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=prbs_order bundle=control
    #pragma HLS INTERFACE s_axilite port=corr_threshold bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_guard bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_train bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_factor bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_mode bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=raw_detections bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_detections bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    #pragma HLS DATAFLOW
    
    // Internal stream between correlator and CFAR
    hls::stream<axis_range_t> internal_range;
    #pragma HLS STREAM variable=internal_range depth=1024
    
    // Run correlator
    zero_dsp_correlator(
        s_axis_rx,
        internal_range,
        num_range_bins,
        prbs_order,
        corr_threshold,
        enable,
        busy,
        raw_detections
    );
    
    // Copy to output and run CFAR
    cfar_detector(
        internal_range,
        m_axis_detections,
        num_range_bins,
        cfar_guard,
        cfar_train,
        cfar_factor,
        cfar_mode,
        enable,
        cfar_detections
    );
}
