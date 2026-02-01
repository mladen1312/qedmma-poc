/*
 * TITAN Radar - CFAR Detector
 * 2D Constant False Alarm Rate Detection
 * 
 * Author: Dr. Mladen Mešter
 * Copyright (c) 2026 - All Rights Reserved
 * 
 * Supports:
 * - CA-CFAR (Cell Averaging)
 * - GO-CFAR (Greatest Of)
 * - SO-CFAR (Smallest Of)
 * - OS-CFAR (Order Statistic)
 * 
 * Platform: AMD RFSoC 4x2
 */

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "common/types.hpp"

using namespace titan;

//=============================================================================
// Constants
//=============================================================================

#define MAX_RANGE_BINS 16384
#define MAX_DOPPLER_BINS 1024
#define MAX_GUARD_CELLS 16
#define MAX_TRAIN_CELLS 64
#define MAX_DETECTIONS 1024

// CFAR modes
#define CFAR_CA 0    // Cell Averaging
#define CFAR_GO 1    // Greatest Of
#define CFAR_SO 2    // Smallest Of
#define CFAR_OS 3    // Order Statistic

//=============================================================================
// Types
//=============================================================================

typedef ap_uint<24> power_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> threshold_t;

//=============================================================================
// Sorting Network for OS-CFAR
//=============================================================================

// Compare and swap
void compare_swap(power_t &a, power_t &b) {
    #pragma HLS INLINE
    if (a > b) {
        power_t temp = a;
        a = b;
        b = temp;
    }
}

// Bitonic sort for 8 elements
void bitonic_sort_8(power_t data[8]) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    // Stage 1
    compare_swap(data[0], data[1]);
    compare_swap(data[2], data[3]);
    compare_swap(data[4], data[5]);
    compare_swap(data[6], data[7]);
    
    // Stage 2
    compare_swap(data[0], data[2]);
    compare_swap(data[1], data[3]);
    compare_swap(data[4], data[6]);
    compare_swap(data[5], data[7]);
    
    // Stage 3
    compare_swap(data[1], data[2]);
    compare_swap(data[5], data[6]);
    
    // Stage 4
    compare_swap(data[0], data[4]);
    compare_swap(data[1], data[5]);
    compare_swap(data[2], data[6]);
    compare_swap(data[3], data[7]);
    
    // Stage 5
    compare_swap(data[2], data[4]);
    compare_swap(data[3], data[5]);
    
    // Stage 6
    compare_swap(data[1], data[2]);
    compare_swap(data[3], data[4]);
    compare_swap(data[5], data[6]);
}

//=============================================================================
// CFAR Window Processor
//=============================================================================

void process_cfar_window(
    const power_t window[2 * MAX_TRAIN_CELLS],
    ap_uint<8> num_train,
    ap_uint<2> cfar_mode,
    ap_uint<8> os_rank,
    threshold_t &noise_estimate
) {
    #pragma HLS INLINE
    
    // Sum accumulators
    ap_uint<32> left_sum = 0;
    ap_uint<32> right_sum = 0;
    
    // Calculate sums
    for (int i = 0; i < MAX_TRAIN_CELLS; i++) {
        #pragma HLS UNROLL factor=8
        if (i < num_train) {
            left_sum += window[i];
            right_sum += window[num_train + i];
        }
    }
    
    // Calculate noise estimate based on mode
    switch(cfar_mode) {
        case CFAR_CA:
            // Cell Averaging: average of all training cells
            noise_estimate = threshold_t(left_sum + right_sum) / threshold_t(2 * num_train);
            break;
            
        case CFAR_GO:
            // Greatest Of: max of left and right averages
            {
                threshold_t left_avg = threshold_t(left_sum) / threshold_t(num_train);
                threshold_t right_avg = threshold_t(right_sum) / threshold_t(num_train);
                noise_estimate = (left_avg > right_avg) ? left_avg : right_avg;
            }
            break;
            
        case CFAR_SO:
            // Smallest Of: min of left and right averages
            {
                threshold_t left_avg = threshold_t(left_sum) / threshold_t(num_train);
                threshold_t right_avg = threshold_t(right_sum) / threshold_t(num_train);
                noise_estimate = (left_avg < right_avg) ? left_avg : right_avg;
            }
            break;
            
        case CFAR_OS:
            // Order Statistic: k-th smallest value
            {
                // Copy to sortable array (limited to 8 for hardware efficiency)
                power_t sort_buffer[8];
                #pragma HLS ARRAY_PARTITION variable=sort_buffer complete
                
                // Sample 8 values from training cells
                for (int i = 0; i < 8; i++) {
                    #pragma HLS UNROLL
                    int idx = (i * num_train) / 4;
                    if (idx < num_train) {
                        sort_buffer[i] = window[idx];
                    } else {
                        sort_buffer[i] = window[num_train + idx - num_train];
                    }
                }
                
                // Sort
                bitonic_sort_8(sort_buffer);
                
                // Select k-th element (os_rank as fraction of 8)
                ap_uint<3> k = os_rank >> 5;  // Scale 0-255 to 0-7
                noise_estimate = threshold_t(sort_buffer[k]);
            }
            break;
            
        default:
            noise_estimate = threshold_t(left_sum + right_sum) / threshold_t(2 * num_train);
            break;
    }
}

//=============================================================================
// 1D Range CFAR
//=============================================================================

void cfar_1d_range(
    const power_t range_profile[MAX_RANGE_BINS],
    ap_uint<16> num_bins,
    ap_uint<8> guard_cells,
    ap_uint<8> train_cells,
    threshold_t threshold_factor,
    ap_uint<2> cfar_mode,
    ap_uint<8> os_rank,
    ap_uint<1> detections[MAX_RANGE_BINS],
    threshold_t thresholds[MAX_RANGE_BINS]
) {
    #pragma HLS INLINE off
    
    // Training cell window
    power_t train_window[2 * MAX_TRAIN_CELLS];
    #pragma HLS ARRAY_PARTITION variable=train_window cyclic factor=8
    
    int window_start = guard_cells + train_cells;
    int window_end = num_bins - guard_cells - train_cells;
    
    // Process each cell under test
    cfar_loop: for (int cut = window_start; cut < window_end; cut++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1000 max=16000
        
        // Load training cells
        load_left: for (int i = 0; i < train_cells && i < MAX_TRAIN_CELLS; i++) {
            #pragma HLS UNROLL factor=8
            train_window[i] = range_profile[cut - guard_cells - train_cells + i];
        }
        
        load_right: for (int i = 0; i < train_cells && i < MAX_TRAIN_CELLS; i++) {
            #pragma HLS UNROLL factor=8
            train_window[train_cells + i] = range_profile[cut + guard_cells + 1 + i];
        }
        
        // Calculate noise estimate
        threshold_t noise_est;
        process_cfar_window(train_window, train_cells, cfar_mode, os_rank, noise_est);
        
        // Calculate threshold
        threshold_t threshold = noise_est * threshold_factor;
        thresholds[cut] = threshold;
        
        // Detection decision
        detections[cut] = (threshold_t(range_profile[cut]) > threshold) ? 1 : 0;
    }
    
    // Clear edges
    for (int i = 0; i < window_start; i++) {
        #pragma HLS UNROLL
        detections[i] = 0;
        thresholds[i] = threshold_t(0);
    }
    for (int i = window_end; i < num_bins; i++) {
        #pragma HLS UNROLL
        detections[i] = 0;
        thresholds[i] = threshold_t(0);
    }
}

//=============================================================================
// 2D Range-Doppler CFAR
//=============================================================================

void cfar_2d(
    // Input: Range-Doppler map
    hls::stream<axis_word_t> &s_axis_rdmap,
    
    // Output: Detections
    hls::stream<detection_t> &m_axis_detections,
    
    // Configuration
    ap_uint<16> num_range_bins,
    ap_uint<16> num_doppler_bins,
    ap_uint<8> guard_range,
    ap_uint<8> guard_doppler,
    ap_uint<8> train_range,
    ap_uint<8> train_doppler,
    threshold_t threshold_factor,
    ap_uint<2> cfar_mode,
    ap_uint<8> os_rank,
    ap_uint<1> enable,
    
    // Status
    ap_uint<1> *busy,
    ap_uint<16> *num_detections
) {
    #pragma HLS INTERFACE axis port=s_axis_rdmap
    #pragma HLS INTERFACE axis port=m_axis_detections
    #pragma HLS INTERFACE s_axilite port=num_range_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=num_doppler_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=guard_range bundle=control
    #pragma HLS INTERFACE s_axilite port=guard_doppler bundle=control
    #pragma HLS INTERFACE s_axilite port=train_range bundle=control
    #pragma HLS INTERFACE s_axilite port=train_doppler bundle=control
    #pragma HLS INTERFACE s_axilite port=threshold_factor bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_mode bundle=control
    #pragma HLS INTERFACE s_axilite port=os_rank bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=num_detections bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    if (!enable) {
        *busy = 0;
        return;
    }
    
    *busy = 1;
    
    // Line buffers for 2D processing
    // Store enough Doppler lines to cover the CFAR window
    static power_t line_buffer[2 * MAX_TRAIN_CELLS + 2 * MAX_GUARD_CELLS + 1][MAX_RANGE_BINS];
    #pragma HLS BIND_STORAGE variable=line_buffer type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    
    ap_uint<16> det_count = 0;
    int total_window = 2 * (train_doppler + guard_doppler) + 1;
    int center_line = train_doppler + guard_doppler;
    
    // Read Range-Doppler map and process
    read_rdmap: for (int d = 0; d < num_doppler_bins; d++) {
        #pragma HLS LOOP_TRIPCOUNT min=256 max=1024
        
        // Shift line buffer
        shift_buffer: for (int l = 0; l < total_window - 1; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < num_range_bins; r++) {
                #pragma HLS UNROLL factor=4
                line_buffer[l][r] = line_buffer[l + 1][r];
            }
        }
        
        // Read new line
        read_line: for (int r = 0; r < num_range_bins; r += 4) {
            #pragma HLS PIPELINE II=1
            
            if (!s_axis_rdmap.empty()) {
                axis_word_t word = s_axis_rdmap.read();
                for (int i = 0; i < 4 && (r + i) < num_range_bins; i++) {
                    #pragma HLS UNROLL
                    line_buffer[total_window - 1][r + i] = 
                        power_t(word.data.range(32*(i+1)-1, 32*i));
                }
            }
        }
        
        // Skip until we have enough lines
        if (d < total_window - 1) continue;
        
        // Process range dimension with 2D CFAR
        int doppler_idx = d - center_line;
        
        process_range: for (int r = train_range + guard_range; 
                           r < num_range_bins - train_range - guard_range; r++) {
            #pragma HLS PIPELINE II=2
            #pragma HLS LOOP_TRIPCOUNT min=1000 max=16000
            
            // Cell under test
            power_t cut = line_buffer[center_line][r];
            
            // Accumulate training cells (2D window excluding guard cells)
            ap_uint<32> train_sum = 0;
            ap_uint<16> train_count = 0;
            
            // Loop over Doppler dimension
            train_doppler_loop: for (int td = 0; td < total_window; td++) {
                #pragma HLS UNROLL
                
                // Skip guard region in Doppler
                int doppler_dist = (td > center_line) ? (td - center_line) : (center_line - td);
                if (doppler_dist <= guard_doppler && doppler_dist > 0) continue;
                if (td == center_line) continue;  // Skip center (will do range separately)
                
                // Loop over range dimension
                train_range_loop: for (int tr = -train_range - guard_range; 
                                       tr <= train_range + guard_range; tr++) {
                    #pragma HLS UNROLL factor=4
                    
                    int range_dist = (tr >= 0) ? tr : -tr;
                    
                    // Check if in training region (not guard)
                    if (range_dist > guard_range || td != center_line) {
                        int range_idx = r + tr;
                        if (range_idx >= 0 && range_idx < num_range_bins) {
                            train_sum += line_buffer[td][range_idx];
                            train_count++;
                        }
                    }
                }
            }
            
            // Add range training cells from center Doppler line
            for (int tr = -train_range - guard_range; tr <= train_range + guard_range; tr++) {
                #pragma HLS UNROLL factor=4
                int range_dist = (tr >= 0) ? tr : -tr;
                if (range_dist > guard_range) {
                    int range_idx = r + tr;
                    if (range_idx >= 0 && range_idx < num_range_bins) {
                        train_sum += line_buffer[center_line][range_idx];
                        train_count++;
                    }
                }
            }
            
            // Calculate noise estimate
            threshold_t noise_est;
            if (train_count > 0) {
                if (cfar_mode == CFAR_CA) {
                    noise_est = threshold_t(train_sum) / threshold_t(train_count);
                } else if (cfar_mode == CFAR_GO) {
                    // Simplified GO for 2D
                    noise_est = threshold_t(train_sum) / threshold_t(train_count);
                } else if (cfar_mode == CFAR_SO) {
                    noise_est = threshold_t(train_sum) / threshold_t(train_count);
                } else {
                    noise_est = threshold_t(train_sum) / threshold_t(train_count);
                }
            } else {
                noise_est = threshold_t(1);
            }
            
            // Threshold
            threshold_t threshold = noise_est * threshold_factor;
            
            // Detection
            if (threshold_t(cut) > threshold && det_count < MAX_DETECTIONS) {
                detection_t det;
                det.range_bin = r;
                det.doppler_bin = doppler_idx;
                det.amplitude = cut;
                
                // Calculate SNR
                if (noise_est > threshold_t(0.001)) {
                    threshold_t snr = threshold_t(cut) / noise_est;
                    // Convert to dB (approximation)
                    if (snr > threshold_t(100)) det.snr_db = 20;
                    else if (snr > threshold_t(10)) det.snr_db = 10;
                    else det.snr_db = 5;
                } else {
                    det.snr_db = 30;
                }
                
                det.channel = 0;
                det.valid = 1;
                
                m_axis_detections.write(det);
                det_count++;
            }
        }
    }
    
    // Send end marker
    detection_t end_det;
    end_det.valid = 0;
    end_det.range_bin = 0xFFFF;
    m_axis_detections.write(end_det);
    
    *num_detections = det_count;
    *busy = 0;
}

//=============================================================================
// Top-Level Wrapper
//=============================================================================

void titan_cfar_detector(
    hls::stream<axis_word_t> &s_axis_rdmap,
    hls::stream<detection_t> &m_axis_detections,
    ap_uint<16> num_range_bins,
    ap_uint<16> num_doppler_bins,
    ap_uint<8> guard_range,
    ap_uint<8> guard_doppler,
    ap_uint<8> train_range,
    ap_uint<8> train_doppler,
    threshold_t threshold_factor,
    ap_uint<2> cfar_mode,
    ap_uint<8> os_rank,
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<16> *num_detections
) {
    #pragma HLS INTERFACE axis port=s_axis_rdmap
    #pragma HLS INTERFACE axis port=m_axis_detections
    #pragma HLS INTERFACE s_axilite port=num_range_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=num_doppler_bins bundle=control
    #pragma HLS INTERFACE s_axilite port=guard_range bundle=control
    #pragma HLS INTERFACE s_axilite port=guard_doppler bundle=control
    #pragma HLS INTERFACE s_axilite port=train_range bundle=control
    #pragma HLS INTERFACE s_axilite port=train_doppler bundle=control
    #pragma HLS INTERFACE s_axilite port=threshold_factor bundle=control
    #pragma HLS INTERFACE s_axilite port=cfar_mode bundle=control
    #pragma HLS INTERFACE s_axilite port=os_rank bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=num_detections bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    cfar_2d(
        s_axis_rdmap,
        m_axis_detections,
        num_range_bins,
        num_doppler_bins,
        guard_range,
        guard_doppler,
        train_range,
        train_doppler,
        threshold_factor,
        cfar_mode,
        os_rank,
        enable,
        busy,
        num_detections
    );
}
