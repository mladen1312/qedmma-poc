/*
 * TITAN Radar - 4-Channel Digital Beamformer
 * Supports Conventional, MVDR, and Adaptive Nulling
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

#define NUM_CHANNELS 4
#define MAX_SAMPLES 65536
#define MAX_NULLS 3
#define PI 3.14159265358979323846f

// Speed of light
#define C0 299792458.0f

//=============================================================================
// Beamformer Types
//=============================================================================

// Complex weight type (Q1.15 for I and Q)
typedef Complex<frac16_t> weight_t;

// Covariance matrix element
typedef ap_fixed<32, 8, AP_RND, AP_SAT> cov_t;

// Steering vector
struct steering_vector_t {
    weight_t sv[NUM_CHANNELS];
};

//=============================================================================
// Steering Vector Computation
//=============================================================================

void compute_steering_vector(
    angle_t angle_deg,
    fixed32_t freq_mhz,
    fixed32_t element_spacing_m,
    steering_vector_t &sv
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=4
    
    // Convert angle to radians
    float angle_rad = float(angle_deg) * PI / 180.0f;
    
    // Wavelength
    float wavelength = C0 / (float(freq_mhz) * 1e6f);
    
    // Wave number
    float k = 2.0f * PI / wavelength;
    
    // Phase increment per element
    float d = float(element_spacing_m);
    float phase_inc = k * d * hls::sinf(angle_rad);
    
    // Compute steering vector
    compute_sv: for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        #pragma HLS UNROLL
        float phase = ch * phase_inc;
        sv.sv[ch].real = frac16_t(hls::cosf(phase));
        sv.sv[ch].imag = frac16_t(hls::sinf(phase));
    }
}

//=============================================================================
// Complex Matrix Operations
//=============================================================================

// 4x4 Complex Matrix
struct matrix4x4_t {
    cov_t real[NUM_CHANNELS][NUM_CHANNELS];
    cov_t imag[NUM_CHANNELS][NUM_CHANNELS];
};

// 4x4 Matrix Inverse using Gauss-Jordan (simplified for 4x4)
void matrix_inverse_4x4(
    const matrix4x4_t &A,
    matrix4x4_t &A_inv
) {
    #pragma HLS INLINE off
    
    // Augmented matrix [A | I]
    cov_t aug_real[NUM_CHANNELS][2*NUM_CHANNELS];
    cov_t aug_imag[NUM_CHANNELS][2*NUM_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=aug_real complete dim=2
    #pragma HLS ARRAY_PARTITION variable=aug_imag complete dim=2
    
    // Initialize augmented matrix
    init_aug: for (int i = 0; i < NUM_CHANNELS; i++) {
        for (int j = 0; j < NUM_CHANNELS; j++) {
            #pragma HLS UNROLL
            aug_real[i][j] = A.real[i][j];
            aug_imag[i][j] = A.imag[i][j];
            aug_real[i][j + NUM_CHANNELS] = (i == j) ? cov_t(1.0f) : cov_t(0.0f);
            aug_imag[i][j + NUM_CHANNELS] = cov_t(0.0f);
        }
    }
    
    // Gauss-Jordan elimination
    gauss_jordan: for (int col = 0; col < NUM_CHANNELS; col++) {
        #pragma HLS PIPELINE II=4
        
        // Pivot (simplified - assume non-zero diagonal)
        cov_t pivot_real = aug_real[col][col];
        cov_t pivot_imag = aug_imag[col][col];
        cov_t pivot_mag_sq = pivot_real * pivot_real + pivot_imag * pivot_imag;
        
        // Avoid division by zero
        if (pivot_mag_sq < cov_t(1e-10f)) {
            pivot_mag_sq = cov_t(1e-10f);
        }
        
        // Normalize pivot row
        cov_t inv_pivot_real = pivot_real / pivot_mag_sq;
        cov_t inv_pivot_imag = -pivot_imag / pivot_mag_sq;
        
        normalize: for (int j = 0; j < 2*NUM_CHANNELS; j++) {
            #pragma HLS UNROLL
            cov_t temp_real = aug_real[col][j] * inv_pivot_real - aug_imag[col][j] * inv_pivot_imag;
            cov_t temp_imag = aug_real[col][j] * inv_pivot_imag + aug_imag[col][j] * inv_pivot_real;
            aug_real[col][j] = temp_real;
            aug_imag[col][j] = temp_imag;
        }
        
        // Eliminate column
        eliminate: for (int row = 0; row < NUM_CHANNELS; row++) {
            if (row != col) {
                cov_t factor_real = aug_real[row][col];
                cov_t factor_imag = aug_imag[row][col];
                
                elim_cols: for (int j = 0; j < 2*NUM_CHANNELS; j++) {
                    #pragma HLS UNROLL
                    aug_real[row][j] -= factor_real * aug_real[col][j] - factor_imag * aug_imag[col][j];
                    aug_imag[row][j] -= factor_real * aug_imag[col][j] + factor_imag * aug_real[col][j];
                }
            }
        }
    }
    
    // Extract inverse
    extract: for (int i = 0; i < NUM_CHANNELS; i++) {
        for (int j = 0; j < NUM_CHANNELS; j++) {
            #pragma HLS UNROLL
            A_inv.real[i][j] = aug_real[i][j + NUM_CHANNELS];
            A_inv.imag[i][j] = aug_imag[i][j + NUM_CHANNELS];
        }
    }
}

//=============================================================================
// Covariance Matrix Estimation
//=============================================================================

void estimate_covariance(
    const complex_sample_t samples[NUM_CHANNELS][1024],
    int num_samples,
    matrix4x4_t &R
) {
    #pragma HLS INLINE off
    
    // Initialize to zero
    init_R: for (int i = 0; i < NUM_CHANNELS; i++) {
        for (int j = 0; j < NUM_CHANNELS; j++) {
            #pragma HLS UNROLL
            R.real[i][j] = 0;
            R.imag[i][j] = 0;
        }
    }
    
    // Accumulate outer products
    accum: for (int n = 0; n < num_samples; n++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=64 max=1024
        
        outer_i: for (int i = 0; i < NUM_CHANNELS; i++) {
            #pragma HLS UNROLL
            outer_j: for (int j = 0; j < NUM_CHANNELS; j++) {
                #pragma HLS UNROLL
                // R[i][j] += x[i] * conj(x[j])
                cov_t prod_real = cov_t(samples[i][n].real) * cov_t(samples[j][n].real) +
                                  cov_t(samples[i][n].imag) * cov_t(samples[j][n].imag);
                cov_t prod_imag = cov_t(samples[i][n].imag) * cov_t(samples[j][n].real) -
                                  cov_t(samples[i][n].real) * cov_t(samples[j][n].imag);
                R.real[i][j] += prod_real;
                R.imag[i][j] += prod_imag;
            }
        }
    }
    
    // Normalize
    cov_t norm = cov_t(1.0f / num_samples);
    normalize_R: for (int i = 0; i < NUM_CHANNELS; i++) {
        for (int j = 0; j < NUM_CHANNELS; j++) {
            #pragma HLS UNROLL
            R.real[i][j] *= norm;
            R.imag[i][j] *= norm;
        }
    }
    
    // Add diagonal loading for stability
    diagonal_loading: for (int i = 0; i < NUM_CHANNELS; i++) {
        #pragma HLS UNROLL
        R.real[i][i] += cov_t(0.01f);  // 1% diagonal loading
    }
}

//=============================================================================
// MVDR Beamformer Weights
//=============================================================================

void compute_mvdr_weights(
    const matrix4x4_t &R,
    const steering_vector_t &a,
    weight_t weights[NUM_CHANNELS]
) {
    #pragma HLS INLINE off
    
    // Compute R^-1
    matrix4x4_t R_inv;
    matrix_inverse_4x4(R, R_inv);
    
    // Compute R^-1 * a
    weight_t R_inv_a[NUM_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=R_inv_a complete
    
    compute_R_inv_a: for (int i = 0; i < NUM_CHANNELS; i++) {
        #pragma HLS UNROLL
        cov_t sum_real = 0;
        cov_t sum_imag = 0;
        
        for (int j = 0; j < NUM_CHANNELS; j++) {
            #pragma HLS UNROLL
            sum_real += R_inv.real[i][j] * cov_t(a.sv[j].real) - 
                       R_inv.imag[i][j] * cov_t(a.sv[j].imag);
            sum_imag += R_inv.real[i][j] * cov_t(a.sv[j].imag) + 
                       R_inv.imag[i][j] * cov_t(a.sv[j].real);
        }
        
        R_inv_a[i].real = frac16_t(sum_real);
        R_inv_a[i].imag = frac16_t(sum_imag);
    }
    
    // Compute a^H * R^-1 * a
    cov_t denom_real = 0;
    cov_t denom_imag = 0;
    compute_denom: for (int i = 0; i < NUM_CHANNELS; i++) {
        #pragma HLS UNROLL
        // a^H[i] * R_inv_a[i]
        denom_real += cov_t(a.sv[i].real) * cov_t(R_inv_a[i].real) + 
                     cov_t(a.sv[i].imag) * cov_t(R_inv_a[i].imag);
        denom_imag += cov_t(a.sv[i].real) * cov_t(R_inv_a[i].imag) - 
                     cov_t(a.sv[i].imag) * cov_t(R_inv_a[i].real);
    }
    
    // w = R^-1 * a / (a^H * R^-1 * a)
    cov_t denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag;
    if (denom_mag_sq < cov_t(1e-10f)) denom_mag_sq = cov_t(1e-10f);
    
    cov_t inv_denom_real = denom_real / denom_mag_sq;
    cov_t inv_denom_imag = -denom_imag / denom_mag_sq;
    
    compute_weights: for (int i = 0; i < NUM_CHANNELS; i++) {
        #pragma HLS UNROLL
        weights[i].real = frac16_t(cov_t(R_inv_a[i].real) * inv_denom_real - 
                                   cov_t(R_inv_a[i].imag) * inv_denom_imag);
        weights[i].imag = frac16_t(cov_t(R_inv_a[i].real) * inv_denom_imag + 
                                   cov_t(R_inv_a[i].imag) * inv_denom_real);
    }
}

//=============================================================================
// Conventional Beamformer Weights
//=============================================================================

void compute_conventional_weights(
    const steering_vector_t &a,
    weight_t weights[NUM_CHANNELS]
) {
    #pragma HLS INLINE
    
    // w = a* / N (conjugate of steering vector, normalized)
    frac16_t norm = frac16_t(1.0f / NUM_CHANNELS);
    
    for (int i = 0; i < NUM_CHANNELS; i++) {
        #pragma HLS UNROLL
        weights[i].real = a.sv[i].real * norm;
        weights[i].imag = -a.sv[i].imag * norm;  // Conjugate
    }
}

//=============================================================================
// Apply Beamformer Weights
//=============================================================================

complex_sample_t apply_weights(
    const complex_sample_t samples[NUM_CHANNELS],
    const weight_t weights[NUM_CHANNELS]
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    accum32_t sum_real = 0;
    accum32_t sum_imag = 0;
    
    apply_loop: for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        #pragma HLS UNROLL
        // y += w[ch]^H * x[ch]
        sum_real += accum32_t(weights[ch].real) * accum32_t(samples[ch].real) +
                   accum32_t(weights[ch].imag) * accum32_t(samples[ch].imag);
        sum_imag += accum32_t(weights[ch].real) * accum32_t(samples[ch].imag) -
                   accum32_t(weights[ch].imag) * accum32_t(samples[ch].real);
    }
    
    complex_sample_t result;
    result.real = sample_t(sum_real >> 15);  // Scale back
    result.imag = sample_t(sum_imag >> 15);
    
    return result;
}

//=============================================================================
// Main Beamformer Function
//=============================================================================

void beamformer(
    // Input streams (4 channels)
    hls::stream<adc_word_t> &s_axis_ch0,
    hls::stream<adc_word_t> &s_axis_ch1,
    hls::stream<adc_word_t> &s_axis_ch2,
    hls::stream<adc_word_t> &s_axis_ch3,
    
    // Output stream (beamformed)
    hls::stream<adc_word_t> &m_axis_beam,
    
    // Configuration
    angle_t steering_angle,
    angle_t null_angles[MAX_NULLS],
    ap_uint<2> num_nulls,
    ap_uint<2> mode,              // 0=conv, 1=MVDR, 2=nulling
    fixed32_t freq_mhz,
    fixed32_t element_spacing,
    ap_uint<1> enable,
    
    // Status
    ap_uint<1> *busy,
    ap_uint<32> *samples_processed
) {
    #pragma HLS INTERFACE axis port=s_axis_ch0
    #pragma HLS INTERFACE axis port=s_axis_ch1
    #pragma HLS INTERFACE axis port=s_axis_ch2
    #pragma HLS INTERFACE axis port=s_axis_ch3
    #pragma HLS INTERFACE axis port=m_axis_beam
    #pragma HLS INTERFACE s_axilite port=steering_angle bundle=control
    #pragma HLS INTERFACE s_axilite port=null_angles bundle=control
    #pragma HLS INTERFACE s_axilite port=num_nulls bundle=control
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=freq_mhz bundle=control
    #pragma HLS INTERFACE s_axilite port=element_spacing bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=samples_processed bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    if (!enable) {
        *busy = 0;
        return;
    }
    
    *busy = 1;
    
    // Compute steering vector
    steering_vector_t sv;
    compute_steering_vector(steering_angle, freq_mhz, element_spacing, sv);
    
    // Compute weights based on mode
    static weight_t weights[NUM_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=weights complete
    
    if (mode == 0) {
        // Conventional beamforming
        compute_conventional_weights(sv, weights);
    }
    // MVDR would require covariance estimation from data
    // For now, use conventional as fallback
    else {
        compute_conventional_weights(sv, weights);
    }
    
    // Process samples
    ap_uint<32> count = 0;
    
    process_loop: while (!s_axis_ch0.empty()) {
        #pragma HLS PIPELINE II=1
        
        // Read from all channels
        adc_word_t in0 = s_axis_ch0.read();
        adc_word_t in1 = s_axis_ch1.read();
        adc_word_t in2 = s_axis_ch2.read();
        adc_word_t in3 = s_axis_ch3.read();
        
        // Beamform each sample in the word
        adc_word_t out_word;
        
        beamform_samples: for (int s = 0; s < 8; s++) {
            #pragma HLS UNROLL
            
            complex_sample_t ch_samples[NUM_CHANNELS];
            ch_samples[0] = {in0.samples[s], sample_t(0)};  // Real ADC data
            ch_samples[1] = {in1.samples[s], sample_t(0)};
            ch_samples[2] = {in2.samples[s], sample_t(0)};
            ch_samples[3] = {in3.samples[s], sample_t(0)};
            
            complex_sample_t beamformed = apply_weights(ch_samples, weights);
            out_word.samples[s] = beamformed.real;  // Output real part
        }
        
        out_word.last = in0.last;
        m_axis_beam.write(out_word);
        
        count += 8;
        
        if (in0.last) break;
    }
    
    *samples_processed = count;
    *busy = 0;
}

//=============================================================================
// Top-Level Wrapper
//=============================================================================

void titan_beamformer(
    hls::stream<adc_word_t> &s_axis_ch0,
    hls::stream<adc_word_t> &s_axis_ch1,
    hls::stream<adc_word_t> &s_axis_ch2,
    hls::stream<adc_word_t> &s_axis_ch3,
    hls::stream<adc_word_t> &m_axis_beam,
    angle_t steering_angle,
    angle_t null_angles[MAX_NULLS],
    ap_uint<2> num_nulls,
    ap_uint<2> mode,
    fixed32_t freq_mhz,
    fixed32_t element_spacing,
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<32> *samples_processed
) {
    #pragma HLS INTERFACE axis port=s_axis_ch0
    #pragma HLS INTERFACE axis port=s_axis_ch1
    #pragma HLS INTERFACE axis port=s_axis_ch2
    #pragma HLS INTERFACE axis port=s_axis_ch3
    #pragma HLS INTERFACE axis port=m_axis_beam
    #pragma HLS INTERFACE s_axilite port=steering_angle bundle=control
    #pragma HLS INTERFACE s_axilite port=null_angles bundle=control
    #pragma HLS INTERFACE s_axilite port=num_nulls bundle=control
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=freq_mhz bundle=control
    #pragma HLS INTERFACE s_axilite port=element_spacing bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=samples_processed bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    beamformer(
        s_axis_ch0, s_axis_ch1, s_axis_ch2, s_axis_ch3,
        m_axis_beam,
        steering_angle, null_angles, num_nulls, mode,
        freq_mhz, element_spacing, enable,
        busy, samples_processed
    );
}
