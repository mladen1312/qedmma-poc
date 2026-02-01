/*
 * TITAN Radar - Waveform Generator
 * PRBS, LFM, and Custom Waveform Generation
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

#define MAX_PULSE_SAMPLES 65536
#define PRBS_15_LEN 32767
#define PRBS_20_LEN 1048575
#define PI 3.14159265358979323846f

// Waveform types
#define WAVEFORM_PRBS_15 0
#define WAVEFORM_PRBS_20 1
#define WAVEFORM_LFM_UP 2
#define WAVEFORM_LFM_DOWN 3
#define WAVEFORM_BPSK 4
#define WAVEFORM_CW 5

//=============================================================================
// NCO (Numerically Controlled Oscillator)
//=============================================================================

class NCO {
public:
    ap_uint<32> phase;
    ap_uint<32> phase_inc;
    
    NCO() : phase(0), phase_inc(0) {}
    
    void set_frequency(ap_uint<32> freq_word) {
        #pragma HLS INLINE
        phase_inc = freq_word;
    }
    
    void reset() {
        #pragma HLS INLINE
        phase = 0;
    }
    
    // Get sine and cosine outputs
    void get_sincos(frac16_t &sin_out, frac16_t &cos_out) {
        #pragma HLS INLINE
        #pragma HLS PIPELINE II=1
        
        // Use top 10 bits as LUT index
        ap_uint<10> lut_idx = phase.range(31, 22);
        
        // Simple quarter-wave symmetry LUT
        // Full implementation would use ROM
        float angle = float(lut_idx) / 1024.0f * 2.0f * PI;
        sin_out = frac16_t(hls::sinf(angle));
        cos_out = frac16_t(hls::cosf(angle));
        
        // Update phase
        phase += phase_inc;
    }
    
    frac16_t get_sin() {
        #pragma HLS INLINE
        ap_uint<10> lut_idx = phase.range(31, 22);
        float angle = float(lut_idx) / 1024.0f * 2.0f * PI;
        phase += phase_inc;
        return frac16_t(hls::sinf(angle));
    }
};

//=============================================================================
// PRBS Generator
//=============================================================================

class PRBSGenerator {
private:
    ap_uint<32> state;
    ap_uint<8> order;
    
public:
    PRBSGenerator() : state(1), order(15) {}
    
    void set_order(ap_uint<8> new_order) {
        #pragma HLS INLINE
        order = new_order;
        state = 1;
    }
    
    void reset() {
        #pragma HLS INLINE
        state = 1;
    }
    
    // Generate next PRBS bit
    ap_uint<1> next_bit() {
        #pragma HLS INLINE
        #pragma HLS PIPELINE II=1
        
        ap_uint<1> output = state & 1;
        ap_uint<1> feedback;
        
        // Feedback taps for different PRBS orders
        switch(order) {
            case 7:
                feedback = ((state >> 6) ^ (state >> 5)) & 1;
                break;
            case 9:
                feedback = ((state >> 8) ^ (state >> 4)) & 1;
                break;
            case 11:
                feedback = ((state >> 10) ^ (state >> 8)) & 1;
                break;
            case 15:
                feedback = ((state >> 14) ^ (state >> 13)) & 1;
                break;
            case 20:
                feedback = ((state >> 19) ^ (state >> 16)) & 1;
                break;
            case 23:
                feedback = ((state >> 22) ^ (state >> 17)) & 1;
                break;
            default:
                feedback = ((state >> 14) ^ (state >> 13)) & 1;
                break;
        }
        
        state = ((state << 1) | feedback) & ((1 << order) - 1);
        if (state == 0) state = 1;  // Prevent lock-up
        
        return output;
    }
    
    // Generate BPSK symbol (+1 or -1)
    frac16_t next_symbol() {
        #pragma HLS INLINE
        return next_bit() ? frac16_t(0.9) : frac16_t(-0.9);
    }
};

//=============================================================================
// LFM (Linear Frequency Modulation) Generator
//=============================================================================

class LFMGenerator {
private:
    ap_uint<48> phase;
    ap_uint<32> freq;
    ap_int<32> chirp_rate;
    ap_uint<32> start_freq;
    
public:
    LFMGenerator() : phase(0), freq(0), chirp_rate(0), start_freq(0) {}
    
    void configure(
        ap_uint<32> start_freq_word,
        ap_uint<32> bandwidth_word,
        ap_uint<32> num_samples,
        bool up_chirp
    ) {
        #pragma HLS INLINE
        
        start_freq = start_freq_word;
        freq = start_freq_word;
        phase = 0;
        
        // Chirp rate = bandwidth / num_samples
        if (up_chirp) {
            chirp_rate = ap_int<32>(bandwidth_word / num_samples);
        } else {
            chirp_rate = ap_int<32>(-ap_int<32>(bandwidth_word / num_samples));
        }
    }
    
    void reset() {
        #pragma HLS INLINE
        phase = 0;
        freq = start_freq;
    }
    
    void get_sample(frac16_t &I, frac16_t &Q) {
        #pragma HLS INLINE
        #pragma HLS PIPELINE II=1
        
        // Phase to sine/cosine
        ap_uint<10> lut_idx = phase.range(47, 38);
        float angle = float(lut_idx) / 1024.0f * 2.0f * PI;
        
        I = frac16_t(hls::cosf(angle));
        Q = frac16_t(hls::sinf(angle));
        
        // Update phase and frequency
        phase += freq;
        freq += chirp_rate;
    }
};

//=============================================================================
// Pulse Shaping Filter
//=============================================================================

void raised_cosine_filter(
    frac16_t &sample,
    ap_uint<8> position,
    ap_uint<8> rise_samples
) {
    #pragma HLS INLINE
    
    if (position < rise_samples) {
        // Rising edge
        float t = float(position) / float(rise_samples);
        frac16_t window = frac16_t(0.5f * (1.0f - hls::cosf(PI * t)));
        sample = sample * window;
    }
    // Flat top - no change
    // Falling edge handled similarly at end of pulse
}

//=============================================================================
// Main Waveform Generator
//=============================================================================

void waveform_generator(
    // Output stream to DAC
    hls::stream<dac_word_t> &m_axis_dac,
    
    // Configuration
    ap_uint<4> waveform_type,
    ap_uint<8> prbs_order,
    ap_uint<32> nco_freq_word,        // NCO frequency in phase words
    ap_uint<32> chirp_bandwidth,       // LFM bandwidth
    ap_uint<32> pulse_samples,         // Samples per pulse
    ap_uint<32> pri_samples,           // Samples per PRI
    ap_uint<16> num_pulses,            // Pulses to generate
    ap_uint<8> samples_per_chip,       // Upsampling factor for PRBS
    ap_uint<1> enable,
    
    // Status
    ap_uint<1> *busy,
    ap_uint<32> *samples_generated
) {
    #pragma HLS INTERFACE axis port=m_axis_dac
    #pragma HLS INTERFACE s_axilite port=waveform_type bundle=control
    #pragma HLS INTERFACE s_axilite port=prbs_order bundle=control
    #pragma HLS INTERFACE s_axilite port=nco_freq_word bundle=control
    #pragma HLS INTERFACE s_axilite port=chirp_bandwidth bundle=control
    #pragma HLS INTERFACE s_axilite port=pulse_samples bundle=control
    #pragma HLS INTERFACE s_axilite port=pri_samples bundle=control
    #pragma HLS INTERFACE s_axilite port=num_pulses bundle=control
    #pragma HLS INTERFACE s_axilite port=samples_per_chip bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=samples_generated bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    if (!enable) {
        *busy = 0;
        return;
    }
    
    *busy = 1;
    
    // Initialize generators
    static PRBSGenerator prbs;
    static LFMGenerator lfm;
    static NCO nco;
    
    prbs.set_order(prbs_order);
    nco.set_frequency(nco_freq_word);
    
    bool use_lfm = (waveform_type == WAVEFORM_LFM_UP || waveform_type == WAVEFORM_LFM_DOWN);
    if (use_lfm) {
        lfm.configure(
            nco_freq_word,
            chirp_bandwidth,
            pulse_samples,
            (waveform_type == WAVEFORM_LFM_UP)
        );
    }
    
    ap_uint<32> total_samples = 0;
    ap_uint<8> chip_counter = 0;
    frac16_t current_chip = frac16_t(0);
    
    // Generate pulses
    pulse_loop: for (int pulse = 0; pulse < num_pulses; pulse++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=1024
        
        // Reset generators at start of each pulse
        if (use_lfm) {
            lfm.reset();
        }
        nco.reset();
        chip_counter = 0;
        
        // Generate one PRI worth of samples
        pri_loop: for (ap_uint<32> sample = 0; sample < pri_samples; sample += 8) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1000 max=100000
            
            dac_word_t out_word;
            
            // Generate 8 samples per word
            sample_loop: for (int s = 0; s < 8; s++) {
                #pragma HLS UNROLL
                
                ap_uint<32> current_sample = sample + s;
                frac16_t baseband_i, baseband_q;
                frac16_t output;
                
                if (current_sample < pulse_samples) {
                    // Within pulse
                    switch(waveform_type) {
                        case WAVEFORM_PRBS_15:
                        case WAVEFORM_PRBS_20:
                        case WAVEFORM_BPSK:
                            // Generate chip value
                            if (chip_counter == 0) {
                                current_chip = prbs.next_symbol();
                            }
                            chip_counter++;
                            if (chip_counter >= samples_per_chip) {
                                chip_counter = 0;
                            }
                            
                            // Modulate onto carrier
                            baseband_i = current_chip;
                            baseband_q = frac16_t(0);
                            break;
                            
                        case WAVEFORM_LFM_UP:
                        case WAVEFORM_LFM_DOWN:
                            lfm.get_sample(baseband_i, baseband_q);
                            break;
                            
                        case WAVEFORM_CW:
                        default:
                            baseband_i = frac16_t(0.9);
                            baseband_q = frac16_t(0);
                            break;
                    }
                    
                    // Mix with carrier (if not LFM which has built-in carrier)
                    if (!use_lfm) {
                        frac16_t cos_val, sin_val;
                        nco.get_sincos(sin_val, cos_val);
                        
                        // I*cos - Q*sin
                        output = baseband_i * cos_val - baseband_q * sin_val;
                    } else {
                        output = baseband_i;  // LFM already has carrier
                    }
                    
                } else {
                    // Dead time between pulses
                    output = frac16_t(0);
                }
                
                // Scale to 14-bit DAC range
                out_word.samples[s] = dac_sample_t(output * frac16_t(8191));
            }
            
            out_word.last = (pulse == num_pulses - 1 && 
                            sample + 8 >= pri_samples) ? 1 : 0;
            
            m_axis_dac.write(out_word);
            total_samples += 8;
        }
    }
    
    *samples_generated = total_samples;
    *busy = 0;
}

//=============================================================================
// Reference Waveform Generator (for correlator)
//=============================================================================

void generate_reference_waveform(
    frac16_t ref_buffer[MAX_PULSE_SAMPLES],
    ap_uint<4> waveform_type,
    ap_uint<8> prbs_order,
    ap_uint<32> num_samples,
    ap_uint<8> samples_per_chip
) {
    #pragma HLS INLINE off
    
    PRBSGenerator prbs;
    prbs.set_order(prbs_order);
    
    ap_uint<8> chip_counter = 0;
    frac16_t current_chip = prbs.next_symbol();
    
    gen_loop: for (ap_uint<32> i = 0; i < num_samples && i < MAX_PULSE_SAMPLES; i++) {
        #pragma HLS PIPELINE II=1
        
        if (waveform_type == WAVEFORM_PRBS_15 || 
            waveform_type == WAVEFORM_PRBS_20 ||
            waveform_type == WAVEFORM_BPSK) {
            
            ref_buffer[i] = current_chip;
            
            chip_counter++;
            if (chip_counter >= samples_per_chip) {
                chip_counter = 0;
                current_chip = prbs.next_symbol();
            }
        } else {
            ref_buffer[i] = frac16_t(1.0);
        }
    }
}

//=============================================================================
// Top-Level Wrapper
//=============================================================================

void titan_waveform_generator(
    hls::stream<dac_word_t> &m_axis_dac,
    ap_uint<4> waveform_type,
    ap_uint<8> prbs_order,
    ap_uint<32> nco_freq_word,
    ap_uint<32> chirp_bandwidth,
    ap_uint<32> pulse_samples,
    ap_uint<32> pri_samples,
    ap_uint<16> num_pulses,
    ap_uint<8> samples_per_chip,
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<32> *samples_generated
) {
    #pragma HLS INTERFACE axis port=m_axis_dac
    #pragma HLS INTERFACE s_axilite port=waveform_type bundle=control
    #pragma HLS INTERFACE s_axilite port=prbs_order bundle=control
    #pragma HLS INTERFACE s_axilite port=nco_freq_word bundle=control
    #pragma HLS INTERFACE s_axilite port=chirp_bandwidth bundle=control
    #pragma HLS INTERFACE s_axilite port=pulse_samples bundle=control
    #pragma HLS INTERFACE s_axilite port=pri_samples bundle=control
    #pragma HLS INTERFACE s_axilite port=num_pulses bundle=control
    #pragma HLS INTERFACE s_axilite port=samples_per_chip bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=samples_generated bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    waveform_generator(
        m_axis_dac,
        waveform_type,
        prbs_order,
        nco_freq_word,
        chirp_bandwidth,
        pulse_samples,
        pri_samples,
        num_pulses,
        samples_per_chip,
        enable,
        busy,
        samples_generated
    );
}
