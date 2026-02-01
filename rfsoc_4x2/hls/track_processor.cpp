/*
 * TITAN Radar - Track Processor
 * Extended Kalman Filter for Multi-Target Tracking
 * 
 * Author: Dr. Mladen Mešter
 * Copyright (c) 2026 - All Rights Reserved
 * 
 * Platform: AMD RFSoC 4x2
 * 
 * Supports:
 * - 256 simultaneous tracks
 * - 6-state EKF (range, velocity, acceleration, azimuth, azimuth rate, az accel)
 * - M-of-N track confirmation logic
 * - Gating and data association
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

#define MAX_TRACKS 256
#define MAX_DETECTIONS 512
#define STATE_DIM 6
#define MEAS_DIM 3

// Track states
#define TRACK_FREE 0
#define TRACK_TENTATIVE 1
#define TRACK_CONFIRMED 2
#define TRACK_COASTING 3

// Default parameters
#define DEFAULT_CONFIRM_HITS 3
#define DEFAULT_DELETE_MISSES 5
#define DEFAULT_COAST_LIMIT 10

//=============================================================================
// Types
//=============================================================================

// State vector: [range, velocity, acceleration, azimuth, az_rate, az_accel]
typedef ap_fixed<32, 16, AP_RND, AP_SAT> state_t;

// Covariance matrix element
typedef ap_fixed<32, 8, AP_RND, AP_SAT> cov_elem_t;

// 6x6 matrix type
struct matrix6x6_t {
    cov_elem_t data[STATE_DIM][STATE_DIM];
};

// 6x3 matrix type (Kalman gain)
struct matrix6x3_t {
    cov_elem_t data[STATE_DIM][MEAS_DIM];
};

// 3x6 matrix type (measurement matrix)
struct matrix3x6_t {
    cov_elem_t data[MEAS_DIM][STATE_DIM];
};

// 3x3 matrix type (innovation covariance)
struct matrix3x3_t {
    cov_elem_t data[MEAS_DIM][MEAS_DIM];
};

// State vector
struct state_vector_t {
    state_t x[STATE_DIM];  // [r, v, a, az, az_rate, az_accel]
};

// Measurement vector
struct measurement_t {
    state_t z[MEAS_DIM];   // [range, velocity, azimuth]
    magnitude_t amplitude;
    ap_uint<1> valid;
};

// Track record
struct track_record_t {
    ap_uint<8> id;
    ap_uint<2> state;           // FREE, TENTATIVE, CONFIRMED, COASTING
    state_vector_t x;           // State estimate
    matrix6x6_t P;              // Covariance matrix
    ap_uint<8> hits;
    ap_uint<8> misses;
    ap_uint<8> coast_count;
    ap_uint<16> age;
    magnitude_t last_amplitude;
    ap_uint<8> quality;         // 0-255 quality metric
    ap_uint<1> valid;
};

// Detection-to-track association
struct association_t {
    ap_uint<8> track_id;
    ap_uint<16> det_id;
    cov_elem_t distance;        // Mahalanobis distance
    ap_uint<1> valid;
};

//=============================================================================
// Matrix Operations
//=============================================================================

// 6x6 Matrix multiply: C = A * B
void mat6x6_mult(
    const matrix6x6_t &A,
    const matrix6x6_t &B,
    matrix6x6_t &C
) {
    #pragma HLS INLINE
    
    for (int i = 0; i < STATE_DIM; i++) {
        #pragma HLS UNROLL factor=2
        for (int j = 0; j < STATE_DIM; j++) {
            #pragma HLS UNROLL factor=2
            cov_elem_t sum = 0;
            for (int k = 0; k < STATE_DIM; k++) {
                #pragma HLS UNROLL
                sum += A.data[i][k] * B.data[k][j];
            }
            C.data[i][j] = sum;
        }
    }
}

// 6x6 Matrix add: C = A + B
void mat6x6_add(
    const matrix6x6_t &A,
    const matrix6x6_t &B,
    matrix6x6_t &C
) {
    #pragma HLS INLINE
    
    for (int i = 0; i < STATE_DIM; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < STATE_DIM; j++) {
            #pragma HLS UNROLL
            C.data[i][j] = A.data[i][j] + B.data[i][j];
        }
    }
}

// 6x6 Matrix transpose
void mat6x6_transpose(
    const matrix6x6_t &A,
    matrix6x6_t &AT
) {
    #pragma HLS INLINE
    
    for (int i = 0; i < STATE_DIM; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < STATE_DIM; j++) {
            #pragma HLS UNROLL
            AT.data[i][j] = A.data[j][i];
        }
    }
}

// 3x3 Matrix inverse (simplified for small matrix)
void mat3x3_inverse(
    const matrix3x3_t &A,
    matrix3x3_t &A_inv
) {
    #pragma HLS INLINE
    
    // Compute determinant
    cov_elem_t det = 
        A.data[0][0] * (A.data[1][1] * A.data[2][2] - A.data[1][2] * A.data[2][1]) -
        A.data[0][1] * (A.data[1][0] * A.data[2][2] - A.data[1][2] * A.data[2][0]) +
        A.data[0][2] * (A.data[1][0] * A.data[2][1] - A.data[1][1] * A.data[2][0]);
    
    // Avoid division by zero
    if (det < cov_elem_t(1e-10) && det > cov_elem_t(-1e-10)) {
        det = cov_elem_t(1e-10);
    }
    
    cov_elem_t inv_det = cov_elem_t(1.0) / det;
    
    // Adjugate matrix / det
    A_inv.data[0][0] = (A.data[1][1] * A.data[2][2] - A.data[1][2] * A.data[2][1]) * inv_det;
    A_inv.data[0][1] = (A.data[0][2] * A.data[2][1] - A.data[0][1] * A.data[2][2]) * inv_det;
    A_inv.data[0][2] = (A.data[0][1] * A.data[1][2] - A.data[0][2] * A.data[1][1]) * inv_det;
    A_inv.data[1][0] = (A.data[1][2] * A.data[2][0] - A.data[1][0] * A.data[2][2]) * inv_det;
    A_inv.data[1][1] = (A.data[0][0] * A.data[2][2] - A.data[0][2] * A.data[2][0]) * inv_det;
    A_inv.data[1][2] = (A.data[0][2] * A.data[1][0] - A.data[0][0] * A.data[1][2]) * inv_det;
    A_inv.data[2][0] = (A.data[1][0] * A.data[2][1] - A.data[1][1] * A.data[2][0]) * inv_det;
    A_inv.data[2][1] = (A.data[0][1] * A.data[2][0] - A.data[0][0] * A.data[2][1]) * inv_det;
    A_inv.data[2][2] = (A.data[0][0] * A.data[1][1] - A.data[0][1] * A.data[1][0]) * inv_det;
}

//=============================================================================
// Kalman Filter Functions
//=============================================================================

// Initialize state transition matrix F for constant acceleration model
void init_F_matrix(
    matrix6x6_t &F,
    cov_elem_t dt
) {
    #pragma HLS INLINE
    
    // Zero matrix
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            F.data[i][j] = (i == j) ? cov_elem_t(1.0) : cov_elem_t(0.0);
        }
    }
    
    cov_elem_t dt2 = dt * dt / cov_elem_t(2.0);
    
    // Range dynamics: r' = r + v*dt + 0.5*a*dt^2
    F.data[0][1] = dt;      // dr/dv
    F.data[0][2] = dt2;     // dr/da
    
    // Velocity dynamics: v' = v + a*dt
    F.data[1][2] = dt;      // dv/da
    
    // Azimuth dynamics
    F.data[3][4] = dt;      // daz/daz_rate
    F.data[3][5] = dt2;     // daz/daz_accel
    F.data[4][5] = dt;      // daz_rate/daz_accel
}

// Initialize process noise matrix Q
void init_Q_matrix(
    matrix6x6_t &Q,
    cov_elem_t dt,
    cov_elem_t sigma_a,      // Acceleration noise
    cov_elem_t sigma_az      // Azimuth acceleration noise
) {
    #pragma HLS INLINE
    
    cov_elem_t dt2 = dt * dt;
    cov_elem_t dt3 = dt2 * dt;
    cov_elem_t dt4 = dt3 * dt;
    
    // Zero matrix
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            Q.data[i][j] = cov_elem_t(0.0);
        }
    }
    
    cov_elem_t var_a = sigma_a * sigma_a;
    cov_elem_t var_az = sigma_az * sigma_az;
    
    // Range/velocity block
    Q.data[0][0] = dt4 / cov_elem_t(4.0) * var_a;
    Q.data[0][1] = dt3 / cov_elem_t(2.0) * var_a;
    Q.data[0][2] = dt2 / cov_elem_t(2.0) * var_a;
    Q.data[1][0] = Q.data[0][1];
    Q.data[1][1] = dt2 * var_a;
    Q.data[1][2] = dt * var_a;
    Q.data[2][0] = Q.data[0][2];
    Q.data[2][1] = Q.data[1][2];
    Q.data[2][2] = var_a;
    
    // Azimuth block
    Q.data[3][3] = dt4 / cov_elem_t(4.0) * var_az;
    Q.data[3][4] = dt3 / cov_elem_t(2.0) * var_az;
    Q.data[3][5] = dt2 / cov_elem_t(2.0) * var_az;
    Q.data[4][3] = Q.data[3][4];
    Q.data[4][4] = dt2 * var_az;
    Q.data[4][5] = dt * var_az;
    Q.data[5][3] = Q.data[3][5];
    Q.data[5][4] = Q.data[4][5];
    Q.data[5][5] = var_az;
}

// Measurement matrix H (maps state to measurement)
void init_H_matrix(matrix3x6_t &H) {
    #pragma HLS INLINE
    
    // z = [range, velocity, azimuth] = H * x
    for (int i = 0; i < MEAS_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            H.data[i][j] = cov_elem_t(0.0);
        }
    }
    
    H.data[0][0] = cov_elem_t(1.0);  // range
    H.data[1][1] = cov_elem_t(1.0);  // velocity
    H.data[2][3] = cov_elem_t(1.0);  // azimuth
}

// Kalman predict step
void kalman_predict(
    track_record_t &track,
    const matrix6x6_t &F,
    const matrix6x6_t &Q
) {
    #pragma HLS INLINE off
    
    // x_pred = F * x
    state_vector_t x_pred;
    for (int i = 0; i < STATE_DIM; i++) {
        #pragma HLS UNROLL
        state_t sum = 0;
        for (int j = 0; j < STATE_DIM; j++) {
            #pragma HLS UNROLL
            sum += state_t(F.data[i][j]) * track.x.x[j];
        }
        x_pred.x[i] = sum;
    }
    track.x = x_pred;
    
    // P_pred = F * P * F' + Q
    matrix6x6_t FP, FT, FPFT;
    mat6x6_mult(F, track.P, FP);
    mat6x6_transpose(F, FT);
    mat6x6_mult(FP, FT, FPFT);
    mat6x6_add(FPFT, Q, track.P);
}

// Kalman update step
void kalman_update(
    track_record_t &track,
    const measurement_t &meas,
    const matrix3x6_t &H,
    const matrix3x3_t &R
) {
    #pragma HLS INLINE off
    
    // Innovation: y = z - H*x
    state_t y[MEAS_DIM];
    for (int i = 0; i < MEAS_DIM; i++) {
        #pragma HLS UNROLL
        state_t Hx = 0;
        for (int j = 0; j < STATE_DIM; j++) {
            #pragma HLS UNROLL
            Hx += state_t(H.data[i][j]) * track.x.x[j];
        }
        y[i] = meas.z[i] - Hx;
    }
    
    // Innovation covariance: S = H*P*H' + R
    matrix3x3_t S;
    for (int i = 0; i < MEAS_DIM; i++) {
        for (int j = 0; j < MEAS_DIM; j++) {
            #pragma HLS PIPELINE
            cov_elem_t sum = R.data[i][j];
            for (int k = 0; k < STATE_DIM; k++) {
                for (int l = 0; l < STATE_DIM; l++) {
                    sum += H.data[i][k] * track.P.data[k][l] * H.data[j][l];
                }
            }
            S.data[i][j] = sum;
        }
    }
    
    // S inverse
    matrix3x3_t S_inv;
    mat3x3_inverse(S, S_inv);
    
    // Kalman gain: K = P*H'*S^-1
    matrix6x3_t K;
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < MEAS_DIM; j++) {
            #pragma HLS PIPELINE
            cov_elem_t sum = 0;
            for (int k = 0; k < STATE_DIM; k++) {
                for (int l = 0; l < MEAS_DIM; l++) {
                    sum += track.P.data[i][k] * H.data[l][k] * S_inv.data[l][j];
                }
            }
            K.data[i][j] = sum;
        }
    }
    
    // State update: x = x + K*y
    for (int i = 0; i < STATE_DIM; i++) {
        #pragma HLS UNROLL
        state_t Ky = 0;
        for (int j = 0; j < MEAS_DIM; j++) {
            #pragma HLS UNROLL
            Ky += state_t(K.data[i][j]) * y[j];
        }
        track.x.x[i] += Ky;
    }
    
    // Covariance update: P = (I - K*H)*P
    matrix6x6_t KH, I_KH;
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            #pragma HLS UNROLL
            cov_elem_t kh = 0;
            for (int k = 0; k < MEAS_DIM; k++) {
                kh += K.data[i][k] * H.data[k][j];
            }
            I_KH.data[i][j] = ((i == j) ? cov_elem_t(1.0) : cov_elem_t(0.0)) - kh;
        }
    }
    
    matrix6x6_t P_new;
    mat6x6_mult(I_KH, track.P, P_new);
    track.P = P_new;
}

//=============================================================================
// Gating and Association
//=============================================================================

cov_elem_t compute_mahalanobis_distance(
    const track_record_t &track,
    const measurement_t &meas,
    const matrix3x6_t &H,
    const matrix3x3_t &R
) {
    #pragma HLS INLINE
    
    // Innovation: y = z - H*x
    state_t y[MEAS_DIM];
    for (int i = 0; i < MEAS_DIM; i++) {
        #pragma HLS UNROLL
        state_t Hx = 0;
        for (int j = 0; j < STATE_DIM; j++) {
            Hx += state_t(H.data[i][j]) * track.x.x[j];
        }
        y[i] = meas.z[i] - Hx;
    }
    
    // Simplified distance (just Euclidean for now)
    // Full Mahalanobis would use S^-1
    cov_elem_t dist = 0;
    dist += cov_elem_t(y[0] * y[0]) / cov_elem_t(1000.0);   // Range (meters)
    dist += cov_elem_t(y[1] * y[1]) / cov_elem_t(100.0);    // Velocity
    dist += cov_elem_t(y[2] * y[2]) / cov_elem_t(10.0);     // Azimuth
    
    return dist;
}

//=============================================================================
// Track Management
//=============================================================================

void init_track(
    track_record_t &track,
    ap_uint<8> id,
    const measurement_t &meas
) {
    #pragma HLS INLINE
    
    track.id = id;
    track.state = TRACK_TENTATIVE;
    track.hits = 1;
    track.misses = 0;
    track.coast_count = 0;
    track.age = 0;
    track.last_amplitude = meas.amplitude;
    track.quality = 128;
    track.valid = 1;
    
    // Initialize state from measurement
    track.x.x[0] = meas.z[0];  // range
    track.x.x[1] = meas.z[1];  // velocity
    track.x.x[2] = state_t(0); // acceleration
    track.x.x[3] = meas.z[2];  // azimuth
    track.x.x[4] = state_t(0); // azimuth rate
    track.x.x[5] = state_t(0); // azimuth accel
    
    // Initialize covariance (large uncertainty)
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            track.P.data[i][j] = (i == j) ? cov_elem_t(1000.0) : cov_elem_t(0.0);
        }
    }
}

void update_track_status(
    track_record_t &track,
    bool associated,
    ap_uint<4> confirm_hits,
    ap_uint<4> delete_misses
) {
    #pragma HLS INLINE
    
    track.age++;
    
    if (associated) {
        track.hits++;
        track.misses = 0;
        track.coast_count = 0;
        
        // Update quality
        if (track.quality < 250) track.quality += 5;
        
        // State transition
        if (track.state == TRACK_TENTATIVE && track.hits >= confirm_hits) {
            track.state = TRACK_CONFIRMED;
        }
        else if (track.state == TRACK_COASTING) {
            track.state = TRACK_CONFIRMED;
        }
    }
    else {
        track.misses++;
        track.coast_count++;
        
        // Reduce quality
        if (track.quality > 10) track.quality -= 10;
        
        // State transition
        if (track.state == TRACK_CONFIRMED && track.misses >= 2) {
            track.state = TRACK_COASTING;
        }
        if (track.misses >= delete_misses || track.coast_count >= DEFAULT_COAST_LIMIT) {
            track.state = TRACK_FREE;
            track.valid = 0;
        }
    }
}

//=============================================================================
// Main Track Processor
//=============================================================================

void track_processor(
    // Input: CFAR detections
    hls::stream<detection_t> &s_axis_detections,
    
    // Output: Track reports
    hls::stream<track_t> &m_axis_tracks,
    
    // Configuration
    ap_uint<8> max_tracks,
    state_t gate_range,
    state_t gate_velocity,
    ap_uint<4> confirm_hits,
    ap_uint<4> delete_misses,
    cov_elem_t dt,                    // Update interval (seconds)
    ap_uint<1> enable,
    
    // Status
    ap_uint<1> *busy,
    ap_uint<8> *num_tracks,
    ap_uint<16> *total_detections
) {
    #pragma HLS INTERFACE axis port=s_axis_detections
    #pragma HLS INTERFACE axis port=m_axis_tracks
    #pragma HLS INTERFACE s_axilite port=max_tracks bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_range bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_velocity bundle=control
    #pragma HLS INTERFACE s_axilite port=confirm_hits bundle=control
    #pragma HLS INTERFACE s_axilite port=delete_misses bundle=control
    #pragma HLS INTERFACE s_axilite port=dt bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=num_tracks bundle=control
    #pragma HLS INTERFACE s_axilite port=total_detections bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Track database
    static track_record_t tracks[MAX_TRACKS];
    #pragma HLS BIND_STORAGE variable=tracks type=RAM_2P impl=BRAM
    
    // Detection buffer
    static measurement_t detections[MAX_DETECTIONS];
    #pragma HLS BIND_STORAGE variable=detections type=RAM_2P impl=BRAM
    
    // Association matrix
    static bool associated_det[MAX_DETECTIONS];
    static bool associated_track[MAX_TRACKS];
    
    if (!enable) {
        *busy = 0;
        return;
    }
    
    *busy = 1;
    
    // Initialize matrices
    matrix6x6_t F, Q;
    matrix3x6_t H;
    matrix3x3_t R;
    
    init_F_matrix(F, dt);
    init_Q_matrix(Q, dt, cov_elem_t(5.0), cov_elem_t(0.1));  // 5 m/s^2, 0.1 rad/s^2
    init_H_matrix(H);
    
    // Measurement noise
    for (int i = 0; i < MEAS_DIM; i++) {
        for (int j = 0; j < MEAS_DIM; j++) {
            R.data[i][j] = cov_elem_t(0.0);
        }
    }
    R.data[0][0] = cov_elem_t(100.0);   // Range variance (10m std)
    R.data[1][1] = cov_elem_t(4.0);     // Velocity variance (2m/s std)
    R.data[2][2] = cov_elem_t(0.01);    // Azimuth variance (~5 deg std)
    
    // Read detections
    ap_uint<16> num_det = 0;
    read_det: while (!s_axis_detections.empty() && num_det < MAX_DETECTIONS) {
        #pragma HLS PIPELINE II=1
        
        detection_t det = s_axis_detections.read();
        if (det.valid) {
            // Convert detection to measurement
            detections[num_det].z[0] = state_t(det.range_bin * 15);  // 15m resolution
            detections[num_det].z[1] = state_t(det.doppler_bin - 512);  // Center at 0
            detections[num_det].z[2] = state_t(0);  // Azimuth from beamformer
            detections[num_det].amplitude = det.amplitude;
            detections[num_det].valid = 1;
            associated_det[num_det] = false;
            num_det++;
        }
    }
    
    *total_detections = num_det;
    
    // Initialize association flags
    for (int t = 0; t < MAX_TRACKS; t++) {
        #pragma HLS UNROLL factor=16
        associated_track[t] = false;
    }
    
    // Predict all tracks
    predict: for (int t = 0; t < max_tracks; t++) {
        #pragma HLS PIPELINE II=4
        
        if (tracks[t].valid && tracks[t].state != TRACK_FREE) {
            kalman_predict(tracks[t], F, Q);
        }
    }
    
    // Data association (nearest neighbor)
    associate: for (int t = 0; t < max_tracks; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=256
        
        if (!tracks[t].valid || tracks[t].state == TRACK_FREE) continue;
        
        cov_elem_t best_dist = cov_elem_t(1e10);
        int best_det = -1;
        
        find_best: for (int d = 0; d < num_det; d++) {
            #pragma HLS PIPELINE II=2
            
            if (associated_det[d]) continue;
            
            cov_elem_t dist = compute_mahalanobis_distance(
                tracks[t], detections[d], H, R);
            
            // Gating
            cov_elem_t gate_threshold = cov_elem_t(16.0);  // Chi-squared threshold
            if (dist < gate_threshold && dist < best_dist) {
                best_dist = dist;
                best_det = d;
            }
        }
        
        if (best_det >= 0) {
            // Update track with associated detection
            kalman_update(tracks[t], detections[best_det], H, R);
            tracks[t].last_amplitude = detections[best_det].amplitude;
            associated_det[best_det] = true;
            associated_track[t] = true;
        }
    }
    
    // Update track states
    update_tracks: for (int t = 0; t < max_tracks; t++) {
        #pragma HLS PIPELINE II=1
        
        if (tracks[t].valid) {
            update_track_status(tracks[t], associated_track[t], 
                               confirm_hits, delete_misses);
        }
    }
    
    // Initialize new tracks from unassociated detections
    ap_uint<8> next_id = 0;
    for (int t = 0; t < max_tracks; t++) {
        if (!tracks[t].valid || tracks[t].state == TRACK_FREE) {
            next_id = t;
            break;
        }
    }
    
    init_new: for (int d = 0; d < num_det; d++) {
        #pragma HLS PIPELINE II=2
        
        if (!associated_det[d] && next_id < max_tracks) {
            // Find free slot
            for (int t = next_id; t < max_tracks; t++) {
                if (!tracks[t].valid || tracks[t].state == TRACK_FREE) {
                    init_track(tracks[t], t, detections[d]);
                    next_id = t + 1;
                    break;
                }
            }
        }
    }
    
    // Output confirmed tracks
    ap_uint<8> track_count = 0;
    output: for (int t = 0; t < max_tracks; t++) {
        #pragma HLS PIPELINE II=1
        
        if (tracks[t].valid && 
            (tracks[t].state == TRACK_CONFIRMED || tracks[t].state == TRACK_COASTING)) {
            
            track_t out;
            out.track_id = tracks[t].id;
            out.range = range_t(tracks[t].x.x[0]);
            out.velocity = velocity_t(tracks[t].x.x[1]);
            out.acceleration = velocity_t(tracks[t].x.x[2]);
            out.azimuth = angle_t(tracks[t].x.x[3]);
            out.azimuth_rate = angle_t(tracks[t].x.x[4]);
            out.amplitude = tracks[t].last_amplitude;
            out.quality = tracks[t].quality;
            out.hits = tracks[t].hits;
            out.misses = tracks[t].misses;
            out.age_frames = tracks[t].age;
            out.state = tracks[t].state;
            out.valid = 1;
            
            m_axis_tracks.write(out);
            track_count++;
        }
    }
    
    *num_tracks = track_count;
    *busy = 0;
}

//=============================================================================
// Top-Level Wrapper
//=============================================================================

void titan_track_processor(
    hls::stream<detection_t> &s_axis_detections,
    hls::stream<track_t> &m_axis_tracks,
    ap_uint<8> max_tracks,
    state_t gate_range,
    state_t gate_velocity,
    ap_uint<4> confirm_hits,
    ap_uint<4> delete_misses,
    cov_elem_t dt,
    ap_uint<1> enable,
    ap_uint<1> *busy,
    ap_uint<8> *num_tracks,
    ap_uint<16> *total_detections
) {
    #pragma HLS INTERFACE axis port=s_axis_detections
    #pragma HLS INTERFACE axis port=m_axis_tracks
    #pragma HLS INTERFACE s_axilite port=max_tracks bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_range bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_velocity bundle=control
    #pragma HLS INTERFACE s_axilite port=confirm_hits bundle=control
    #pragma HLS INTERFACE s_axilite port=delete_misses bundle=control
    #pragma HLS INTERFACE s_axilite port=dt bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=busy bundle=control
    #pragma HLS INTERFACE s_axilite port=num_tracks bundle=control
    #pragma HLS INTERFACE s_axilite port=total_detections bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    track_processor(
        s_axis_detections,
        m_axis_tracks,
        max_tracks,
        gate_range,
        gate_velocity,
        confirm_hits,
        delete_misses,
        dt,
        enable,
        busy,
        num_tracks,
        total_detections
    );
}
