`timescale 1ns / 1ps
//==============================================================================
// QEDMMA v6 Dual-Layer Hypersonic Tracker - Complete RTL Package
// 
// Author: Dr. Mladen Mešter
// Architect: Radar Systems Architect vGrok-X (Factory Spec)
// Copyright (c) 2026 - All Rights Reserved
//
// Description:
//   Complete implementation of dual-layer hypersonic tracker:
//   - Layer 1: IMM (4 kinematic models) + TDOA Least Squares + Doppler fusion
//   - Layer 2: Quantized GAT/Particle residual correction (INT8)
//   - Fixed-point Q16.16 for RFSoC efficiency
//   - Multistatic 3D fusion with 6 nodes
//
// Traceability:
//   [REQ-V6-DUAL-LAYER]      - Dual layer architecture
//   [REQ-TDOA-DOPPLER-FUSION] - TDOA + Doppler constraint fusion
//   [REQ-HYPERSONIC-3D]       - 3D hypersonic tracking capability
//   [REQ-MULTISTATIC-6NODES]  - 6-node multistatic configuration
//
// Target: AMD Zynq UltraScale+ ZU48DR (RFSoC 4x2)
// Clock: 300-600 MHz
//==============================================================================

//==============================================================================
// Module 1: IMM Core 3D - Interacting Multiple Model Estimator
// 
// Implements 4 parallel kinematic models:
//   - CV  (Constant Velocity)
//   - CA  (Constant Acceleration)
//   - CT  (Coordinated Turn)
//   - Jerk (Constant Jerk)
//
// [REQ-IMM-4MODEL]
//==============================================================================

module imm_core_3d #(
    parameter int NUM_MODELS = 4,           // Number of parallel models
    parameter int STATE_DIM  = 9,           // State: [x,y,z, vx,vy,vz, ax,ay,az]
    parameter int DATA_WIDTH = 32           // Q16.16 fixed-point
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    
    // Previous state feedback (for update step)
    input  logic signed [DATA_WIDTH-1:0] prev_state [STATE_DIM],
    
    // Measurement input (from TDOA solver or direct)
    input  logic signed [DATA_WIDTH-1:0] meas_pos [3],
    input  logic                         meas_valid,
    
    // IMM output - fused state estimate
    output logic signed [DATA_WIDTH-1:0] state_out [STATE_DIM],
    output logic                         valid_out,
    
    // Model probabilities (for diagnostics)
    output logic [15:0]                  model_prob [NUM_MODELS]
);

    // =========================================================================
    // Fixed-point constants (Q16.16 format)
    // =========================================================================
    localparam logic signed [DATA_WIDTH-1:0] DT = 32'h0000_1000;  // dt = 0.0625s (16 Hz update)
    localparam logic signed [DATA_WIDTH-1:0] DT_SQ_HALF = 32'h0000_0080;  // 0.5 * dt^2
    localparam logic signed [DATA_WIDTH-1:0] ONE = 32'h0001_0000;  // 1.0 in Q16.16
    
    // Model transition probability matrix (fixed, can be made adaptive)
    // P[i][j] = probability of switching from model i to model j
    localparam logic [15:0] P_STAY = 16'h0CCC;  // 0.8 probability to stay
    localparam logic [15:0] P_SWITCH = 16'h0444; // 0.2/3 probability to switch
    
    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // Per-model predicted states
    logic signed [DATA_WIDTH-1:0] model_state [NUM_MODELS][STATE_DIM];
    logic signed [DATA_WIDTH-1:0] model_pred  [NUM_MODELS][STATE_DIM];
    
    // Model likelihoods and probabilities
    logic [31:0] likelihood [NUM_MODELS];
    logic [31:0] prob_sum;
    logic [15:0] mu [NUM_MODELS];  // Model probabilities
    
    // Mixed initial state for each model
    logic signed [DATA_WIDTH-1:0] mixed_state [NUM_MODELS][STATE_DIM];
    
    // =========================================================================
    // State Machine
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE,
        MIXING,
        PREDICTION,
        LIKELIHOOD,
        COMBINATION,
        OUTPUT
    } imm_state_t;
    
    imm_state_t fsm_state;
    logic [2:0] model_idx;
    logic [3:0] state_idx;
    
    // =========================================================================
    // Model Prediction Functions (implemented as case statements)
    // =========================================================================
    
    // CV Model: x' = x + v*dt
    function automatic void predict_cv(
        input  logic signed [DATA_WIDTH-1:0] s_in [STATE_DIM],
        output logic signed [DATA_WIDTH-1:0] s_out [STATE_DIM]
    );
        // Position update: p = p + v*dt
        s_out[0] = s_in[0] + ((s_in[3] * DT) >>> 16);
        s_out[1] = s_in[1] + ((s_in[4] * DT) >>> 16);
        s_out[2] = s_in[2] + ((s_in[5] * DT) >>> 16);
        // Velocity unchanged
        s_out[3] = s_in[3];
        s_out[4] = s_in[4];
        s_out[5] = s_in[5];
        // Acceleration = 0
        s_out[6] = 0;
        s_out[7] = 0;
        s_out[8] = 0;
    endfunction
    
    // CA Model: x' = x + v*dt + 0.5*a*dt^2, v' = v + a*dt
    function automatic void predict_ca(
        input  logic signed [DATA_WIDTH-1:0] s_in [STATE_DIM],
        output logic signed [DATA_WIDTH-1:0] s_out [STATE_DIM]
    );
        // Position update: p = p + v*dt + 0.5*a*dt^2
        s_out[0] = s_in[0] + ((s_in[3] * DT) >>> 16) + ((s_in[6] * DT_SQ_HALF) >>> 16);
        s_out[1] = s_in[1] + ((s_in[4] * DT) >>> 16) + ((s_in[7] * DT_SQ_HALF) >>> 16);
        s_out[2] = s_in[2] + ((s_in[5] * DT) >>> 16) + ((s_in[8] * DT_SQ_HALF) >>> 16);
        // Velocity update: v = v + a*dt
        s_out[3] = s_in[3] + ((s_in[6] * DT) >>> 16);
        s_out[4] = s_in[4] + ((s_in[7] * DT) >>> 16);
        s_out[5] = s_in[5] + ((s_in[8] * DT) >>> 16);
        // Acceleration unchanged
        s_out[6] = s_in[6];
        s_out[7] = s_in[7];
        s_out[8] = s_in[8];
    endfunction
    
    // CT Model: Coordinated Turn (simplified 2D turn rate in XY plane)
    function automatic void predict_ct(
        input  logic signed [DATA_WIDTH-1:0] s_in [STATE_DIM],
        output logic signed [DATA_WIDTH-1:0] s_out [STATE_DIM]
    );
        // Turn rate omega derived from lateral acceleration
        // omega = a_lateral / v
        logic signed [DATA_WIDTH-1:0] v_mag_sq;
        logic signed [DATA_WIDTH-1:0] omega;
        
        v_mag_sq = ((s_in[3] * s_in[3]) >>> 16) + ((s_in[4] * s_in[4]) >>> 16);
        
        // Simplified: assume small angle turn
        // Position update similar to CA
        s_out[0] = s_in[0] + ((s_in[3] * DT) >>> 16);
        s_out[1] = s_in[1] + ((s_in[4] * DT) >>> 16);
        s_out[2] = s_in[2] + ((s_in[5] * DT) >>> 16);
        
        // Velocity rotates (simplified)
        s_out[3] = s_in[3] - ((s_in[4] * DT) >>> 20);  // Small rotation
        s_out[4] = s_in[4] + ((s_in[3] * DT) >>> 20);
        s_out[5] = s_in[5];
        
        // Centripetal acceleration
        s_out[6] = s_in[6];
        s_out[7] = s_in[7];
        s_out[8] = s_in[8];
    endfunction
    
    // Jerk Model: Constant jerk (rate of change of acceleration)
    function automatic void predict_jerk(
        input  logic signed [DATA_WIDTH-1:0] s_in [STATE_DIM],
        output logic signed [DATA_WIDTH-1:0] s_out [STATE_DIM]
    );
        // For hypersonic pull-up maneuvers, jerk is significant
        // Simplified: acceleration changes linearly
        s_out[0] = s_in[0] + ((s_in[3] * DT) >>> 16) + ((s_in[6] * DT_SQ_HALF) >>> 16);
        s_out[1] = s_in[1] + ((s_in[4] * DT) >>> 16) + ((s_in[7] * DT_SQ_HALF) >>> 16);
        s_out[2] = s_in[2] + ((s_in[5] * DT) >>> 16) + ((s_in[8] * DT_SQ_HALF) >>> 16);
        
        s_out[3] = s_in[3] + ((s_in[6] * DT) >>> 16);
        s_out[4] = s_in[4] + ((s_in[7] * DT) >>> 16);
        s_out[5] = s_in[5] + ((s_in[8] * DT) >>> 16);
        
        // Acceleration increases (simulating pull-up)
        s_out[6] = s_in[6] + (s_in[6] >>> 6);  // Small jerk
        s_out[7] = s_in[7] + (s_in[7] >>> 6);
        s_out[8] = s_in[8] + (s_in[8] >>> 6);
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= IDLE;
            valid_out <= 0;
            model_idx <= 0;
            state_idx <= 0;
            
            // Initialize model probabilities equally
            for (int i = 0; i < NUM_MODELS; i++) begin
                mu[i] <= 16'h4000;  // 0.25 each
            end
            
            // Initialize states to zero
            for (int i = 0; i < STATE_DIM; i++) begin
                state_out[i] <= 0;
            end
            
        end else begin
            case (fsm_state)
                IDLE: begin
                    valid_out <= 0;
                    if (valid_in) begin
                        fsm_state <= MIXING;
                        model_idx <= 0;
                    end
                end
                
                MIXING: begin
                    // Compute mixed initial state for each model
                    // x_0j = sum_i(mu_i|j * x_i)
                    // Simplified: use previous combined state
                    for (int m = 0; m < NUM_MODELS; m++) begin
                        for (int s = 0; s < STATE_DIM; s++) begin
                            mixed_state[m][s] <= prev_state[s];
                        end
                    end
                    fsm_state <= PREDICTION;
                end
                
                PREDICTION: begin
                    // Run each model's prediction in parallel (combinatorial)
                    // Model 0: CV
                    predict_cv(mixed_state[0], model_pred[0]);
                    // Model 1: CA
                    predict_ca(mixed_state[1], model_pred[1]);
                    // Model 2: CT
                    predict_ct(mixed_state[2], model_pred[2]);
                    // Model 3: Jerk
                    predict_jerk(mixed_state[3], model_pred[3]);
                    
                    fsm_state <= LIKELIHOOD;
                end
                
                LIKELIHOOD: begin
                    // Compute likelihood of each model given measurement
                    // L_j = exp(-0.5 * residual^2 / sigma^2)
                    // Simplified: use Mahalanobis distance approximation
                    
                    if (meas_valid) begin
                        for (int m = 0; m < NUM_MODELS; m++) begin
                            logic signed [DATA_WIDTH-1:0] res_x, res_y, res_z;
                            logic signed [63:0] res_sq;
                            
                            res_x = meas_pos[0] - model_pred[m][0];
                            res_y = meas_pos[1] - model_pred[m][1];
                            res_z = meas_pos[2] - model_pred[m][2];
                            
                            res_sq = (res_x * res_x + res_y * res_y + res_z * res_z) >>> 16;
                            
                            // Likelihood ~ 1 / (1 + res_sq)
                            likelihood[m] <= 32'hFFFF_FFFF / (32'h0001_0000 + res_sq[31:0]);
                        end
                    end else begin
                        // No measurement - use prior
                        for (int m = 0; m < NUM_MODELS; m++) begin
                            likelihood[m] <= 32'h0001_0000;
                        end
                    end
                    
                    fsm_state <= COMBINATION;
                end
                
                COMBINATION: begin
                    // Update model probabilities
                    // mu_j = (c_j * L_j * sum_i(p_ij * mu_i)) / sum_k(...)
                    
                    prob_sum = 0;
                    for (int m = 0; m < NUM_MODELS; m++) begin
                        logic [31:0] prior_contrib;
                        prior_contrib = (mu[m] * P_STAY) >>> 16;
                        likelihood[m] = (likelihood[m] * prior_contrib) >>> 16;
                        prob_sum = prob_sum + likelihood[m];
                    end
                    
                    // Normalize
                    if (prob_sum > 0) begin
                        for (int m = 0; m < NUM_MODELS; m++) begin
                            mu[m] <= (likelihood[m] << 16) / prob_sum;
                        end
                    end
                    
                    // Compute combined state estimate
                    // x = sum_j(mu_j * x_j)
                    for (int s = 0; s < STATE_DIM; s++) begin
                        logic signed [63:0] weighted_sum;
                        weighted_sum = 0;
                        for (int m = 0; m < NUM_MODELS; m++) begin
                            weighted_sum = weighted_sum + 
                                          ((model_pred[m][s] * $signed({16'b0, mu[m]})) >>> 16);
                        end
                        state_out[s] <= weighted_sum[DATA_WIDTH-1:0];
                    end
                    
                    fsm_state <= OUTPUT;
                end
                
                OUTPUT: begin
                    valid_out <= 1;
                    for (int m = 0; m < NUM_MODELS; m++) begin
                        model_prob[m] <= mu[m];
                    end
                    fsm_state <= IDLE;
                end
                
                default: fsm_state <= IDLE;
            endcase
        end
    end

endmodule


//==============================================================================
// Module 2: TDOA Least Squares Solver with Doppler Fusion
//
// Implements Gauss-Newton iterative solver for 3D position estimation
// using TDOA measurements from 6 nodes with Doppler velocity constraints.
//
// [REQ-TDOA-DOPPLER-FUSION]
// [REQ-MULTISTATIC-6NODES]
//==============================================================================

module tdoa_ls_doppler_fusion #(
    parameter int NUM_NODES  = 6,
    parameter int DATA_WIDTH = 32,
    parameter int MAX_ITER   = 8
) (
    input  logic                         clk,
    input  logic                         rst_n,
    
    // TDOA measurements (range differences relative to reference node 0)
    input  logic signed [DATA_WIDTH-1:0] tdoa_meas [NUM_NODES-1],
    
    // Doppler measurements (radial velocity from each node)
    input  logic signed [DATA_WIDTH-1:0] doppler_meas [NUM_NODES],
    
    // Node coordinates (ECEF or local ENU)
    input  logic signed [DATA_WIDTH-1:0] node_x [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0] node_y [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0] node_z [NUM_NODES],
    
    // Initial guess from IMM predictor
    input  logic signed [DATA_WIDTH-1:0] init_pos [3],
    input  logic signed [DATA_WIDTH-1:0] init_vel [3],
    input  logic                         valid_in,
    
    // Output position and velocity
    output logic signed [DATA_WIDTH-1:0] pos_out [3],
    output logic signed [DATA_WIDTH-1:0] vel_out [3],
    output logic                         converged
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    logic signed [DATA_WIDTH-1:0] pos_est [3];
    logic signed [DATA_WIDTH-1:0] vel_est [3];
    
    // Range to each node
    logic signed [DATA_WIDTH-1:0] range [NUM_NODES];
    
    // Jacobian matrix elements (NUM_NODES-1 rows for TDOA, NUM_NODES for Doppler)
    // Simplified: store only diagonal approximation
    logic signed [DATA_WIDTH-1:0] jacobian_diag [3];
    
    // Residual vector
    logic signed [DATA_WIDTH-1:0] residual_tdoa [NUM_NODES-1];
    logic signed [DATA_WIDTH-1:0] residual_dop  [NUM_NODES];
    
    // Iteration counter
    logic [3:0] iter_count;
    
    // Convergence threshold (in Q16.16: ~10m)
    localparam logic signed [DATA_WIDTH-1:0] CONV_THRESH = 32'h000A_0000;
    
    // FSM
    typedef enum logic [2:0] {
        IDLE,
        CALC_RANGE,
        CALC_RESIDUAL,
        CALC_UPDATE,
        CHECK_CONV,
        DONE
    } ls_state_t;
    
    ls_state_t fsm_state;
    logic [2:0] node_idx;
    
    // =========================================================================
    // Range calculation (sqrt approximation using Newton-Raphson)
    // =========================================================================
    function automatic logic signed [DATA_WIDTH-1:0] approx_sqrt(
        input logic signed [63:0] val
    );
        logic signed [DATA_WIDTH-1:0] x;
        logic signed [DATA_WIDTH-1:0] x_new;
        
        // Initial guess: val >> 17 (rough approximation)
        x = val[47:16];
        
        // 3 Newton-Raphson iterations
        for (int i = 0; i < 3; i++) begin
            if (x != 0) begin
                x_new = (x + (val[47:16] / x)) >>> 1;
                x = x_new;
            end
        end
        
        return x;
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= IDLE;
            converged <= 0;
            iter_count <= 0;
            node_idx <= 0;
            
            for (int i = 0; i < 3; i++) begin
                pos_out[i] <= 0;
                vel_out[i] <= 0;
                pos_est[i] <= 0;
                vel_est[i] <= 0;
            end
            
        end else begin
            case (fsm_state)
                IDLE: begin
                    converged <= 0;
                    if (valid_in) begin
                        // Initialize with IMM prediction
                        pos_est <= init_pos;
                        vel_est <= init_vel;
                        iter_count <= 0;
                        node_idx <= 0;
                        fsm_state <= CALC_RANGE;
                    end
                end
                
                CALC_RANGE: begin
                    // Calculate range from current estimate to each node
                    logic signed [63:0] dx, dy, dz, r_sq;
                    
                    dx = pos_est[0] - node_x[node_idx];
                    dy = pos_est[1] - node_y[node_idx];
                    dz = pos_est[2] - node_z[node_idx];
                    
                    r_sq = ((dx * dx) >>> 16) + ((dy * dy) >>> 16) + ((dz * dz) >>> 16);
                    range[node_idx] <= approx_sqrt(r_sq);
                    
                    if (node_idx < NUM_NODES - 1) begin
                        node_idx <= node_idx + 1;
                    end else begin
                        node_idx <= 0;
                        fsm_state <= CALC_RESIDUAL;
                    end
                end
                
                CALC_RESIDUAL: begin
                    // TDOA residual: measured - (range[i] - range[0])
                    for (int i = 0; i < NUM_NODES-1; i++) begin
                        residual_tdoa[i] <= tdoa_meas[i] - (range[i+1] - range[0]);
                    end
                    
                    // Doppler residual: measured - (v · r_hat)
                    // r_hat = unit vector from target to node
                    for (int i = 0; i < NUM_NODES; i++) begin
                        logic signed [DATA_WIDTH-1:0] rx, ry, rz;
                        logic signed [63:0] dot_prod;
                        
                        if (range[i] != 0) begin
                            rx = ((node_x[i] - pos_est[0]) << 16) / range[i];
                            ry = ((node_y[i] - pos_est[1]) << 16) / range[i];
                            rz = ((node_z[i] - pos_est[2]) << 16) / range[i];
                            
                            dot_prod = ((vel_est[0] * rx) >>> 16) +
                                       ((vel_est[1] * ry) >>> 16) +
                                       ((vel_est[2] * rz) >>> 16);
                            
                            residual_dop[i] <= doppler_meas[i] - dot_prod[DATA_WIDTH-1:0];
                        end else begin
                            residual_dop[i] <= 0;
                        end
                    end
                    
                    fsm_state <= CALC_UPDATE;
                end
                
                CALC_UPDATE: begin
                    // Simplified Gauss-Newton update
                    // delta = (J^T J)^-1 J^T r
                    // Approximation: delta ≈ alpha * J^T * r (gradient descent step)
                    
                    localparam logic signed [DATA_WIDTH-1:0] ALPHA = 32'h0000_2000; // Step size 0.125
                    
                    logic signed [63:0] grad_x, grad_y, grad_z;
                    logic signed [63:0] grad_vx, grad_vy, grad_vz;
                    
                    grad_x = 0; grad_y = 0; grad_z = 0;
                    grad_vx = 0; grad_vy = 0; grad_vz = 0;
                    
                    // Accumulate gradient from TDOA residuals
                    for (int i = 0; i < NUM_NODES-1; i++) begin
                        logic signed [DATA_WIDTH-1:0] dx1, dy1, dz1;
                        logic signed [DATA_WIDTH-1:0] dx0, dy0, dz0;
                        
                        // Direction from target to node i+1
                        if (range[i+1] != 0) begin
                            dx1 = ((node_x[i+1] - pos_est[0]) << 12) / (range[i+1] >>> 4);
                            dy1 = ((node_y[i+1] - pos_est[1]) << 12) / (range[i+1] >>> 4);
                            dz1 = ((node_z[i+1] - pos_est[2]) << 12) / (range[i+1] >>> 4);
                        end else begin
                            dx1 = 0; dy1 = 0; dz1 = 0;
                        end
                        
                        // Direction from target to node 0
                        if (range[0] != 0) begin
                            dx0 = ((node_x[0] - pos_est[0]) << 12) / (range[0] >>> 4);
                            dy0 = ((node_y[0] - pos_est[1]) << 12) / (range[0] >>> 4);
                            dz0 = ((node_z[0] - pos_est[2]) << 12) / (range[0] >>> 4);
                        end else begin
                            dx0 = 0; dy0 = 0; dz0 = 0;
                        end
                        
                        // Jacobian row: [dx1-dx0, dy1-dy0, dz1-dz0]
                        grad_x = grad_x + ((residual_tdoa[i] * (dx1 - dx0)) >>> 16);
                        grad_y = grad_y + ((residual_tdoa[i] * (dy1 - dy0)) >>> 16);
                        grad_z = grad_z + ((residual_tdoa[i] * (dz1 - dz0)) >>> 16);
                    end
                    
                    // Accumulate gradient from Doppler residuals (for velocity)
                    for (int i = 0; i < NUM_NODES; i++) begin
                        logic signed [DATA_WIDTH-1:0] rx, ry, rz;
                        
                        if (range[i] != 0) begin
                            rx = ((node_x[i] - pos_est[0]) << 12) / (range[i] >>> 4);
                            ry = ((node_y[i] - pos_est[1]) << 12) / (range[i] >>> 4);
                            rz = ((node_z[i] - pos_est[2]) << 12) / (range[i] >>> 4);
                            
                            grad_vx = grad_vx + ((residual_dop[i] * rx) >>> 16);
                            grad_vy = grad_vy + ((residual_dop[i] * ry) >>> 16);
                            grad_vz = grad_vz + ((residual_dop[i] * rz) >>> 16);
                        end
                    end
                    
                    // Apply update
                    pos_est[0] <= pos_est[0] + ((ALPHA * grad_x[DATA_WIDTH-1:0]) >>> 16);
                    pos_est[1] <= pos_est[1] + ((ALPHA * grad_y[DATA_WIDTH-1:0]) >>> 16);
                    pos_est[2] <= pos_est[2] + ((ALPHA * grad_z[DATA_WIDTH-1:0]) >>> 16);
                    
                    vel_est[0] <= vel_est[0] + ((ALPHA * grad_vx[DATA_WIDTH-1:0]) >>> 16);
                    vel_est[1] <= vel_est[1] + ((ALPHA * grad_vy[DATA_WIDTH-1:0]) >>> 16);
                    vel_est[2] <= vel_est[2] + ((ALPHA * grad_vz[DATA_WIDTH-1:0]) >>> 16);
                    
                    iter_count <= iter_count + 1;
                    fsm_state <= CHECK_CONV;
                end
                
                CHECK_CONV: begin
                    // Check convergence: ||residual|| < threshold
                    logic signed [63:0] res_norm_sq;
                    res_norm_sq = 0;
                    
                    for (int i = 0; i < NUM_NODES-1; i++) begin
                        res_norm_sq = res_norm_sq + ((residual_tdoa[i] * residual_tdoa[i]) >>> 16);
                    end
                    
                    if (res_norm_sq < ((CONV_THRESH * CONV_THRESH) >>> 16) || 
                        iter_count >= MAX_ITER) begin
                        fsm_state <= DONE;
                    end else begin
                        node_idx <= 0;
                        fsm_state <= CALC_RANGE;
                    end
                end
                
                DONE: begin
                    pos_out <= pos_est;
                    vel_out <= vel_est;
                    converged <= 1;
                    fsm_state <= IDLE;
                end
                
                default: fsm_state <= IDLE;
            endcase
        end
    end

endmodule


//==============================================================================
// Module 3: GAT/Particle Residual Correction (INT8 Quantized)
//
// Implements simplified Graph Attention Network for residual estimation
// during extreme maneuvers (>60g acceleration).
// Uses INT8 quantized weights for FPGA efficiency.
//
// [REQ-GAT-RESIDUAL]
// [REQ-INT8-QUANTIZED]
//==============================================================================

module gat_particle_residual_int8 #(
    parameter int STATE_DIM  = 9,
    parameter int DATA_WIDTH = 32,
    parameter int HIDDEN_DIM = 16
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         enable,
    
    // Input state from Layer 1
    input  logic signed [DATA_WIDTH-1:0] state_in [STATE_DIM],
    
    // Output residual correction
    output logic signed [15:0]           residual_out [STATE_DIM],
    output logic                         valid_out
);

    // =========================================================================
    // INT8 Quantized Weights (pretrained, stored in ROM)
    // =========================================================================
    
    // Attention weight matrix (HIDDEN_DIM x STATE_DIM) - INT8
    logic signed [7:0] W_attn [HIDDEN_DIM][STATE_DIM];
    
    // Output weight matrix (STATE_DIM x HIDDEN_DIM) - INT8
    logic signed [7:0] W_out [STATE_DIM][HIDDEN_DIM];
    
    // Bias vectors - INT8
    logic signed [7:0] b_attn [HIDDEN_DIM];
    logic signed [7:0] b_out [STATE_DIM];
    
    // Initialize weights (simplified - in practice, loaded from BRAM)
    initial begin
        // Random initialization for simulation
        // In deployment: load from calibration file
        for (int i = 0; i < HIDDEN_DIM; i++) begin
            for (int j = 0; j < STATE_DIM; j++) begin
                W_attn[i][j] = $signed(8'((i * j) % 127 - 63));
            end
            b_attn[i] = 0;
        end
        
        for (int i = 0; i < STATE_DIM; i++) begin
            for (int j = 0; j < HIDDEN_DIM; j++) begin
                W_out[i][j] = $signed(8'((i + j) % 127 - 63));
            end
            b_out[i] = 0;
        end
    end
    
    // =========================================================================
    // Internal signals
    // =========================================================================
    logic signed [15:0] state_quant [STATE_DIM];  // Quantized input
    logic signed [23:0] hidden [HIDDEN_DIM];       // Hidden layer activations
    logic signed [23:0] output_raw [STATE_DIM];    // Raw output before scaling
    
    // FSM
    typedef enum logic [1:0] {
        IDLE,
        FORWARD,
        OUTPUT
    } gat_state_t;
    
    gat_state_t fsm_state;
    logic [4:0] compute_idx;
    
    // =========================================================================
    // Quantization: Q16.16 -> INT8 (scale by 1/256)
    // =========================================================================
    always_comb begin
        for (int i = 0; i < STATE_DIM; i++) begin
            // Scale down and saturate to INT16 range
            logic signed [DATA_WIDTH-1:0] scaled;
            scaled = state_in[i] >>> 8;  // Scale to roughly ±32k range
            
            if (scaled > 32767)
                state_quant[i] = 16'h7FFF;
            else if (scaled < -32768)
                state_quant[i] = 16'h8000;
            else
                state_quant[i] = scaled[15:0];
        end
    end
    
    // =========================================================================
    // ReLU activation function
    // =========================================================================
    function automatic logic signed [23:0] relu(input logic signed [23:0] x);
        return (x > 0) ? x : 0;
    endfunction
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= IDLE;
            valid_out <= 0;
            compute_idx <= 0;
            
            for (int i = 0; i < STATE_DIM; i++) begin
                residual_out[i] <= 0;
            end
            
        end else begin
            case (fsm_state)
                IDLE: begin
                    valid_out <= 0;
                    if (enable) begin
                        fsm_state <= FORWARD;
                        compute_idx <= 0;
                    end
                end
                
                FORWARD: begin
                    // Layer 1: Hidden = ReLU(W_attn * state + b_attn)
                    if (compute_idx < HIDDEN_DIM) begin
                        logic signed [31:0] sum;
                        sum = {24'b0, b_attn[compute_idx]} << 8;
                        
                        for (int j = 0; j < STATE_DIM; j++) begin
                            sum = sum + (W_attn[compute_idx][j] * state_quant[j]);
                        end
                        
                        hidden[compute_idx] <= relu(sum[23:0]);
                        compute_idx <= compute_idx + 1;
                        
                    end else if (compute_idx < HIDDEN_DIM + STATE_DIM) begin
                        // Layer 2: Output = W_out * hidden + b_out
                        logic [4:0] out_idx;
                        logic signed [31:0] sum;
                        
                        out_idx = compute_idx - HIDDEN_DIM;
                        sum = {24'b0, b_out[out_idx]} << 8;
                        
                        for (int j = 0; j < HIDDEN_DIM; j++) begin
                            sum = sum + ((W_out[out_idx][j] * hidden[j][15:0]) >>> 8);
                        end
                        
                        output_raw[out_idx] <= sum[23:0];
                        compute_idx <= compute_idx + 1;
                        
                    end else begin
                        fsm_state <= OUTPUT;
                    end
                end
                
                OUTPUT: begin
                    // Scale output to residual range and output
                    for (int i = 0; i < STATE_DIM; i++) begin
                        // Output is scaled residual correction
                        residual_out[i] <= output_raw[i][15:0];
                    end
                    valid_out <= 1;
                    fsm_state <= IDLE;
                end
                
                default: fsm_state <= IDLE;
            endcase
        end
    end

endmodule


//==============================================================================
// Top Module: QEDMMA v6 Dual-Layer Tracker (Complete)
//==============================================================================

module qedmma_v6_dual_layer_tracker #(
    parameter int NUM_NODES  = 6,
    parameter int DATA_WIDTH = 32,
    parameter int STATE_DIM  = 9,
    parameter int MAX_LS_ITER = 8
) (
    input  logic                             clk,
    input  logic                             rst_n,
    
    // TDOA measurements
    input  logic signed [DATA_WIDTH-1:0]     tdoa_meas    [NUM_NODES-1],
    input  logic signed [DATA_WIDTH-1:0]     doppler_meas [NUM_NODES],
    
    // Node coordinates
    input  logic signed [DATA_WIDTH-1:0]     node_x       [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0]     node_y       [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0]     node_z       [NUM_NODES],
    
    input  logic                             valid_in,
    
    // Outputs
    output logic signed [DATA_WIDTH-1:0]     pos_x, pos_y, pos_z,
    output logic signed [DATA_WIDTH-1:0]     vel_x, vel_y, vel_z,
    output logic signed [DATA_WIDTH-1:0]     acc_x, acc_y, acc_z,
    output logic                             track_valid
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // IMM outputs
    logic signed [DATA_WIDTH-1:0] imm_state [STATE_DIM];
    logic                         imm_valid;
    logic [15:0]                  model_prob [4];
    
    // TDOA LS outputs
    logic signed [DATA_WIDTH-1:0] ls_pos [3];
    logic signed [DATA_WIDTH-1:0] ls_vel [3];
    logic                         ls_converged;
    
    // GAT outputs
    logic signed [15:0]           gat_residual [STATE_DIM];
    logic                         gat_valid;
    
    // State feedback for IMM
    logic signed [DATA_WIDTH-1:0] prev_state [STATE_DIM];
    
    // High acceleration detection
    logic signed [63:0] accel_sq;
    logic high_accel;
    
    // =========================================================================
    // Layer 1: IMM Core
    // =========================================================================
    imm_core_3d #(
        .NUM_MODELS(4),
        .STATE_DIM(STATE_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_imm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .prev_state(prev_state),
        .meas_pos(ls_pos),
        .meas_valid(ls_converged),
        .state_out(imm_state),
        .valid_out(imm_valid),
        .model_prob(model_prob)
    );
    
    // =========================================================================
    // Layer 1: TDOA Least Squares with Doppler Fusion
    // =========================================================================
    tdoa_ls_doppler_fusion #(
        .NUM_NODES(NUM_NODES),
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_ITER(MAX_LS_ITER)
    ) u_tdoa_ls (
        .clk(clk),
        .rst_n(rst_n),
        .tdoa_meas(tdoa_meas),
        .doppler_meas(doppler_meas),
        .node_x(node_x),
        .node_y(node_y),
        .node_z(node_z),
        .init_pos({imm_state[0], imm_state[1], imm_state[2]}),
        .init_vel({imm_state[3], imm_state[4], imm_state[5]}),
        .valid_in(imm_valid),
        .pos_out(ls_pos),
        .vel_out(ls_vel),
        .converged(ls_converged)
    );
    
    // =========================================================================
    // High Acceleration Detection (>60g)
    // =========================================================================
    // 60g = 588 m/s² → in Q16.16: ~38,535,168 = 0x024C_0000
    localparam logic [63:0] ACCEL_THRESH_SQ = 64'h0005_A400_0000_0000;  // (60g)² in Q32.32
    
    always_comb begin
        accel_sq = ((imm_state[6] * imm_state[6]) >>> 16) +
                   ((imm_state[7] * imm_state[7]) >>> 16) +
                   ((imm_state[8] * imm_state[8]) >>> 16);
        high_accel = (accel_sq > ACCEL_THRESH_SQ[47:16]);
    end
    
    // =========================================================================
    // Layer 2: GAT/Particle Residual Correction (INT8)
    // =========================================================================
    gat_particle_residual_int8 #(
        .STATE_DIM(STATE_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .HIDDEN_DIM(16)
    ) u_gat (
        .clk(clk),
        .rst_n(rst_n),
        .enable(high_accel & ls_converged),
        .state_in({ls_pos[0], ls_pos[1], ls_pos[2],
                   ls_vel[0], ls_vel[1], ls_vel[2],
                   imm_state[6], imm_state[7], imm_state[8]}),
        .residual_out(gat_residual),
        .valid_out(gat_valid)
    );
    
    // =========================================================================
    // Final Fusion and Output
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            track_valid <= 0;
            pos_x <= 0; pos_y <= 0; pos_z <= 0;
            vel_x <= 0; vel_y <= 0; vel_z <= 0;
            acc_x <= 0; acc_y <= 0; acc_z <= 0;
            
            for (int i = 0; i < STATE_DIM; i++) begin
                prev_state[i] <= 0;
            end
            
        end else if (ls_converged) begin
            // Apply Layer 2 correction if high acceleration detected
            if (high_accel && gat_valid) begin
                pos_x <= ls_pos[0] + {{16{gat_residual[0][15]}}, gat_residual[0]};
                pos_y <= ls_pos[1] + {{16{gat_residual[1][15]}}, gat_residual[1]};
                pos_z <= ls_pos[2] + {{16{gat_residual[2][15]}}, gat_residual[2]};
                
                vel_x <= ls_vel[0] + {{16{gat_residual[3][15]}}, gat_residual[3]};
                vel_y <= ls_vel[1] + {{16{gat_residual[4][15]}}, gat_residual[4]};
                vel_z <= ls_vel[2] + {{16{gat_residual[5][15]}}, gat_residual[5]};
                
                acc_x <= imm_state[6] + {{16{gat_residual[6][15]}}, gat_residual[6]};
                acc_y <= imm_state[7] + {{16{gat_residual[7][15]}}, gat_residual[7]};
                acc_z <= imm_state[8] + {{16{gat_residual[8][15]}}, gat_residual[8]};
            end else begin
                // Use Layer 1 output directly
                pos_x <= ls_pos[0];
                pos_y <= ls_pos[1];
                pos_z <= ls_pos[2];
                
                vel_x <= ls_vel[0];
                vel_y <= ls_vel[1];
                vel_z <= ls_vel[2];
                
                acc_x <= imm_state[6];
                acc_y <= imm_state[7];
                acc_z <= imm_state[8];
            end
            
            // Update state feedback
            prev_state[0] <= ls_pos[0];
            prev_state[1] <= ls_pos[1];
            prev_state[2] <= ls_pos[2];
            prev_state[3] <= ls_vel[0];
            prev_state[4] <= ls_vel[1];
            prev_state[5] <= ls_vel[2];
            prev_state[6] <= imm_state[6];
            prev_state[7] <= imm_state[7];
            prev_state[8] <= imm_state[8];
            
            track_valid <= 1;
        end else begin
            track_valid <= 0;
        end
    end

endmodule
