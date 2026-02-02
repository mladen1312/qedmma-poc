`timescale 1ns / 1ps
// ==============================================================================
// QEDMMA v6 - TDOA Least Squares Solver with Doppler Fusion
// 
// Description:
//   Iterative Gauss-Newton solver for 3D localization using:
//   - TDOA (Time Difference of Arrival) from 6 distributed nodes
//   - Doppler velocity constraints from each node
//   
//   The TDOA equations are:
//     Δt_ij × c = ||r - s_i|| - ||r - s_j||
//   
//   Doppler provides radial velocity constraint:
//     v_r,i = v · (r - s_i) / ||r - s_i||
//
//   Gauss-Newton iteration:
//     δx = (J'WJ)^-1 × J'W × residual
//
// Traceability:
//   [REQ-TDOA-DOPPLER-FUSION] Combined position + velocity estimation
//   [REQ-MULTISTATIC-6NODES]  6-node baseline ≥600 km for hypersonic
//   [REQ-GAUSS-NEWTON]        Iterative refinement, max 8 iterations
//
// Author: Dr. Mladen Mešter / Radar Systems Architect vGrok-X
// Copyright (c) 2026 - All Rights Reserved
// ==============================================================================

module tdoa_ls_doppler_fusion #(
    parameter int NUM_NODES   = 6,          // Number of multistatic receivers
    parameter int DATA_WIDTH  = 32,         // Q16.16 fixed-point
    parameter int FRAC_BITS   = 16,         // Fractional bits
    parameter int MAX_ITER    = 8,          // Maximum Gauss-Newton iterations
    parameter int CONV_THRESH = 32'h0000_0100  // Convergence: ~0.004 (meters)
) (
    input  logic                             clk,
    input  logic                             rst_n,
    
    // TDOA measurements (relative to node 0 as reference)
    input  logic signed [DATA_WIDTH-1:0]     tdoa_meas [NUM_NODES-1],  // Δd = Δt × c (meters)
    
    // Doppler measurements (radial velocity from each node)
    input  logic signed [DATA_WIDTH-1:0]     doppler_meas [NUM_NODES],  // v_radial (m/s)
    
    // Node positions (calibrated, fixed)
    input  logic signed [DATA_WIDTH-1:0]     node_x [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0]     node_y [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0]     node_z [NUM_NODES],
    
    // Initial guess from IMM predictor
    input  logic signed [DATA_WIDTH-1:0]     init_pos [3],
    input  logic signed [DATA_WIDTH-1:0]     init_vel [3],
    
    input  logic                             valid_in,
    
    // Output: Refined position and velocity
    output logic signed [DATA_WIDTH-1:0]     pos_out [3],
    output logic signed [DATA_WIDTH-1:0]     vel_out [3],
    output logic                             converged,
    output logic                             valid_out
);

    // =========================================================================
    // Internal State
    // =========================================================================
    
    // Current estimates
    logic signed [DATA_WIDTH-1:0] pos_est [3];
    logic signed [DATA_WIDTH-1:0] vel_est [3];
    
    // Range from target to each node
    logic signed [DATA_WIDTH-1:0] range [NUM_NODES];
    
    // Unit vectors (target → node)
    logic signed [DATA_WIDTH-1:0] unit_x [NUM_NODES];
    logic signed [DATA_WIDTH-1:0] unit_y [NUM_NODES];
    logic signed [DATA_WIDTH-1:0] unit_z [NUM_NODES];
    
    // TDOA residuals
    logic signed [DATA_WIDTH-1:0] res_tdoa [NUM_NODES-1];
    
    // Doppler residuals
    logic signed [DATA_WIDTH-1:0] res_dopp [NUM_NODES];
    
    // Jacobian matrices (TDOA and Doppler)
    logic signed [DATA_WIDTH-1:0] J_tdoa [NUM_NODES-1][3];
    logic signed [DATA_WIDTH-1:0] J_dopp [NUM_NODES][3];
    
    // Normal equation matrices
    logic signed [DATA_WIDTH-1:0] JtWJ [3][3];  // 3×3 (position only)
    logic signed [DATA_WIDTH-1:0] JtWr [3];     // 3×1
    
    // Position update
    logic signed [DATA_WIDTH-1:0] delta_pos [3];
    
    // Iteration counter
    logic [3:0] iter_count;
    logic iteration_done;
    
    // =========================================================================
    // FSM
    // =========================================================================
    
    typedef enum logic [3:0] {
        IDLE,
        INIT,
        CALC_RANGE,
        CALC_UNIT,
        CALC_RESIDUAL,
        CALC_JACOBIAN,
        FORM_NORMAL,
        SOLVE_DELTA,
        UPDATE_EST,
        CHECK_CONV,
        REFINE_VEL,
        OUTPUT
    } state_t;
    
    state_t fsm_state, fsm_next;
    
    // =========================================================================
    // Node index for sequential processing
    // =========================================================================
    
    logic [2:0] node_idx;
    
    // =========================================================================
    // FSM: Sequential Logic
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= IDLE;
            valid_out <= 0;
            converged <= 0;
            iter_count <= 0;
            node_idx <= 0;
            
            for (int i = 0; i < 3; i++) begin
                pos_est[i] <= 0;
                vel_est[i] <= 0;
                pos_out[i] <= 0;
                vel_out[i] <= 0;
            end
            
        end else begin
            fsm_state <= fsm_next;
            
            case (fsm_state)
                // ---------------------------------------------------------
                IDLE: begin
                    valid_out <= 0;
                    converged <= 0;
                    if (valid_in) begin
                        iter_count <= 0;
                        node_idx <= 0;
                    end
                end
                
                // ---------------------------------------------------------
                INIT: begin
                    // Initialize with IMM prediction
                    pos_est[0] <= init_pos[0];
                    pos_est[1] <= init_pos[1];
                    pos_est[2] <= init_pos[2];
                    vel_est[0] <= init_vel[0];
                    vel_est[1] <= init_vel[1];
                    vel_est[2] <= init_vel[2];
                    node_idx <= 0;
                end
                
                // ---------------------------------------------------------
                CALC_RANGE: begin
                    // Calculate ||r - s_i|| for current node
                    logic signed [DATA_WIDTH-1:0] dx, dy, dz;
                    logic signed [2*DATA_WIDTH-1:0] dist_sq;
                    
                    dx = pos_est[0] - node_x[node_idx];
                    dy = pos_est[1] - node_y[node_idx];
                    dz = pos_est[2] - node_z[node_idx];
                    
                    dist_sq = dx*dx + dy*dy + dz*dz;
                    
                    // Fixed-point sqrt using Newton-Raphson
                    range[node_idx] <= fp_sqrt(dist_sq[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]);
                    
                    // Next node or advance state
                    if (node_idx < NUM_NODES - 1) begin
                        node_idx <= node_idx + 1;
                    end else begin
                        node_idx <= 0;
                    end
                end
                
                // ---------------------------------------------------------
                CALC_UNIT: begin
                    // Calculate unit vector from target to node
                    logic signed [DATA_WIDTH-1:0] dx, dy, dz;
                    
                    dx = pos_est[0] - node_x[node_idx];
                    dy = pos_est[1] - node_y[node_idx];
                    dz = pos_est[2] - node_z[node_idx];
                    
                    if (range[node_idx] > 32'h0000_1000) begin  // Avoid div by zero
                        unit_x[node_idx] <= fp_div(dx, range[node_idx]);
                        unit_y[node_idx] <= fp_div(dy, range[node_idx]);
                        unit_z[node_idx] <= fp_div(dz, range[node_idx]);
                    end else begin
                        unit_x[node_idx] <= 32'h0001_0000;  // Default
                        unit_y[node_idx] <= 0;
                        unit_z[node_idx] <= 0;
                    end
                    
                    if (node_idx < NUM_NODES - 1) begin
                        node_idx <= node_idx + 1;
                    end else begin
                        node_idx <= 0;
                    end
                end
                
                // ---------------------------------------------------------
                CALC_RESIDUAL: begin
                    // TDOA residuals: measured - predicted
                    // res_i = tdoa_meas[i] - (range[i+1] - range[0])
                    for (int i = 0; i < NUM_NODES-1; i++) begin
                        res_tdoa[i] <= tdoa_meas[i] - (range[i+1] - range[0]);
                    end
                    
                    // Doppler residuals
                    // res_i = doppler_meas[i] - (vel · unit_i)
                    for (int i = 0; i < NUM_NODES; i++) begin
                        logic signed [DATA_WIDTH-1:0] v_radial_pred;
                        v_radial_pred = fp_mult(vel_est[0], unit_x[i]) +
                                       fp_mult(vel_est[1], unit_y[i]) +
                                       fp_mult(vel_est[2], unit_z[i]);
                        res_dopp[i] <= doppler_meas[i] - v_radial_pred;
                    end
                end
                
                // ---------------------------------------------------------
                CALC_JACOBIAN: begin
                    // TDOA Jacobian: d(TDOA_i)/d(pos) = unit_{i+1} - unit_0
                    for (int i = 0; i < NUM_NODES-1; i++) begin
                        J_tdoa[i][0] <= unit_x[i+1] - unit_x[0];
                        J_tdoa[i][1] <= unit_y[i+1] - unit_y[0];
                        J_tdoa[i][2] <= unit_z[i+1] - unit_z[0];
                    end
                    
                    // Doppler Jacobian: d(v_r)/d(pos) = (partial dependence on unit vector)
                    // Simplified: use unit vectors directly
                    for (int i = 0; i < NUM_NODES; i++) begin
                        J_dopp[i][0] <= unit_x[i];
                        J_dopp[i][1] <= unit_y[i];
                        J_dopp[i][2] <= unit_z[i];
                    end
                end
                
                // ---------------------------------------------------------
                FORM_NORMAL: begin
                    // Form J'WJ and J'Wr
                    // Weight: TDOA = 1.0, Doppler = 0.1
                    localparam W_TDOA = 32'h0001_0000;  // 1.0
                    localparam W_DOPP = 32'h0000_199A;  // 0.1
                    
                    // Initialize to zero
                    for (int i = 0; i < 3; i++) begin
                        JtWr[i] <= 0;
                        for (int j = 0; j < 3; j++) begin
                            JtWJ[i][j] <= 0;
                        end
                    end
                    
                    // Accumulate TDOA
                    for (int n = 0; n < NUM_NODES-1; n++) begin
                        for (int i = 0; i < 3; i++) begin
                            JtWr[i] <= JtWr[i] + fp_mult(fp_mult(J_tdoa[n][i], W_TDOA), res_tdoa[n]);
                            for (int j = 0; j < 3; j++) begin
                                JtWJ[i][j] <= JtWJ[i][j] + fp_mult(fp_mult(J_tdoa[n][i], W_TDOA), J_tdoa[n][j]);
                            end
                        end
                    end
                    
                    // Regularization: add small diagonal for stability
                    for (int i = 0; i < 3; i++) begin
                        JtWJ[i][i] <= JtWJ[i][i] + 32'h0000_1000;  // 0.0625
                    end
                end
                
                // ---------------------------------------------------------
                SOLVE_DELTA: begin
                    // Solve 3×3 system using diagonal approximation
                    // Full solution would use LU decomposition
                    for (int i = 0; i < 3; i++) begin
                        if (JtWJ[i][i] > 32'h0000_0100) begin
                            delta_pos[i] <= fp_div(JtWr[i], JtWJ[i][i]);
                        end else begin
                            delta_pos[i] <= 0;
                        end
                    end
                end
                
                // ---------------------------------------------------------
                UPDATE_EST: begin
                    // Apply position update with damping (0.5) for stability
                    pos_est[0] <= pos_est[0] + (delta_pos[0] >>> 1);
                    pos_est[1] <= pos_est[1] + (delta_pos[1] >>> 1);
                    pos_est[2] <= pos_est[2] + (delta_pos[2] >>> 1);
                    
                    iter_count <= iter_count + 1;
                end
                
                // ---------------------------------------------------------
                CHECK_CONV: begin
                    // Check convergence: ||delta|| < threshold
                    logic signed [2*DATA_WIDTH-1:0] delta_sq;
                    delta_sq = delta_pos[0]*delta_pos[0] + 
                               delta_pos[1]*delta_pos[1] + 
                               delta_pos[2]*delta_pos[2];
                    
                    if (delta_sq < (CONV_THRESH * CONV_THRESH)) begin
                        converged <= 1;
                    end else if (iter_count >= MAX_ITER) begin
                        converged <= 1;  // Max iterations
                    end else begin
                        converged <= 0;
                        node_idx <= 0;  // Restart for next iteration
                    end
                end
                
                // ---------------------------------------------------------
                REFINE_VEL: begin
                    // Velocity estimation using Doppler residuals
                    // v_update = J_dopp' × res_dopp (simplified)
                    logic signed [DATA_WIDTH-1:0] v_update [3];
                    
                    v_update[0] = 0;
                    v_update[1] = 0;
                    v_update[2] = 0;
                    
                    for (int n = 0; n < NUM_NODES; n++) begin
                        v_update[0] = v_update[0] + fp_mult(J_dopp[n][0], res_dopp[n]);
                        v_update[1] = v_update[1] + fp_mult(J_dopp[n][1], res_dopp[n]);
                        v_update[2] = v_update[2] + fp_mult(J_dopp[n][2], res_dopp[n]);
                    end
                    
                    // Update velocity with damping
                    vel_est[0] <= vel_est[0] + fp_div(v_update[0], NUM_NODES << FRAC_BITS);
                    vel_est[1] <= vel_est[1] + fp_div(v_update[1], NUM_NODES << FRAC_BITS);
                    vel_est[2] <= vel_est[2] + fp_div(v_update[2], NUM_NODES << FRAC_BITS);
                end
                
                // ---------------------------------------------------------
                OUTPUT: begin
                    pos_out[0] <= pos_est[0];
                    pos_out[1] <= pos_est[1];
                    pos_out[2] <= pos_est[2];
                    vel_out[0] <= vel_est[0];
                    vel_out[1] <= vel_est[1];
                    vel_out[2] <= vel_est[2];
                    valid_out <= 1;
                end
            endcase
        end
    end
    
    // =========================================================================
    // FSM: Next-state logic
    // =========================================================================
    
    always_comb begin
        fsm_next = fsm_state;
        
        case (fsm_state)
            IDLE:          if (valid_in) fsm_next = INIT;
            INIT:          fsm_next = CALC_RANGE;
            CALC_RANGE:    if (node_idx == NUM_NODES-1) fsm_next = CALC_UNIT; else fsm_next = CALC_RANGE;
            CALC_UNIT:     if (node_idx == NUM_NODES-1) fsm_next = CALC_RESIDUAL; else fsm_next = CALC_UNIT;
            CALC_RESIDUAL: fsm_next = CALC_JACOBIAN;
            CALC_JACOBIAN: fsm_next = FORM_NORMAL;
            FORM_NORMAL:   fsm_next = SOLVE_DELTA;
            SOLVE_DELTA:   fsm_next = UPDATE_EST;
            UPDATE_EST:    fsm_next = CHECK_CONV;
            CHECK_CONV:    if (converged) fsm_next = REFINE_VEL; else fsm_next = CALC_RANGE;
            REFINE_VEL:    fsm_next = OUTPUT;
            OUTPUT:        fsm_next = IDLE;
            default:       fsm_next = IDLE;
        endcase
    end
    
    // =========================================================================
    // Fixed-Point Functions
    // =========================================================================
    
    // Q16.16 multiplication
    function automatic logic signed [DATA_WIDTH-1:0] fp_mult(
        input logic signed [DATA_WIDTH-1:0] a,
        input logic signed [DATA_WIDTH-1:0] b
    );
        logic signed [2*DATA_WIDTH-1:0] product;
        product = a * b;
        return product[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];
    endfunction
    
    // Q16.16 division
    function automatic logic signed [DATA_WIDTH-1:0] fp_div(
        input logic signed [DATA_WIDTH-1:0] a,
        input logic signed [DATA_WIDTH-1:0] b
    );
        logic signed [2*DATA_WIDTH-1:0] dividend;
        dividend = a << FRAC_BITS;
        return dividend / b;
    endfunction
    
    // Q16.16 square root (Newton-Raphson, 4 iterations)
    function automatic logic signed [DATA_WIDTH-1:0] fp_sqrt(
        input logic signed [DATA_WIDTH-1:0] x
    );
        logic signed [DATA_WIDTH-1:0] y;
        
        if (x <= 0) return 0;
        
        // Initial guess
        y = x >>> 1;
        if (y < 32'h0001_0000) y = 32'h0001_0000;
        
        // Newton-Raphson iterations
        for (int i = 0; i < 4; i++) begin
            y = (y + fp_div(x, y)) >>> 1;
        end
        
        return y;
    endfunction

endmodule
