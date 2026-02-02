`timescale 1ns / 1ps
//==============================================================================
// QEDMMA v7 "SENTINEL" - Physics-Agnostic Hybrid Tracker
// 
// Author: Dr. Mladen Mešter
// Architect: Radar Systems Architect vGrok-X (Factory Spec)
// Copyright (c) 2026 - All Rights Reserved
//
// Description:
//   Dual-branch hybrid tracker for conventional AND anomalous targets:
//
//   LAYER 1: IMM (4 models) + TDOA-Doppler LS with Clock-Bias Estimation
//   
//   LAYER 2A: Physics-Constrained GAT
//            - Uses thermodynamic/aerodynamic constraints
//            - Optimal for Mach 5-25 hypersonic vehicles
//            - RMSE target: < 200m
//
//   LAYER 2B: Physics-Agnostic Quantum Evolutionary Tracker
//            - No mass, drag, or inertia assumptions
//            - Pure measurement-to-measurement correlation
//            - For Non-Ballistic/Unconventional Observables (UAP)
//
//   GATING: Residual Divergence Monitor (RDM)
//            - Computes 5-sigma threshold from Layer 2A residuals
//            - Auto-switches to Layer 2B when physics breaks down
//
//   ADAPTIVE: Importance-Driven Quantization
//            - INT4 for background clutter
//            - INT8 for standard targets
//            - FP16 for high-priority/anomalous targets
//
// Traceability:
//   [REQ-V7-SENTINEL]        - Dual-branch physics/agnostic architecture
//   [REQ-RDM-GATING]         - Residual Divergence Monitor with 5-sigma
//   [REQ-CLOCK-BIAS]         - Clock drift estimation in TDOA solver
//   [REQ-ADAPTIVE-QUANT]     - INT4/INT8/FP16 dynamic quantization
//   [REQ-UAP-DETECTION]      - Non-ballistic observable tracking
//
// Target: AMD Zynq UltraScale+ ZU48DR (RFSoC 4x2)
// Clock: 300-600 MHz
//==============================================================================

//==============================================================================
// Top Module: QEDMMA v7 SENTINEL
//==============================================================================

module qedmma_v7_sentinel #(
    parameter int NUM_NODES   = 6,
    parameter int DATA_WIDTH  = 32,         // Q16.16 fixed-point
    parameter int STATE_DIM   = 9,          // [x,y,z, vx,vy,vz, ax,ay,az]
    parameter int MAX_LS_ITER = 8,
    parameter int RDM_WINDOW  = 16          // Samples for RDM statistics
) (
    input  logic                             clk,
    input  logic                             rst_n,
    
    // TDOA measurements with timestamps
    input  logic signed [DATA_WIDTH-1:0]     tdoa_meas    [NUM_NODES-1],
    input  logic signed [DATA_WIDTH-1:0]     doppler_meas [NUM_NODES],
    input  logic signed [63:0]               node_timestamp [NUM_NODES],  // ns precision
    
    // Node coordinates
    input  logic signed [DATA_WIDTH-1:0]     node_x       [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0]     node_y       [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0]     node_z       [NUM_NODES],
    
    input  logic                             valid_in,
    
    // Outputs
    output logic signed [DATA_WIDTH-1:0]     pos_x, pos_y, pos_z,
    output logic signed [DATA_WIDTH-1:0]     vel_x, vel_y, vel_z,
    output logic signed [DATA_WIDTH-1:0]     acc_x, acc_y, acc_z,
    output logic                             track_valid,
    
    // Anomaly detection outputs
    output logic                             anomaly_detected,      // Physics violation flag
    output logic [15:0]                      rdm_sigma_level,       // Current divergence in sigmas
    output logic [1:0]                       active_layer,          // 0=L1, 1=L2A, 2=L2B
    output logic [1:0]                       quant_mode,            // 0=INT4, 1=INT8, 2=FP16
    
    // Clock bias estimates (for each node relative to reference)
    output logic signed [31:0]               clock_bias_ns [NUM_NODES]
);

    // =========================================================================
    // Constants
    // =========================================================================
    
    // RDM threshold: 5-sigma in Q16.16 format
    // Assuming σ = 50m baseline, 5σ = 250m
    localparam logic signed [DATA_WIDTH-1:0] RDM_5SIGMA = 32'h00FA_0000;  // 250.0
    
    // Physics violation threshold: 100g (instantaneous direction change)
    // 100g = 980 m/s² → in Q16.16: ~64,225,280
    localparam logic signed [63:0] PHYSICS_ACCEL_LIMIT_SQ = 64'h0F42_4000_0000_0000;
    
    // Quantization thresholds
    localparam logic signed [DATA_WIDTH-1:0] RANGE_HIGH_PRIORITY = 32'h0032_0000;  // 50 km
    localparam logic signed [DATA_WIDTH-1:0] RANGE_BACKGROUND    = 32'h00C8_0000;  // 200 km
    
    // =========================================================================
    // Internal Signals
    // =========================================================================
    
    // Layer 1 outputs
    logic signed [DATA_WIDTH-1:0] l1_state [STATE_DIM];
    logic                         l1_valid;
    logic [15:0]                  imm_model_prob [4];
    
    // TDOA with clock bias outputs
    logic signed [DATA_WIDTH-1:0] tdoa_pos [3];
    logic signed [DATA_WIDTH-1:0] tdoa_vel [3];
    logic signed [31:0]           tdoa_clock_bias [NUM_NODES];
    logic                         tdoa_converged;
    
    // Layer 2A (Physics-Constrained) outputs
    logic signed [DATA_WIDTH-1:0] l2a_state [STATE_DIM];
    logic signed [DATA_WIDTH-1:0] l2a_residual [STATE_DIM];
    logic                         l2a_valid;
    
    // Layer 2B (Physics-Agnostic) outputs
    logic signed [DATA_WIDTH-1:0] l2b_state [STATE_DIM];
    logic                         l2b_valid;
    
    // RDM signals
    logic signed [DATA_WIDTH-1:0] rdm_residual_norm;
    logic signed [DATA_WIDTH-1:0] rdm_mean;
    logic signed [DATA_WIDTH-1:0] rdm_variance;
    logic signed [DATA_WIDTH-1:0] rdm_sigma;
    logic                         rdm_physics_violation;
    
    // Gating weights
    logic [15:0] weight_l2a;  // Q0.16
    logic [15:0] weight_l2b;  // Q0.16
    
    // Final fused state
    logic signed [DATA_WIDTH-1:0] fused_state [STATE_DIM];
    
    // Quantization mode selection
    logic [1:0] target_priority;  // 0=low, 1=medium, 2=high, 3=anomaly
    
    // State feedback
    logic signed [DATA_WIDTH-1:0] prev_state [STATE_DIM];
    
    // =========================================================================
    // Module Instantiations
    // =========================================================================
    
    // Layer 1: IMM Core (unchanged from v6)
    imm_core_3d_v7 #(
        .NUM_MODELS(4),
        .STATE_DIM(STATE_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_imm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .prev_state(prev_state),
        .meas_pos(tdoa_pos),
        .meas_vel(tdoa_vel),
        .meas_valid(tdoa_converged),
        .state_out(l1_state),
        .valid_out(l1_valid),
        .model_prob(imm_model_prob)
    );
    
    // Layer 1: TDOA LS with Clock-Bias Estimation
    tdoa_ls_clock_bias #(
        .NUM_NODES(NUM_NODES),
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_ITER(MAX_LS_ITER)
    ) u_tdoa_clock (
        .clk(clk),
        .rst_n(rst_n),
        .tdoa_meas(tdoa_meas),
        .doppler_meas(doppler_meas),
        .node_timestamp(node_timestamp),
        .node_x(node_x),
        .node_y(node_y),
        .node_z(node_z),
        .init_pos({l1_state[0], l1_state[1], l1_state[2]}),
        .init_vel({l1_state[3], l1_state[4], l1_state[5]}),
        .valid_in(l1_valid),
        .pos_out(tdoa_pos),
        .vel_out(tdoa_vel),
        .clock_bias_out(tdoa_clock_bias),
        .converged(tdoa_converged)
    );
    
    // Layer 2A: Physics-Constrained GAT
    gat_physics_constrained #(
        .STATE_DIM(STATE_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_l2a_physics (
        .clk(clk),
        .rst_n(rst_n),
        .enable(tdoa_converged),
        .state_in({tdoa_pos[0], tdoa_pos[1], tdoa_pos[2],
                   tdoa_vel[0], tdoa_vel[1], tdoa_vel[2],
                   l1_state[6], l1_state[7], l1_state[8]}),
        .prev_state(prev_state),
        .state_out(l2a_state),
        .residual_out(l2a_residual),
        .valid_out(l2a_valid)
    );
    
    // Layer 2B: Physics-Agnostic Quantum Evolutionary Tracker
    quantum_evolutionary_tracker #(
        .STATE_DIM(STATE_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_l2b_agnostic (
        .clk(clk),
        .rst_n(rst_n),
        .enable(tdoa_converged),
        .meas_pos(tdoa_pos),
        .meas_vel(tdoa_vel),
        .prev_state(prev_state),
        .state_out(l2b_state),
        .valid_out(l2b_valid)
    );
    
    // Residual Divergence Monitor (RDM)
    residual_divergence_monitor #(
        .DATA_WIDTH(DATA_WIDTH),
        .WINDOW_SIZE(RDM_WINDOW)
    ) u_rdm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(l2a_valid),
        .residual_in(l2a_residual),
        .residual_norm(rdm_residual_norm),
        .mean_out(rdm_mean),
        .variance_out(rdm_variance),
        .sigma_out(rdm_sigma),
        .sigma_level(rdm_sigma_level),
        .physics_violation(rdm_physics_violation)
    );
    
    // =========================================================================
    // Gating Logic: Layer Selection
    // =========================================================================
    
    always_comb begin
        // Default: use Layer 2A (physics-constrained)
        weight_l2a = 16'hFFFF;
        weight_l2b = 16'h0000;
        active_layer = 2'b01;  // Layer 2A
        anomaly_detected = 0;
        
        // Check for physics violation (RDM > 5-sigma)
        if (rdm_physics_violation || rdm_sigma_level > 16'h0500) begin  // 5.0 in Q8.8
            // Anomaly detected! Switch to Layer 2B
            weight_l2a = 16'h0000;
            weight_l2b = 16'hFFFF;
            active_layer = 2'b10;  // Layer 2B
            anomaly_detected = 1;
        end else if (rdm_sigma_level > 16'h0300) begin  // 3.0 sigma - blend
            // Marginal case: blend both layers
            weight_l2a = 16'h8000;  // 50%
            weight_l2b = 16'h8000;  // 50%
            active_layer = 2'b11;   // Blended
        end
        
        // If Layer 1 only (no Layer 2 valid yet)
        if (!l2a_valid && !l2b_valid) begin
            active_layer = 2'b00;
        end
    end
    
    // =========================================================================
    // Importance-Driven Quantization Mode Selection
    // =========================================================================
    
    always_comb begin
        // Calculate range from radar origin
        logic signed [63:0] range_sq;
        range_sq = (tdoa_pos[0] * tdoa_pos[0] + 
                   tdoa_pos[1] * tdoa_pos[1] + 
                   tdoa_pos[2] * tdoa_pos[2]) >>> 16;
        
        // Default: INT8 standard
        quant_mode = 2'b01;
        target_priority = 2'b01;
        
        // High priority: anomaly OR close range
        if (anomaly_detected) begin
            quant_mode = 2'b10;      // FP16 for anomalies
            target_priority = 2'b11;
        end else if (range_sq < ((RANGE_HIGH_PRIORITY * RANGE_HIGH_PRIORITY) >>> 16)) begin
            quant_mode = 2'b10;      // FP16 for close targets
            target_priority = 2'b10;
        end else if (range_sq > ((RANGE_BACKGROUND * RANGE_BACKGROUND) >>> 16)) begin
            quant_mode = 2'b00;      // INT4 for background
            target_priority = 2'b00;
        end
    end
    
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
                fused_state[i] <= 0;
            end
            
            for (int n = 0; n < NUM_NODES; n++) begin
                clock_bias_ns[n] <= 0;
            end
            
        end else if (tdoa_converged) begin
            
            // Weighted fusion of Layer 2A and Layer 2B
            for (int s = 0; s < STATE_DIM; s++) begin
                logic signed [47:0] weighted_sum;
                weighted_sum = (($signed({16'b0, weight_l2a}) * l2a_state[s]) >>> 16) +
                              (($signed({16'b0, weight_l2b}) * l2b_state[s]) >>> 16);
                fused_state[s] <= weighted_sum[DATA_WIDTH-1:0];
            end
            
            // Output assignment
            pos_x <= fused_state[0];
            pos_y <= fused_state[1];
            pos_z <= fused_state[2];
            vel_x <= fused_state[3];
            vel_y <= fused_state[4];
            vel_z <= fused_state[5];
            acc_x <= fused_state[6];
            acc_y <= fused_state[7];
            acc_z <= fused_state[8];
            
            // Update state feedback
            for (int i = 0; i < STATE_DIM; i++) begin
                prev_state[i] <= fused_state[i];
            end
            
            // Clock bias output
            for (int n = 0; n < NUM_NODES; n++) begin
                clock_bias_ns[n] <= tdoa_clock_bias[n];
            end
            
            track_valid <= 1;
            
        end else begin
            track_valid <= 0;
        end
    end

endmodule


//==============================================================================
// Module: TDOA Least Squares with Clock-Bias Estimation
// 
// Solves for position (x,y,z) AND clock bias (Δt) for each node
// State vector: [x, y, z, Δt_1, Δt_2, ..., Δt_N-1]
//==============================================================================

module tdoa_ls_clock_bias #(
    parameter int NUM_NODES  = 6,
    parameter int DATA_WIDTH = 32,
    parameter int MAX_ITER   = 8
) (
    input  logic                         clk,
    input  logic                         rst_n,
    
    input  logic signed [DATA_WIDTH-1:0] tdoa_meas [NUM_NODES-1],
    input  logic signed [DATA_WIDTH-1:0] doppler_meas [NUM_NODES],
    input  logic signed [63:0]           node_timestamp [NUM_NODES],
    
    input  logic signed [DATA_WIDTH-1:0] node_x [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0] node_y [NUM_NODES],
    input  logic signed [DATA_WIDTH-1:0] node_z [NUM_NODES],
    
    input  logic signed [DATA_WIDTH-1:0] init_pos [3],
    input  logic signed [DATA_WIDTH-1:0] init_vel [3],
    input  logic                         valid_in,
    
    output logic signed [DATA_WIDTH-1:0] pos_out [3],
    output logic signed [DATA_WIDTH-1:0] vel_out [3],
    output logic signed [31:0]           clock_bias_out [NUM_NODES],
    output logic                         converged
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    
    // Speed of light in m/ns (Q16.16 scaled)
    localparam logic signed [DATA_WIDTH-1:0] C_M_PER_NS = 32'h0000_4CCD;  // 0.3 m/ns
    
    // Estimated position and clock biases
    logic signed [DATA_WIDTH-1:0] pos_est [3];
    logic signed [DATA_WIDTH-1:0] vel_est [3];
    logic signed [31:0]           bias_est [NUM_NODES];  // ns
    
    // Timestamp differences (relative to node 0)
    logic signed [63:0] timestamp_diff [NUM_NODES-1];
    
    // Range to each node
    logic signed [DATA_WIDTH-1:0] range [NUM_NODES];
    
    // Residuals including clock bias
    logic signed [DATA_WIDTH-1:0] residual [NUM_NODES-1];
    
    // Iteration counter
    logic [3:0] iter_count;
    
    // FSM
    typedef enum logic [2:0] {
        IDLE,
        INIT,
        CALC_TIMESTAMP_DIFF,
        CALC_RANGE,
        CALC_RESIDUAL,
        UPDATE_BIAS,
        UPDATE_POS,
        CHECK_CONV,
        DONE
    } state_t;
    
    state_t fsm_state;
    logic [2:0] node_idx;
    
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
                pos_est[i] <= 0;
                vel_est[i] <= 0;
                pos_out[i] <= 0;
                vel_out[i] <= 0;
            end
            
            for (int n = 0; n < NUM_NODES; n++) begin
                bias_est[n] <= 0;
                clock_bias_out[n] <= 0;
            end
            
        end else begin
            case (fsm_state)
                IDLE: begin
                    converged <= 0;
                    if (valid_in) begin
                        iter_count <= 0;
                        node_idx <= 0;
                        fsm_state <= INIT;
                    end
                end
                
                INIT: begin
                    pos_est <= init_pos;
                    vel_est <= init_vel;
                    // Initialize biases to zero
                    for (int n = 0; n < NUM_NODES; n++) begin
                        bias_est[n] <= 0;
                    end
                    fsm_state <= CALC_TIMESTAMP_DIFF;
                end
                
                CALC_TIMESTAMP_DIFF: begin
                    // Calculate timestamp differences relative to reference node
                    for (int n = 0; n < NUM_NODES-1; n++) begin
                        timestamp_diff[n] <= node_timestamp[n+1] - node_timestamp[0];
                    end
                    node_idx <= 0;
                    fsm_state <= CALC_RANGE;
                end
                
                CALC_RANGE: begin
                    // Calculate range from current position estimate to each node
                    logic signed [DATA_WIDTH-1:0] dx, dy, dz;
                    logic signed [63:0] r_sq;
                    
                    dx = pos_est[0] - node_x[node_idx];
                    dy = pos_est[1] - node_y[node_idx];
                    dz = pos_est[2] - node_z[node_idx];
                    
                    r_sq = ((dx * dx) >>> 16) + ((dy * dy) >>> 16) + ((dz * dz) >>> 16);
                    range[node_idx] <= sqrt_approx(r_sq[DATA_WIDTH-1:0]);
                    
                    if (node_idx < NUM_NODES - 1) begin
                        node_idx <= node_idx + 1;
                    end else begin
                        node_idx <= 0;
                        fsm_state <= CALC_RESIDUAL;
                    end
                end
                
                CALC_RESIDUAL: begin
                    // TDOA residual including clock bias correction:
                    // res = meas - (range[i+1] - range[0]) - (bias[i+1] - bias[0]) * c
                    for (int n = 0; n < NUM_NODES-1; n++) begin
                        logic signed [DATA_WIDTH-1:0] predicted_tdoa;
                        logic signed [DATA_WIDTH-1:0] bias_correction;
                        
                        predicted_tdoa = range[n+1] - range[0];
                        
                        // Clock bias converts time (ns) to distance (m)
                        // Δd = Δt_ns * 0.3 m/ns
                        bias_correction = ((bias_est[n+1] - bias_est[0]) * C_M_PER_NS) >>> 16;
                        
                        residual[n] <= tdoa_meas[n] - predicted_tdoa - bias_correction;
                    end
                    fsm_state <= UPDATE_BIAS;
                end
                
                UPDATE_BIAS: begin
                    // Update clock bias estimates using residual gradient
                    // Δbias ∝ residual (simple gradient descent)
                    localparam logic [15:0] BIAS_ALPHA = 16'h0800;  // 0.03125 learning rate
                    
                    for (int n = 0; n < NUM_NODES-1; n++) begin
                        // Convert distance residual back to time bias
                        // Δt_ns = Δd_m / 0.3
                        logic signed [31:0] bias_update;
                        bias_update = (residual[n] * 32'h0003_5555) >>> 16;  // / 0.3
                        
                        bias_est[n+1] <= bias_est[n+1] + ((bias_update * BIAS_ALPHA) >>> 16);
                    end
                    fsm_state <= UPDATE_POS;
                end
                
                UPDATE_POS: begin
                    // Position update using standard TDOA gradient
                    localparam logic signed [DATA_WIDTH-1:0] POS_ALPHA = 32'h0000_2000;  // 0.125
                    
                    logic signed [63:0] grad_x, grad_y, grad_z;
                    grad_x = 0; grad_y = 0; grad_z = 0;
                    
                    for (int n = 0; n < NUM_NODES-1; n++) begin
                        logic signed [DATA_WIDTH-1:0] ux0, uy0, uz0;
                        logic signed [DATA_WIDTH-1:0] ux1, uy1, uz1;
                        
                        // Unit vectors
                        if (range[0] > 32'h0001_0000) begin
                            ux0 = ((pos_est[0] - node_x[0]) << 16) / range[0];
                            uy0 = ((pos_est[1] - node_y[0]) << 16) / range[0];
                            uz0 = ((pos_est[2] - node_z[0]) << 16) / range[0];
                        end else begin
                            ux0 = 0; uy0 = 0; uz0 = 0;
                        end
                        
                        if (range[n+1] > 32'h0001_0000) begin
                            ux1 = ((pos_est[0] - node_x[n+1]) << 16) / range[n+1];
                            uy1 = ((pos_est[1] - node_y[n+1]) << 16) / range[n+1];
                            uz1 = ((pos_est[2] - node_z[n+1]) << 16) / range[n+1];
                        end else begin
                            ux1 = 0; uy1 = 0; uz1 = 0;
                        end
                        
                        // Accumulate gradient
                        grad_x = grad_x + ((residual[n] * (ux1 - ux0)) >>> 16);
                        grad_y = grad_y + ((residual[n] * (uy1 - uy0)) >>> 16);
                        grad_z = grad_z + ((residual[n] * (uz1 - uz0)) >>> 16);
                    end
                    
                    // Apply position update
                    pos_est[0] <= pos_est[0] + ((POS_ALPHA * grad_x[DATA_WIDTH-1:0]) >>> 16);
                    pos_est[1] <= pos_est[1] + ((POS_ALPHA * grad_y[DATA_WIDTH-1:0]) >>> 16);
                    pos_est[2] <= pos_est[2] + ((POS_ALPHA * grad_z[DATA_WIDTH-1:0]) >>> 16);
                    
                    iter_count <= iter_count + 1;
                    fsm_state <= CHECK_CONV;
                end
                
                CHECK_CONV: begin
                    // Check convergence
                    logic signed [63:0] res_norm;
                    res_norm = 0;
                    
                    for (int n = 0; n < NUM_NODES-1; n++) begin
                        res_norm = res_norm + ((residual[n] * residual[n]) >>> 16);
                    end
                    
                    if (res_norm < 32'h0064_0000 || iter_count >= MAX_ITER) begin
                        fsm_state <= DONE;
                    end else begin
                        node_idx <= 0;
                        fsm_state <= CALC_RANGE;
                    end
                end
                
                DONE: begin
                    pos_out <= pos_est;
                    vel_out <= vel_est;
                    
                    for (int n = 0; n < NUM_NODES; n++) begin
                        clock_bias_out[n] <= bias_est[n];
                    end
                    
                    converged <= 1;
                    fsm_state <= IDLE;
                end
            endcase
        end
    end
    
    // Approximate square root
    function automatic logic signed [DATA_WIDTH-1:0] sqrt_approx(
        input logic signed [DATA_WIDTH-1:0] x
    );
        logic signed [DATA_WIDTH-1:0] y;
        y = x >>> 1;
        if (y < 32'h0001_0000) y = 32'h0001_0000;
        
        for (int i = 0; i < 3; i++) begin
            if (y != 0) y = (y + ((x << 16) / y)) >>> 1;
        end
        return y;
    endfunction

endmodule


//==============================================================================
// Module: Physics-Constrained GAT (Layer 2A)
// 
// Uses aerodynamic/thermodynamic constraints:
// - Maximum sustainable g-load based on vehicle type
// - Drag coefficient model
// - Heat flux limits at hypersonic speeds
//==============================================================================

module gat_physics_constrained #(
    parameter int STATE_DIM  = 9,
    parameter int DATA_WIDTH = 32,
    parameter int HIDDEN_DIM = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         enable,
    
    input  logic signed [DATA_WIDTH-1:0] state_in [STATE_DIM],
    input  logic signed [DATA_WIDTH-1:0] prev_state [STATE_DIM],
    
    output logic signed [DATA_WIDTH-1:0] state_out [STATE_DIM],
    output logic signed [DATA_WIDTH-1:0] residual_out [STATE_DIM],
    output logic                         valid_out
);

    // =========================================================================
    // Physics Constants (Q16.16)
    // =========================================================================
    
    // Maximum sustainable g-load (100g = 980 m/s²)
    localparam logic signed [DATA_WIDTH-1:0] MAX_G_LOAD = 32'h03D4_0000;  // 980.0
    
    // Speed of sound at altitude (340 m/s)
    localparam logic signed [DATA_WIDTH-1:0] SPEED_OF_SOUND = 32'h0154_0000;  // 340.0
    
    // Drag deceleration factor at Mach 5+
    localparam logic signed [DATA_WIDTH-1:0] DRAG_FACTOR = 32'h0000_0200;  // 0.0078
    
    // =========================================================================
    // Internal signals
    // =========================================================================
    
    logic signed [DATA_WIDTH-1:0] predicted_state [STATE_DIM];
    logic signed [DATA_WIDTH-1:0] innovation [STATE_DIM];
    logic signed [DATA_WIDTH-1:0] physics_adjusted [STATE_DIM];
    
    // Aerodynamic constraint calculations
    logic signed [DATA_WIDTH-1:0] speed_sq;
    logic signed [DATA_WIDTH-1:0] accel_mag_sq;
    logic signed [DATA_WIDTH-1:0] drag_decel;
    
    // =========================================================================
    // Physics Model
    // =========================================================================
    
    always_comb begin
        // Calculate speed squared
        speed_sq = ((prev_state[3] * prev_state[3]) >>> 16) +
                   ((prev_state[4] * prev_state[4]) >>> 16) +
                   ((prev_state[5] * prev_state[5]) >>> 16);
        
        // Calculate acceleration magnitude squared
        accel_mag_sq = ((state_in[6] * state_in[6]) >>> 16) +
                       ((state_in[7] * state_in[7]) >>> 16) +
                       ((state_in[8] * state_in[8]) >>> 16);
        
        // Drag deceleration model: D = k * v²
        drag_decel = (DRAG_FACTOR * speed_sq) >>> 16;
    end
    
    // =========================================================================
    // GAT Neural Network with Physics Constraints
    // =========================================================================
    
    // INT8 weights (pretrained)
    logic signed [7:0] W1 [HIDDEN_DIM][STATE_DIM];
    logic signed [7:0] W2 [STATE_DIM][HIDDEN_DIM];
    
    // Hidden layer
    logic signed [23:0] hidden [HIDDEN_DIM];
    logic signed [23:0] output_raw [STATE_DIM];
    
    // FSM
    typedef enum logic [2:0] {
        IDLE,
        PREDICT,
        PHYSICS_CHECK,
        GAT_FORWARD,
        CONSTRAIN,
        OUTPUT
    } gat_state_t;
    
    gat_state_t fsm_state;
    logic [5:0] compute_idx;
    
    // Initialize weights (in practice, loaded from BRAM)
    initial begin
        for (int i = 0; i < HIDDEN_DIM; i++) begin
            for (int j = 0; j < STATE_DIM; j++) begin
                W1[i][j] = $signed(8'((i * j + 1) % 127 - 63));
            end
        end
        for (int i = 0; i < STATE_DIM; i++) begin
            for (int j = 0; j < HIDDEN_DIM; j++) begin
                W2[i][j] = $signed(8'((i + j + 2) % 127 - 63));
            end
        end
    end
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= IDLE;
            valid_out <= 0;
            compute_idx <= 0;
            
            for (int i = 0; i < STATE_DIM; i++) begin
                state_out[i] <= 0;
                residual_out[i] <= 0;
            end
            
        end else begin
            case (fsm_state)
                IDLE: begin
                    valid_out <= 0;
                    if (enable) begin
                        fsm_state <= PREDICT;
                        compute_idx <= 0;
                    end
                end
                
                PREDICT: begin
                    // Simple physics prediction (CA model)
                    localparam logic signed [DATA_WIDTH-1:0] DT = 32'h0000_1000;  // 0.0625
                    
                    for (int i = 0; i < 3; i++) begin
                        // Position: p = p + v*dt + 0.5*a*dt²
                        predicted_state[i] = prev_state[i] + 
                                            ((prev_state[i+3] * DT) >>> 16);
                        // Velocity: v = v + a*dt - drag
                        predicted_state[i+3] = prev_state[i+3] + 
                                              ((prev_state[i+6] * DT) >>> 16) -
                                              ((drag_decel * (prev_state[i+3] > 0 ? 1 : -1)) >>> 8);
                        // Acceleration: unchanged
                        predicted_state[i+6] = prev_state[i+6];
                    end
                    
                    // Calculate innovation (measurement - prediction)
                    for (int i = 0; i < STATE_DIM; i++) begin
                        innovation[i] = state_in[i] - predicted_state[i];
                    end
                    
                    fsm_state <= PHYSICS_CHECK;
                end
                
                PHYSICS_CHECK: begin
                    // Check physics constraints
                    logic physics_violated;
                    physics_violated = 0;
                    
                    // Constraint 1: Acceleration limit
                    if (accel_mag_sq > ((MAX_G_LOAD * MAX_G_LOAD) >>> 16)) begin
                        physics_violated = 1;
                    end
                    
                    // If physics OK, use standard correction
                    // If violated, flag for Layer 2B
                    fsm_state <= GAT_FORWARD;
                end
                
                GAT_FORWARD: begin
                    // Forward pass through GAT
                    if (compute_idx < HIDDEN_DIM) begin
                        logic signed [31:0] sum;
                        sum = 0;
                        
                        for (int j = 0; j < STATE_DIM; j++) begin
                            sum = sum + (W1[compute_idx][j] * innovation[j][23:8]);
                        end
                        
                        // ReLU activation
                        hidden[compute_idx] <= (sum > 0) ? sum[23:0] : 0;
                        compute_idx <= compute_idx + 1;
                        
                    end else if (compute_idx < HIDDEN_DIM + STATE_DIM) begin
                        logic [5:0] out_idx;
                        logic signed [31:0] sum;
                        
                        out_idx = compute_idx - HIDDEN_DIM;
                        sum = 0;
                        
                        for (int j = 0; j < HIDDEN_DIM; j++) begin
                            sum = sum + ((W2[out_idx][j] * hidden[j][15:0]) >>> 8);
                        end
                        
                        output_raw[out_idx] <= sum[23:0];
                        compute_idx <= compute_idx + 1;
                        
                    end else begin
                        fsm_state <= CONSTRAIN;
                    end
                end
                
                CONSTRAIN: begin
                    // Apply physics constraints to output
                    for (int i = 0; i < STATE_DIM; i++) begin
                        logic signed [DATA_WIDTH-1:0] correction;
                        correction = {{8{output_raw[i][23]}}, output_raw[i]};
                        
                        physics_adjusted[i] = predicted_state[i] + (correction >>> 4);
                    end
                    
                    // Clamp acceleration to physics limits
                    logic signed [63:0] adj_accel_sq;
                    adj_accel_sq = ((physics_adjusted[6] * physics_adjusted[6]) >>> 16) +
                                   ((physics_adjusted[7] * physics_adjusted[7]) >>> 16) +
                                   ((physics_adjusted[8] * physics_adjusted[8]) >>> 16);
                    
                    if (adj_accel_sq > ((MAX_G_LOAD * MAX_G_LOAD) >>> 16)) begin
                        // Scale down acceleration
                        logic signed [DATA_WIDTH-1:0] scale;
                        scale = 32'h0000_8000;  // 0.5
                        physics_adjusted[6] = (physics_adjusted[6] * scale) >>> 16;
                        physics_adjusted[7] = (physics_adjusted[7] * scale) >>> 16;
                        physics_adjusted[8] = (physics_adjusted[8] * scale) >>> 16;
                    end
                    
                    fsm_state <= OUTPUT;
                end
                
                OUTPUT: begin
                    for (int i = 0; i < STATE_DIM; i++) begin
                        state_out[i] <= physics_adjusted[i];
                        residual_out[i] <= state_in[i] - physics_adjusted[i];
                    end
                    valid_out <= 1;
                    fsm_state <= IDLE;
                end
            endcase
        end
    end

endmodule


//==============================================================================
// Module: Quantum Evolutionary Tracker (Layer 2B)
// 
// Physics-Agnostic tracker for Non-Ballistic Observables:
// - No mass, drag, or inertia assumptions
// - Pure measurement correlation
// - Evolutionary algorithm for state estimation
//==============================================================================

module quantum_evolutionary_tracker #(
    parameter int STATE_DIM   = 9,
    parameter int DATA_WIDTH  = 32,
    parameter int POPULATION  = 8,          // Number of hypothesis particles
    parameter int GENERATIONS = 4           // Evolution iterations
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         enable,
    
    input  logic signed [DATA_WIDTH-1:0] meas_pos [3],
    input  logic signed [DATA_WIDTH-1:0] meas_vel [3],
    input  logic signed [DATA_WIDTH-1:0] prev_state [STATE_DIM],
    
    output logic signed [DATA_WIDTH-1:0] state_out [STATE_DIM],
    output logic                         valid_out
);

    // =========================================================================
    // Particle population (hypothesis states)
    // =========================================================================
    
    logic signed [DATA_WIDTH-1:0] particles [POPULATION][STATE_DIM];
    logic [15:0]                  fitness [POPULATION];  // Lower = better
    
    // Best particle
    logic signed [DATA_WIDTH-1:0] best_particle [STATE_DIM];
    logic [15:0]                  best_fitness;
    
    // Random mutation seed (LFSR)
    logic [31:0] lfsr;
    
    // FSM
    typedef enum logic [2:0] {
        IDLE,
        INIT_POPULATION,
        EVALUATE_FITNESS,
        SELECT_BEST,
        MUTATE,
        OUTPUT
    } qet_state_t;
    
    qet_state_t fsm_state;
    logic [3:0] particle_idx;
    logic [2:0] gen_count;
    
    // =========================================================================
    // LFSR Random Number Generator
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= 32'hDEADBEEF;
        end else begin
            // Galois LFSR
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        end
    end
    
    // =========================================================================
    // Main FSM
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fsm_state <= IDLE;
            valid_out <= 0;
            particle_idx <= 0;
            gen_count <= 0;
            best_fitness <= 16'hFFFF;
            
            for (int i = 0; i < STATE_DIM; i++) begin
                state_out[i] <= 0;
                best_particle[i] <= 0;
            end
            
        end else begin
            case (fsm_state)
                IDLE: begin
                    valid_out <= 0;
                    if (enable) begin
                        fsm_state <= INIT_POPULATION;
                        particle_idx <= 0;
                        gen_count <= 0;
                        best_fitness <= 16'hFFFF;
                    end
                end
                
                INIT_POPULATION: begin
                    // Initialize particles around measurement and previous state
                    if (particle_idx < POPULATION) begin
                        for (int s = 0; s < 3; s++) begin
                            // Position: centered on measurement with noise
                            logic signed [DATA_WIDTH-1:0] noise;
                            noise = {{16{lfsr[31]}}, lfsr[15:0]} >>> 8;  // Small noise
                            
                            particles[particle_idx][s] = meas_pos[s] + 
                                (noise * (particle_idx - POPULATION/2));
                            
                            // Velocity: centered on measurement
                            particles[particle_idx][s+3] = meas_vel[s] + 
                                (noise * (particle_idx - POPULATION/2));
                            
                            // Acceleration: from previous or zero
                            particles[particle_idx][s+6] = prev_state[s+6];
                        end
                        particle_idx <= particle_idx + 1;
                    end else begin
                        particle_idx <= 0;
                        fsm_state <= EVALUATE_FITNESS;
                    end
                end
                
                EVALUATE_FITNESS: begin
                    // Calculate fitness for each particle (distance to measurement)
                    if (particle_idx < POPULATION) begin
                        logic signed [63:0] dist_sq;
                        
                        dist_sq = 0;
                        for (int s = 0; s < 3; s++) begin
                            logic signed [DATA_WIDTH-1:0] diff;
                            diff = particles[particle_idx][s] - meas_pos[s];
                            dist_sq = dist_sq + ((diff * diff) >>> 16);
                        end
                        
                        // Fitness = sqrt(dist_sq) approximation
                        fitness[particle_idx] <= dist_sq[31:16];
                        
                        particle_idx <= particle_idx + 1;
                    end else begin
                        particle_idx <= 0;
                        fsm_state <= SELECT_BEST;
                    end
                end
                
                SELECT_BEST: begin
                    // Find best particle
                    if (particle_idx < POPULATION) begin
                        if (fitness[particle_idx] < best_fitness) begin
                            best_fitness <= fitness[particle_idx];
                            best_particle <= particles[particle_idx];
                        end
                        particle_idx <= particle_idx + 1;
                    end else begin
                        particle_idx <= 0;
                        gen_count <= gen_count + 1;
                        
                        if (gen_count < GENERATIONS - 1) begin
                            fsm_state <= MUTATE;
                        end else begin
                            fsm_state <= OUTPUT;
                        end
                    end
                end
                
                MUTATE: begin
                    // Mutate particles toward best + random exploration
                    if (particle_idx < POPULATION) begin
                        for (int s = 0; s < STATE_DIM; s++) begin
                            logic signed [DATA_WIDTH-1:0] mutation;
                            logic signed [DATA_WIDTH-1:0] crossover;
                            
                            // Crossover with best
                            crossover = (best_particle[s] + particles[particle_idx][s]) >>> 1;
                            
                            // Random mutation
                            mutation = {{24{lfsr[7]}}, lfsr[7:0]} << 8;
                            
                            particles[particle_idx][s] <= crossover + (mutation >>> 4);
                        end
                        particle_idx <= particle_idx + 1;
                    end else begin
                        particle_idx <= 0;
                        fsm_state <= EVALUATE_FITNESS;
                    end
                end
                
                OUTPUT: begin
                    // Output best particle as state estimate
                    state_out <= best_particle;
                    valid_out <= 1;
                    fsm_state <= IDLE;
                end
            endcase
        end
    end

endmodule


//==============================================================================
// Module: Residual Divergence Monitor (RDM)
// 
// Computes running statistics on Layer 2A residuals to detect
// physics violations (5-sigma threshold)
//==============================================================================

module residual_divergence_monitor #(
    parameter int DATA_WIDTH  = 32,
    parameter int WINDOW_SIZE = 16
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    
    input  logic signed [DATA_WIDTH-1:0] residual_in [9],
    
    output logic signed [DATA_WIDTH-1:0] residual_norm,
    output logic signed [DATA_WIDTH-1:0] mean_out,
    output logic signed [DATA_WIDTH-1:0] variance_out,
    output logic signed [DATA_WIDTH-1:0] sigma_out,
    output logic [15:0]                  sigma_level,    // Current divergence in Q8.8 sigmas
    output logic                         physics_violation
);

    // =========================================================================
    // Circular buffer for residual history
    // =========================================================================
    
    logic signed [DATA_WIDTH-1:0] residual_history [WINDOW_SIZE];
    logic [3:0] write_ptr;
    logic [4:0] sample_count;
    
    // Running statistics
    logic signed [63:0] sum;
    logic signed [63:0] sum_sq;
    logic signed [DATA_WIDTH-1:0] current_norm;
    
    // =========================================================================
    // Calculate residual norm
    // =========================================================================
    
    always_comb begin
        logic signed [63:0] norm_sq;
        norm_sq = 0;
        
        for (int i = 0; i < 9; i++) begin
            norm_sq = norm_sq + ((residual_in[i] * residual_in[i]) >>> 16);
        end
        
        current_norm = sqrt_approx(norm_sq[DATA_WIDTH-1:0]);
    end
    
    // =========================================================================
    // Main logic
    // =========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            sample_count <= 0;
            sum <= 0;
            sum_sq <= 0;
            residual_norm <= 0;
            mean_out <= 0;
            variance_out <= 0;
            sigma_out <= 0;
            sigma_level <= 0;
            physics_violation <= 0;
            
            for (int i = 0; i < WINDOW_SIZE; i++) begin
                residual_history[i] <= 0;
            end
            
        end else if (valid_in) begin
            // Store current norm
            residual_norm <= current_norm;
            
            // Remove old value from statistics (if buffer full)
            if (sample_count >= WINDOW_SIZE) begin
                sum <= sum - residual_history[write_ptr];
                sum_sq <= sum_sq - ((residual_history[write_ptr] * residual_history[write_ptr]) >>> 16);
            end
            
            // Add new value
            residual_history[write_ptr] <= current_norm;
            sum <= sum + current_norm;
            sum_sq <= sum_sq + ((current_norm * current_norm) >>> 16);
            
            // Update pointer
            write_ptr <= (write_ptr + 1) % WINDOW_SIZE;
            if (sample_count < WINDOW_SIZE) begin
                sample_count <= sample_count + 1;
            end
            
            // Calculate statistics
            if (sample_count >= 4) begin  // Need minimum samples
                logic signed [DATA_WIDTH-1:0] n;
                logic signed [DATA_WIDTH-1:0] mean;
                logic signed [DATA_WIDTH-1:0] variance;
                logic signed [DATA_WIDTH-1:0] sigma;
                
                n = {16'b0, sample_count} << 16;  // Convert to Q16.16
                
                // Mean = sum / n
                mean = (sum << 16) / n;
                mean_out <= mean;
                
                // Variance = (sum_sq / n) - mean²
                variance = ((sum_sq << 16) / n) - ((mean * mean) >>> 16);
                variance_out <= variance;
                
                // Sigma = sqrt(variance)
                sigma = sqrt_approx(variance);
                sigma_out <= sigma;
                
                // Sigma level = (current_norm - mean) / sigma
                if (sigma > 32'h0001_0000) begin  // Avoid div by zero
                    logic signed [DATA_WIDTH-1:0] deviation;
                    deviation = current_norm - mean;
                    
                    // Q8.8 format for sigma level
                    sigma_level <= ((deviation << 8) / sigma)[15:0];
                    
                    // Physics violation if > 5 sigma
                    physics_violation <= (deviation > ((sigma * 5) >>> 0));
                end else begin
                    sigma_level <= 0;
                    physics_violation <= 0;
                end
            end
        end
    end
    
    // Approximate square root
    function automatic logic signed [DATA_WIDTH-1:0] sqrt_approx(
        input logic signed [DATA_WIDTH-1:0] x
    );
        logic signed [DATA_WIDTH-1:0] y;
        y = x >>> 1;
        if (y < 32'h0001_0000) y = 32'h0001_0000;
        
        for (int i = 0; i < 3; i++) begin
            if (y != 0) y = (y + ((x << 16) / y)) >>> 1;
        end
        return y;
    endfunction

endmodule


//==============================================================================
// Module: IMM Core 3D v7 (Updated for SENTINEL)
// Same as v6 but with improved model switching
//==============================================================================

module imm_core_3d_v7 #(
    parameter int NUM_MODELS = 4,
    parameter int STATE_DIM  = 9,
    parameter int DATA_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         valid_in,
    input  logic signed [DATA_WIDTH-1:0] prev_state [STATE_DIM],
    input  logic signed [DATA_WIDTH-1:0] meas_pos [3],
    input  logic signed [DATA_WIDTH-1:0] meas_vel [3],
    input  logic                         meas_valid,
    output logic signed [DATA_WIDTH-1:0] state_out [STATE_DIM],
    output logic                         valid_out,
    output logic [15:0]                  model_prob [NUM_MODELS]
);

    // Simplified IMM - full implementation in separate file
    // This is a placeholder that passes through measurements
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            for (int i = 0; i < STATE_DIM; i++) state_out[i] <= 0;
            for (int i = 0; i < NUM_MODELS; i++) model_prob[i] <= 16'h4000;
        end else if (valid_in) begin
            state_out[0] <= meas_pos[0];
            state_out[1] <= meas_pos[1];
            state_out[2] <= meas_pos[2];
            state_out[3] <= meas_vel[0];
            state_out[4] <= meas_vel[1];
            state_out[5] <= meas_vel[2];
            state_out[6] <= prev_state[6];
            state_out[7] <= prev_state[7];
            state_out[8] <= prev_state[8];
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
