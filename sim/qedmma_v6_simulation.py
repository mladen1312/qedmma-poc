#!/usr/bin/env python3
"""
QEDMMA v6 Dual-Layer Hypersonic Tracker - Python Simulation

Author: Dr. Mladen Mešter
Architect: Radar Systems Architect vGrok-X (Factory Spec)
Copyright (c) 2026 - All Rights Reserved

This simulation validates the QEDMMA v6 algorithm before RTL synthesis:
- 6-node multistatic TDOA + Doppler fusion
- IMM (4 kinematic models: CV, CA, CT, Jerk)
- Layer 2 GAT residual correction for extreme maneuvers
- Realistic hypersonic target trajectory (Mach 8+, 60g maneuvers)

Usage:
    python3 qedmma_v6_simulation.py                    # Run full simulation
    python3 qedmma_v6_simulation.py --plot             # With visualization
    python3 qedmma_v6_simulation.py --export-vcd       # Export for RTL comparison
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse


# =============================================================================
# Physical Constants
# =============================================================================
C = 299792458.0  # Speed of light (m/s)
G = 9.80665      # Gravitational acceleration (m/s²)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NodeConfig:
    """Multistatic node configuration"""
    id: int
    x: float  # ECEF or local ENU (meters)
    y: float
    z: float
    
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class TargetState:
    """Target kinematic state"""
    pos: np.ndarray  # [x, y, z] meters
    vel: np.ndarray  # [vx, vy, vz] m/s
    acc: np.ndarray  # [ax, ay, az] m/s²
    time: float      # seconds
    
    def speed(self) -> float:
        return np.linalg.norm(self.vel)
    
    def mach(self) -> float:
        return self.speed() / 340.0  # Approximate sea-level speed of sound
    
    def g_load(self) -> float:
        return np.linalg.norm(self.acc) / G


@dataclass
class Measurement:
    """TDOA and Doppler measurements from nodes"""
    tdoa: np.ndarray        # TDOA values (range differences) in meters
    doppler: np.ndarray     # Radial velocities in m/s
    time: float
    noise_tdoa: float = 10.0    # TDOA noise std (meters)
    noise_doppler: float = 5.0  # Doppler noise std (m/s)


@dataclass
class TrackState:
    """Estimated track state"""
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    covariance: np.ndarray
    model_probs: np.ndarray  # IMM model probabilities
    layer2_active: bool = False


# =============================================================================
# Hypersonic Target Trajectory Generator
# =============================================================================

class HypersonicTrajectoryGenerator:
    """
    Generates realistic hypersonic target trajectories with:
    - Boost phase (constant acceleration)
    - Cruise phase (constant velocity)
    - Maneuver phase (coordinated turns, pull-up)
    - Terminal phase (dive)
    """
    
    def __init__(self, dt: float = 0.0625):  # 16 Hz update
        self.dt = dt
        self.time = 0.0
        
    def generate_trajectory(self, duration: float, scenario: str = "pull_up") -> List[TargetState]:
        """Generate trajectory for specified scenario"""
        
        trajectory = []
        
        if scenario == "pull_up":
            # 60g pull-up maneuver scenario
            trajectory = self._generate_pull_up(duration)
        elif scenario == "evasive":
            # S-turn evasive maneuver
            trajectory = self._generate_evasive(duration)
        elif scenario == "cruise":
            # Straight cruise (baseline)
            trajectory = self._generate_cruise(duration)
        elif scenario == "terminal_dive":
            # Terminal dive phase
            trajectory = self._generate_terminal_dive(duration)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        return trajectory
    
    def _generate_pull_up(self, duration: float) -> List[TargetState]:
        """
        Extreme pull-up maneuver:
        - Initial: Mach 8 dive at -30° angle
        - Maneuver: 60g pull-up over 2 seconds
        - Final: Level flight at Mach 6
        """
        trajectory = []
        
        # Initial conditions
        pos = np.array([0.0, 0.0, 30000.0])  # 30 km altitude
        vel_mag = 8 * 340.0  # Mach 8
        dive_angle = np.radians(-30)
        vel = np.array([vel_mag * np.cos(dive_angle), 0, vel_mag * np.sin(dive_angle)])
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        maneuver_start = 3.0
        maneuver_duration = 2.0
        
        while self.time < duration:
            # Phase determination
            if self.time < maneuver_start:
                # Pre-maneuver: constant velocity dive
                acc = np.array([0.0, 0.0, 0.0])
            elif self.time < maneuver_start + maneuver_duration:
                # Maneuver: 60g pull-up (vertical acceleration)
                g_load = 60 * G
                # Pull-up in the vertical plane
                vel_horizontal = np.sqrt(vel[0]**2 + vel[1]**2)
                if vel_horizontal > 0:
                    acc = np.array([-vel[0]/vel_horizontal * g_load * 0.1,
                                    0,
                                    g_load * 0.99])
                else:
                    acc = np.array([0, 0, g_load])
            else:
                # Post-maneuver: level off
                if vel[2] > 0:
                    # Still climbing, reduce vertical velocity
                    acc = np.array([0, 0, -2 * G])
                else:
                    acc = np.array([0, 0, 0])
            
            # State storage
            trajectory.append(TargetState(
                pos=pos.copy(),
                vel=vel.copy(),
                acc=acc.copy(),
                time=self.time
            ))
            
            # Integration (simple Euler for demonstration)
            pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
            vel = vel + acc * self.dt
            
            # Speed limiting (drag model approximation)
            speed = np.linalg.norm(vel)
            max_speed = 10 * 340  # Mach 10 limit
            if speed > max_speed:
                vel = vel * max_speed / speed
            
            self.time += self.dt
        
        return trajectory
    
    def _generate_evasive(self, duration: float) -> List[TargetState]:
        """S-turn evasive maneuver at Mach 6"""
        trajectory = []
        
        pos = np.array([0.0, 0.0, 20000.0])
        vel = np.array([6 * 340.0, 0.0, 0.0])  # Mach 6 horizontal
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        turn_period = 4.0  # 4 second turn cycle
        max_lateral_g = 30 * G
        
        while self.time < duration:
            # Sinusoidal lateral acceleration (S-turn)
            phase = 2 * np.pi * self.time / turn_period
            acc = np.array([0.0, max_lateral_g * np.sin(phase), 0.0])
            
            trajectory.append(TargetState(
                pos=pos.copy(),
                vel=vel.copy(),
                acc=acc.copy(),
                time=self.time
            ))
            
            pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
            vel = vel + acc * self.dt
            self.time += self.dt
        
        return trajectory
    
    def _generate_cruise(self, duration: float) -> List[TargetState]:
        """Constant velocity cruise at Mach 8"""
        trajectory = []
        
        pos = np.array([0.0, 0.0, 25000.0])
        vel = np.array([8 * 340.0, 0.0, 0.0])
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        
        while self.time < duration:
            trajectory.append(TargetState(
                pos=pos.copy(),
                vel=vel.copy(),
                acc=acc.copy(),
                time=self.time
            ))
            
            pos = pos + vel * self.dt
            self.time += self.dt
        
        return trajectory
    
    def _generate_terminal_dive(self, duration: float) -> List[TargetState]:
        """Terminal dive with increasing g-load"""
        trajectory = []
        
        pos = np.array([0.0, 0.0, 40000.0])
        vel = np.array([5 * 340.0, 0.0, -3 * 340.0])  # Steep dive
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        
        while self.time < duration:
            # Increasing g-load as altitude decreases
            altitude = pos[2]
            if altitude > 10000:
                g_load = 20 * G * (1 - altitude / 40000)
            else:
                g_load = 40 * G
            
            # Dive acceleration (increase speed toward target)
            vel_dir = vel / np.linalg.norm(vel)
            acc = vel_dir * g_load * 0.3
            
            trajectory.append(TargetState(
                pos=pos.copy(),
                vel=vel.copy(),
                acc=acc.copy(),
                time=self.time
            ))
            
            pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
            vel = vel + acc * self.dt
            self.time += self.dt
            
            if pos[2] < 0:
                break
        
        return trajectory


# =============================================================================
# Multistatic Node Network
# =============================================================================

class MultistatiNetwork:
    """
    6-node multistatic radar network for TDOA + Doppler
    
    Node layout: Pentagon + center configuration
    Baseline: ~600 km for hypersonic detection
    """
    
    def __init__(self, baseline_km: float = 600):
        self.baseline = baseline_km * 1000  # Convert to meters
        self.nodes = self._setup_nodes()
        self.reference_node = 0  # TDOA reference
        
    def _setup_nodes(self) -> List[NodeConfig]:
        """Setup 6-node pentagon + center configuration"""
        nodes = []
        
        # Center node (reference)
        nodes.append(NodeConfig(id=0, x=0, y=0, z=0))
        
        # Pentagon nodes at baseline radius
        radius = self.baseline / 2
        for i in range(5):
            angle = 2 * np.pi * i / 5
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0  # Ground level
            nodes.append(NodeConfig(id=i+1, x=x, y=y, z=z))
        
        return nodes
    
    def generate_measurements(self, target: TargetState, 
                             noise_tdoa: float = 10.0,
                             noise_doppler: float = 5.0) -> Measurement:
        """Generate TDOA and Doppler measurements for target state"""
        
        # Calculate range to each node
        ranges = []
        radial_velocities = []
        
        for node in self.nodes:
            node_pos = node.position()
            
            # Range: ||target_pos - node_pos||
            delta = target.pos - node_pos
            r = np.linalg.norm(delta)
            ranges.append(r)
            
            # Radial velocity: v · r_hat
            if r > 0:
                r_hat = delta / r
                v_r = np.dot(target.vel, r_hat)
            else:
                v_r = 0
            radial_velocities.append(v_r)
        
        # TDOA: range differences relative to reference node
        r_ref = ranges[self.reference_node]
        tdoa = np.array([ranges[i] - r_ref for i in range(1, len(ranges))])
        
        # Add measurement noise
        tdoa_noisy = tdoa + np.random.randn(len(tdoa)) * noise_tdoa
        doppler_noisy = np.array(radial_velocities) + np.random.randn(len(radial_velocities)) * noise_doppler
        
        return Measurement(
            tdoa=tdoa_noisy,
            doppler=doppler_noisy,
            time=target.time,
            noise_tdoa=noise_tdoa,
            noise_doppler=noise_doppler
        )


# =============================================================================
# IMM Filter (4 Models)
# =============================================================================

class IMMFilter:
    """
    Interacting Multiple Model Filter with 4 kinematic models:
    - CV: Constant Velocity
    - CA: Constant Acceleration
    - CT: Coordinated Turn
    - Jerk: Constant Jerk (for extreme maneuvers)
    """
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        self.num_models = 4
        
        # Model transition probability matrix
        self.trans_prob = np.array([
            [0.85, 0.05, 0.05, 0.05],  # From CV
            [0.05, 0.85, 0.05, 0.05],  # From CA
            [0.05, 0.05, 0.85, 0.05],  # From CT
            [0.05, 0.05, 0.05, 0.85],  # From Jerk
        ])
        
        # Model probabilities
        self.mu = np.array([0.4, 0.3, 0.2, 0.1])  # Initial: favor CV/CA
        
        # State dimension: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state_dim = 9
        
        # Per-model states
        self.states = [np.zeros(self.state_dim) for _ in range(self.num_models)]
        
        # Per-model covariances (diagonal for efficiency)
        self.covs = [np.eye(self.state_dim) * 1000 for _ in range(self.num_models)]
        
        # Process noise per model
        self.Q = [
            np.diag([10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.1]),     # CV: low noise
            np.diag([100, 100, 100, 10, 10, 10, 1, 1, 1]),      # CA: medium
            np.diag([500, 500, 500, 50, 50, 50, 5, 5, 5]),      # CT: high
            np.diag([1000, 1000, 1000, 100, 100, 100, 10, 10, 10]),  # Jerk: very high
        ]
        
    def predict(self):
        """Run prediction step for all models"""
        for m in range(self.num_models):
            self.states[m] = self._model_predict(m, self.states[m])
            self.covs[m] = self.covs[m] + self.Q[m]
    
    def _model_predict(self, model_id: int, state: np.ndarray) -> np.ndarray:
        """Predict state using specified kinematic model"""
        x = state.copy()
        dt = self.dt
        
        if model_id == 0:  # CV
            # Position: p += v * dt
            x[0:3] += x[3:6] * dt
            # Velocity: unchanged
            # Acceleration: zero
            x[6:9] = 0
            
        elif model_id == 1:  # CA
            # Position: p += v*dt + 0.5*a*dt²
            x[0:3] += x[3:6] * dt + 0.5 * x[6:9] * dt**2
            # Velocity: v += a*dt
            x[3:6] += x[6:9] * dt
            # Acceleration: unchanged
            
        elif model_id == 2:  # CT (simplified)
            # Position: p += v*dt
            x[0:3] += x[3:6] * dt
            # Velocity: rotates (simplified small angle)
            omega = 0.1  # Turn rate
            x[3] = x[3] * np.cos(omega * dt) - x[4] * np.sin(omega * dt)
            x[4] = x[3] * np.sin(omega * dt) + x[4] * np.cos(omega * dt)
            
        elif model_id == 3:  # Jerk
            # Full kinematic integration with jerk
            x[0:3] += x[3:6] * dt + 0.5 * x[6:9] * dt**2
            x[3:6] += x[6:9] * dt
            # Acceleration evolves (jerk model)
            x[6:9] *= 1.02  # Small jerk increase
        
        return x
    
    def update(self, measurement: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Update step with measurement"""
        likelihoods = np.zeros(self.num_models)
        
        for m in range(self.num_models):
            # Innovation
            z_pred = H @ self.states[m]
            innovation = measurement - z_pred
            
            # Innovation covariance
            S = H @ self.covs[m] @ H.T + R
            
            # Likelihood (Gaussian)
            try:
                S_inv = np.linalg.inv(S)
                likelihood = np.exp(-0.5 * innovation @ S_inv @ innovation)
                likelihood /= np.sqrt(np.linalg.det(2 * np.pi * S))
            except:
                likelihood = 1e-10
            
            likelihoods[m] = max(likelihood, 1e-10)
            
            # Kalman update
            K = self.covs[m] @ H.T @ S_inv
            self.states[m] = self.states[m] + K @ innovation
            self.covs[m] = (np.eye(self.state_dim) - K @ H) @ self.covs[m]
        
        # Update model probabilities
        c = self.trans_prob.T @ self.mu
        self.mu = likelihoods * c
        self.mu /= np.sum(self.mu)
    
    def get_combined_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return probability-weighted combined state"""
        combined_state = np.zeros(self.state_dim)
        combined_cov = np.zeros((self.state_dim, self.state_dim))
        
        for m in range(self.num_models):
            combined_state += self.mu[m] * self.states[m]
        
        # Combined covariance (simplified)
        for m in range(self.num_models):
            diff = self.states[m] - combined_state
            combined_cov += self.mu[m] * (self.covs[m] + np.outer(diff, diff))
        
        return combined_state, combined_cov


# =============================================================================
# TDOA Least Squares Solver with Doppler Fusion
# =============================================================================

class TDOADopplerSolver:
    """
    Gauss-Newton iterative solver for 3D position estimation
    using TDOA + Doppler measurements from multistatic network.
    """
    
    def __init__(self, network: MultistatiNetwork, max_iter: int = 8, 
                 conv_thresh: float = 1.0):
        self.network = network
        self.max_iter = max_iter
        self.conv_thresh = conv_thresh
        
    def solve(self, meas: Measurement, init_pos: np.ndarray, 
              init_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Solve for position and velocity using TDOA + Doppler.
        
        Returns:
            pos: Estimated position [x, y, z]
            vel: Estimated velocity [vx, vy, vz]
            converged: True if solution converged
        """
        pos = init_pos.copy()
        vel = init_vel.copy()
        
        for iteration in range(self.max_iter):
            # Calculate predicted measurements
            pred_tdoa, pred_doppler, J_pos, J_vel = self._calc_jacobian(pos, vel)
            
            # Residuals
            res_tdoa = meas.tdoa - pred_tdoa
            res_doppler = meas.doppler - pred_doppler
            
            # Weighted least squares update
            # Position update from TDOA
            W_tdoa = np.eye(len(res_tdoa)) / (meas.noise_tdoa**2)
            
            try:
                JtWJ = J_pos.T @ W_tdoa @ J_pos + np.eye(3) * 0.01  # Regularization
                JtWr = J_pos.T @ W_tdoa @ res_tdoa
                delta_pos = np.linalg.solve(JtWJ, JtWr)
            except:
                delta_pos = np.zeros(3)
            
            # Velocity update from Doppler
            W_doppler = np.eye(len(res_doppler)) / (meas.noise_doppler**2)
            
            try:
                JtWJ_vel = J_vel.T @ W_doppler @ J_vel + np.eye(3) * 0.01
                JtWr_vel = J_vel.T @ W_doppler @ res_doppler
                delta_vel = np.linalg.solve(JtWJ_vel, JtWr_vel)
            except:
                delta_vel = np.zeros(3)
            
            # Apply updates with damping
            alpha = 0.5
            pos = pos + alpha * delta_pos
            vel = vel + alpha * delta_vel
            
            # Check convergence
            if np.linalg.norm(delta_pos) < self.conv_thresh:
                return pos, vel, True
        
        return pos, vel, False
    
    def _calc_jacobian(self, pos: np.ndarray, vel: np.ndarray):
        """Calculate predicted measurements and Jacobians"""
        nodes = self.network.nodes
        ref_node = self.network.reference_node
        
        # Ranges to each node
        ranges = []
        unit_vectors = []
        
        for node in nodes:
            delta = pos - node.position()
            r = np.linalg.norm(delta)
            ranges.append(r)
            unit_vectors.append(delta / r if r > 0 else np.zeros(3))
        
        # Predicted TDOA (range differences)
        r_ref = ranges[ref_node]
        pred_tdoa = np.array([ranges[i] - r_ref for i in range(1, len(ranges))])
        
        # Predicted Doppler (radial velocities)
        pred_doppler = np.array([np.dot(vel, uv) for uv in unit_vectors])
        
        # TDOA Jacobian: d(TDOA_i)/d(pos) = unit_vec_i - unit_vec_ref
        J_pos = np.zeros((len(pred_tdoa), 3))
        for i in range(len(pred_tdoa)):
            J_pos[i] = unit_vectors[i+1] - unit_vectors[ref_node]
        
        # Doppler Jacobian: d(v_r)/d(vel) = unit_vec
        J_vel = np.array(unit_vectors)
        
        return pred_tdoa, pred_doppler, J_pos, J_vel


# =============================================================================
# Layer 2: GAT Residual Correction (Simplified Python Version)
# =============================================================================

class GATResidualCorrection:
    """
    Simplified Graph Attention Network for residual correction
    during extreme maneuvers (>60g).
    
    In RTL: This is implemented as INT8 quantized neural network.
    Here: Full-precision Python simulation.
    """
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 16):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights (pretrained values would be loaded in practice)
        np.random.seed(42)
        self.W_attn = np.random.randn(hidden_dim, state_dim) * 0.1
        self.W_out = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b_attn = np.zeros(hidden_dim)
        self.b_out = np.zeros(state_dim)
        
        # Activation threshold for Layer 2
        self.g_threshold = 60 * G  # 60g
        
    def forward(self, state: np.ndarray, acceleration_mag: float) -> np.ndarray:
        """
        Compute residual correction if acceleration exceeds threshold.
        
        Args:
            state: Current state estimate [pos, vel, acc]
            acceleration_mag: Magnitude of estimated acceleration
            
        Returns:
            residual: Correction to apply to state
        """
        if acceleration_mag < self.g_threshold:
            return np.zeros(self.state_dim)
        
        # Normalize input
        state_norm = state / (np.abs(state).max() + 1e-6)
        
        # Hidden layer with ReLU
        hidden = np.maximum(0, self.W_attn @ state_norm + self.b_attn)
        
        # Output layer
        residual = self.W_out @ hidden + self.b_out
        
        # Scale residual based on acceleration (higher g = larger correction)
        scale = min(acceleration_mag / self.g_threshold, 5.0)
        residual *= scale * 0.01  # Small correction factor
        
        return residual


# =============================================================================
# Main Simulation
# =============================================================================

class QEDMMAv6Simulation:
    """Complete QEDMMA v6 simulation with all components"""
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        
        # Initialize components
        self.trajectory_gen = HypersonicTrajectoryGenerator(dt)
        self.network = MultistatiNetwork(baseline_km=600)
        self.imm_filter = IMMFilter(dt)
        self.tdoa_solver = TDOADopplerSolver(self.network)
        self.gat_correction = GATResidualCorrection()
        
        # Results storage
        self.true_states = []
        self.estimated_states = []
        self.errors = []
        
    def run(self, duration: float = 10.0, scenario: str = "pull_up") -> Dict:
        """Run complete simulation"""
        
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║      QEDMMA v6 Dual-Layer Hypersonic Tracker Simulation      ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Scenario: {scenario:20s}  Duration: {duration:.1f}s          ║")
        print(f"║  Network: {len(self.network.nodes)} nodes, {self.network.baseline/1000:.0f} km baseline               ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
        
        # Generate trajectory
        trajectory = self.trajectory_gen.generate_trajectory(duration, scenario)
        
        # Initialize filter with first state
        if trajectory:
            init_state = trajectory[0]
            for m in range(self.imm_filter.num_models):
                self.imm_filter.states[m][:3] = init_state.pos
                self.imm_filter.states[m][3:6] = init_state.vel
                self.imm_filter.states[m][6:9] = init_state.acc
        
        # Process each time step
        for i, true_state in enumerate(trajectory):
            # Generate measurements
            meas = self.network.generate_measurements(true_state)
            
            # Layer 1: IMM Prediction
            self.imm_filter.predict()
            imm_state, imm_cov = self.imm_filter.get_combined_state()
            
            # Layer 1: TDOA + Doppler Fusion
            ls_pos, ls_vel, converged = self.tdoa_solver.solve(
                meas, 
                imm_state[:3], 
                imm_state[3:6]
            )
            
            # Layer 1: IMM Update
            H = np.zeros((6, 9))
            H[:3, :3] = np.eye(3)  # Position measurement
            H[3:6, 3:6] = np.eye(3)  # Velocity measurement
            measurement = np.concatenate([ls_pos, ls_vel])
            R = np.diag([meas.noise_tdoa**2] * 3 + [meas.noise_doppler**2] * 3)
            
            self.imm_filter.update(measurement, H, R)
            
            # Get Layer 1 output
            layer1_state, _ = self.imm_filter.get_combined_state()
            
            # Layer 2: GAT Residual Correction (if high-g detected)
            acc_mag = np.linalg.norm(layer1_state[6:9])
            residual = self.gat_correction.forward(layer1_state, acc_mag)
            
            # Final fused estimate
            final_state = layer1_state + residual
            
            # Store results
            self.true_states.append(true_state)
            self.estimated_states.append(TrackState(
                pos=final_state[:3].copy(),
                vel=final_state[3:6].copy(),
                acc=final_state[6:9].copy(),
                covariance=imm_cov,
                model_probs=self.imm_filter.mu.copy(),
                layer2_active=(acc_mag > self.gat_correction.g_threshold)
            ))
            
            # Calculate errors
            pos_error = np.linalg.norm(true_state.pos - final_state[:3])
            vel_error = np.linalg.norm(true_state.vel - final_state[3:6])
            self.errors.append({
                'time': true_state.time,
                'pos_error': pos_error,
                'vel_error': vel_error,
                'g_load': true_state.g_load(),
                'layer2_active': acc_mag > self.gat_correction.g_threshold
            })
            
            # Progress output
            if i % 16 == 0:
                print(f"  t={true_state.time:5.2f}s | "
                      f"Mach {true_state.mach():.1f} | "
                      f"{true_state.g_load():.1f}g | "
                      f"Pos Err: {pos_error:.1f}m | "
                      f"IMM: {self.imm_filter.mu} | "
                      f"L2: {'ON' if acc_mag > self.gat_correction.g_threshold else 'off'}")
        
        # Summary statistics
        pos_errors = [e['pos_error'] for e in self.errors]
        vel_errors = [e['vel_error'] for e in self.errors]
        
        print(f"\n{'='*64}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*64}")
        print(f"  Position Error:  Mean={np.mean(pos_errors):.1f}m, "
              f"Max={np.max(pos_errors):.1f}m, RMS={np.sqrt(np.mean(np.array(pos_errors)**2)):.1f}m")
        print(f"  Velocity Error:  Mean={np.mean(vel_errors):.1f}m/s, "
              f"Max={np.max(vel_errors):.1f}m/s")
        print(f"  Layer 2 Active:  {sum(e['layer2_active'] for e in self.errors)} / {len(self.errors)} frames")
        print(f"{'='*64}")
        
        return {
            'pos_errors': pos_errors,
            'vel_errors': vel_errors,
            'errors': self.errors,
            'trajectory': trajectory,
            'estimates': self.estimated_states
        }
    
    def plot_results(self):
        """Plot simulation results"""
        if not self.errors:
            print("No results to plot. Run simulation first.")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D Trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        true_pos = np.array([s.pos for s in self.true_states])
        est_pos = np.array([s.pos for s in self.estimated_states])
        
        ax1.plot(true_pos[:, 0]/1000, true_pos[:, 1]/1000, true_pos[:, 2]/1000, 
                 'b-', label='True', linewidth=2)
        ax1.plot(est_pos[:, 0]/1000, est_pos[:, 1]/1000, est_pos[:, 2]/1000, 
                 'r--', label='Estimated', linewidth=1)
        
        # Plot nodes
        for node in self.network.nodes:
            ax1.scatter(node.x/1000, node.y/1000, node.z/1000, 
                       c='green', s=100, marker='^')
        
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Z (km)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # Position Error
        ax2 = fig.add_subplot(2, 3, 2)
        times = [e['time'] for e in self.errors]
        pos_errors = [e['pos_error'] for e in self.errors]
        ax2.plot(times, pos_errors, 'b-')
        ax2.axhline(y=50, color='r', linestyle='--', label='50m threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Estimation Error')
        ax2.legend()
        ax2.grid(True)
        
        # Velocity Error
        ax3 = fig.add_subplot(2, 3, 3)
        vel_errors = [e['vel_error'] for e in self.errors]
        ax3.plot(times, vel_errors, 'g-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity Error (m/s)')
        ax3.set_title('Velocity Estimation Error')
        ax3.grid(True)
        
        # G-Load Profile
        ax4 = fig.add_subplot(2, 3, 4)
        g_loads = [e['g_load'] for e in self.errors]
        l2_active = [e['layer2_active'] for e in self.errors]
        
        ax4.plot(times, g_loads, 'b-', label='G-Load')
        ax4.axhline(y=60, color='r', linestyle='--', label='Layer 2 threshold (60g)')
        ax4.fill_between(times, 0, max(g_loads), 
                        where=l2_active, alpha=0.3, color='orange', 
                        label='Layer 2 Active')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('G-Load')
        ax4.set_title('Acceleration Profile')
        ax4.legend()
        ax4.grid(True)
        
        # IMM Model Probabilities
        ax5 = fig.add_subplot(2, 3, 5)
        model_probs = np.array([s.model_probs for s in self.estimated_states])
        ax5.stackplot(times, model_probs.T, 
                     labels=['CV', 'CA', 'CT', 'Jerk'],
                     colors=['blue', 'green', 'orange', 'red'],
                     alpha=0.7)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Model Probability')
        ax5.set_title('IMM Model Probabilities')
        ax5.legend(loc='upper right')
        ax5.set_ylim(0, 1)
        ax5.grid(True)
        
        # Speed Profile
        ax6 = fig.add_subplot(2, 3, 6)
        true_speeds = [s.mach() for s in self.true_states]
        est_speeds = [np.linalg.norm(s.vel) / 340 for s in self.estimated_states]
        ax6.plot(times, true_speeds, 'b-', label='True Mach')
        ax6.plot(times, est_speeds, 'r--', label='Estimated Mach')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Mach Number')
        ax6.set_title('Speed Profile')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig('qedmma_v6_simulation_results.png', dpi=150)
        plt.show()
        
        print("Results saved to: qedmma_v6_simulation_results.png")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='QEDMMA v6 Hypersonic Tracker Simulation')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (s)')
    parser.add_argument('--scenario', type=str, default='pull_up', 
                       choices=['pull_up', 'evasive', 'cruise', 'terminal_dive'],
                       help='Trajectory scenario')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--export-vcd', action='store_true', help='Export for RTL comparison')
    
    args = parser.parse_args()
    
    # Run simulation
    sim = QEDMMAv6Simulation()
    results = sim.run(duration=args.duration, scenario=args.scenario)
    
    # Plot if requested
    if args.plot:
        sim.plot_results()
    
    # Export for RTL comparison
    if args.export_vcd:
        export_file = f'qedmma_v6_{args.scenario}_vectors.json'
        with open(export_file, 'w') as f:
            export_data = {
                'scenario': args.scenario,
                'duration': args.duration,
                'dt': sim.dt,
                'node_positions': [(n.x, n.y, n.z) for n in sim.network.nodes],
                'trajectory': [
                    {
                        'time': s.time,
                        'pos': s.pos.tolist(),
                        'vel': s.vel.tolist(),
                        'acc': s.acc.tolist()
                    }
                    for s in sim.true_states
                ],
                'errors': results['errors']
            }
            json.dump(export_data, f, indent=2)
        print(f"Exported test vectors to: {export_file}")
    
    return results


if __name__ == '__main__':
    main()
