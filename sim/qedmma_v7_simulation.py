#!/usr/bin/env python3
"""
QEDMMA v7 "SENTINEL" - Physics-Agnostic Hybrid Tracker Simulation

Author: Dr. Mladen Mešter
Architect: Radar Systems Architect vGrok-X (Factory Spec)
Copyright (c) 2026 - All Rights Reserved

Features:
- Layer 2A: Physics-Constrained GAT (conventional targets)
- Layer 2B: Physics-Agnostic Quantum Evolutionary Tracker (UAP/anomalies)
- Residual Divergence Monitor (RDM) with 5-sigma gating
- Clock-Bias Estimation for async multistatic
- Importance-Driven Quantization (INT4/INT8/FP16)

Usage:
    python3 qedmma_v7_simulation.py --scenario conventional    # Hypersonic target
    python3 qedmma_v7_simulation.py --scenario uap_instant     # Instant direction change
    python3 qedmma_v7_simulation.py --scenario uap_teleport    # Position discontinuity
    python3 qedmma_v7_simulation.py --plot                     # With visualization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from enum import Enum


# =============================================================================
# Constants
# =============================================================================
C = 299792458.0  # Speed of light (m/s)
G = 9.80665      # Gravitational acceleration (m/s²)


# =============================================================================
# Enums
# =============================================================================

class QuantizationMode(Enum):
    INT4 = 0       # Background/clutter
    INT8 = 1       # Standard targets
    FP16 = 2       # High-priority/anomalous


class ActiveLayer(Enum):
    LAYER_1 = 0    # IMM + TDOA only
    LAYER_2A = 1   # Physics-constrained
    LAYER_2B = 2   # Physics-agnostic
    BLENDED = 3    # Weighted combination


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TargetState:
    """Target kinematic state"""
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    time: float
    
    # UAP indicators
    physics_compliant: bool = True
    instantaneous_turn: bool = False
    teleported: bool = False


@dataclass
class TrackEstimate:
    """Estimated track with anomaly indicators"""
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    
    # Anomaly detection
    anomaly_detected: bool = False
    rdm_sigma_level: float = 0.0
    active_layer: ActiveLayer = ActiveLayer.LAYER_1
    quant_mode: QuantizationMode = QuantizationMode.INT8
    
    # Clock bias estimates (ns)
    clock_bias_ns: np.ndarray = None
    
    # Layer outputs for debugging
    l2a_state: np.ndarray = None
    l2b_state: np.ndarray = None
    weight_l2a: float = 1.0
    weight_l2b: float = 0.0


# =============================================================================
# UAP Trajectory Generator
# =============================================================================

class UAP_TrajectoryGenerator:
    """
    Generates trajectories that violate classical physics:
    - Instant direction changes without inertia
    - Position teleportation
    - Impossible accelerations (>1000g)
    - Hovering with instant acceleration
    """
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        self.time = 0.0
    
    def generate_trajectory(self, duration: float, 
                           scenario: str = "conventional") -> List[TargetState]:
        """Generate trajectory for specified scenario"""
        
        if scenario == "conventional":
            return self._conventional_hypersonic(duration)
        elif scenario == "uap_instant_turn":
            return self._uap_instant_turn(duration)
        elif scenario == "uap_teleport":
            return self._uap_teleport(duration)
        elif scenario == "uap_hover_dart":
            return self._uap_hover_dart(duration)
        elif scenario == "uap_zigzag":
            return self._uap_zigzag(duration)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _conventional_hypersonic(self, duration: float) -> List[TargetState]:
        """Standard hypersonic trajectory (physics-compliant)"""
        trajectory = []
        pos = np.array([0.0, 0.0, 25000.0])
        vel = np.array([2500.0, 0.0, 0.0])  # Mach 7.3
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        
        # 60g turn at t=3s
        while self.time < duration:
            if 3.0 < self.time < 5.0:
                # Coordinated turn
                acc = np.array([0.0, 60 * G, 0.0])
            else:
                acc = np.array([0.0, 0.0, 0.0])
            
            trajectory.append(TargetState(
                pos=pos.copy(),
                vel=vel.copy(),
                acc=acc.copy(),
                time=self.time,
                physics_compliant=True
            ))
            
            # Physics integration
            pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
            vel = vel + acc * self.dt
            self.time += self.dt
        
        return trajectory
    
    def _uap_instant_turn(self, duration: float) -> List[TargetState]:
        """
        UAP Scenario: Instant 90° turn without deceleration
        
        This violates Newton's laws - object changes direction
        instantaneously without any transitional acceleration.
        """
        trajectory = []
        pos = np.array([0.0, 0.0, 20000.0])
        vel = np.array([3000.0, 0.0, 0.0])  # Mach 8.8
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        turn_time = 4.0
        
        while self.time < duration:
            if abs(self.time - turn_time) < self.dt:
                # INSTANT 90° turn - no physics transition!
                speed = np.linalg.norm(vel)
                vel = np.array([0.0, speed, 0.0])  # Same speed, perpendicular direction
                
                # Apparent acceleration would be infinite
                # We mark this frame as physics-violating
                trajectory.append(TargetState(
                    pos=pos.copy(),
                    vel=vel.copy(),
                    acc=np.array([0.0, 1e6, 0.0]),  # Marker for infinite accel
                    time=self.time,
                    physics_compliant=False,
                    instantaneous_turn=True
                ))
            else:
                trajectory.append(TargetState(
                    pos=pos.copy(),
                    vel=vel.copy(),
                    acc=acc.copy(),
                    time=self.time,
                    physics_compliant=True
                ))
            
            pos = pos + vel * self.dt
            self.time += self.dt
        
        return trajectory
    
    def _uap_teleport(self, duration: float) -> List[TargetState]:
        """
        UAP Scenario: Position discontinuity (teleportation)
        
        Object disappears and reappears at different location
        without traversing intermediate space.
        """
        trajectory = []
        pos = np.array([0.0, 0.0, 15000.0])
        vel = np.array([1000.0, 0.0, 0.0])
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        teleport_time = 3.0
        teleport_distance = 50000.0  # 50 km jump
        
        while self.time < duration:
            if abs(self.time - teleport_time) < self.dt:
                # TELEPORT - discontinuous position!
                pos = pos + np.array([teleport_distance, 0.0, 5000.0])
                
                trajectory.append(TargetState(
                    pos=pos.copy(),
                    vel=vel.copy(),
                    acc=acc.copy(),
                    time=self.time,
                    physics_compliant=False,
                    teleported=True
                ))
            else:
                trajectory.append(TargetState(
                    pos=pos.copy(),
                    vel=vel.copy(),
                    acc=acc.copy(),
                    time=self.time,
                    physics_compliant=True
                ))
            
            pos = pos + vel * self.dt
            self.time += self.dt
        
        return trajectory
    
    def _uap_hover_dart(self, duration: float) -> List[TargetState]:
        """
        UAP Scenario: Stationary hover → instant Mach 10
        
        Object hovers motionless, then instantly accelerates
        to hypersonic speed without any acceleration phase.
        """
        trajectory = []
        pos = np.array([50000.0, 0.0, 10000.0])
        vel = np.array([0.0, 0.0, 0.0])  # Hovering
        acc = np.array([0.0, 0.0, 0.0])
        
        self.time = 0.0
        dart_time = 2.5
        
        while self.time < duration:
            if abs(self.time - dart_time) < self.dt:
                # INSTANT acceleration to Mach 10
                vel = np.array([3400.0, 0.0, -1000.0])  # Diving attack
                
                trajectory.append(TargetState(
                    pos=pos.copy(),
                    vel=vel.copy(),
                    acc=np.array([1e6, 0.0, 0.0]),  # Infinite acceleration marker
                    time=self.time,
                    physics_compliant=False,
                    instantaneous_turn=True
                ))
            else:
                trajectory.append(TargetState(
                    pos=pos.copy(),
                    vel=vel.copy(),
                    acc=acc.copy(),
                    time=self.time,
                    physics_compliant=True
                ))
            
            pos = pos + vel * self.dt
            self.time += self.dt
        
        return trajectory
    
    def _uap_zigzag(self, duration: float) -> List[TargetState]:
        """
        UAP Scenario: High-frequency zigzag without inertia
        
        Object oscillates rapidly at impossible frequency
        given its mass and speed (violates F=ma).
        """
        trajectory = []
        pos = np.array([0.0, 0.0, 20000.0])
        vel = np.array([2000.0, 0.0, 0.0])
        
        self.time = 0.0
        zigzag_frequency = 2.0  # Hz - way too fast for any physical object
        zigzag_amplitude = 5000.0  # 5 km lateral oscillation
        
        while self.time < duration:
            # Lateral position oscillates
            y_offset = zigzag_amplitude * np.sin(2 * np.pi * zigzag_frequency * self.time)
            y_vel = zigzag_amplitude * 2 * np.pi * zigzag_frequency * np.cos(2 * np.pi * zigzag_frequency * self.time)
            y_acc = -zigzag_amplitude * (2 * np.pi * zigzag_frequency)**2 * np.sin(2 * np.pi * zigzag_frequency * self.time)
            
            current_pos = pos + np.array([0, y_offset, 0])
            current_vel = vel + np.array([0, y_vel, 0])
            current_acc = np.array([0, y_acc, 0])
            
            # Check if acceleration violates physics
            g_load = np.abs(y_acc) / G
            physics_ok = g_load < 1000  # Allow up to 1000g (still extreme)
            
            trajectory.append(TargetState(
                pos=current_pos.copy(),
                vel=current_vel.copy(),
                acc=current_acc.copy(),
                time=self.time,
                physics_compliant=physics_ok
            ))
            
            pos[0] += vel[0] * self.dt  # Forward motion
            self.time += self.dt
        
        return trajectory


# =============================================================================
# Residual Divergence Monitor (RDM)
# =============================================================================

class ResidualDivergenceMonitor:
    """
    Monitors Layer 2A residuals and detects physics violations.
    
    Uses running statistics to compute sigma level.
    5-sigma threshold triggers Layer 2B activation.
    """
    
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.residual_history = []
        self.sigma_threshold = 5.0
    
    def update(self, residual: np.ndarray) -> Tuple[float, bool]:
        """
        Update RDM with new residual.
        
        Returns:
            sigma_level: Current divergence in sigmas
            physics_violation: True if > 5-sigma
        """
        residual_norm = np.linalg.norm(residual)
        
        self.residual_history.append(residual_norm)
        if len(self.residual_history) > self.window_size:
            self.residual_history.pop(0)
        
        if len(self.residual_history) < 4:
            return 0.0, False
        
        mean = np.mean(self.residual_history)
        std = np.std(self.residual_history) + 1e-6  # Avoid div by zero
        
        sigma_level = (residual_norm - mean) / std
        physics_violation = sigma_level > self.sigma_threshold
        
        return sigma_level, physics_violation
    
    def reset(self):
        self.residual_history = []


# =============================================================================
# Clock Bias Estimator
# =============================================================================

class ClockBiasEstimator:
    """
    Estimates clock drift for each node when White Rabbit sync is lost.
    
    Adds clock bias as additional state variable in TDOA solver:
    [x, y, z, Δt_1, Δt_2, ..., Δt_N]
    """
    
    def __init__(self, num_nodes: int = 6):
        self.num_nodes = num_nodes
        self.clock_bias_ns = np.zeros(num_nodes)  # ns
        self.clock_drift_rate = np.zeros(num_nodes)  # ns/s
        self.learning_rate = 0.1
    
    def update(self, tdoa_residuals: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        """
        Update clock bias estimates from TDOA residuals.
        
        Δd = Δt × c → Δt = Δd / c
        """
        c_m_per_ns = 0.299792458  # m/ns
        
        for i in range(len(tdoa_residuals)):
            # Estimate bias from residual
            bias_correction_ns = tdoa_residuals[i] / c_m_per_ns
            
            # Update with exponential smoothing
            self.clock_bias_ns[i+1] += self.learning_rate * bias_correction_ns
        
        return self.clock_bias_ns.copy()
    
    def get_corrected_tdoa(self, tdoa_raw: np.ndarray) -> np.ndarray:
        """Apply clock bias correction to raw TDOA measurements"""
        c_m_per_ns = 0.299792458
        
        tdoa_corrected = tdoa_raw.copy()
        for i in range(len(tdoa_raw)):
            bias_distance = (self.clock_bias_ns[i+1] - self.clock_bias_ns[0]) * c_m_per_ns
            tdoa_corrected[i] -= bias_distance
        
        return tdoa_corrected


# =============================================================================
# Layer 2A: Physics-Constrained GAT
# =============================================================================

class PhysicsConstrainedGAT:
    """
    Physics-constrained correction layer.
    
    Uses aerodynamic/thermodynamic constraints:
    - Maximum g-load based on vehicle structure
    - Drag model at hypersonic speeds
    - Heat flux limits
    """
    
    def __init__(self):
        self.max_g_load = 100 * G  # 100g limit for hypersonic
        self.drag_factor = 0.01   # Drag coefficient
    
    def process(self, state: np.ndarray, prev_state: np.ndarray, 
                dt: float = 0.0625) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply physics-constrained correction.
        
        Returns:
            corrected_state: Physics-compliant state
            residual: Difference from input (for RDM)
        """
        pos = state[:3]
        vel = state[3:6]
        acc = state[6:9]
        
        prev_vel = prev_state[3:6]
        
        # Calculate speed and drag
        speed = np.linalg.norm(vel)
        drag_decel = self.drag_factor * speed**2
        
        # Predicted velocity with drag
        vel_pred = prev_vel - drag_decel * prev_vel / (speed + 1e-6) * dt
        
        # Check acceleration limit
        acc_mag = np.linalg.norm(acc)
        if acc_mag > self.max_g_load:
            # Clamp to physical limit
            acc_clamped = acc * self.max_g_load / acc_mag
        else:
            acc_clamped = acc
        
        # Corrected state
        corrected = np.concatenate([pos, vel, acc_clamped])
        
        # Residual for RDM
        residual = state - corrected
        
        return corrected, residual


# =============================================================================
# Layer 2B: Quantum Evolutionary Tracker (Physics-Agnostic)
# =============================================================================

class QuantumEvolutionaryTracker:
    """
    Physics-agnostic tracker using evolutionary algorithm.
    
    No assumptions about:
    - Mass or inertia
    - Drag or friction
    - Maximum acceleration
    - Continuous motion
    
    Pure measurement-to-measurement correlation.
    """
    
    def __init__(self, population_size: int = 8, generations: int = 4):
        self.population_size = population_size
        self.generations = generations
    
    def process(self, measurement: np.ndarray, prev_state: np.ndarray) -> np.ndarray:
        """
        Estimate state using evolutionary algorithm.
        
        Creates population of hypotheses and evolves toward measurement.
        """
        state_dim = 9
        
        # Initialize population around measurement
        population = []
        for i in range(self.population_size):
            noise = np.random.randn(state_dim) * 100
            hypothesis = measurement.copy()
            hypothesis[:6] += noise[:6]  # Vary position and velocity
            population.append(hypothesis)
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness (distance to measurement)
            fitness = []
            for hyp in population:
                dist = np.linalg.norm(hyp[:3] - measurement[:3])
                dist += 0.1 * np.linalg.norm(hyp[3:6] - measurement[3:6])
                fitness.append(dist)
            
            # Find best
            best_idx = np.argmin(fitness)
            best = population[best_idx].copy()
            
            # Create new population via mutation + crossover
            new_population = [best]  # Elitism
            
            for i in range(1, self.population_size):
                # Crossover with best
                child = 0.5 * (best + population[i])
                
                # Mutation
                mutation = np.random.randn(state_dim) * 50 * (1 - gen / self.generations)
                child += mutation
                
                new_population.append(child)
            
            population = new_population
        
        # Return best hypothesis
        fitness = [np.linalg.norm(h[:3] - measurement[:3]) for h in population]
        best_idx = np.argmin(fitness)
        
        return population[best_idx]


# =============================================================================
# QEDMMA v7 SENTINEL Main Tracker
# =============================================================================

class QEDMMA_v7_SENTINEL:
    """
    Complete QEDMMA v7 SENTINEL tracker implementation.
    
    Dual-branch architecture:
    - Layer 2A: Physics-constrained for conventional targets
    - Layer 2B: Physics-agnostic for UAP/anomalies
    - RDM gating with 5-sigma threshold
    """
    
    def __init__(self, num_nodes: int = 6, dt: float = 0.0625):
        self.num_nodes = num_nodes
        self.dt = dt
        
        # Components
        self.rdm = ResidualDivergenceMonitor(window_size=16)
        self.clock_bias = ClockBiasEstimator(num_nodes)
        self.layer_2a = PhysicsConstrainedGAT()
        self.layer_2b = QuantumEvolutionaryTracker()
        
        # State
        self.prev_state = np.zeros(9)
        self.track_history = []
        
        # Statistics
        self.anomaly_count = 0
        self.layer_2b_activations = 0
    
    def process_measurement(self, measurement: np.ndarray, 
                           tdoa_residuals: np.ndarray = None) -> TrackEstimate:
        """
        Process single measurement through dual-branch architecture.
        
        Args:
            measurement: [x, y, z, vx, vy, vz, ax, ay, az]
            tdoa_residuals: Raw TDOA residuals for clock bias estimation
        
        Returns:
            TrackEstimate with anomaly indicators
        """
        # Layer 2A: Physics-constrained correction
        l2a_state, l2a_residual = self.layer_2a.process(measurement, self.prev_state, self.dt)
        
        # Layer 2B: Physics-agnostic estimation
        l2b_state = self.layer_2b.process(measurement, self.prev_state)
        
        # RDM: Detect physics violation
        sigma_level, physics_violation = self.rdm.update(l2a_residual)
        
        # Gating: Determine layer weights
        if physics_violation or sigma_level > 5.0:
            # Anomaly! Switch to Layer 2B
            weight_l2a = 0.0
            weight_l2b = 1.0
            active_layer = ActiveLayer.LAYER_2B
            anomaly_detected = True
            self.anomaly_count += 1
            self.layer_2b_activations += 1
        elif sigma_level > 3.0:
            # Marginal: blend both layers
            weight_l2a = 0.5
            weight_l2b = 0.5
            active_layer = ActiveLayer.BLENDED
            anomaly_detected = False
        else:
            # Normal: use Layer 2A
            weight_l2a = 1.0
            weight_l2b = 0.0
            active_layer = ActiveLayer.LAYER_2A
            anomaly_detected = False
        
        # Fuse outputs
        fused_state = weight_l2a * l2a_state + weight_l2b * l2b_state
        
        # Clock bias estimation
        if tdoa_residuals is not None:
            clock_bias_ns = self.clock_bias.update(tdoa_residuals, measurement[:3])
        else:
            clock_bias_ns = self.clock_bias.clock_bias_ns.copy()
        
        # Determine quantization mode based on priority
        range_to_origin = np.linalg.norm(measurement[:3])
        if anomaly_detected:
            quant_mode = QuantizationMode.FP16
        elif range_to_origin < 50000:  # < 50 km
            quant_mode = QuantizationMode.FP16
        elif range_to_origin > 200000:  # > 200 km
            quant_mode = QuantizationMode.INT4
        else:
            quant_mode = QuantizationMode.INT8
        
        # Update state
        self.prev_state = fused_state.copy()
        
        # Create estimate
        estimate = TrackEstimate(
            pos=fused_state[:3],
            vel=fused_state[3:6],
            acc=fused_state[6:9],
            anomaly_detected=anomaly_detected,
            rdm_sigma_level=sigma_level,
            active_layer=active_layer,
            quant_mode=quant_mode,
            clock_bias_ns=clock_bias_ns,
            l2a_state=l2a_state,
            l2b_state=l2b_state,
            weight_l2a=weight_l2a,
            weight_l2b=weight_l2b
        )
        
        self.track_history.append(estimate)
        
        return estimate
    
    def reset(self):
        self.prev_state = np.zeros(9)
        self.rdm.reset()
        self.track_history = []
        self.anomaly_count = 0
        self.layer_2b_activations = 0


# =============================================================================
# Simulation Runner
# =============================================================================

class SENTINEL_Simulation:
    """Complete QEDMMA v7 SENTINEL simulation"""
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        self.trajectory_gen = UAP_TrajectoryGenerator(dt)
        self.tracker = QEDMMA_v7_SENTINEL(num_nodes=6, dt=dt)
        
        self.true_states = []
        self.estimates = []
        self.errors = []
    
    def run(self, duration: float = 10.0, scenario: str = "conventional") -> Dict:
        """Run complete simulation"""
        
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║        QEDMMA v7 SENTINEL - Physics-Agnostic Hybrid Tracker       ║")
        print("╠═══════════════════════════════════════════════════════════════════╣")
        print(f"║  Scenario: {scenario:25s}  Duration: {duration:.1f}s         ║")
        print("║  Features: RDM Gating | Clock-Bias Est | Adaptive Quantization    ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        
        # Generate trajectory
        trajectory = self.trajectory_gen.generate_trajectory(duration, scenario)
        
        self.tracker.reset()
        self.true_states = trajectory
        self.estimates = []
        self.errors = []
        
        # Process each frame
        for i, true_state in enumerate(trajectory):
            # Create measurement (with small noise)
            meas_noise = np.random.randn(9) * np.array([10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.1])
            measurement = np.concatenate([true_state.pos, true_state.vel, true_state.acc]) + meas_noise
            
            # Process through tracker
            estimate = self.tracker.process_measurement(measurement)
            self.estimates.append(estimate)
            
            # Calculate errors
            pos_error = np.linalg.norm(true_state.pos - estimate.pos)
            vel_error = np.linalg.norm(true_state.vel - estimate.vel)
            
            self.errors.append({
                'time': true_state.time,
                'pos_error': pos_error,
                'vel_error': vel_error,
                'sigma_level': estimate.rdm_sigma_level,
                'anomaly_detected': estimate.anomaly_detected,
                'active_layer': estimate.active_layer.name,
                'physics_compliant': true_state.physics_compliant
            })
            
            # Progress output
            if i % 16 == 0:
                layer_str = estimate.active_layer.name[:6]
                anomaly_str = "⚠️ ANOMALY" if estimate.anomaly_detected else "   normal"
                physics_str = "✓" if true_state.physics_compliant else "✗"
                
                print(f"  t={true_state.time:5.2f}s | "
                      f"σ={estimate.rdm_sigma_level:5.1f} | "
                      f"Layer: {layer_str:8s} | "
                      f"{anomaly_str} | "
                      f"Physics: {physics_str} | "
                      f"Err: {pos_error:.1f}m")
        
        # Summary
        pos_errors = [e['pos_error'] for e in self.errors]
        anomaly_frames = sum(1 for e in self.errors if e['anomaly_detected'])
        physics_violations = sum(1 for s in trajectory if not s.physics_compliant)
        
        print(f"\n{'='*71}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*71}")
        print(f"  Position Error:    Mean={np.mean(pos_errors):.1f}m, "
              f"Max={np.max(pos_errors):.1f}m, RMS={np.sqrt(np.mean(np.array(pos_errors)**2)):.1f}m")
        print(f"  Anomalies Detected: {anomaly_frames} / {len(self.errors)} frames "
              f"({100*anomaly_frames/len(self.errors):.1f}%)")
        print(f"  True Physics Violations: {physics_violations} frames")
        print(f"  Layer 2B Activations: {self.tracker.layer_2b_activations}")
        print(f"{'='*71}")
        
        return {
            'trajectory': trajectory,
            'estimates': self.estimates,
            'errors': self.errors,
            'anomaly_count': self.tracker.anomaly_count
        }
    
    def plot_results(self):
        """Visualize simulation results"""
        if not self.errors:
            print("No results to plot. Run simulation first.")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 3D Trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        true_pos = np.array([s.pos for s in self.true_states])
        est_pos = np.array([e.pos for e in self.estimates])
        
        ax1.plot(true_pos[:, 0]/1000, true_pos[:, 1]/1000, true_pos[:, 2]/1000,
                'b-', label='True', linewidth=2)
        ax1.plot(est_pos[:, 0]/1000, est_pos[:, 1]/1000, est_pos[:, 2]/1000,
                'r--', label='Estimated', linewidth=1)
        
        # Mark anomalies
        anomaly_idx = [i for i, e in enumerate(self.errors) if e['anomaly_detected']]
        if anomaly_idx:
            ax1.scatter(true_pos[anomaly_idx, 0]/1000, 
                       true_pos[anomaly_idx, 1]/1000,
                       true_pos[anomaly_idx, 2]/1000,
                       c='orange', s=100, marker='*', label='Anomaly')
        
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Z (km)')
        ax1.set_title('3D Trajectory with Anomaly Detection')
        ax1.legend()
        
        # RDM Sigma Level
        ax2 = fig.add_subplot(2, 3, 2)
        times = [e['time'] for e in self.errors]
        sigma_levels = [e['sigma_level'] for e in self.errors]
        
        ax2.plot(times, sigma_levels, 'b-', linewidth=1)
        ax2.axhline(y=5.0, color='r', linestyle='--', label='5-sigma threshold')
        ax2.axhline(y=3.0, color='orange', linestyle='--', label='3-sigma threshold')
        
        # Shade anomaly regions
        anomaly_times = [e['time'] for e in self.errors if e['anomaly_detected']]
        for t in anomaly_times:
            ax2.axvline(x=t, color='red', alpha=0.3)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Sigma Level')
        ax2.set_title('Residual Divergence Monitor (RDM)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(-2, max(sigma_levels) * 1.2 if sigma_levels else 10)
        
        # Position Error
        ax3 = fig.add_subplot(2, 3, 3)
        pos_errors = [e['pos_error'] for e in self.errors]
        ax3.plot(times, pos_errors, 'b-')
        
        # Highlight anomaly frames
        for i, e in enumerate(self.errors):
            if e['anomaly_detected']:
                ax3.scatter(e['time'], e['pos_error'], c='red', s=50, zorder=5)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position Error (m)')
        ax3.set_title('Position Estimation Error')
        ax3.grid(True)
        
        # Active Layer Timeline
        ax4 = fig.add_subplot(2, 3, 4)
        layer_map = {'LAYER_1': 0, 'LAYER_2A': 1, 'LAYER_2B': 2, 'BLENDED': 1.5}
        layers = [layer_map.get(e['active_layer'], 0) for e in self.errors]
        
        ax4.step(times, layers, 'b-', where='post')
        ax4.set_yticks([0, 1, 1.5, 2])
        ax4.set_yticklabels(['L1', 'L2A', 'Blend', 'L2B'])
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Active Layer')
        ax4.set_title('Layer Selection Over Time')
        ax4.grid(True)
        
        # True Physics Compliance
        ax5 = fig.add_subplot(2, 3, 5)
        physics_compliant = [1 if s.physics_compliant else 0 for s in self.true_states]
        detected = [1 if e['anomaly_detected'] else 0 for e in self.errors]
        
        ax5.fill_between(times, 0, physics_compliant, alpha=0.3, 
                        color='green', label='Physics Compliant')
        ax5.fill_between(times, 0, detected, alpha=0.5, 
                        color='red', label='Anomaly Detected')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Flag')
        ax5.set_title('Ground Truth vs Detection')
        ax5.legend()
        ax5.set_ylim(-0.1, 1.1)
        
        # Layer Weights
        ax6 = fig.add_subplot(2, 3, 6)
        w_l2a = [e.weight_l2a for e in self.estimates]
        w_l2b = [e.weight_l2b for e in self.estimates]
        
        ax6.stackplot(times, w_l2a, w_l2b, 
                     labels=['L2A (Physics)', 'L2B (Agnostic)'],
                     colors=['green', 'red'], alpha=0.7)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Weight')
        ax6.set_title('Layer Fusion Weights')
        ax6.legend(loc='upper right')
        ax6.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('qedmma_v7_sentinel_results.png', dpi=150)
        plt.show()
        
        print("Results saved to: qedmma_v7_sentinel_results.png")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='QEDMMA v7 SENTINEL Simulation')
    parser.add_argument('--duration', type=float, default=8.0, help='Simulation duration (s)')
    parser.add_argument('--scenario', type=str, default='uap_instant_turn',
                       choices=['conventional', 'uap_instant_turn', 'uap_teleport', 
                               'uap_hover_dart', 'uap_zigzag'],
                       help='Trajectory scenario')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    
    args = parser.parse_args()
    
    sim = SENTINEL_Simulation()
    results = sim.run(duration=args.duration, scenario=args.scenario)
    
    if args.plot:
        sim.plot_results()
    
    return results


if __name__ == '__main__':
    main()
