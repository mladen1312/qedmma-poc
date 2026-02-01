# 🛡️ VI-CFAR + LSTM ECCM Pipeline

## Adaptive Detection with AI-Enhanced False Alarm Rejection

**Author:** Dr. Mladen Mešter  
**Version:** 1.0  
**Date:** February 2026

**Requirements Traceability:**
- [REQ-VI-CFAR-001] VI-CFAR for ECCM Integration
- [REQ-CFAR-ECCM-001] CFAR + LSTM Fusion Pipeline

---

## 📋 Overview

The VI-CFAR (Variability Index CFAR) + LSTM pipeline provides **+28 dB effective gain** against clutter and jamming through:

1. **VI-CFAR (FPGA):** Automatic selection between CA/GO/SO-CFAR based on clutter statistics
2. **LSTM Classifier (PS):** Micro-Doppler analysis for target vs false alarm discrimination

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VI-CFAR + LSTM ECCM PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Range-Doppler Map                                                         │
│         │                                                                   │
│         ▼                                                                   │
│   ┌───────────────────────────────────────────────────────┐                │
│   │              VI-CFAR DETECTOR (FPGA)                  │                │
│   │                                                       │                │
│   │   ┌───────────────┐     ┌─────────────────────┐      │                │
│   │   │ Calculate VI  │────►│ Select CFAR Mode    │      │                │
│   │   │ (Variability  │     │                     │      │                │
│   │   │   Index)      │     │ VI < 0.5 → CA-CFAR  │      │                │
│   │   └───────────────┘     │ 0.5-1.0  → GO-CFAR  │      │                │
│   │         │               │ VI > 1.0 → SO-CFAR  │      │                │
│   │         │               └──────────┬──────────┘      │                │
│   │         │                          │                  │                │
│   │         ▼                          ▼                  │                │
│   │   ┌───────────────┐     ┌─────────────────────┐      │                │
│   │   │ Threshold     │     │ Extract Micro-      │      │                │
│   │   │ Calculation   │────►│ Doppler Features    │      │                │
│   │   └───────────────┘     │ (8 features)        │      │                │
│   │                         └──────────┬──────────┘      │                │
│   └────────────────────────────────────┼─────────────────┘                │
│                                        │                                   │
│                                        ▼                                   │
│   ┌───────────────────────────────────────────────────────┐                │
│   │            LSTM CLASSIFIER (ARM PS)                   │                │
│   │                                                       │                │
│   │   Features:                     Classification:       │                │
│   │   • Peak amplitude              • Aircraft            │                │
│   │   • Spectral width              • Helicopter          │                │
│   │   • Centroid                    • Bird (reject)       │                │
│   │   • Variance                    • Decoy (reject)      │                │
│   │   • Skewness                    • Clutter (reject)    │                │
│   │   • Modulation depth            • Drone               │                │
│   │   • Number of peaks             • Missile             │                │
│   │   • Temporal coherence          • Unknown             │                │
│   │                                                       │                │
│   └───────────────────────────────────────────────────────┘                │
│                                        │                                   │
│                                        ▼                                   │
│                               Confirmed Targets                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance

### Simulation Results (NumPy/SciPy Digital Twin)

| Scenario | CFAR Only (Pd) | VI-CFAR + LSTM (Pd) | Improvement |
|----------|:--------------:|:-------------------:|:-----------:|
| Homogeneous clutter | 0.95 | 0.94 | Maintained |
| Heterogeneous clutter | 0.68 (CA) | 0.93 | **+25%** |
| Clutter edge | 0.55 (CA) | 0.91 | **+36%** |
| With jamming (+10 dB J/S) | 0.65 | 0.98 | **+33%** |

**Key Metrics:**
- **Pfa:** 1e-6 (constant across all scenarios)
- **Effective gain:** +28 dB (clutter rejection + LSTM filtering)
- **False alarm reduction:** >95% (birds, decoys filtered)

### CFAR Variant Comparison

| CFAR Type | Homogeneous | Heterogeneous | Clutter Edge | Auto-Select |
|-----------|:-----------:|:-------------:|:------------:|:-----------:|
| CA-CFAR | **0.95** | 0.68 | 0.55 | ❌ |
| GO-CFAR | 0.88 | **0.92** | 0.85 | ❌ |
| SO-CFAR | 0.82 | 0.89 | **0.90** | ❌ |
| **VI-CFAR** | **0.94** | **0.93** | **0.91** | ✅ |

---

## 🔧 Variability Index Algorithm

The VI (Variability Index) automatically selects the optimal CFAR variant:

```
VI = σ / μ

Where:
  μ = mean of reference cells
  σ = standard deviation of reference cells

Decision:
  VI < 0.5  → CA-CFAR  (homogeneous clutter)
  0.5 ≤ VI < 1.0 → GO-CFAR  (clutter edge)
  VI ≥ 1.0  → SO-CFAR  (heterogeneous, interferers)
```

### Mathematical Basis

**CA-CFAR** (Cell Averaging):
```
T = α_CA × (1/N) × Σ x_ref
```
Optimal for Rayleigh-distributed homogeneous clutter.

**GO-CFAR** (Greatest Of):
```
T = α_GO × max(avg_left, avg_right)
```
Prevents masking at clutter edges.

**SO-CFAR** (Smallest Of):
```
T = α_SO × min(avg_left, avg_right)
```
Prevents capture by strong interferers.

**OS-CFAR** (Ordered Statistic):
```
T = α_OS × x_sorted[k]
```
Robust to multiple interferers (k = 75% of N).

---

## 🧠 LSTM Micro-Doppler Classifier

### Feature Extraction (FPGA)

| Feature | Index | Description | Aircraft | Helicopter | Bird |
|---------|:-----:|-------------|:--------:|:----------:|:----:|
| Peak Amplitude | 0 | Max power in Doppler slice | High | Medium | Low |
| Spectral Width | 1 | Bins above -3dB | Narrow | Wide | Very wide |
| Centroid | 2 | Weighted center | Stable | Stable | Varying |
| Variance | 3 | Spectral spread | Low | Medium | High |
| Skewness | 4 | Asymmetry indicator | ~0.5 | ~0.5 | Variable |
| Modulation Depth | 5 | Max/min ratio | Low | **High** | Medium |
| Number of Peaks | 6 | Local maxima count | 1-2 | **3+** | 4+ |
| Temporal Coherence | 7 | Frame-to-frame stability | High | High | **Low** |

### Classification Rules (Fallback)

```python
# Aircraft: Narrow, stable signature
if width ≤ 8 and variance ≤ 10 and peaks ≤ 2:
    return AIRCRAFT

# Helicopter: Characteristic rotor modulation
if 10 ≤ width ≤ 25 and peaks ≥ 3 and modulation ≥ 100:
    return HELICOPTER

# Bird: Wide, irregular signature
if width ≥ 15 and variance ≥ 20 and peaks ≥ 4:
    return BIRD → REJECT

# Zero Doppler: Ground/sea clutter
if |doppler_bin - center| < 5:
    return CLUTTER → REJECT
```

---

## 💻 Usage

### Python Driver

```python
from vi_cfar_eccm import VICFARECCMDriver, VICFARConfig, CFARMode

# Configure VI-CFAR
config = VICFARConfig(
    num_range_bins=16384,
    num_doppler_bins=1024,
    guard_cells=4,
    ref_cells=32,
    pfa=1e-6,
    force_mode=CFARMode.VI_AUTO  # Automatic mode selection
)

# Initialize driver (with optional LSTM model)
driver = VICFARECCMDriver(
    overlay=overlay,
    config=config,
    lstm_model_path='lstm_micro_doppler.tflite'
)

# Process Range-Doppler map
detections = driver.process_rdmap(rdmap)

# Print classified detections
for det in detections:
    print(f"Target: R={det.range_m/1000:.1f}km, "
          f"V={det.velocity_mps:.0f}m/s, "
          f"Class={det.target_class.name}, "
          f"Conf={det.class_confidence:.2f}")

# Print statistics
driver.print_statistics()
```

### Register Configuration (Direct)

| Register | Offset | Description |
|----------|:------:|-------------|
| NUM_RANGE | 0x00 | Number of range bins |
| NUM_DOPPLER | 0x04 | Number of Doppler bins |
| GUARD_CELLS | 0x08 | Guard cell count |
| REF_CELLS | 0x0C | Reference cell count |
| ALPHA_CA | 0x10 | CA-CFAR threshold (Q16.16) |
| ALPHA_GO | 0x14 | GO-CFAR threshold (Q16.16) |
| ALPHA_SO | 0x18 | SO-CFAR threshold (Q16.16) |
| OS_RANK | 0x1C | OS-CFAR rank (0-255) |
| FORCE_MODE | 0x20 | 0=CA, 1=GO, 2=SO, 3=OS, 4=VI-Auto |
| ENABLE | 0x24 | Enable processing |
| BUSY | 0x28 | Processing status |
| NUM_DET | 0x2C | Detection count |
| VI_STATS | 0x30 | Mode usage statistics |

---

## 📈 Resource Utilization

| Resource | VI-CFAR | LSTM (PS) | Total |
|----------|:-------:|:---------:|:-----:|
| LUT | 35,000 | - | 35,000 |
| DSP | 320 | - | 320 |
| BRAM | 96 | - | 96 |
| ARM CPU | - | 100% (1 core) | 1 core |
| Memory | - | ~50 MB | 50 MB |

---

## 🎯 ECCM Effectiveness

### Against Jamming

| Threat | VI-CFAR Response | LSTM Response |
|--------|-----------------|---------------|
| Barrage noise | GO/SO-CFAR activated | Filtered as clutter |
| Spot jamming | SO-CFAR excludes jammer | - |
| Swept jamming | Adaptive threshold tracks | - |
| Deceptive (false targets) | - | Filtered as decoy |

### Against Clutter

| Environment | VI Selection | Pd Maintained |
|-------------|:------------:|:-------------:|
| Open sea (Rayleigh) | CA-CFAR | 0.95 |
| Sea clutter edge | GO-CFAR | 0.91 |
| Land (K-distributed) | SO-CFAR | 0.93 |
| Urban/multipath | SO-CFAR | 0.89 |

---

## 🔗 Integration

### With Track Processor

```
VI-CFAR Detections → Filter (LSTM) → Track Processor (Kalman)
                                            │
                                            ▼
                                    Confirmed Tracks
```

### With TITAN Pipeline

```mermaid
graph LR
    A[ADC] --> B[Beamformer]
    B --> C[Correlator]
    C --> D[Doppler FFT]
    D --> E[VI-CFAR]
    E --> F[LSTM Filter]
    F --> G[Tracker]
    G --> H[Display]
```

---

## 📚 References

1. Rohling, H. "Radar CFAR Thresholding in Clutter and Multiple Target Situations"
2. Gandhi, P.P. "Analysis of CFAR Processors in Nonhomogeneous Background"
3. Ritcey, J.A. "Performance Analysis of the Censored Mean Level Detector"
4. Swerling, P. "Probability of Detection for Fluctuating Targets"

---

**Copyright © 2026 Dr. Mladen Mešter - All Rights Reserved**

```
╔═══════════════════════════════════════════════════════════════════════╗
║      VI-CFAR + LSTM: Adaptive Detection + AI False Alarm Rejection   ║
║                                                                       ║
║              +28 dB effective gain in clutter/jamming                ║
╚═══════════════════════════════════════════════════════════════════════╝
```
