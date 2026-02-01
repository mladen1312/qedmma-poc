# 🔥 TITAN Radar - RFSoC 4x2 PYNQ Overlay

## Complete Radar-on-Chip for QEDMMA

**Author:** Dr. Mladen Mešter  
**Platform:** AMD RFSoC 4x2 (Zynq UltraScale+ ZU48DR)  
**Price:** $2,149 Academic

---

## 📊 Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **ADC** | 4× 5 GSPS, 14-bit, DC-6 GHz |
| **DAC** | 2× 9.85 GSPS, 14-bit |
| **Logic Cells** | 930,000 |
| **DSP Slices** | 4,272 |
| **Block RAM** | 38.8 Mb |
| **UltraRAM** | 22.5 Mb |
| **DDR4** | 8 GB (4GB PS + 4GB PL) |
| **High-Speed I/O** | 100 GbE QSFP28 |

---

## 📁 Directory Structure

```
rfsoc_4x2/
├── drivers/
│   └── titan_radar.py       # Main PYNQ driver
├── notebooks/
│   └── titan_radar_demo.ipynb   # Jupyter demonstration
├── hls/
│   └── zero_dsp_correlator.cpp  # HLS correlator source
├── tcl/
│   └── build_titan_overlay.tcl  # Vivado build script
└── README.md
```

---

## 🚀 Quick Start

### 1. Build Overlay (on development machine)

```bash
# Open Vivado
vivado -mode batch -source tcl/build_titan_overlay.tcl
```

### 2. Deploy to RFSoC 4x2

```bash
# Copy files to board
scp titan_radar.bit titan_radar.hwh xilinx@rfsoc4x2:/home/xilinx/
scp -r drivers/ notebooks/ xilinx@rfsoc4x2:/home/xilinx/titan_radar/
```

### 3. Run on RFSoC 4x2

```python
from titan_radar import TitanRadarOverlay, TitanConfig

# Initialize
radar = TitanRadarOverlay('titan_radar.bit')
radar.configure()
radar.start()

# Process
result = radar.process_cpi()
print(f"Detections: {len(result['detections'])}")

# Cleanup
radar.stop()
```

---

## ⚡ TITAN Radar Capabilities

| Parameter | Value |
|-----------|-------|
| Frequency | DC - 6 GHz (direct sampling) |
| Range Bins | 16,384 |
| Range Resolution | 15 m (@ 10 Mchip/s) |
| Max Range | 2,457 km theoretical |
| Doppler Bins | 1,024 |
| Beamforming | 4-channel, ±60° steering |
| ECCM | 3 simultaneous nulls |
| Max Tracks | 256 |
| Update Rate | 1,000+ Hz possible |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RFSoC 4x2 BOARD                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  RF DATA CONVERTERS                                   │  │
│  │  DAC0 ──► SMA ──► PA ──► TX Antenna                  │  │
│  │  ADC0-3 ◄── SMA ◄── LNA ◄── RX Array (4-element)     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  PROGRAMMABLE LOGIC (FPGA)                            │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │  │
│  │  │Waveform │ │Correlator│ │ Doppler │ │    CFAR     │ │  │
│  │  │Generator│→│(Zero-DSP)│→│   FFT   │→│  Detector   │ │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘ │  │
│  │       │                                      │        │  │
│  │  ┌─────────┐                          ┌──────────┐   │  │
│  │  │Beamformer│                          │ Tracker  │   │  │
│  │  │ (4-ch)  │                          │ (Kalman) │   │  │
│  │  └─────────┘                          └──────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  PROCESSOR SYSTEM (ARM)                               │  │
│  │  4× Cortex-A53 @ 1.5 GHz + 2× Cortex-R5 @ 600 MHz    │  │
│  │  Ubuntu/PYNQ + FreeRTOS                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│              QSFP28 (100 GbE) for data offload              │
└─────────────────────────────────────────────────────────────┘
```

---

## 💰 Bill of Materials

| Component | Price |
|-----------|:-----:|
| RFSoC 4x2 Board | €2,000 |
| PA Module (100W VHF) | €200 |
| 4× LNA (SPF5189Z) | €50 |
| TX Antenna (Yagi) | €40 |
| RX Array (4× Yagi) | €160 |
| Cables & connectors | €100 |
| Enclosure | €100 |
| Power supplies | €150 |
| **TOTAL** | **~€2,800** |

---

## 📈 Comparison

| Platform | ADC | DSP | Price | Verdict |
|----------|:---:|:---:|:-----:|---------|
| PlutoSDR | 61 MSPS | 80 | €230 | Starter |
| KV260 | External | 1,248 | €230 | Good processing |
| bladeRF xA9 | 122 MSPS | 684 | €860 | Native VHF |
| **RFSoC 4x2** | **5,000 MSPS** | **4,272** | **€2,000** | **COMPLETE RADAR** |

**RFSoC 4x2 = 82× faster ADC and 53× more DSP than PlutoSDR!**

---

## ⚠️ Requirements

### To Purchase RFSoC 4x2:
- University or Research Institute affiliation required
- Apply via AMD University Program
- URL: https://www.amd.com/en/corporate/university-program

### To Build Overlay:
- Vivado 2024.1 or later
- Vitis HLS 2024.1
- Linux development machine (Ubuntu 22.04 recommended)

### To Run:
- RFSoC 4x2 board with PYNQ image
- Python 3.10+
- NumPy, Matplotlib

---

## 📚 References

- [RFSoC-PYNQ Documentation](http://www.rfsoc-pynq.io/)
- [Real Digital RFSoC 4x2](https://www.realdigital.org/hardware/rfsoc-4x2)
- [AMD University Program](https://www.amd.com/en/corporate/university-program)
- [Xilinx RFSoC GitHub](https://github.com/Xilinx/RFSoC-PYNQ)

---

**Copyright © 2026 Dr. Mladen Mešter - All Rights Reserved**
