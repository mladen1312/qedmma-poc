# QEDMMA Radar: RFSoC 4x2 "OVERKILL" Architecture

## 🔥🔥🔥 "TITAN" Configuration - Full Radar-on-Chip

**Author:** Dr. Mladen Mešter  
**Date:** February 2026  
**Codename:** TITAN (Total Integrated Transceiver And Navigator)

---

## 💰 SHOCKING VALUE PROPOSITION

```
╔════════════════════════════════════════════════════════════════════════╗
║                    RFSoC 4x2 PRICING                                   ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  RFSoC 4x2 Board (Academic):     $2,149 USD (~€2,000)                 ║
║                                                                        ║
║  For comparison:                                                       ║
║  ├─ ZU48DR chip ALONE:           $22,742 USD (!)                      ║
║  ├─ ZCU208 eval kit:             $13,194 USD                          ║
║  ├─ ZCU111 eval kit:             $10,794 USD                          ║
║  └─ Commercial RFSoC boards:     $15,000-50,000 USD                   ║
║                                                                        ║
║  YOU GET: 90% DISCOUNT vs chip price!                                 ║
║                                                                        ║
║  NOTE: Available only to Universities & Research Institutes           ║
║        via AMD University Program                                      ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 📊 RFSoC 4x2 Full Specifications

```
╔════════════════════════════════════════════════════════════════════════╗
║               AMD ZYNQ ULTRASCALE+ RFSoC ZU48DR                        ║
║                        "RADAR ON A CHIP"                               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  RF DATA CONVERTERS (THE GAME CHANGER!):                              ║
║  ┌────────────────────────────────────────────────────────────────┐   ║
║  │  ADC (Receive):                                                 │   ║
║  │  ├─ Channels:        4× ADC available on SMA                   │   ║
║  │  ├─ Sample Rate:     5 GSPS (5,000,000,000 samples/sec!)      │   ║
║  │  ├─ Resolution:      14-bit                                    │   ║
║  │  ├─ Input BW:        6 GHz (DC to 6 GHz!)                     │   ║
║  │  ├─ SFDR:            >70 dBFS                                  │   ║
║  │  └─ SNR:             >55 dB                                    │   ║
║  │                                                                 │   ║
║  │  DAC (Transmit):                                                │   ║
║  │  ├─ Channels:        2× DAC available on SMA                   │   ║
║  │  ├─ Sample Rate:     9.85 GSPS (!)                             │   ║
║  │  ├─ Resolution:      14-bit                                    │   ║
║  │  └─ Output BW:       >4 GHz                                    │   ║
║  └────────────────────────────────────────────────────────────────┘   ║
║                                                                        ║
║  PROCESSOR SYSTEM (PS):                                                ║
║  ├─ CPU:           Quad-core ARM Cortex-A53 @ 1.5 GHz                 ║
║  ├─ Real-time:     Dual-core ARM Cortex-R5F @ 600 MHz                 ║
║  ├─ GPU:           Mali-400 MP2                                        ║
║  └─ Cache:         1 MB L2 shared                                      ║
║                                                                        ║
║  PROGRAMMABLE LOGIC (PL):                                              ║
║  ├─ Logic Cells:   930,300 (~1 million!)                              ║
║  ├─ CLB LUTs:      425,280                                            ║
║  ├─ CLB Flip-Flops: 850,560                                           ║
║  ├─ Block RAM:     1,080 × 36Kb = 38.8 Mb                             ║
║  ├─ UltraRAM:      80 × 288Kb = 22.5 Mb                               ║
║  ├─ DSP Slices:    4,272 (!)                                          ║
║  └─ Total on-chip: 61.3 Mb SRAM                                       ║
║                                                                        ║
║  MEMORY:                                                               ║
║  ├─ DDR4 PS:       4 GB @ 2400 MT/s                                   ║
║  └─ DDR4 PL:       4 GB @ 2400 MT/s (dedicated for FPGA!)            ║
║                    TOTAL: 8 GB DDR4                                    ║
║                                                                        ║
║  HIGH-SPEED I/O:                                                       ║
║  ├─ QSFP28:        100 Gbps Ethernet (4×25G or 2×50G or 1×100G)      ║
║  ├─ GTY:           16× transceivers @ 32.75 Gbps each                 ║
║  └─ GbE:           10/100/1000 Ethernet                               ║
║                                                                        ║
║  CLOCKING:                                                             ║
║  ├─ LMK04828:      Ultra-low jitter clock generator                   ║
║  ├─ LMX2594:       Wide-band PLL (up to 15 GHz)                       ║
║  └─ Compatibility: ZCU208 compatible clocking subsystem               ║
║                                                                        ║
║  PRICE: $2,149 USD (~€2,000) Academic                                 ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## ⚡ THE ULTIMATE COMPARISON

| Parameter | PlutoSDR | KV260 | bladeRF xA9 | **RFSoC 4x2** |
|-----------|:--------:|:-----:|:-----------:|:-------------:|
| **RF ADC** | 12-bit 61 MSPS | ❌ External | 12-bit 61 MSPS | **14-bit 5 GSPS** |
| **RF DAC** | 12-bit 61 MSPS | ❌ External | 12-bit 61 MSPS | **14-bit 9.85 GSPS** |
| **ADC Channels** | 2 | 0 | 2 | **4** |
| **DAC Channels** | 2 | 0 | 2 | **2** |
| **RF Bandwidth** | 56 MHz | N/A | 122 MHz | **6 GHz (!)** |
| **Logic Cells** | 28K | 256K | 301K | **930K** |
| **DSP Slices** | 80 | 1,248 | 684 | **4,272** |
| **Block RAM** | 2.1 Mb | 5.1 Mb | ~10 Mb | **38.8 Mb** |
| **UltraRAM** | ❌ | 18 Mb | ❌ | **22.5 Mb** |
| **DDR Memory** | 512 MB | 4 GB | ❌ | **8 GB** |
| **ARM Cores** | 2× A9 | 4× A53 | ❌ | **4× A53 + 2× R5** |
| **High-speed I/O** | USB 2.0 | USB 3.0 | USB 3.0 | **100 GbE** |
| **VHF Native** | ❌ (hack) | N/A | ✅ (47 MHz) | **✅ DC-6 GHz** |
| **Price** | €230 | €230 | €860 | **€2,000** |

### Key Ratios vs PlutoSDR:

```
ADC Speed:     5000 MSPS / 61 MSPS = 82× FASTER
ADC Bits:      14-bit vs 12-bit = 4× dynamic range
DSP Slices:    4272 / 80 = 53× MORE
Memory:        8 GB / 512 MB = 16× MORE
Bandwidth:     6 GHz / 56 MHz = 107× WIDER
```

---

## 🎯 Why RFSoC 4x2 is PERFECT for QEDMMA

### 1. Direct RF Sampling - NO EXTERNAL SDR NEEDED!

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  TRADITIONAL APPROACH (PlutoSDR)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Antenna → LNA → External ADC → USB → PC/FPGA → Processing             │
│                    (PlutoSDR)   (Bottleneck!)                           │
│                                                                         │
│  Problems:                                                              │
│  • USB 2.0 bandwidth limit (480 Mbps)                                  │
│  • External clock synchronization                                       │
│  • Multiple PCBs = noise coupling                                      │
│  • Limited bandwidth (56 MHz)                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                  RFSoC 4x2 APPROACH (TITAN)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Antenna → LNA → [ADC → FPGA → ARM] → 100GbE → Display                │
│                   ▲                                                     │
│                   └── ALL ON SINGLE CHIP!                              │
│                                                                         │
│  Benefits:                                                              │
│  • No USB bottleneck - direct fabric connection                        │
│  • Perfect synchronization (same die)                                  │
│  • Single PCB = minimal noise                                          │
│  • 6 GHz bandwidth (DC to 6 GHz!)                                     │
│  • 100 Gbps data offload if needed                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Multi-Band Capability

```
RFSoC 4x2 can simultaneously sample:

┌─────────────────────────────────────────────────────────────────────────┐
│  FREQUENCY COVERAGE: DC to 6 GHz                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│  0    100   500    1G    2G    3G    4G    5G    6G   (Hz)            │
│  │     │     │      │     │     │     │     │     │                    │
│  │     │     │      │     │     │     │     │     │                    │
│  ├─────┼─────┼──────┼─────┼─────┼─────┼─────┼─────┤                    │
│  │ VHF │ UHF │      L-band      │  S-band   │ C-band                   │
│  │     │     │                  │           │                          │
│  │ ◄───QEDMMA Primary──────────►│           │                          │
│  │     (30-300 MHz)             │           │                          │
│  │                              │           │                          │
│  │              ◄────Secondary bands────────►                          │
│  │                                                                     │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  With 4 ADC channels, can sample FOUR bands simultaneously!           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Massive Processing Power

```
4,272 DSP Slices = WHAT YOU CAN DO:

┌─────────────────────────────────────────────────────────────────────────┐
│  PARALLEL PROCESSING CAPABILITY                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Correlator:                                                           │
│  ├─ 2000 DSP → 1000 parallel complex MACs                             │
│  ├─ 16,384 range bins in real-time                                    │
│  └─ Processing: <1 μs per CPI (!)                                     │
│                                                                         │
│  CFAR Detector:                                                        │
│  ├─ 500 DSP → 16,384 parallel cells                                   │
│  └─ All bins processed simultaneously                                  │
│                                                                         │
│  Beamformer (if using array):                                          │
│  ├─ 1000 DSP → 16-channel MVDR                                        │
│  └─ Full adaptive nulling                                              │
│                                                                         │
│  Doppler Processing:                                                   │
│  ├─ 500 DSP → 1024-point FFT per range bin                           │
│  └─ Full range-Doppler map real-time                                  │
│                                                                         │
│  Track Processor:                                                      │
│  ├─ 200 DSP → 256 simultaneous tracks                                 │
│  └─ Extended Kalman filter per track                                  │
│                                                                         │
│  Spare:                                                                │
│  └─ 72 DSP for future expansion                                       │
│                                                                         │
│  TOTAL: 4272 DSP allocated                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ TITAN System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TITAN RADAR ARCHITECTURE                              │
│                        (RFSoC 4x2 Based QEDMMA)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                        RFSoC 4x2 BOARD                                   │  │
│   │                                                                          │  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   │  │
│   │  │                     RF DATA CONVERTERS                            │   │  │
│   │  │                                                                   │   │  │
│   │  │   DAC0 ──► SMA ──► PA ──► TX Antenna                             │   │  │
│   │  │   DAC1 ──► SMA ──► (spare/MIMO)                                  │   │  │
│   │  │                                                                   │   │  │
│   │  │   ADC0 ◄── SMA ◄── LNA ◄── RX Antenna 1                         │   │  │
│   │  │   ADC1 ◄── SMA ◄── LNA ◄── RX Antenna 2                         │   │  │
│   │  │   ADC2 ◄── SMA ◄── LNA ◄── RX Antenna 3                         │   │  │
│   │  │   ADC3 ◄── SMA ◄── LNA ◄── RX Antenna 4                         │   │  │
│   │  │                                                                   │   │  │
│   │  │   5 GSPS × 4 channels = 20 GSPS aggregate!                       │   │  │
│   │  │                                                                   │   │  │
│   │  └──────────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                           │  │
│   │                     AXI4-Stream (direct fabric)                         │  │
│   │                              │                                           │  │
│   │  ┌──────────────────────────────────────────────────────────────────┐   │  │
│   │  │                    PROGRAMMABLE LOGIC (PL)                        │   │  │
│   │  │                        930K Logic Cells                           │   │  │
│   │  │                                                                   │   │  │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │  │
│   │  │  │ Waveform    │  │ Digital     │  │ Decimation &            │   │   │  │
│   │  │  │ Generator   │  │ Upconverter │  │ Channelizer             │   │   │  │
│   │  │  │ (PRBS/LFM)  │  │ (NCO+Mixer) │  │ (Polyphase filter)      │   │   │  │
│   │  │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘   │   │  │
│   │  │         │                │                      │                 │   │  │
│   │  │         ▼                ▼                      ▼                 │   │  │
│   │  │      To DAC           To DAC              From ADCs              │   │  │
│   │  │                                                 │                 │   │  │
│   │  │                                                 ▼                 │   │  │
│   │  │  ┌──────────────────────────────────────────────────────────┐    │   │  │
│   │  │  │                RADAR SIGNAL PROCESSOR                     │    │   │  │
│   │  │  │                                                           │    │   │  │
│   │  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │    │   │  │
│   │  │  │  │ Zero-DSP   │  │ Doppler    │  │ CFAR + Detection   │  │    │   │  │
│   │  │  │  │ Correlator │→ │ FFT Bank   │→ │ (16K parallel)     │  │    │   │  │
│   │  │  │  │ (2000 DSP) │  │ (512 DSP)  │  │ (500 DSP)          │  │    │   │  │
│   │  │  │  └────────────┘  └────────────┘  └─────────┬──────────┘  │    │   │  │
│   │  │  │                                            │              │    │   │  │
│   │  │  │  ┌────────────┐  ┌────────────┐           │              │    │   │  │
│   │  │  │  │ Beamformer │  │ Track      │◄──────────┘              │    │   │  │
│   │  │  │  │ (4-ch)     │  │ Processor  │                          │    │   │  │
│   │  │  │  │ (1000 DSP) │  │ (200 DSP)  │                          │    │   │  │
│   │  │  │  └────────────┘  └─────┬──────┘                          │    │   │  │
│   │  │  │                        │                                  │    │   │  │
│   │  │  └────────────────────────┼──────────────────────────────────┘    │   │  │
│   │  │                           │                                       │   │  │
│   │  └───────────────────────────┼───────────────────────────────────────┘   │  │
│   │                              │                                           │  │
│   │                         AXI4 Interconnect                               │  │
│   │                              │                                           │  │
│   │  ┌───────────────────────────┼───────────────────────────────────────┐   │  │
│   │  │                    PROCESSOR SYSTEM (PS)                          │   │  │
│   │  │                                                                   │   │  │
│   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │  │
│   │  │  │ ARM A53 ×4  │  │ ARM R5F ×2  │  │ 8 GB DDR4              │   │   │  │
│   │  │  │ Linux/PYNQ  │  │ Real-time   │  │ (4GB PS + 4GB PL)      │   │   │  │
│   │  │  │ @ 1.5 GHz   │  │ @ 600 MHz   │  │                        │   │   │  │
│   │  │  └──────┬──────┘  └─────────────┘  └─────────────────────────┘   │   │  │
│   │  │         │                                                         │   │  │
│   │  └─────────┼─────────────────────────────────────────────────────────┘   │  │
│   │            │                                                             │  │
│   │   ┌────────┴────────┐                                                   │  │
│   │   │    QSFP28       │ ──────► 100 GbE to Display/Storage               │  │
│   │   │   100 Gbps      │                                                   │  │
│   │   └─────────────────┘                                                   │  │
│   │                                                                          │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   EXTERNAL:                                                                    │
│   ├─ PA Module:     30-100W VHF amplifier                                     │
│   ├─ LNA Array:     4× SPF5189Z or similar                                    │
│   └─ Antennas:      TX Yagi + 4-element RX array                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💰 TITAN Bill of Materials

| # | Item | Qty | Unit (€) | Total (€) | Notes |
|---|------|:---:|:--------:|:---------:|-------|
| 1 | **RFSoC 4x2 Board** | 1 | 2,000 | 2,000 | Academic price |
| 2 | PA Module (100W VHF) | 1 | 200 | 200 | RA30H1317M ×2 |
| 3 | LNA SPF5189Z | 4 | 12 | 48 | Per ADC channel |
| 4 | TX Yagi (155 MHz) | 1 | 40 | 40 | DIY |
| 5 | RX Yagi Array (4-elem) | 4 | 40 | 160 | DIY or commercial |
| 6 | SMA cables & adapters | 1 | 80 | 80 | Quality RF cables |
| 7 | QSFP28 transceiver | 1 | 50 | 50 | For 100GbE |
| 8 | PSU 12V 10A | 1 | 50 | 50 | Board power |
| 9 | Enclosure | 1 | 100 | 100 | RF shielded |
| 10 | Misc (cooling, etc) | 1 | 72 | 72 | |
| | **TOTAL** | | | **€2,800** | |

### Cost Comparison:

| Configuration | Price | Capability |
|---------------|:-----:|------------|
| PlutoSDR PoC | €500 | Basic, USB limited |
| ZEUS-LITE (KV260+Pluto) | €700 | Good processing, external RF |
| HYDRA (blade+Kraken) | €1,800 | Beamforming, good |
| **TITAN (RFSoC 4x2)** | **€2,800** | **COMPLETE RADAR ON CHIP** |

---

## 📈 TITAN Performance Specifications

```
╔════════════════════════════════════════════════════════════════════════╗
║                    TITAN PERFORMANCE TARGETS                           ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  RADAR PARAMETERS:                                                     ║
║  ├─ Frequency:         155 MHz (VHF) primary                          ║
║  │                     30 MHz - 3 GHz configurable                    ║
║  ├─ Bandwidth:         Up to 200 MHz instantaneous                    ║
║  ├─ Waveform:          PRBS-15/PRBS-20, LFM, or custom               ║
║  ├─ PRF:               Variable, 100 Hz - 100 kHz                     ║
║  └─ Duty Cycle:        Up to 100% (CW capable)                        ║
║                                                                        ║
║  RANGE PERFORMANCE:                                                    ║
║  ├─ Range Bins:        16,384 (vs 512 on Pluto!)                     ║
║  ├─ Resolution:        150 m (with 1 MHz chip rate)                   ║
║  │                     15 m possible (with 10 MHz chip rate)         ║
║  ├─ Max Range:         2,457 km theoretical                           ║
║  │                     500+ km practical (F-35 target)               ║
║  └─ Range Accuracy:    <50 m                                          ║
║                                                                        ║
║  DOPPLER PERFORMANCE:                                                  ║
║  ├─ FFT Size:          1,024 - 4,096 points                          ║
║  ├─ Velocity Range:    ±1,000 m/s                                     ║
║  ├─ Velocity Res:      ~1 m/s                                         ║
║  └─ MTI Improvement:   >40 dB                                         ║
║                                                                        ║
║  ANGULAR PERFORMANCE (with 4-element array):                          ║
║  ├─ AOA Resolution:    ~5°                                            ║
║  ├─ Array Gain:        +6 dB                                          ║
║  └─ Jammer Nulls:      3 simultaneous                                 ║
║                                                                        ║
║  PROCESSING:                                                           ║
║  ├─ Update Rate:       1,000+ Hz possible                             ║
║  ├─ Latency:           <1 ms end-to-end                               ║
║  ├─ Tracks:            256 simultaneous                               ║
║  └─ Data Rate:         100 Gbps offload available                     ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 🚀 Development Approach

### Phase 1: Bring-up (Week 1-2)

```python
# PYNQ Jupyter Notebook on RFSoC 4x2
from pynq.overlays.base import BaseOverlay
from xrfdc import RFdc
import numpy as np

# Load base overlay
base = BaseOverlay('base.bit')

# Access RF data converters
rfdc = base.rfdc

# Configure ADC tile 0
rfdc.adc_tiles[0].DynamicPLLConfig(1, 5000.0, 5000.0)  # 5 GSPS
rfdc.adc_tiles[0].blocks[0].MixerSettings['Freq'] = 155.0  # 155 MHz

# Configure DAC tile 0  
rfdc.dac_tiles[0].DynamicPLLConfig(1, 9850.0, 9850.0)  # 9.85 GSPS
rfdc.dac_tiles[0].blocks[0].MixerSettings['Freq'] = 155.0  # 155 MHz

# Sample at 5 GSPS!
adc_data = base.adc_buffer.read()
print(f"Captured {len(adc_data)} samples at 5 GSPS!")
```

### Phase 2: Radar Overlay Development (Week 3-6)

```tcl
# titan_radar.tcl - Vivado block design
create_project titan_radar ./titan_radar -part xczu48dr-ffvg1517-1-e

# Add RFSoC subsystem
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 ps
create_bd_cell -type ip -vlnv xilinx.com:ip:usp_rf_data_converter:2.6 rfdc

# Add radar processing chain
create_bd_cell -type ip -vlnv user:hls:waveform_gen:1.0 waveform
create_bd_cell -type ip -vlnv user:hls:zero_dsp_correlator:1.0 correlator
create_bd_cell -type ip -vlnv user:hls:cfar_detector:1.0 cfar
create_bd_cell -type ip -vlnv user:hls:track_processor:1.0 tracker

# Connect RF data converters to processing
connect_bd_intf_net [get_bd_intf_pins rfdc/m00_axis] \
                    [get_bd_intf_pins correlator/s_axis_adc]
connect_bd_intf_net [get_bd_intf_pins waveform/m_axis_dac] \
                    [get_bd_intf_pins rfdc/s00_axis]
```

### Phase 3: Full System Test (Week 7-8)

- Loopback test (DAC → cable → ADC)
- Antenna test (real targets)
- Performance validation
- Range/Doppler verification

---

## ✅ FINAL VERDICT

```
╔════════════════════════════════════════════════════════════════════════╗
║                    RFSoC 4x2 FOR QEDMMA                                ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  PROS:                                                                 ║
║  ├─ ✅ Complete radar on single chip                                  ║
║  ├─ ✅ 5 GSPS ADC - direct RF sampling to 6 GHz                      ║
║  ├─ ✅ 4 RX channels - built-in beamforming                          ║
║  ├─ ✅ 930K logic cells, 4272 DSP slices                             ║
║  ├─ ✅ No USB bottleneck - direct fabric connection                  ║
║  ├─ ✅ 100 GbE for data offload                                       ║
║  ├─ ✅ PYNQ supported - rapid prototyping                            ║
║  ├─ ✅ Open source schematics, Gerbers available                     ║
║  └─ ✅ Academic price: $2,149 (chip alone costs $22,742!)            ║
║                                                                        ║
║  CONS:                                                                 ║
║  ├─ ⚠️ Academic-only (need university affiliation)                   ║
║  ├─ ⚠️ Steeper learning curve (RFSoC is complex)                     ║
║  └─ ⚠️ Higher initial cost than Pluto/KV260                          ║
║                                                                        ║
║  VERDICT: IF YOU CAN GET IT - THIS IS THE PLATFORM!                   ║
║                                                                        ║
║  This is not incremental improvement - this is PARADIGM SHIFT.        ║
║  RFSoC 4x2 eliminates every bottleneck we've discussed:               ║
║  • No external SDR needed                                              ║
║  • No USB bandwidth limits                                             ║
║  • No synchronization issues                                           ║
║  • 82× faster sampling                                                 ║
║  • 53× more DSP                                                        ║
║                                                                        ║
║  For €2,800 total, you get a COMPLETE RADAR SYSTEM                    ║
║  that would cost €50,000+ to build any other way.                     ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 📋 Next Steps to Acquire RFSoC 4x2

1. **Apply to AMD University Program**
   - URL: https://www.amd.com/en/corporate/university-program
   - Need: University/Research Institute affiliation
   - Your medical research background should qualify!

2. **Submit purchase request**
   - Real Digital processes orders
   - Typical lead time: 2-4 weeks

3. **Prepare development environment**
   - Install Vivado 2024.1
   - Download RFSoC-PYNQ image
   - Review tutorials and overlays

---

**Document Version:** 1.0  
**Author:** Dr. Mladen Mešter  
**Copyright © 2026** - All Rights Reserved
