# TITAN VHF Anti-Stealth Radar - Complete Bill of Materials
# Version 2.0.0 - Production Ready
# Author: Dr. Mladen Mešter
# Generated: 2026-02-02

## ══════════════════════════════════════════════════════════════════════════════
##                           EXECUTIVE SUMMARY
## ══════════════════════════════════════════════════════════════════════════════

**Total System Cost: €2,847 - €3,295** (depending on antenna configuration)

| Category | Budget | Standard | Enhanced |
|----------|-------:|--------:|--------:|
| RFSoC 4x2 Platform | €2,000 | €2,000 | €2,000 |
| Power Amplifier | €85 | €85 | €145 |
| LNA Modules (4×) | €48 | €48 | €80 |
| Filters & Passives | €102 | €122 | €165 |
| Antennas | €225 | €375 | €600 |
| Cables & Connectors | €130 | €145 | €175 |
| Power System | €95 | €105 | €125 |
| Enclosure & Thermal | €120 | €140 | €180 |
| **TOTAL** | **€2,805** | **€3,020** | **€3,470** |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      1. CORE PLATFORM - RFSoC 4x2
## ══════════════════════════════════════════════════════════════════════════════

### 1.1 Main Board

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| RFSoC 4x2 Development Board | RFSOC4X2 | 1 | €2,000 | €2,000 | AMD/Avnet | AMD University Program pricing |

**Specifications:**
- Zynq UltraScale+ ZU48DR
- 4× ADC: 5 GSPS, 14-bit
- 2× DAC: 9.85 GSPS, 14-bit
- 930K logic cells, 4,272 DSP slices
- 4 GB DDR4, 100GbE QSFP28

**Ordering:**
1. Apply to AMD University Program: https://www.amd.com/en/corporate/university-program
2. Request RFSoC 4x2 board
3. Approval typically 2-4 weeks

**Alternative Sources:**
| Alternative | Price | Notes |
|-------------|------:|-------|
| Xilinx ZCU111 | €8,000 | More channels, higher cost |
| Xilinx ZCU216 | €12,000 | Production grade |
| Used RFSoC4x2 | €1,500-1,800 | Check eBay, university surplus |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      2. RF FRONTEND - TRANSMIT PATH
## ══════════════════════════════════════════════════════════════════════════════

### 2.1 Power Amplifier

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| VHF Power Amplifier Module | RA60H1317M | 1 | €85 | €85 | Mouser/AliExpress | 60W, 134-174 MHz |

**RA60H1317M Specifications:**
- Frequency: 134-174 MHz
- Output Power: 60W @ 12.5V
- Gain: 16 dB typical
- Efficiency: 60% typical
- Supply: 12.5V, 10A peak
- Package: H2M (flanged)

**Alternative PAs:**

| Part Number | Power | Freq Range | Price | Supplier | Notes |
|-------------|------:|------------|------:|----------|-------|
| RA30H1317M | 30W | 134-174 MHz | €45 | Mouser | Lower power option |
| MRF300AN | 300W | 1.8-250 MHz | €150 | Mouser | Higher power, needs matching |
| BLF188XR | 1200W | 1.8-500 MHz | €180 | LDMOS | Very high power |
| RA60H1317M1 | 60W | 134-174 MHz | €90 | Mouser | Updated version |

### 2.2 TX Driver Stage

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| VHF Driver Amplifier | MAR-6+ | 1 | €3 | €3 | Mini-Circuits | 20 dB gain, 50 mW |
| RF Transformer | T1-6T-KK81+ | 1 | €5 | €5 | Mini-Circuits | 1:1 impedance |

### 2.3 TX Filtering

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| VHF Bandpass Filter | SBP-150+ | 1 | €35 | €35 | Mini-Circuits | 127-173 MHz, SMA |
| Low Pass Filter | SLP-200+ | 1 | €22 | €22 | Mini-Circuits | DC-190 MHz harmonic suppression |

**TX Path Subtotal: €150**

---

## ══════════════════════════════════════════════════════════════════════════════
##                      3. RF FRONTEND - RECEIVE PATH
## ══════════════════════════════════════════════════════════════════════════════

### 3.1 Low Noise Amplifiers (4× for Array)

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| VHF LNA Module | SPF5189Z | 4 | €12 | €48 | AliExpress | 50-4000 MHz, NF 0.6 dB |

**SPF5189Z Specifications:**
- Frequency: 50 MHz - 4 GHz
- Noise Figure: 0.6 dB @ 150 MHz
- Gain: 18.7 dB
- P1dB: +22 dBm
- Supply: 3.3-5V, 70 mA

**Alternative LNAs:**

| Part Number | NF | Gain | Price | Supplier | Notes |
|-------------|---:|-----:|------:|----------|-------|
| PGA-103+ | 0.5 dB | 15 dB | €8 | Mini-Circuits | Excellent NF |
| PSA4-5043+ | 0.75 dB | 21 dB | €10 | Mini-Circuits | Higher gain |
| SKY67151-396LF | 0.4 dB | 20 dB | €6 | Skyworks | Best NF, SMD |
| Custom MMIC | 0.3 dB | 25 dB | €25 | Design | Ultimate performance |

### 3.2 RX Filtering (4× for Array)

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| VHF Bandpass Filter | SBP-150+ | 4 | €35 | €140 | Mini-Circuits | 127-173 MHz |

**Note:** Consider single cavity filter + splitter for cost reduction.

### 3.3 Bias Tees (4× for Array)

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| Bias Tee | ZFBT-4R2GW+ | 4 | €8 | €32 | Mini-Circuits | 10-4200 MHz |

**Alternative:** AliExpress bias tees: €3 each

### 3.4 RX Protection

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| TVS Diode Array | PESD5V0S1BL | 4 | €0.50 | €2 | Nexperia | ESD protection |
| RF Limiter | JLML-01-1+ | 4 | €4 | €16 | Mini-Circuits | Optional, protects ADC |

**RX Path Subtotal: €238**

---

## ══════════════════════════════════════════════════════════════════════════════
##                      4. ANTENNA SYSTEM
## ══════════════════════════════════════════════════════════════════════════════

### 4.1 Configuration Options

#### Option A: Budget (€225)
| Item | Qty | Unit Price | Total | Notes |
|------|----:|----------:|------:|-------|
| TX: 3-element Yagi | 1 | €45 | €45 | 6 dBi, homebrew possible |
| RX: 3-element Yagi | 4 | €45 | €180 | 4-element array |

#### Option B: Standard (€375) - RECOMMENDED
| Item | Qty | Unit Price | Total | Notes |
|------|----:|----------:|------:|-------|
| TX: 5-element Yagi | 1 | €75 | €75 | 9 dBi |
| RX: 5-element Yagi | 4 | €75 | €300 | 15 dBi array gain |

#### Option C: Enhanced (€600)
| Item | Qty | Unit Price | Total | Notes |
|------|----:|----------:|------:|-------|
| TX: 7-element Yagi | 1 | €120 | €120 | 11 dBi |
| RX: 7-element Yagi | 4 | €120 | €480 | 17 dBi array gain |

### 4.2 Commercial VHF Yagi Sources

| Supplier | Model | Elements | Gain | Price | URL |
|----------|-------|----------|------|------:|-----|
| Diamond | A144S10 | 10 | 13.1 dBi | €180 | HRO |
| Cushcraft | A148-3S | 3 | 6.5 dBi | €65 | DX Engineering |
| M2 Antennas | 2M5 | 5 | 9.5 dBi | €120 | M2 Antenna |
| InnovAntennas | 2m LFA | 5 | 9.8 dBi | €150 | InnovAntennas |
| Homebrew | Yagi | 5-7 | 9-11 dBi | €30 | DIY |

### 4.3 Homebrew Yagi Bill of Materials

**5-Element Yagi @ 155 MHz (DIY Cost: ~€30)**

| Item | Qty | Unit Price | Total | Notes |
|------|----:|----------:|------:|-------|
| Aluminum tube 12mm × 2m | 3 | €5 | €15 | Elements |
| Aluminum square 25mm × 2m | 1 | €8 | €8 | Boom |
| SO-239 connector | 1 | €2 | €2 | Feedpoint |
| Stainless hardware | 1 set | €5 | €5 | Mounting |

**Element Dimensions (155 MHz):**
- Reflector: 100.5 cm
- Driven: 91.5 cm (with gamma match)
- Director 1: 88.0 cm
- Director 2: 86.2 cm
- Director 3: 84.5 cm
- Spacing: See titan_antenna_design.py

### 4.4 Antenna Mounting

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| Mast (aluminum, 6m) | Generic | 1 | €80 | €80 | Local |
| Rotator (optional) | Yaesu G-450C | 1 | €350 | €350 | Ham radio |
| Mast clamps | Generic | 5 | €5 | €25 | Local |
| Guy wire kit | Stainless | 1 | €30 | €30 | Local |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      5. CABLES & CONNECTORS
## ══════════════════════════════════════════════════════════════════════════════

### 5.1 Coaxial Cables

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| LMR-400 (30m) | LMR-400 | 30m | €2/m | €60 | Mouser | Main feedlines |
| RG-316 (5m) | RG-316 | 5m | €1.50/m | €8 | Mouser | Internal connections |
| RG-402 Semi-rigid (2m) | RG-402 | 2m | €5/m | €10 | Mouser | Board connections |

**Cable Loss @ 155 MHz:**
| Cable Type | Loss per 100m |
|------------|---------------|
| LMR-400 | 2.7 dB |
| RG-213 | 4.5 dB |
| RG-58 | 10 dB |

### 5.2 RF Connectors

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| N-Type Male (LMR-400) | Generic | 10 | €3 | €30 | AliExpress |
| SMA Male (RG-316) | Generic | 20 | €1 | €20 | AliExpress |
| SMA Female PCB | Generic | 10 | €0.50 | €5 | AliExpress |
| N-to-SMA adapter | Generic | 5 | €3 | €15 | AliExpress |

### 5.3 RF Accessories

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| 30 dB Attenuator (N) | HAT-30+ | 2 | €25 | €50 | Mini-Circuits |
| DC Block | BLK-89-S+ | 4 | €8 | €32 | Mini-Circuits |
| Termination 50Ω | TERM-50 | 4 | €5 | €20 | Generic |

**Cables & Connectors Subtotal: €250**

---

## ══════════════════════════════════════════════════════════════════════════════
##                      6. POWER SYSTEM
## ══════════════════════════════════════════════════════════════════════════════

### 6.1 Main Power Supplies

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| 12V 30A PSU | S-360-12 | 1 | €35 | €35 | AliExpress | RFSoC + peripherals |
| 28V 15A PSU | S-400-28 | 1 | €45 | €45 | AliExpress | PA supply |

### 6.2 Power Distribution

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| Fuse holder + fuses | Generic | 1 | €10 | €10 | Local |
| Terminal blocks | Generic | 10 | €0.50 | €5 | Local |
| Power switch 20A | Generic | 1 | €5 | €5 | Local |
| Power inlet IEC | Generic | 1 | €3 | €3 | Local |

### 6.3 Bias Power (LNAs)

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| DC-DC 5V 3A | LM2596 | 1 | €3 | €3 | AliExpress | LNA bias |
| Capacitors 100µF | Generic | 4 | €0.25 | €1 | Local |

**Power System Subtotal: €107**

---

## ══════════════════════════════════════════════════════════════════════════════
##                      7. ENCLOSURE & THERMAL
## ══════════════════════════════════════════════════════════════════════════════

### 7.1 Main Enclosure

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| Aluminum case 400×300×150 | Generic | 1 | €60 | €60 | AliExpress | Main electronics |
| RF shield internal | Copper sheet | 1 | €15 | €15 | Local | EMI isolation |

### 7.2 Thermal Management

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| Heatsink (PA) | 100×100×40mm | 1 | €15 | €15 | AliExpress |
| Fan 80mm | Generic | 2 | €5 | €10 | AliExpress |
| Thermal paste | Arctic MX-4 | 1 | €8 | €8 | Amazon |
| Thermal pad | Generic | 4 | €2 | €8 | AliExpress |

### 7.3 Environmental Protection

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| Weatherproof box (outdoor) | IP65 | 1 | €25 | €25 | Local |
| Cable glands | PG9/PG11 | 10 | €1 | €10 | Local |
| Silicone sealant | Generic | 1 | €5 | €5 | Local |

**Enclosure Subtotal: €156**

---

## ══════════════════════════════════════════════════════════════════════════════
##                      8. MISCELLANEOUS
## ══════════════════════════════════════════════════════════════════════════════

### 8.1 Storage & Peripherals

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| SD Card 64GB (Class 10) | SanDisk | 2 | €12 | €24 | Amazon |
| USB-Ethernet adapter | Generic | 1 | €10 | €10 | Amazon |
| HDMI cable | Generic | 1 | €5 | €5 | Local |

### 8.2 Test Equipment (Optional but Recommended)

| Item | Part Number | Qty | Unit Price | Total | Supplier | Notes |
|------|-------------|----:|----------:|------:|----------|-------|
| NanoVNA | NanoVNA-H4 | 1 | €60 | €60 | AliExpress | Antenna tuning |
| SWR Meter | Diamond SX-200 | 1 | €80 | €80 | Ham radio | TX monitoring |
| Dummy Load 100W | Generic | 1 | €25 | €25 | AliExpress | Testing |

### 8.3 Safety Equipment

| Item | Part Number | Qty | Unit Price | Total | Supplier |
|------|-------------|----:|----------:|------:|----------|
| Lightning arrestor (N) | Polyphaser | 2 | €35 | €70 | RF supplier |
| Grounding rod | Copper 1.5m | 1 | €15 | €15 | Local |
| Ground wire 10mm² | 10m | 1 | €10 | €10 | Local |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      9. COMPLETE SYSTEM SUMMARY
## ══════════════════════════════════════════════════════════════════════════════

### Standard Configuration Total

| Category | Cost |
|----------|-----:|
| 1. RFSoC 4x2 Platform | €2,000 |
| 2. TX Path (PA, filters) | €150 |
| 3. RX Path (4× LNA, filters) | €238 |
| 4. Antennas (Standard) | €375 |
| 5. Cables & Connectors | €250 |
| 6. Power System | €107 |
| 7. Enclosure & Thermal | €156 |
| 8. Miscellaneous | €39 |
| **SUBTOTAL** | **€3,315** |
| Contingency (10%) | €332 |
| **TOTAL WITH CONTINGENCY** | **€3,647** |

### Volume Pricing (10 units)

| Category | 1× | 10× | Savings |
|----------|---:|----:|--------:|
| RFSoC 4x2 | €2,000 | €1,800 | 10% |
| RF Components | €388 | €310 | 20% |
| Antennas | €375 | €300 | 20% |
| Other | €552 | €470 | 15% |
| **Total per unit** | €3,315 | €2,880 | **13%** |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      10. SUPPLIER INFORMATION
## ══════════════════════════════════════════════════════════════════════════════

### Primary Suppliers

| Supplier | Categories | Lead Time | Notes |
|----------|------------|-----------|-------|
| **Mouser** | RF, semiconductors | 2-5 days | Best selection |
| **DigiKey** | RF, semiconductors | 2-5 days | Alternative to Mouser |
| **Mini-Circuits** | RF modules, filters | 3-7 days | Direct for volume |
| **AliExpress** | LNAs, PSUs, enclosures | 2-4 weeks | Budget option |
| **Amazon** | Cables, storage | 1-2 days | Fast delivery |
| **Local electronics** | Passives, hardware | Same day | Small parts |

### AMD University Program

**Application Process:**
1. Visit: https://www.amd.com/en/corporate/university-program
2. Submit application with research proposal
3. Wait 2-4 weeks for approval
4. Order at academic pricing

### Ham Radio Suppliers (Antennas)

| Supplier | Country | URL |
|----------|---------|-----|
| DX Engineering | USA | dxengineering.com |
| Ham Radio Outlet | USA | hamradio.com |
| Martin Lynch | UK | hamradio.co.uk |
| WiMo | Germany | wimo.com |
| Difona | Germany | difona.de |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      11. PROCUREMENT TIMELINE
## ══════════════════════════════════════════════════════════════════════════════

```
Week 1:  ├── Apply to AMD University Program
         ├── Order RF components (Mouser/DigiKey)
         └── Order enclosure, thermal (AliExpress)

Week 2:  ├── Order LNAs, bias tees (AliExpress)
         └── Source/build antennas

Week 3:  ├── RF components arrive (Mouser)
         └── Order cables, connectors

Week 4:  ├── RFSoC 4x2 approval expected
         └── Order power supplies

Week 5:  ├── RFSoC 4x2 ships
         └── AliExpress items arrive

Week 6:  ├── RFSoC 4x2 arrives
         ├── Begin assembly
         └── Antenna installation

Week 7:  ├── System integration
         ├── Initial testing (loopback)
         └── Software deployment

Week 8:  ├── Full system test
         └── Calibration
```

---

## ══════════════════════════════════════════════════════════════════════════════
##                      12. RISK ASSESSMENT
## ══════════════════════════════════════════════════════════════════════════════

### Component Risks

| Component | Risk | Mitigation |
|-----------|------|------------|
| RFSoC 4x2 | University program rejection | Apply early, have ZCU111 backup |
| RA60H1317M PA | End of life risk | Stock spares, MRF300AN alternative |
| SPF5189Z LNA | Quality variation | Test before integration |
| LMR-400 cable | Counterfeit | Buy from authorized distributor |

### Supply Chain

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Extended lead times | Medium | Medium | Order early, buffer stock |
| Price increases | Low | Low | Lock prices with PO |
| Component shortage | Low | High | Identify alternatives |
| Shipping delays | Medium | Medium | Use multiple suppliers |

---

## ══════════════════════════════════════════════════════════════════════════════
##                      APPENDIX: PART NUMBER CROSS-REFERENCE
## ══════════════════════════════════════════════════════════════════════════════

| Function | Primary | Alternative 1 | Alternative 2 |
|----------|---------|---------------|---------------|
| PA | RA60H1317M | RA30H1317M | MRF300AN |
| LNA | SPF5189Z | PGA-103+ | PSA4-5043+ |
| BPF | SBP-150+ | ZABP-156+ | Custom cavity |
| Bias Tee | ZFBT-4R2GW+ | Generic | DIY inductor |
| Attenuator | HAT-30+ | VAT-30+ | Resistive pad |

---

**Document Version:** 2.0.0
**Last Updated:** 2026-02-02
**Author:** Dr. Mladen Mešter
**Copyright:** © 2026 - All Rights Reserved
