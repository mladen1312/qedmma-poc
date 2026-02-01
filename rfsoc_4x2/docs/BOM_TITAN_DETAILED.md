# 💰 TITAN Radar - Detailed Bill of Materials

## Complete Component List with Purchase Links

**Author:** Dr. Mladen Mešter  
**Version:** 2.0.0  
**Date:** February 2026  
**Total System Cost:** ~€2,900

---

## 📋 Table of Contents

1. [Summary](#summary)
2. [RFSoC Platform](#1-rfsoc-platform)
3. [RF Frontend](#2-rf-frontend)
4. [Antennas](#3-antennas)
5. [Cables & Connectors](#4-cables--connectors)
6. [Power System](#5-power-system)
7. [Enclosure & Thermal](#6-enclosure--thermal)
8. [Miscellaneous](#7-miscellaneous)
9. [Tools Required](#8-tools-required)
10. [Alternative Components](#9-alternative-components)

---

## Summary

| Category | Cost (€) | % of Total |
|----------|:--------:|:----------:|
| RFSoC Platform | 2,000 | 69% |
| RF Frontend | 285 | 10% |
| Antennas | 225 | 8% |
| Cables & Connectors | 130 | 4% |
| Power System | 95 | 3% |
| Enclosure & Thermal | 120 | 4% |
| Miscellaneous | 50 | 2% |
| **TOTAL** | **€2,905** | **100%** |

---

## 1. RFSoC Platform

### RFSoC 4x2 Development Board

| Item | Specification | Qty | Price | Source |
|------|--------------|:---:|:-----:|--------|
| **RFSoC 4x2 Board** | AMD ZU48DR, 5 GSPS ADC, 9.85 GSPS DAC | 1 | €2,000 | [AMD University Program](https://www.amd.com/en/corporate/university-program) |

**Notes:**
- Price: $2,149 USD (academic pricing)
- Requires university/research affiliation
- Apply via AMD University Program
- Lead time: 2-4 weeks
- Includes: Board, power supply, USB cables, SD card

**Alternative Sources:**
- [Real Digital](https://www.realdigital.org/hardware/rfsoc-4x2) - Original manufacturer
- Commercial price (if no academic access): ~$10,000+

---

## 2. RF Frontend

### 2.1 Power Amplifier (TX)

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **RA60H1317M** | 60W, 134-174 MHz, 12.5V | 1 | 85 | 85 | [Mouser](https://www.mouser.com/ProductDetail/Mitsubishi-Electric/RA60H1317M-101) |

**Alternative PAs:**

| Part | Power | Freq | Price | Notes |
|------|:-----:|:----:|:-----:|-------|
| RA30H1317M | 30W | 134-174 MHz | €45 | Lower cost option |
| RA60H1317M | 60W | 134-174 MHz | €85 | **Recommended** |
| MRF1K50H | 1.5kW | 1.8-500 MHz | €350 | High power option |

**Purchase Links:**
- [Mouser RA60H1317M](https://www.mouser.com/ProductDetail/Mitsubishi-Electric/RA60H1317M-101)
- [DigiKey RA60H1317M](https://www.digikey.com/en/products/detail/mitsubishi-electric/RA60H1317M-101/3905044)
- [AliExpress (clones)](https://www.aliexpress.com/wholesale?SearchText=ra60h1317m) - €40-60, quality varies

### 2.2 Low Noise Amplifiers (RX)

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **SPF5189Z Module** | 50-4000 MHz, NF 0.6 dB, G 18 dB | 4 | 12 | 48 | [AliExpress](https://www.aliexpress.com/wholesale?SearchText=spf5189z) |

**Alternative LNAs:**

| Part | Freq | NF | Gain | Price | Notes |
|------|:----:|:--:|:----:|:-----:|-------|
| SPF5189Z | 50-4000 MHz | 0.6 dB | 18 dB | €12 | **Recommended** |
| PGA-103+ | 50-4000 MHz | 0.5 dB | 20 dB | €35 | Higher performance |
| SAV-541+ | DC-8 GHz | 1.0 dB | 13 dB | €25 | Wider bandwidth |

**Purchase Links:**
- [AliExpress SPF5189Z](https://www.aliexpress.com/wholesale?SearchText=spf5189z+module)
- [eBay SPF5189Z](https://www.ebay.com/sch/i.html?_nkw=spf5189z)
- [LCSC SPF5189Z IC](https://www.lcsc.com/search?q=spf5189z)

### 2.3 Bandpass Filters

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **VHF BPF 140-170 MHz** | 3dB BW 30 MHz, IL <2 dB | 2 | 35 | 70 | [Mini-Circuits](https://www.minicircuits.com/WebStore/Filters.html) |

**Recommended Filters:**

| Part | Type | Freq | BW | IL | Price | Notes |
|------|:----:|:----:|:--:|:--:|:-----:|-------|
| SBP-150+ | BPF | 127-173 MHz | 46 MHz | 1.2 dB | €35 | Mini-Circuits |
| VBFZ-150-S+ | BPF | 130-170 MHz | 40 MHz | 0.5 dB | €65 | Low loss |
| Custom LC | BPF | 140-170 MHz | 30 MHz | 1.5 dB | €15 | DIY option |

**Purchase Links:**
- [Mini-Circuits Filters](https://www.minicircuits.com/WebStore/Filters.html)
- [Pasternack VHF Filters](https://www.pasternack.com/rf-bandpass-filters-category.aspx)

### 2.4 Bias Tees

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **Bias Tee** | 10-6000 MHz, 500 mA | 4 | 8 | 32 | [AliExpress](https://www.aliexpress.com/wholesale?SearchText=bias+tee+sma) |

**Purchase Links:**
- [AliExpress Bias Tee](https://www.aliexpress.com/wholesale?SearchText=bias+tee+sma)
- [Mini-Circuits ZFBT-4R2GW+](https://www.minicircuits.com/pdfs/ZFBT-4R2GW+.pdf) - €45, professional

### 2.5 RF Switches (Optional)

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **SPDT RF Switch** | DC-6 GHz, 60 dB isolation | 1 | 50 | 50 | [Mini-Circuits](https://www.minicircuits.com/WebStore/Switches.html) |

---

## 3. Antennas

### 3.1 TX Antenna

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **VHF Yagi (TX)** | 155 MHz, 3-element, 6 dBd | 1 | 45 | 45 | [Wimo](https://www.wimo.com/en/) |

**Options:**

| Type | Gain | Price | Notes |
|------|:----:|:-----:|-------|
| 3-element Yagi | 6 dBd | €45 | Commercial |
| 5-element Yagi | 9 dBd | €85 | Higher gain |
| DIY Yagi | 6 dBd | €15 | Build from plans |
| Dipole | 2.15 dBi | €10 | Simple option |

**Purchase Links:**
- [Wimo VHF Antennas](https://www.wimo.com/en/antennas/vhf-uhf-antennas)
- [Diamond A144S10](https://www.diamondantenna.net/) - 144 MHz Yagi
- [M2 Antennas](https://www.m2inc.com/) - Professional VHF

### 3.2 RX Array (4 elements)

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **VHF Yagi (RX)** | 155 MHz, 3-element, 6 dBd | 4 | 45 | 180 | [Wimo](https://www.wimo.com/en/) |

**Array Specifications:**
- Element spacing: λ/2 = 0.97 m @ 155 MHz
- Total aperture: ~3 m
- Array gain: +6 dB over single element
- Beamwidth: ~25°

**DIY Option:**
- Aluminum tubing + PVC boom
- Material cost: ~€40 for 4 antennas
- Plans available in `/docs/ANTENNA_DESIGNS.md`

---

## 4. Cables & Connectors

### 4.1 SMA Cables

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| SMA-SMA 30cm | RG316, M-M | 8 | 5 | 40 | [Pasternack](https://www.pasternack.com/) |
| SMA-SMA 1m | RG58, M-M | 4 | 8 | 32 | [Pasternack](https://www.pasternack.com/) |
| SMA-N Adapter | M-M | 6 | 5 | 30 | [Various](https://www.aliexpress.com/) |

**Purchase Links:**
- [Pasternack Cables](https://www.pasternack.com/rf-coaxial-cables-category.aspx)
- [Fairview Microwave](https://www.fairviewmicrowave.com/)
- [AliExpress SMA Cables](https://www.aliexpress.com/wholesale?SearchText=sma+cable+rg316)

### 4.2 Antenna Cables

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| LMR-400 10m | N-M to SMA-M | 5 | 15 | 75 | [Wimo](https://www.wimo.com/) |

**Cable Options:**

| Type | Loss @150 MHz | Cost/m | Notes |
|------|:------------:|:------:|-------|
| RG58 | 5.6 dB/100m | €0.50 | Budget |
| RG213 | 3.2 dB/100m | €1.50 | Medium |
| LMR-400 | 1.5 dB/100m | €3.00 | **Recommended** |
| 7/8" Hardline | 0.5 dB/100m | €15.00 | Professional |

---

## 5. Power System

### 5.1 Main Power Supply

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **12V 15A PSU** | Mean Well LRS-200-12 | 1 | 45 | 45 | [Mouser](https://www.mouser.com/) |

**Purchase Links:**
- [Mouser Mean Well](https://www.mouser.com/c/power/ac-dc-converters/enclosed-ac-dc-power-supplies/?m=Mean%20Well)
- [DigiKey Mean Well](https://www.digikey.com/en/products/filter/ac-dc-converters/130)

### 5.2 PA Power Supply

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **28V 5A PSU** | For PA bias | 1 | 35 | 35 | [Mouser](https://www.mouser.com/) |

### 5.3 Power Distribution

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| Fuse holder | 5×20mm | 3 | 2 | 6 | Various |
| Fuses | 5A, 10A, 15A | 10 | 0.50 | 5 | Various |
| DC connectors | Anderson PP | 5 | 2 | 10 | Various |

---

## 6. Enclosure & Thermal

### 6.1 Enclosure

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| **RF Enclosure** | Aluminum, 300×200×100mm | 1 | 80 | 80 | [Hammond](https://www.hammfg.com/) |

**Options:**

| Type | Size | Price | Notes |
|------|:----:|:-----:|-------|
| Hammond 1590Z | 150×100×50mm | €25 | Small, board only |
| Hammond 1590DD | 188×120×56mm | €40 | Medium |
| Custom Al box | 300×200×100mm | €80 | Full system |
| 19" Rack 2U | 482×88×300mm | €120 | Professional |

**Purchase Links:**
- [Hammond Enclosures](https://www.hammfg.com/electronics/small-case/diecast/1590)
- [Takachi (Europe)](https://www.takachi-enclosure.com/)
- [AliExpress Aluminum Box](https://www.aliexpress.com/wholesale?SearchText=aluminum+enclosure+rf)

### 6.2 Thermal Management

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| Heatsink (PA) | 100×100×35mm | 1 | 15 | 15 | [Mouser](https://www.mouser.com/) |
| Fan 80mm | 12V, 3000 RPM | 2 | 8 | 16 | [Noctua](https://noctua.at/) |
| Thermal paste | Arctic MX-4 | 1 | 8 | 8 | [Amazon](https://www.amazon.com/) |

---

## 7. Miscellaneous

| Item | Specification | Qty | Unit (€) | Total (€) | Source |
|------|--------------|:---:|:--------:|:---------:|--------|
| SD Card 64GB | Class 10, A2 | 1 | 15 | 15 | Amazon |
| Ethernet cable | Cat6, 2m | 2 | 5 | 10 | Various |
| USB cables | Various | 1 | 10 | 10 | Various |
| Mounting hardware | Screws, standoffs | 1 | 15 | 15 | Various |

---

## 8. Tools Required

### Essential Tools

| Tool | Purpose | Approx. Cost |
|------|---------|:------------:|
| Soldering station | Assembly | €50-200 |
| Multimeter | Testing | €30-100 |
| SMA torque wrench | Connectors | €25 |
| Wire strippers | Cables | €15 |
| Heat gun | Heatshrink | €25 |

### Recommended Tools

| Tool | Purpose | Approx. Cost |
|------|---------|:------------:|
| Spectrum analyzer | RF testing | €300-2000 |
| NanoVNA | Antenna tuning | €50 |
| Oscilloscope | Debug | €300-1000 |
| Power meter | TX power | €100-500 |

---

## 9. Alternative Components

### Budget Options (Total ~€2,200)

| Original | Alternative | Savings |
|----------|-------------|:-------:|
| RA60H1317M (€85) | RA30H1317M (€45) | €40 |
| Commercial Yagi ×5 (€225) | DIY Yagi ×5 (€60) | €165 |
| LMR-400 cables (€75) | RG58 cables (€25) | €50 |
| Hammond enclosure (€80) | Generic Al box (€40) | €40 |

### High-Performance Options (Total ~€4,500)

| Original | Upgrade | Extra Cost |
|----------|---------|:----------:|
| RA60H1317M | 300W LDMOS amp | +€400 |
| 3-el Yagi ×5 | 7-el Yagi ×5 | +€300 |
| SPF5189Z | PSA4-5043+ | +€100 |
| Basic enclosure | 19" Rack system | +€800 |

---

## 📊 Cost Breakdown Chart

```
╔════════════════════════════════════════════════════════════════╗
║                    TITAN BOM COST BREAKDOWN                    ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  RFSoC 4x2    ████████████████████████████████████████  €2,000 ║
║  RF Frontend  █████                                       €285 ║
║  Antennas     ████                                        €225 ║
║  Cables       ██                                          €130 ║
║  Power        █                                            €95 ║
║  Enclosure    ██                                          €120 ║
║  Misc         █                                            €50 ║
║                                                                ║
║  TOTAL: €2,905                                                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🛒 Ordering Checklist

### Week 1: Long Lead Items
- [ ] Apply to AMD University Program
- [ ] Order RFSoC 4x2 board

### Week 2: RF Components
- [ ] Order PA module (Mouser/DigiKey)
- [ ] Order LNAs (AliExpress - allow 2-3 weeks)
- [ ] Order filters (Mini-Circuits)

### Week 3: Mechanical
- [ ] Order/build antennas
- [ ] Order enclosure
- [ ] Order cables and connectors

### Week 4: Power & Misc
- [ ] Order power supplies
- [ ] Order cooling components
- [ ] Order miscellaneous items

---

**Copyright © 2026 Dr. Mladen Mešter - All Rights Reserved**
