# QEDMMA PoC - "Garažni Pobunjenik" v3.4

## 🎯 Kompletni Vodič za Samogradnju

**Author:** Dr. Mladen Mešter  
**Version:** 3.4.0 PoC  
**Date:** February 2026  
**Budget:** <€500  
**Build Time:** 2-3 vikenda  
**Copyright © 2026** - All Rights Reserved

---

## 📋 Executive Summary

Ovaj dokument opisuje **potpunu samogradnju** funkcionalnog VHF radara za **<€500** koji demonstrira core QEDMMA fiziku:

| Parametar | Ciljna vrijednost | Napomena |
|-----------|-------------------|----------|
| Frekvencija | 137-174 MHz | VHF band |
| Tx snaga | 25-30 W | RA30H1317M modul |
| Rx NF | 0.6 dB | SPF5189Z "sirotinjski Rydberg" |
| Processing gain | 45-60 dB | PRBS-15/20 correlator |
| Test range | 10-100 km | Avion/dron refleksija |
| Antenna gain | 10-12 dBi | DIY 5-element Yagi |

**Što dokazujemo:**
1. ✅ Zero-DSP correlator radi u praksi
2. ✅ PRBS processing gain (45-60 dB)
3. ✅ VHF anti-stealth princip
4. ✅ Bistatic/multistatic geometrija
5. ✅ Low-cost alternativa kvantnom prijemniku

---

## 💰 Bill of Materials (Live Prices February 2026)

### Core Components

| # | Komponenta | Specifikacija | Dobavljač | Cijena (€) |
|---|------------|---------------|-----------|------------|
| 1 | **ADALM-PLUTO** | Rev C, 2T2R, AD9363 | DigiKey | €230 |
| 2 | **RA30H1317M** | 30W VHF PA, 135-175 MHz | eBay | €75 |
| 3 | **SPF5189Z LNA** | NF 0.6 dB, 50-4000 MHz | AliExpress | €12 |
| 4 | **Bias Tee** | DC-6 GHz, SMA | AliExpress | €8 |
| 5 | **10dB Attenuator** | SMA, 2W | AliExpress | €6 |

### RF Connectors & Cables

| # | Komponenta | Qty | Cijena (€) |
|---|------------|-----|------------|
| 6 | SMA Male-Male cable 30cm | 4 | €16 |
| 7 | SMA Female bulkhead | 4 | €8 |
| 8 | N-Type to SMA adapter | 2 | €10 |
| 9 | RG316 coax 5m | 1 | €12 |

### Antenna Materials (DIY Yagi)

| # | Komponenta | Specifikacija | Izvor | Cijena (€) |
|---|------------|---------------|-------|------------|
| 10 | Alu cijev Ø10mm | 6m (2x3m) | Bauhaus | €15 |
| 11 | PVC cijev Ø32mm | 2m (boom) | Bauhaus | €8 |
| 12 | U-vijci M6 | 10 kom | Bauhaus | €5 |
| 13 | SO-239 chassis mount | 2 | Chipoteka | €6 |

### Power & Cooling

| # | Komponenta | Specifikacija | Cijena (€) |
|---|------------|---------------|------------|
| 14 | PSU 13.8V 10A | Switching | €28 |
| 15 | Heatsink 100x69x36mm | Alu | €12 |
| 16 | Fan 80mm 12V | PC surplus | €5 |
| 17 | Thermal paste | Arctic MX-4 | €6 |

### Misc

| # | Komponenta | Cijena (€) |
|---|------------|------------|
| 18 | Projektna kutija IP65 | €15 |
| 19 | Terminal blokovi, žice | €10 |
| 20 | Lemni materijal | €8 |

### 📊 UKUPNO

| Kategorija | Cijena (€) |
|------------|------------|
| Core Components | €331 |
| RF Connectors | €46 |
| Antenna | €34 |
| Power & Cooling | €51 |
| Misc | €33 |
| **GRAND TOTAL** | **€495** |

---

## 📡 DIY Yagi Antenna Design (155 MHz)

### Element Dimensions

```
λ = 300/155 = 1.935 m

Element         Length (mm)    Position (mm from R)
─────────────────────────────────────────────────
Reflector       1010           0
Driven Element  940            350
Director 1      910            650
Director 2      890            1000
Director 3      870            1450
─────────────────────────────────────────────────
Total boom length: 1450 mm
Gain: ~10.5 dBi
```

### Construction
1. Boom: PVC cijev Ø32mm × 1.6m
2. Elements: Alu cijev Ø10mm
3. Mounting: U-vijci M6 kroz boom
4. Feed: SO-239 + gamma match

---

## 🔧 Hardware Assembly

### Block Diagram

```
LAPTOP (Python)
     │
     │ USB
     ▼
┌──────────────┐
│ ADALM-PLUTO  │
│   AD9363     │
└──┬───────┬───┘
   │       │
  Tx      Rx
   │       │
   ▼       ▲
┌──────┐ ┌──────────┐
│10dB  │ │ Bias Tee │
│Atten │ │   +5V    │
└──┬───┘ └────┬─────┘
   │          │
   ▼          ▼
┌─────────┐ ┌─────────┐
│RA30H1317│ │SPF5189Z │
│  30W PA │ │LNA 0.6dB│
└────┬────┘ └────┬────┘
     │           │
     ▼           ▲
  Tx Yagi     Rx Yagi
     │           │
     └─── RF ────┘
       (target)
```

### PA Wiring (RA30H1317M)

```
Pin     Connection
────────────────────────────
GND     Heatsink, PSU GND
Vgg     +3.5V (via 10k pot)
Vdd     +12.5V (via 3A fuse)
RF_IN   From 10dB attenuator
RF_OUT  To Tx antenna
────────────────────────────

⚠️ CRITICAL: PlutoSDR outputs ~7dBm
   RA30H1317M max input: 0dBm
   → MUST use 10dB attenuator!
```

---

## ⚠️ Legal Notice

- **Amaterska licenca** potrebna za VHF TX
- Za testiranje: faradayev kavez ili HAKOM dozvola
- Ovaj projekt je za edukativne svrhe

---

**Budget: €495 | Range: 10-100 km | Build: 2-3 vikenda**
