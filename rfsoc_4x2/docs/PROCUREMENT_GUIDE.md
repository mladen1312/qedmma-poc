# 🛒 TITAN Radar - Procurement Guide

## Step-by-Step Component Acquisition

**Author:** Dr. Mladen Mešter  
**Version:** 1.0  
**Date:** February 2026

---

## 📅 Procurement Timeline

```
Week 1  ████████████████████  AMD University Program Application
Week 2  ████████████████████  Order RFSoC 4x2 + RF Components (Mouser/DigiKey)
Week 3  ████████████████████  Order LNAs, Bias Tees (AliExpress)
Week 4  ████████████████████  Order/Build Antennas, Enclosure
Week 5  ████████████████████  Cables, Power Supplies, Misc
Week 6  ████████████████████  Assembly and Testing
```

---

## 🎓 Step 1: AMD University Program (Week 1)

### Requirements
- University or Research Institution affiliation
- Valid academic email address
- Research project description

### Application Process

1. **Visit AMD University Program**
   - URL: https://www.amd.com/en/corporate/university-program
   - Click "Apply Now"

2. **Fill Application Form**
   ```
   Institution: [Your University/Research Institute]
   Department: [Engineering/Physics/etc.]
   Project Title: "QEDMMA VHF Radar Research Platform"
   Project Description: 
     "Development of a low-cost VHF radar system for research 
      into anti-stealth detection techniques using direct RF 
      sampling and digital beamforming."
   ```

3. **Wait for Approval** (typically 1-2 weeks)

4. **Order RFSoC 4x2**
   - Price: $2,149 USD (~€2,000)
   - Supplier: Real Digital
   - Lead time: 2-4 weeks

### Alternative (No Academic Access)
- Contact local universities for collaboration
- Purchase through research grants
- Commercial pricing available (~$10,000+)

---

## 🔧 Step 2: RF Components (Week 2)

### Power Amplifier

| Store | Part | Price | Link |
|-------|------|:-----:|------|
| **Mouser** | RA60H1317M-101 | €85 | [Order](https://www.mouser.com/ProductDetail/Mitsubishi-Electric/RA60H1317M-101) |
| DigiKey | RA60H1317M-101 | €87 | [Order](https://www.digikey.com/en/products/detail/mitsubishi-electric/RA60H1317M-101) |

**Order Instructions:**
1. Go to Mouser.com
2. Search: "RA60H1317M"
3. Add to cart
4. Checkout (shipping €15-25 to EU)

### Bandpass Filters

| Store | Part | Price | Link |
|-------|------|:-----:|------|
| **Mini-Circuits** | SBP-150+ | €35 | [Order](https://www.minicircuits.com/WebStore/dashboard.html?model=SBP-150%2B) |

**Order Instructions:**
1. Go to minicircuits.com
2. Create account
3. Search: "SBP-150+"
4. Quantity: 2
5. Checkout

---

## 📦 Step 3: AliExpress Components (Week 2-3)

### LNA Modules (SPF5189Z)

**Search:** "SPF5189Z LNA module"

**Recommended Sellers:**
1. "RF Module Store" - ⭐4.9, €10-12/pc
2. "DYKB Official Store" - ⭐4.8, €11-13/pc

**Order:**
- Quantity: 4 (+ 1-2 spares recommended)
- Total: ~€60

### Bias Tees

**Search:** "Bias Tee SMA 10MHz-6GHz"

**Order:**
- Quantity: 4
- Price: ~€6-10 each
- Total: ~€32

### SMA Adapters

**Search:** "SMA N adapter male female"

**Order:**
- SMA Male to N Female: 4 pcs
- SMA Male to SMA Male: 2 pcs
- Total: ~€15

**⚠️ Note:** AliExpress shipping to EU typically 2-4 weeks. Order early!

---

## 📡 Step 4: Antennas (Week 3-4)

### Option A: Commercial (Recommended for Quick Start)

**Wimo Antennas (Germany)**
- URL: https://www.wimo.com/en/antennas
- Product: VHF Yagi 144-148 MHz (close to 155 MHz)
- Quantity: 5 (1 TX + 4 RX)
- Price: ~€45 each

**Alternative:**
- Diamond A144S10 (144 MHz Yagi)
- M2 Antennas (US, higher shipping)

### Option B: DIY (Cost-Effective)

**Materials Needed:**
| Item | Quantity | Price |
|------|:--------:|:-----:|
| Aluminum tubing 10mm | 10m | €15 |
| PVC pipe 25mm | 3m | €5 |
| N connectors | 5 | €10 |
| Hardware | - | €10 |
| **Total** | | **€40** |

**Build Plans:** See `/docs/ANTENNA_DESIGNS.md`

---

## ⚡ Step 5: Power Supplies (Week 4)

### Main Power Supply (12V)

| Store | Part | Price |
|-------|------|:-----:|
| **Mouser** | Mean Well LRS-200-12 | €45 |
| Amazon | Generic 12V 15A | €25 |

### PA Power Supply (28V)

| Store | Part | Price |
|-------|------|:-----:|
| **Mouser** | Mean Well LRS-100-24 | €35 |

**Note:** RA60H1317M operates at 12.5V, but higher voltage headroom is useful.

---

## 🔌 Step 6: Cables (Week 4-5)

### RF Cables

**Pasternack (USA) or Fairview Microwave:**
- SMA Male-Male 30cm (RG316): 8× €5 = €40
- SMA Male-Male 1m (RG58): 4× €8 = €32

**Antenna Feedlines:**
- LMR-400 with N connectors, 10m: 5× €15 = €75

**Alternative:** Build custom cables with bulk cable + connectors

### Power Cables
- 14 AWG silicone wire (red/black): €15
- Anderson PowerPole connectors: €10

---

## 📦 Step 7: Enclosure & Thermal (Week 4-5)

### Enclosure Options

| Type | Size | Price | Source |
|------|------|:-----:|--------|
| Hammond 1590DD | 188×120×56mm | €40 | Mouser |
| Generic Al Box | 300×200×100mm | €50 | AliExpress |
| **Custom** | 300×200×100mm | €80 | Local machinist |

### Cooling

| Item | Price | Source |
|------|:-----:|--------|
| Heatsink 100×100mm | €15 | Mouser |
| Noctua NF-A8 (2×) | €16 | Amazon |
| Thermal paste | €8 | Amazon |

---

## 🛠️ Step 8: Tools & Miscellaneous

### Essential Tools

| Tool | Price | Purpose |
|------|:-----:|---------|
| Soldering station | €50-100 | Assembly |
| SMA torque wrench | €25 | Connector tightening |
| Multimeter | €30 | Testing |
| Wire stripper | €15 | Cable prep |

### Recommended Tools

| Tool | Price | Purpose |
|------|:-----:|---------|
| NanoVNA | €50 | Antenna tuning |
| RTL-SDR | €30 | Basic spectrum |
| USB power meter | €20 | Current monitoring |

---

## 📊 Order Summary Checklist

### Week 1
- [ ] Submit AMD University Program application
- [ ] Prepare project description

### Week 2
- [ ] Order RFSoC 4x2 (upon approval)
- [ ] Order RA60H1317M from Mouser
- [ ] Order SBP-150+ filters from Mini-Circuits

### Week 3
- [ ] Order SPF5189Z LNAs from AliExpress (×5)
- [ ] Order bias tees from AliExpress (×5)
- [ ] Order SMA adapters from AliExpress

### Week 4
- [ ] Order/build antennas
- [ ] Order enclosure
- [ ] Order power supplies from Mouser

### Week 5
- [ ] Order cables (Pasternack or local)
- [ ] Order cooling components
- [ ] Order miscellaneous (SD card, etc.)

### Week 6+
- [ ] Receive all components
- [ ] Begin assembly
- [ ] Flash PYNQ image
- [ ] Build FPGA overlay
- [ ] Integration testing

---

## 💰 Budget Summary

| Category | Budget (€) | Actual (€) | Status |
|----------|:----------:|:----------:|:------:|
| RFSoC 4x2 | 2,000 | | ⏳ |
| PA | 85 | | ⏳ |
| LNAs | 60 | | ⏳ |
| Filters | 70 | | ⏳ |
| Antennas | 225 | | ⏳ |
| Cables | 150 | | ⏳ |
| Power | 95 | | ⏳ |
| Enclosure | 120 | | ⏳ |
| Misc | 100 | | ⏳ |
| **TOTAL** | **~€2,900** | | |

---

## ⚠️ Important Notes

1. **Lead Times:**
   - RFSoC 4x2: 2-4 weeks
   - AliExpress: 2-4 weeks
   - Mouser/DigiKey: 3-7 days

2. **Customs (EU):**
   - Orders from US >€150 may incur VAT
   - AliExpress often declares low value

3. **Quality:**
   - Buy spare LNAs (cheap, may be DOA)
   - Keep receipts for warranty

4. **Testing:**
   - Test each component before integration
   - Have NanoVNA for antenna verification

---

## 📞 Supplier Contacts

| Supplier | Region | Contact |
|----------|--------|---------|
| AMD University | Global | univ-program@amd.com |
| Mouser | EU | +49 89 520 462 110 |
| Mini-Circuits | USA | sales@minicircuits.com |
| Wimo | DE | info@wimo.com |
| Pasternack | USA | sales@pasternack.com |

---

**Good luck with your build, Dr. Mešter!** 🚀

---

**Copyright © 2026 Dr. Mladen Mešter - All Rights Reserved**
