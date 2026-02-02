#!/usr/bin/env python3
"""
TITAN Radar - Bill of Materials Generator & Cost Analysis
Production-Ready BOM with Supplier Information

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved

Usage:
    python3 titan_bom.py                    # Print BOM summary
    python3 titan_bom.py --csv output.csv   # Export to CSV
    python3 titan_bom.py --html output.html # Export to HTML
    python3 titan_bom.py --detailed         # Show detailed analysis
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import json
from datetime import datetime


#===============================================================================
# Data Models
#===============================================================================

class Category(Enum):
    PLATFORM = "Core Platform"
    TX_PATH = "TX Path"
    RX_PATH = "RX Path"
    ANTENNA = "Antennas"
    CABLES = "Cables & Connectors"
    POWER = "Power System"
    ENCLOSURE = "Enclosure & Thermal"
    MISC = "Miscellaneous"


class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class Component:
    """Single BOM component"""
    name: str
    part_number: str
    category: Category
    quantity: int
    unit_price_eur: float
    supplier: str
    supplier_url: str = ""
    description: str = ""
    alternatives: List[str] = field(default_factory=list)
    lead_time_days: int = 7
    eol_risk: RiskLevel = RiskLevel.LOW
    notes: str = ""
    
    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price_eur


@dataclass
class BOMConfiguration:
    """Complete BOM configuration"""
    name: str
    description: str
    components: List[Component]
    
    @property
    def total_cost(self) -> float:
        return sum(c.total_price for c in self.components)
    
    @property
    def by_category(self) -> Dict[Category, List[Component]]:
        result = {}
        for cat in Category:
            result[cat] = [c for c in self.components if c.category == cat]
        return result
    
    @property
    def category_totals(self) -> Dict[Category, float]:
        return {cat: sum(c.total_price for c in comps) 
                for cat, comps in self.by_category.items()}


#===============================================================================
# TITAN BOM Definitions
#===============================================================================

def create_standard_bom() -> BOMConfiguration:
    """Create standard TITAN BOM configuration"""
    
    components = [
        #-----------------------------------------------------------------------
        # 1. CORE PLATFORM
        #-----------------------------------------------------------------------
        Component(
            name="RFSoC 4x2 Development Board",
            part_number="RFSOC4X2",
            category=Category.PLATFORM,
            quantity=1,
            unit_price_eur=2000,
            supplier="AMD/Avnet",
            supplier_url="https://www.avnet.com/rfsoc4x2",
            description="Zynq UltraScale+ ZU48DR, 4×ADC 5GSPS, 2×DAC 9.85GSPS",
            alternatives=["ZCU111 (€8000)", "ZCU216 (€12000)"],
            lead_time_days=14,
            eol_risk=RiskLevel.LOW,
            notes="Apply to AMD University Program for academic pricing"
        ),
        
        #-----------------------------------------------------------------------
        # 2. TX PATH
        #-----------------------------------------------------------------------
        Component(
            name="VHF Power Amplifier 60W",
            part_number="RA60H1317M",
            category=Category.TX_PATH,
            quantity=1,
            unit_price_eur=85,
            supplier="Mouser",
            supplier_url="https://www.mouser.com/ProductDetail/Mitsubishi/RA60H1317M",
            description="60W, 134-174 MHz, 12.5V supply, 60% efficiency",
            alternatives=["RA30H1317M (€45, 30W)", "MRF300AN (€150, 300W)"],
            lead_time_days=5,
            eol_risk=RiskLevel.MEDIUM,
            notes="Requires heatsink, 10A peak current"
        ),
        Component(
            name="VHF Driver Amplifier",
            part_number="MAR-6+",
            category=Category.TX_PATH,
            quantity=1,
            unit_price_eur=3,
            supplier="Mini-Circuits",
            supplier_url="https://www.minicircuits.com/pdfs/MAR-6+.pdf",
            description="20 dB gain, 50 mW output",
            alternatives=["ERA-3SM+ (€5)"],
            lead_time_days=3
        ),
        Component(
            name="TX Bandpass Filter",
            part_number="SBP-150+",
            category=Category.TX_PATH,
            quantity=1,
            unit_price_eur=35,
            supplier="Mini-Circuits",
            supplier_url="https://www.minicircuits.com/pdfs/SBP-150+.pdf",
            description="127-173 MHz, 50Ω, SMA",
            alternatives=["Custom cavity filter"],
            lead_time_days=5
        ),
        Component(
            name="TX Low Pass Filter",
            part_number="SLP-200+",
            category=Category.TX_PATH,
            quantity=1,
            unit_price_eur=22,
            supplier="Mini-Circuits",
            supplier_url="https://www.minicircuits.com/pdfs/SLP-200+.pdf",
            description="DC-190 MHz, harmonic suppression",
            lead_time_days=5
        ),
        Component(
            name="RF Transformer 1:1",
            part_number="T1-6T-KK81+",
            category=Category.TX_PATH,
            quantity=1,
            unit_price_eur=5,
            supplier="Mini-Circuits",
            lead_time_days=5
        ),
        
        #-----------------------------------------------------------------------
        # 3. RX PATH (4 channels)
        #-----------------------------------------------------------------------
        Component(
            name="VHF Low Noise Amplifier",
            part_number="SPF5189Z",
            category=Category.RX_PATH,
            quantity=4,
            unit_price_eur=12,
            supplier="AliExpress",
            supplier_url="https://www.aliexpress.com/item/SPF5189Z",
            description="50 MHz-4 GHz, NF 0.6 dB, Gain 18.7 dB",
            alternatives=["PGA-103+ (€8)", "PSA4-5043+ (€10)"],
            lead_time_days=21,
            notes="Test before integration - quality varies"
        ),
        Component(
            name="RX Bandpass Filter",
            part_number="SBP-150+",
            category=Category.RX_PATH,
            quantity=4,
            unit_price_eur=35,
            supplier="Mini-Circuits",
            description="127-173 MHz per channel",
            lead_time_days=5,
            notes="Consider single filter + splitter for cost reduction"
        ),
        Component(
            name="Bias Tee",
            part_number="ZFBT-4R2GW+",
            category=Category.RX_PATH,
            quantity=4,
            unit_price_eur=8,
            supplier="Mini-Circuits",
            supplier_url="https://www.minicircuits.com/pdfs/ZFBT-4R2GW+.pdf",
            description="10-4200 MHz, SMA",
            alternatives=["Generic bias tee (€3)"],
            lead_time_days=5
        ),
        Component(
            name="RF Limiter",
            part_number="JLML-01-1+",
            category=Category.RX_PATH,
            quantity=4,
            unit_price_eur=4,
            supplier="Mini-Circuits",
            description="ADC protection",
            lead_time_days=5
        ),
        Component(
            name="TVS Diode Array",
            part_number="PESD5V0S1BL",
            category=Category.RX_PATH,
            quantity=4,
            unit_price_eur=0.50,
            supplier="Mouser",
            description="ESD protection",
            lead_time_days=3
        ),
        
        #-----------------------------------------------------------------------
        # 4. ANTENNAS (Standard config: 5-element Yagi)
        #-----------------------------------------------------------------------
        Component(
            name="TX Antenna: 5-Element Yagi",
            part_number="VHF-YAGI-5",
            category=Category.ANTENNA,
            quantity=1,
            unit_price_eur=75,
            supplier="DX Engineering / DIY",
            supplier_url="https://www.dxengineering.com",
            description="155 MHz, 9 dBi gain, 52° beamwidth",
            alternatives=["3-el (€45, 6dBi)", "7-el (€120, 11dBi)", "DIY (€30)"],
            lead_time_days=14,
            notes="Homebrew possible - see titan_antenna_design.py"
        ),
        Component(
            name="RX Antenna: 5-Element Yagi",
            part_number="VHF-YAGI-5",
            category=Category.ANTENNA,
            quantity=4,
            unit_price_eur=75,
            supplier="DX Engineering / DIY",
            description="4-element receive array, 15 dBi combined",
            alternatives=["3-el (€45)", "7-el (€120)"],
            lead_time_days=14
        ),
        
        #-----------------------------------------------------------------------
        # 5. CABLES & CONNECTORS
        #-----------------------------------------------------------------------
        Component(
            name="LMR-400 Coax Cable",
            part_number="LMR-400",
            category=Category.CABLES,
            quantity=30,  # meters
            unit_price_eur=2,
            supplier="Mouser",
            description="Low loss, 2.7 dB/100m @ 155 MHz",
            alternatives=["RG-213 (€1/m, more loss)"],
            lead_time_days=5
        ),
        Component(
            name="RG-316 Coax Cable",
            part_number="RG-316",
            category=Category.CABLES,
            quantity=5,  # meters
            unit_price_eur=1.50,
            supplier="Mouser",
            description="Internal connections",
            lead_time_days=5
        ),
        Component(
            name="N-Type Male Connector (LMR-400)",
            part_number="N-MALE-400",
            category=Category.CABLES,
            quantity=10,
            unit_price_eur=3,
            supplier="AliExpress",
            description="Weatherproof antenna connections",
            lead_time_days=14
        ),
        Component(
            name="SMA Male Connector",
            part_number="SMA-MALE",
            category=Category.CABLES,
            quantity=20,
            unit_price_eur=1,
            supplier="AliExpress",
            description="Board connections",
            lead_time_days=14
        ),
        Component(
            name="N-to-SMA Adapter",
            part_number="N-SMA-ADAPTER",
            category=Category.CABLES,
            quantity=5,
            unit_price_eur=3,
            supplier="AliExpress",
            lead_time_days=14
        ),
        Component(
            name="30 dB Attenuator (N)",
            part_number="HAT-30+",
            category=Category.CABLES,
            quantity=2,
            unit_price_eur=25,
            supplier="Mini-Circuits",
            description="Testing & loopback",
            lead_time_days=5
        ),
        Component(
            name="DC Block",
            part_number="BLK-89-S+",
            category=Category.CABLES,
            quantity=4,
            unit_price_eur=8,
            supplier="Mini-Circuits",
            lead_time_days=5
        ),
        
        #-----------------------------------------------------------------------
        # 6. POWER SYSTEM
        #-----------------------------------------------------------------------
        Component(
            name="12V 30A Power Supply",
            part_number="S-360-12",
            category=Category.POWER,
            quantity=1,
            unit_price_eur=35,
            supplier="AliExpress",
            description="RFSoC + peripherals",
            lead_time_days=14
        ),
        Component(
            name="28V 15A Power Supply",
            part_number="S-400-28",
            category=Category.POWER,
            quantity=1,
            unit_price_eur=45,
            supplier="AliExpress",
            description="PA supply (if using higher voltage PA)",
            lead_time_days=14,
            notes="Optional - RA60H1317M runs on 12.5V"
        ),
        Component(
            name="DC-DC Converter 5V 3A",
            part_number="LM2596",
            category=Category.POWER,
            quantity=1,
            unit_price_eur=3,
            supplier="AliExpress",
            description="LNA bias supply",
            lead_time_days=14
        ),
        Component(
            name="Fuse Holder + Fuses",
            part_number="FUSE-KIT",
            category=Category.POWER,
            quantity=1,
            unit_price_eur=10,
            supplier="Local",
            lead_time_days=1
        ),
        Component(
            name="Power Entry Module IEC",
            part_number="IEC-INLET",
            category=Category.POWER,
            quantity=1,
            unit_price_eur=5,
            supplier="Local",
            lead_time_days=1
        ),
        
        #-----------------------------------------------------------------------
        # 7. ENCLOSURE & THERMAL
        #-----------------------------------------------------------------------
        Component(
            name="Aluminum Enclosure 400×300×150mm",
            part_number="ALU-CASE-L",
            category=Category.ENCLOSURE,
            quantity=1,
            unit_price_eur=60,
            supplier="AliExpress",
            description="Main electronics housing",
            lead_time_days=21
        ),
        Component(
            name="Heatsink 100×100×40mm",
            part_number="HS-100-40",
            category=Category.ENCLOSURE,
            quantity=1,
            unit_price_eur=15,
            supplier="AliExpress",
            description="PA cooling",
            lead_time_days=14
        ),
        Component(
            name="Cooling Fan 80mm",
            part_number="FAN-80",
            category=Category.ENCLOSURE,
            quantity=2,
            unit_price_eur=5,
            supplier="AliExpress",
            description="Forced air cooling",
            lead_time_days=14
        ),
        Component(
            name="Thermal Paste",
            part_number="ARCTIC-MX4",
            category=Category.ENCLOSURE,
            quantity=1,
            unit_price_eur=8,
            supplier="Amazon",
            lead_time_days=2
        ),
        Component(
            name="RF Shield Copper Sheet",
            part_number="CU-SHEET",
            category=Category.ENCLOSURE,
            quantity=1,
            unit_price_eur=15,
            supplier="Local",
            description="EMI isolation",
            lead_time_days=3
        ),
        Component(
            name="Weatherproof Box IP65",
            part_number="IP65-BOX",
            category=Category.ENCLOSURE,
            quantity=1,
            unit_price_eur=25,
            supplier="Local",
            description="Outdoor RF frontend",
            lead_time_days=5
        ),
        Component(
            name="Cable Glands PG9/PG11",
            part_number="PG-GLANDS",
            category=Category.ENCLOSURE,
            quantity=10,
            unit_price_eur=1,
            supplier="Local",
            lead_time_days=1
        ),
        
        #-----------------------------------------------------------------------
        # 8. MISCELLANEOUS
        #-----------------------------------------------------------------------
        Component(
            name="SD Card 64GB Class 10",
            part_number="SANDISK-64GB",
            category=Category.MISC,
            quantity=2,
            unit_price_eur=12,
            supplier="Amazon",
            description="Boot media + spare",
            lead_time_days=2
        ),
        Component(
            name="Lightning Arrestor (N)",
            part_number="POLYPHASER",
            category=Category.MISC,
            quantity=2,
            unit_price_eur=35,
            supplier="RF supplier",
            description="Antenna protection - ESSENTIAL",
            lead_time_days=7,
            notes="MANDATORY for outdoor installation"
        ),
        Component(
            name="Grounding Kit",
            part_number="GND-KIT",
            category=Category.MISC,
            quantity=1,
            unit_price_eur=25,
            supplier="Local",
            description="Ground rod + wire",
            lead_time_days=3
        ),
    ]
    
    return BOMConfiguration(
        name="TITAN Standard",
        description="Standard configuration with 5-element Yagi antennas",
        components=components
    )


def create_budget_bom() -> BOMConfiguration:
    """Create budget TITAN BOM configuration"""
    standard = create_standard_bom()
    
    # Modify for budget
    budget_components = []
    for c in standard.components:
        if c.category == Category.ANTENNA:
            # Use 3-element Yagis
            new_c = Component(
                name=c.name.replace("5-Element", "3-Element"),
                part_number="VHF-YAGI-3",
                category=c.category,
                quantity=c.quantity,
                unit_price_eur=45,  # Cheaper
                supplier=c.supplier,
                description="3-element Yagi, 6 dBi",
                lead_time_days=c.lead_time_days
            )
            budget_components.append(new_c)
        else:
            budget_components.append(c)
    
    return BOMConfiguration(
        name="TITAN Budget",
        description="Budget configuration with 3-element Yagi antennas",
        components=budget_components
    )


def create_enhanced_bom() -> BOMConfiguration:
    """Create enhanced TITAN BOM configuration"""
    standard = create_standard_bom()
    
    # Modify for enhanced
    enhanced_components = []
    for c in standard.components:
        if c.category == Category.ANTENNA:
            # Use 7-element Yagis
            new_c = Component(
                name=c.name.replace("5-Element", "7-Element"),
                part_number="VHF-YAGI-7",
                category=c.category,
                quantity=c.quantity,
                unit_price_eur=120,  # Higher performance
                supplier=c.supplier,
                description="7-element Yagi, 11 dBi",
                lead_time_days=c.lead_time_days
            )
            enhanced_components.append(new_c)
        elif "LNA" in c.name:
            # Better LNAs
            new_c = Component(
                name="VHF Low Noise Amplifier (Premium)",
                part_number="PGA-103+",
                category=c.category,
                quantity=c.quantity,
                unit_price_eur=20,
                supplier="Mini-Circuits",
                description="NF 0.5 dB, 15 dB gain",
                lead_time_days=5
            )
            enhanced_components.append(new_c)
        else:
            enhanced_components.append(c)
    
    return BOMConfiguration(
        name="TITAN Enhanced",
        description="Enhanced configuration with 7-element Yagis and premium LNAs",
        components=enhanced_components
    )


#===============================================================================
# Report Generation
#===============================================================================

def print_bom_summary(bom: BOMConfiguration):
    """Print BOM summary to console"""
    
    print("\n" + "═" * 80)
    print(f"  TITAN VHF RADAR - BILL OF MATERIALS")
    print(f"  Configuration: {bom.name}")
    print("═" * 80)
    print(f"\n  {bom.description}")
    
    print("\n" + "─" * 80)
    print(f"  {'CATEGORY':<25} {'ITEMS':>6} {'COST (€)':>12}")
    print("─" * 80)
    
    for cat, total in bom.category_totals.items():
        items = len(bom.by_category[cat])
        if items > 0:
            print(f"  {cat.value:<25} {items:>6} {total:>12.2f}")
    
    print("─" * 80)
    print(f"  {'SUBTOTAL':<25} {len(bom.components):>6} {bom.total_cost:>12.2f}")
    print(f"  {'Contingency (10%)':<25} {'':<6} {bom.total_cost * 0.1:>12.2f}")
    print("─" * 80)
    print(f"  {'TOTAL':<25} {'':<6} {bom.total_cost * 1.1:>12.2f}")
    print("═" * 80)


def print_detailed_bom(bom: BOMConfiguration):
    """Print detailed BOM with all components"""
    
    print_bom_summary(bom)
    
    for cat in Category:
        components = bom.by_category[cat]
        if not components:
            continue
            
        print(f"\n\n{'█' * 80}")
        print(f"  {cat.value.upper()}")
        print("█" * 80)
        
        print(f"\n  {'Item':<35} {'P/N':<15} {'Qty':>4} {'Unit €':>8} {'Total €':>10} {'Supplier':<15}")
        print("  " + "─" * 95)
        
        for c in components:
            print(f"  {c.name[:35]:<35} {c.part_number[:15]:<15} {c.quantity:>4} "
                  f"{c.unit_price_eur:>8.2f} {c.total_price:>10.2f} {c.supplier[:15]:<15}")
            
            if c.description:
                print(f"    └─ {c.description}")
            if c.alternatives:
                print(f"    └─ Alternatives: {', '.join(c.alternatives[:2])}")
            if c.notes:
                print(f"    └─ ⚠ {c.notes}")


def export_csv(bom: BOMConfiguration, filename: str):
    """Export BOM to CSV file"""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Category', 'Item', 'Part Number', 'Quantity', 
            'Unit Price (€)', 'Total Price (€)', 'Supplier',
            'Supplier URL', 'Description', 'Alternatives',
            'Lead Time (days)', 'EOL Risk', 'Notes'
        ])
        
        # Data
        for c in bom.components:
            writer.writerow([
                c.category.value, c.name, c.part_number, c.quantity,
                c.unit_price_eur, c.total_price, c.supplier,
                c.supplier_url, c.description, '; '.join(c.alternatives),
                c.lead_time_days, c.eol_risk.value, c.notes
            ])
        
        # Totals
        writer.writerow([])
        writer.writerow(['', '', '', '', 'SUBTOTAL:', bom.total_cost])
        writer.writerow(['', '', '', '', 'Contingency (10%):', bom.total_cost * 0.1])
        writer.writerow(['', '', '', '', 'TOTAL:', bom.total_cost * 1.1])
    
    print(f"✅ CSV exported to: {filename}")


def export_html(bom: BOMConfiguration, filename: str):
    """Export BOM to HTML file"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TITAN Radar BOM - {bom.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .category {{ background-color: #333; color: white; font-weight: bold; }}
        .total {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        .notes {{ font-size: 0.9em; color: #666; }}
        a {{ color: #4CAF50; }}
    </style>
</head>
<body>
    <h1>TITAN VHF Anti-Stealth Radar</h1>
    <h2>Bill of Materials: {bom.name}</h2>
    <p>{bom.description}</p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <table>
        <tr>
            <th>Item</th>
            <th>Part Number</th>
            <th>Qty</th>
            <th>Unit (€)</th>
            <th>Total (€)</th>
            <th>Supplier</th>
            <th>Notes</th>
        </tr>
"""
    
    for cat in Category:
        components = bom.by_category[cat]
        if not components:
            continue
        
        cat_total = sum(c.total_price for c in components)
        html += f'<tr class="category"><td colspan="7">{cat.value} - €{cat_total:.2f}</td></tr>\n'
        
        for c in components:
            supplier_link = f'<a href="{c.supplier_url}">{c.supplier}</a>' if c.supplier_url else c.supplier
            notes = f'{c.description}<br><span class="notes">{c.notes}</span>' if c.notes else c.description
            
            html += f"""<tr>
                <td>{c.name}</td>
                <td>{c.part_number}</td>
                <td>{c.quantity}</td>
                <td>{c.unit_price_eur:.2f}</td>
                <td>{c.total_price:.2f}</td>
                <td>{supplier_link}</td>
                <td>{notes}</td>
            </tr>\n"""
    
    html += f"""
        <tr class="total">
            <td colspan="4">SUBTOTAL</td>
            <td>{bom.total_cost:.2f}</td>
            <td colspan="2"></td>
        </tr>
        <tr>
            <td colspan="4">Contingency (10%)</td>
            <td>{bom.total_cost * 0.1:.2f}</td>
            <td colspan="2"></td>
        </tr>
        <tr class="total">
            <td colspan="4">TOTAL</td>
            <td>{bom.total_cost * 1.1:.2f}</td>
            <td colspan="2"></td>
        </tr>
    </table>
    
    <h3>Supplier Summary</h3>
    <ul>
        <li><strong>Mouser/DigiKey:</strong> RF components, semiconductors (2-5 days)</li>
        <li><strong>Mini-Circuits:</strong> Filters, amplifiers, attenuators (3-7 days)</li>
        <li><strong>AliExpress:</strong> LNAs, enclosures, power supplies (2-4 weeks)</li>
        <li><strong>AMD University Program:</strong> RFSoC 4x2 board (2-4 weeks approval)</li>
    </ul>
    
    <p><em>Copyright © 2026 Dr. Mladen Mešter - All Rights Reserved</em></p>
</body>
</html>"""
    
    with open(filename, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML exported to: {filename}")


def compare_configurations():
    """Compare all BOM configurations"""
    
    budget = create_budget_bom()
    standard = create_standard_bom()
    enhanced = create_enhanced_bom()
    
    print("\n" + "═" * 80)
    print("  TITAN BOM CONFIGURATION COMPARISON")
    print("═" * 80)
    
    print(f"\n  {'Configuration':<20} {'Components':>12} {'Subtotal':>12} {'+10%':>12} {'Est. Range':>15}")
    print("  " + "─" * 75)
    
    configs = [
        (budget, "66 km"),
        (standard, "93 km"),
        (enhanced, "117 km"),
    ]
    
    for bom, range_est in configs:
        print(f"  {bom.name:<20} {len(bom.components):>12} "
              f"€{bom.total_cost:>10.2f} €{bom.total_cost*1.1:>10.2f} {range_est:>15}")
    
    print("  " + "─" * 75)
    print("\n  * Range estimates for 1kW TX, 10m² RCS (stealth @ VHF), PRBS-20")


#===============================================================================
# Main
#===============================================================================

def main():
    parser = argparse.ArgumentParser(description='TITAN Radar BOM Generator')
    parser.add_argument('--config', choices=['budget', 'standard', 'enhanced'], 
                       default='standard', help='BOM configuration')
    parser.add_argument('--csv', type=str, help='Export to CSV file')
    parser.add_argument('--html', type=str, help='Export to HTML file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed BOM')
    parser.add_argument('--compare', action='store_true', help='Compare all configurations')
    
    args = parser.parse_args()
    
    # Create BOM
    if args.config == 'budget':
        bom = create_budget_bom()
    elif args.config == 'enhanced':
        bom = create_enhanced_bom()
    else:
        bom = create_standard_bom()
    
    # Output
    if args.compare:
        compare_configurations()
    elif args.csv:
        export_csv(bom, args.csv)
    elif args.html:
        export_html(bom, args.html)
    elif args.detailed:
        print_detailed_bom(bom)
    else:
        print_bom_summary(bom)
        print("\nUse --detailed for full component list")
        print("Use --csv FILE.csv to export")
        print("Use --html FILE.html for web view")
        print("Use --compare to see all configurations")


if __name__ == "__main__":
    main()
