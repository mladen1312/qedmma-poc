#!/bin/bash
#===============================================================================
# TITAN RADAR - RFSoC 4x2 DEPLOYMENT SCRIPT
# PetaLinux/PYNQ Boot Image Builder
#
# Author: Dr. Mladen Mešter
# Copyright (c) 2026 - All Rights Reserved
#
# This script automates the complete deployment process for TITAN radar
# on the AMD RFSoC 4x2 platform.
#===============================================================================

set -e

#===============================================================================
# Configuration
#===============================================================================

TITAN_VERSION="2.0.0"
BOARD="rfsoc4x2"
VIVADO_VERSION="2023.2"
PETALINUX_VERSION="2023.2"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_DIR="${PROJECT_ROOT}/output"
BITSTREAM_DIR="${PROJECT_ROOT}/bitstreams"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#===============================================================================
# Functions
#===============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  $1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_header "CHECKING PREREQUISITES"
    
    # Check Vivado
    if command -v vivado &> /dev/null; then
        VIVADO_VER=$(vivado -version | head -1 | grep -oP '\d+\.\d+')
        print_step "Vivado ${VIVADO_VER} found"
    else
        print_warn "Vivado not found - bitstream build will be skipped"
    fi
    
    # Check PetaLinux
    if command -v petalinux-build &> /dev/null; then
        print_step "PetaLinux found"
    else
        print_warn "PetaLinux not found - using PYNQ SD image instead"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version)
        print_step "${PYTHON_VER} found"
    else
        print_error "Python3 not found!"
        exit 1
    fi
    
    # Check required Python packages
    print_step "Checking Python packages..."
    python3 -c "import numpy" 2>/dev/null || pip3 install numpy
    python3 -c "import scipy" 2>/dev/null || pip3 install scipy
    python3 -c "import matplotlib" 2>/dev/null || pip3 install matplotlib
}

create_directories() {
    print_header "CREATING BUILD DIRECTORIES"
    
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}/boot"
    mkdir -p "${OUTPUT_DIR}/rootfs"
    mkdir -p "${OUTPUT_DIR}/titan"
    
    print_step "Directories created"
}

build_bitstream() {
    print_header "BUILDING FPGA BITSTREAM"
    
    if [ ! -f "${BITSTREAM_DIR}/titan_radar.bit" ]; then
        if command -v vivado &> /dev/null; then
            print_step "Running Vivado synthesis..."
            cd "${PROJECT_ROOT}/tcl"
            vivado -mode batch -source build_titan_overlay.tcl
            
            # Copy outputs
            cp "${BUILD_DIR}/titan_radar/titan_radar.runs/impl_1/titan_radar_wrapper.bit" \
               "${BITSTREAM_DIR}/titan_radar.bit"
            cp "${BUILD_DIR}/titan_radar/titan_radar.gen/sources_1/bd/titan_radar/hw_handoff/titan_radar.hwh" \
               "${BITSTREAM_DIR}/titan_radar.hwh"
            
            print_step "Bitstream built successfully"
        else
            print_warn "Vivado not available - using pre-built bitstream"
        fi
    else
        print_step "Using existing bitstream"
    fi
}

package_software() {
    print_header "PACKAGING TITAN SOFTWARE"
    
    TITAN_PKG="${OUTPUT_DIR}/titan"
    
    # Copy Python software
    print_step "Copying Python modules..."
    cp "${PROJECT_ROOT}/software/"*.py "${TITAN_PKG}/"
    cp "${PROJECT_ROOT}/drivers/"*.py "${TITAN_PKG}/"
    
    # Copy bitstream
    if [ -f "${BITSTREAM_DIR}/titan_radar.bit" ]; then
        cp "${BITSTREAM_DIR}/titan_radar.bit" "${TITAN_PKG}/"
        cp "${BITSTREAM_DIR}/titan_radar.hwh" "${TITAN_PKG}/"
    fi
    
    # Create requirements.txt
    cat > "${TITAN_PKG}/requirements.txt" << 'EOF'
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
numba>=0.54.0
pynq>=2.7.0
EOF
    
    # Create setup script
    cat > "${TITAN_PKG}/setup.sh" << 'EOF'
#!/bin/bash
# TITAN Radar Setup Script for RFSoC 4x2

echo "Installing TITAN Radar dependencies..."
pip3 install -r requirements.txt

echo "Setting up PYNQ overlay..."
mkdir -p /home/xilinx/pynq/overlays/titan
cp titan_radar.bit /home/xilinx/pynq/overlays/titan/
cp titan_radar.hwh /home/xilinx/pynq/overlays/titan/

echo "Creating systemd service..."
sudo cp titan-radar.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable titan-radar

echo "TITAN Radar setup complete!"
echo "Start with: sudo systemctl start titan-radar"
echo "Or run manually: python3 run_titan.py --mode radar"
EOF
    chmod +x "${TITAN_PKG}/setup.sh"
    
    # Create systemd service
    cat > "${TITAN_PKG}/titan-radar.service" << 'EOF'
[Unit]
Description=TITAN VHF Anti-Stealth Radar
After=network.target

[Service]
Type=simple
User=xilinx
WorkingDirectory=/home/xilinx/titan
ExecStart=/usr/bin/python3 /home/xilinx/titan/run_titan.py --mode radar
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    print_step "Software packaged: ${TITAN_PKG}"
}

create_boot_script() {
    print_header "CREATING BOOT CONFIGURATION"
    
    # Create boot.py for PYNQ auto-start
    cat > "${OUTPUT_DIR}/boot/boot.py" << 'EOF'
#!/usr/bin/env python3
"""
TITAN Radar Boot Script
Runs automatically on PYNQ startup
"""

import sys
import os
import time

# Add TITAN to path
sys.path.insert(0, '/home/xilinx/titan')

def main():
    print("=" * 60)
    print("TITAN VHF Anti-Stealth Radar System")
    print("Version 2.0.0 - RFSoC 4x2 Platform")
    print("=" * 60)
    
    # Wait for system to stabilize
    time.sleep(5)
    
    # Check if auto-start is enabled
    if os.path.exists('/home/xilinx/titan/.autostart'):
        print("Auto-starting TITAN radar...")
        from run_titan import main as titan_main
        titan_main()
    else:
        print("Auto-start disabled. Run manually:")
        print("  python3 /home/xilinx/titan/run_titan.py --mode radar")

if __name__ == "__main__":
    main()
EOF
    
    print_step "Boot script created"
}

create_flash_script() {
    print_header "CREATING FLASH/PROGRAMMING SCRIPTS"
    
    # JTAG programming script
    cat > "${OUTPUT_DIR}/program_jtag.tcl" << 'EOF'
# TITAN Radar - JTAG Programming Script
# Run with: vivado -mode batch -source program_jtag.tcl

open_hw_manager
connect_hw_server -allow_non_jtag

# Find target
open_hw_target

# Get device
set device [lindex [get_hw_devices] 0]
current_hw_device $device

# Program bitstream
set_property PROGRAM.FILE {titan_radar.bit} $device
program_hw_devices $device

# Verify
puts "Programming complete!"
close_hw_target
disconnect_hw_server
close_hw_manager
EOF
    
    # SD card preparation script
    cat > "${OUTPUT_DIR}/prepare_sd.sh" << 'EOF'
#!/bin/bash
# TITAN Radar - SD Card Preparation Script
# Usage: ./prepare_sd.sh /dev/sdX

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 /dev/sdX"
    echo "WARNING: This will erase all data on the target device!"
    exit 1
fi

DEVICE=$1
MOUNT_POINT="/mnt/titan_sd"

echo "WARNING: This will erase all data on ${DEVICE}!"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Unmount if mounted
sudo umount ${DEVICE}* 2>/dev/null || true

# Create partitions
echo "Creating partitions..."
sudo parted -s ${DEVICE} mklabel msdos
sudo parted -s ${DEVICE} mkpart primary fat32 1MiB 512MiB
sudo parted -s ${DEVICE} mkpart primary ext4 512MiB 100%

# Format partitions
echo "Formatting..."
sudo mkfs.vfat -F 32 ${DEVICE}1
sudo mkfs.ext4 ${DEVICE}2

# Mount
sudo mkdir -p ${MOUNT_POINT}/boot
sudo mkdir -p ${MOUNT_POINT}/root
sudo mount ${DEVICE}1 ${MOUNT_POINT}/boot
sudo mount ${DEVICE}2 ${MOUNT_POINT}/root

# Copy PYNQ boot files (must be obtained separately)
echo "Copy PYNQ boot files to ${MOUNT_POINT}/boot"
echo "Then copy TITAN files:"
echo "  cp -r titan/* ${MOUNT_POINT}/root/home/xilinx/titan/"

# Cleanup
# sudo umount ${MOUNT_POINT}/boot
# sudo umount ${MOUNT_POINT}/root

echo "SD card prepared. Complete PYNQ installation manually."
EOF
    chmod +x "${OUTPUT_DIR}/prepare_sd.sh"
    
    print_step "Flash scripts created"
}

create_ota_update() {
    print_header "CREATING OTA UPDATE PACKAGE"
    
    OTA_DIR="${OUTPUT_DIR}/ota"
    mkdir -p "${OTA_DIR}"
    
    # Create update manifest
    cat > "${OTA_DIR}/manifest.json" << EOF
{
    "version": "${TITAN_VERSION}",
    "date": "$(date -Iseconds)",
    "components": {
        "software": {
            "version": "${TITAN_VERSION}",
            "files": [
                "titan_signal_processor.py",
                "titan_rfsoc_driver.py",
                "titan_display.py",
                "titan_doppler_compensated.py",
                "titan_doppler_enhanced.py",
                "run_titan.py"
            ]
        },
        "bitstream": {
            "version": "${TITAN_VERSION}",
            "file": "titan_radar.bit",
            "hwh": "titan_radar.hwh"
        }
    },
    "checksum_algorithm": "sha256"
}
EOF
    
    # Create update script
    cat > "${OTA_DIR}/apply_update.sh" << 'EOF'
#!/bin/bash
# TITAN Radar OTA Update Script

set -e

TITAN_DIR="/home/xilinx/titan"
BACKUP_DIR="/home/xilinx/titan_backup_$(date +%Y%m%d_%H%M%S)"

echo "TITAN Radar OTA Update"
echo "======================"

# Create backup
echo "Creating backup..."
cp -r ${TITAN_DIR} ${BACKUP_DIR}

# Stop service
echo "Stopping TITAN service..."
sudo systemctl stop titan-radar 2>/dev/null || true

# Apply update
echo "Applying update..."
cp -f *.py ${TITAN_DIR}/

if [ -f "titan_radar.bit" ]; then
    echo "Updating bitstream..."
    cp titan_radar.bit /home/xilinx/pynq/overlays/titan/
    cp titan_radar.hwh /home/xilinx/pynq/overlays/titan/
fi

# Restart service
echo "Restarting TITAN service..."
sudo systemctl start titan-radar

echo "Update complete!"
echo "Backup saved to: ${BACKUP_DIR}"
EOF
    chmod +x "${OTA_DIR}/apply_update.sh"
    
    # Copy software to OTA package
    cp "${PROJECT_ROOT}/software/"*.py "${OTA_DIR}/" 2>/dev/null || true
    
    # Create tarball
    cd "${OUTPUT_DIR}"
    tar -czvf "titan_ota_${TITAN_VERSION}.tar.gz" -C ota .
    
    print_step "OTA package created: titan_ota_${TITAN_VERSION}.tar.gz"
}

create_deployment_docs() {
    print_header "CREATING DEPLOYMENT DOCUMENTATION"
    
    cat > "${OUTPUT_DIR}/DEPLOYMENT_GUIDE.md" << 'EOF'
# TITAN Radar - Deployment Guide

## RFSoC 4x2 Platform

### Prerequisites

1. **RFSoC 4x2 Board** (AMD University Program)
2. **PYNQ SD Image** for RFSoC 4x2
3. **Host PC** with:
   - Vivado 2023.2 (optional, for bitstream rebuild)
   - Python 3.8+
   - SSH client

### Quick Start (PYNQ Image)

1. **Flash PYNQ Image to SD Card**
   ```bash
   # Download PYNQ image for RFSoC 4x2 from:
   # http://www.pynq.io/board.html
   
   # Flash to SD card (Linux)
   sudo dd if=pynq_rfsoc4x2.img of=/dev/sdX bs=4M status=progress
   ```

2. **Boot RFSoC 4x2**
   - Insert SD card
   - Connect Ethernet
   - Power on
   - Wait ~2 minutes for boot

3. **Connect via SSH**
   ```bash
   ssh xilinx@pynq   # Password: xilinx
   ```

4. **Install TITAN Software**
   ```bash
   # Copy TITAN package
   scp -r titan/ xilinx@pynq:/home/xilinx/
   
   # SSH to board and run setup
   ssh xilinx@pynq
   cd ~/titan
   ./setup.sh
   ```

5. **Run TITAN Radar**
   ```bash
   # Simulation mode (no RF hardware)
   python3 run_titan.py --mode simulation
   
   # Loopback test (TX→RX with attenuator)
   python3 run_titan.py --mode loopback
   
   # Full radar operation
   python3 run_titan.py --mode radar
   ```

### Hardware Connections

```
RFSoC 4x2 Board Connections:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  DAC0 (SMA) ──► BPF ──► PA ──► TX Antenna         │
│                                                     │
│  ADC0 (SMA) ◄── BPF ◄── LNA ◄── RX Antenna 1      │
│  ADC1 (SMA) ◄── BPF ◄── LNA ◄── RX Antenna 2      │
│  ADC2 (SMA) ◄── BPF ◄── LNA ◄── RX Antenna 3      │
│  ADC3 (SMA) ◄── BPF ◄── LNA ◄── RX Antenna 4      │
│                                                     │
│  Ethernet ────► Network (SSH, Jupyter)             │
│  USB ─────────► Debug console                      │
│  JTAG ────────► Programming (optional)             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Loopback Test Setup

For initial testing without antennas:

```
DAC0 ──► 30dB Attenuator ──► Power Splitter ──┬──► ADC0
                                              ├──► ADC1
                                              ├──► ADC2
                                              └──► ADC3
```

**CRITICAL: Use 30dB attenuator to prevent ADC damage!**

### OTA Updates

```bash
# On host PC
scp titan_ota_2.0.0.tar.gz xilinx@pynq:/tmp/

# On RFSoC 4x2
cd /tmp
tar -xzf titan_ota_2.0.0.tar.gz
./apply_update.sh
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| No network | Check DHCP, try static IP |
| Overlay fails | Verify .bit and .hwh match |
| ADC clipping | Increase attenuation |
| No detections | Check antenna connections |

### Performance Verification

```bash
# Run benchmark
python3 run_titan.py --benchmark --prbs 15

# Expected results:
# - Processing rate: >100 CPIs/second
# - Latency: <10 ms
# - Memory usage: <2 GB
```

EOF
    
    print_step "Documentation created"
}

create_production_package() {
    print_header "CREATING PRODUCTION PACKAGE"
    
    PROD_PKG="${OUTPUT_DIR}/titan_production_${TITAN_VERSION}"
    mkdir -p "${PROD_PKG}"
    
    # Copy all components
    cp -r "${OUTPUT_DIR}/titan" "${PROD_PKG}/"
    cp -r "${OUTPUT_DIR}/boot" "${PROD_PKG}/"
    cp "${OUTPUT_DIR}/program_jtag.tcl" "${PROD_PKG}/"
    cp "${OUTPUT_DIR}/prepare_sd.sh" "${PROD_PKG}/"
    cp "${OUTPUT_DIR}/DEPLOYMENT_GUIDE.md" "${PROD_PKG}/"
    cp "${OUTPUT_DIR}/titan_ota_${TITAN_VERSION}.tar.gz" "${PROD_PKG}/"
    
    # Create final archive
    cd "${OUTPUT_DIR}"
    tar -czvf "titan_production_${TITAN_VERSION}.tar.gz" \
        "titan_production_${TITAN_VERSION}"
    
    print_step "Production package: titan_production_${TITAN_VERSION}.tar.gz"
}

#===============================================================================
# Main
#===============================================================================

main() {
    print_header "TITAN RADAR DEPLOYMENT BUILDER v${TITAN_VERSION}"
    
    echo ""
    echo "Target: RFSoC 4x2 (${BOARD})"
    echo "Vivado: ${VIVADO_VERSION}"
    echo ""
    
    check_prerequisites
    create_directories
    build_bitstream
    package_software
    create_boot_script
    create_flash_script
    create_ota_update
    create_deployment_docs
    create_production_package
    
    print_header "DEPLOYMENT BUILD COMPLETE"
    
    echo ""
    echo "Output files in: ${OUTPUT_DIR}"
    echo ""
    echo "  titan_production_${TITAN_VERSION}.tar.gz  - Complete production package"
    echo "  titan_ota_${TITAN_VERSION}.tar.gz        - OTA update package"
    echo "  DEPLOYMENT_GUIDE.md                       - Installation guide"
    echo ""
    echo "Next steps:"
    echo "  1. Flash PYNQ image to SD card"
    echo "  2. Copy titan/ folder to /home/xilinx/"
    echo "  3. Run ./setup.sh on the board"
    echo "  4. Start radar: python3 run_titan.py --mode radar"
    echo ""
}

main "$@"
