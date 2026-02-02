#!/bin/bash
#===============================================================================
# TITAN Radar - RFSoC 4x2 Deployment Script
# PetaLinux/PYNQ Build and Flash Automation
#
# Author: Dr. Mladen Mešter
# Copyright (c) 2026 - All Rights Reserved
#
# Usage:
#   ./deploy_titan.sh build      - Build complete system
#   ./deploy_titan.sh flash      - Flash to SD card
#   ./deploy_titan.sh package    - Create deployment package
#   ./deploy_titan.sh all        - Full pipeline
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="titan_radar"
VERSION="2.0.0"

# Paths
VIVADO_VERSION="2023.2"
PETALINUX_VERSION="2023.2"
PYNQ_VERSION="3.0.1"

# Build directories
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/output"
BITSTREAM_DIR="${SCRIPT_DIR}/../bitstreams"
SOFTWARE_DIR="${SCRIPT_DIR}/../software"
DRIVERS_DIR="${SCRIPT_DIR}/../drivers"

# Target device
DEVICE="xczu48dr-ffvg1517-2-e"
BOARD="realdigital.org:rfsoc4x2:part0:1.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Vivado
    if ! command -v vivado &> /dev/null; then
        log_warning "Vivado not found. Some features may not work."
    else
        log_success "Vivado found: $(vivado -version | head -1)"
    fi
    
    # Check PetaLinux
    if ! command -v petalinux-build &> /dev/null; then
        log_warning "PetaLinux not found. Using PYNQ SD image instead."
        USE_PYNQ=true
    else
        log_success "PetaLinux found"
        USE_PYNQ=false
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 required but not found!"
        exit 1
    fi
    
    log_success "Prerequisites check complete"
}

#-------------------------------------------------------------------------------
# Build Functions
#-------------------------------------------------------------------------------

build_bitstream() {
    log_info "Building FPGA bitstream..."
    
    if [ -f "${BITSTREAM_DIR}/titan_radar.bit" ]; then
        log_info "Bitstream already exists. Skipping build."
        return 0
    fi
    
    # Build HLS IP cores first
    log_info "Building HLS IP cores..."
    cd "${SCRIPT_DIR}/../hls"
    
    if command -v vitis_hls &> /dev/null; then
        vitis_hls -f run_hls.tcl
    else
        log_warning "Vitis HLS not found. Using pre-built IPs."
    fi
    
    # Build Vivado project
    log_info "Building Vivado project..."
    cd "${SCRIPT_DIR}/../tcl"
    
    if command -v vivado &> /dev/null; then
        vivado -mode batch -source build_titan_overlay.tcl
    else
        log_warning "Vivado not found. Using pre-built bitstream."
    fi
    
    log_success "Bitstream build complete"
}

build_pynq_overlay() {
    log_info "Building PYNQ overlay package..."
    
    mkdir -p "${BUILD_DIR}/overlay"
    
    # Copy bitstream and hardware handoff
    if [ -f "${BITSTREAM_DIR}/titan_radar.bit" ]; then
        cp "${BITSTREAM_DIR}/titan_radar.bit" "${BUILD_DIR}/overlay/"
        cp "${BITSTREAM_DIR}/titan_radar.hwh" "${BUILD_DIR}/overlay/"
    else
        log_warning "Bitstream not found. Creating placeholder."
        touch "${BUILD_DIR}/overlay/titan_radar.bit"
        touch "${BUILD_DIR}/overlay/titan_radar.hwh"
    fi
    
    # Copy Python drivers
    cp "${DRIVERS_DIR}"/*.py "${BUILD_DIR}/overlay/"
    cp "${SOFTWARE_DIR}"/*.py "${BUILD_DIR}/overlay/"
    
    # Create __init__.py
    cat > "${BUILD_DIR}/overlay/__init__.py" << 'EOF'
"""
TITAN Radar - PYNQ Overlay Package
VHF Anti-Stealth Radar System

Author: Dr. Mladen Mešter
Copyright (c) 2026 - All Rights Reserved
"""

from .titan_radar import TITANOverlay
from .titan_signal_processor import TITANProcessor, TITANConfig
from .titan_rfsoc_driver import TITANRFSoC, RFSoCConfig
from .titan_display import TITANDisplay

__version__ = "2.0.0"
__all__ = ['TITANOverlay', 'TITANProcessor', 'TITANConfig', 
           'TITANRFSoC', 'RFSoCConfig', 'TITANDisplay']
EOF

    log_success "PYNQ overlay package created"
}

build_petalinux() {
    log_info "Building PetaLinux image..."
    
    if [ "$USE_PYNQ" = true ]; then
        log_info "Skipping PetaLinux build (using PYNQ)."
        return 0
    fi
    
    mkdir -p "${BUILD_DIR}/petalinux"
    cd "${BUILD_DIR}/petalinux"
    
    # Create PetaLinux project
    if [ ! -d "titan_radar" ]; then
        petalinux-create -t project -n titan_radar --template zynqMP
    fi
    
    cd titan_radar
    
    # Configure hardware
    if [ -f "${BITSTREAM_DIR}/titan_radar.xsa" ]; then
        petalinux-config --get-hw-description="${BITSTREAM_DIR}"
    fi
    
    # Build
    petalinux-build
    
    # Package
    petalinux-package --boot --fsbl images/linux/zynqmp_fsbl.elf \
        --fpga images/linux/system.bit \
        --u-boot images/linux/u-boot.elf \
        --force
    
    log_success "PetaLinux build complete"
}

#-------------------------------------------------------------------------------
# Package Functions
#-------------------------------------------------------------------------------

create_sd_image() {
    log_info "Creating SD card image..."
    
    mkdir -p "${OUTPUT_DIR}"
    
    # Create boot partition files
    mkdir -p "${OUTPUT_DIR}/BOOT"
    mkdir -p "${OUTPUT_DIR}/rootfs"
    
    if [ "$USE_PYNQ" = true ]; then
        # PYNQ-based deployment
        log_info "Preparing PYNQ-based deployment..."
        
        # Copy overlay to PYNQ location
        mkdir -p "${OUTPUT_DIR}/rootfs/home/xilinx/titan_radar"
        cp -r "${BUILD_DIR}/overlay"/* "${OUTPUT_DIR}/rootfs/home/xilinx/titan_radar/"
        
        # Create setup script
        cat > "${OUTPUT_DIR}/rootfs/home/xilinx/titan_radar/setup.sh" << 'EOF'
#!/bin/bash
# TITAN Radar Setup Script for PYNQ

echo "Installing TITAN Radar..."

# Install Python dependencies
pip3 install numpy scipy matplotlib numba

# Copy overlay to PYNQ overlays directory
sudo mkdir -p /usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq/overlays/titan_radar
sudo cp *.bit *.hwh /usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq/overlays/titan_radar/

# Create symlink for easy access
ln -sf /home/xilinx/titan_radar /home/xilinx/jupyter_notebooks/titan_radar

echo "Setup complete! Access TITAN at:"
echo "  http://$(hostname -I | awk '{print $1}'):9090/lab"
EOF
        chmod +x "${OUTPUT_DIR}/rootfs/home/xilinx/titan_radar/setup.sh"
        
    else
        # PetaLinux-based deployment
        log_info "Preparing PetaLinux-based deployment..."
        
        cp "${BUILD_DIR}/petalinux/titan_radar/images/linux/BOOT.BIN" "${OUTPUT_DIR}/BOOT/"
        cp "${BUILD_DIR}/petalinux/titan_radar/images/linux/image.ub" "${OUTPUT_DIR}/BOOT/"
        cp "${BUILD_DIR}/petalinux/titan_radar/images/linux/boot.scr" "${OUTPUT_DIR}/BOOT/"
    fi
    
    log_success "SD card image prepared in ${OUTPUT_DIR}"
}

create_deployment_package() {
    log_info "Creating deployment package..."
    
    PACKAGE_NAME="titan_radar_v${VERSION}_$(date +%Y%m%d)"
    PACKAGE_DIR="${OUTPUT_DIR}/${PACKAGE_NAME}"
    
    mkdir -p "${PACKAGE_DIR}"
    
    # Copy all components
    cp -r "${BUILD_DIR}/overlay" "${PACKAGE_DIR}/"
    cp -r "${OUTPUT_DIR}/BOOT" "${PACKAGE_DIR}/" 2>/dev/null || true
    cp -r "${OUTPUT_DIR}/rootfs" "${PACKAGE_DIR}/" 2>/dev/null || true
    
    # Copy documentation
    mkdir -p "${PACKAGE_DIR}/docs"
    cp "${SCRIPT_DIR}/../docs"/*.md "${PACKAGE_DIR}/docs/" 2>/dev/null || true
    
    # Create README
    cat > "${PACKAGE_DIR}/README.txt" << EOF
===============================================================================
TITAN VHF Anti-Stealth Radar System
Version: ${VERSION}
Build Date: $(date)
===============================================================================

CONTENTS:
---------
  overlay/      - PYNQ overlay files (bitstream + Python drivers)
  BOOT/         - Boot partition files (if PetaLinux build)
  rootfs/       - Root filesystem additions
  docs/         - Documentation

QUICK START (PYNQ):
-------------------
1. Flash PYNQ v${PYNQ_VERSION} SD image to card
2. Copy overlay/ folder to /home/xilinx/titan_radar/
3. Run: cd /home/xilinx/titan_radar && ./setup.sh
4. Access Jupyter at http://<board-ip>:9090

QUICK START (PetaLinux):
------------------------
1. Copy BOOT/* to SD card BOOT partition
2. Extract rootfs to SD card rootfs partition
3. Boot the board
4. Run: python3 /home/root/titan_radar/run_titan.py

SUPPORT:
--------
Author: Dr. Mladen Mešter
Repository: https://github.com/mladen1312/qedmma-poc

===============================================================================
EOF

    # Create archive
    cd "${OUTPUT_DIR}"
    tar -czvf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
    
    log_success "Deployment package created: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
}

#-------------------------------------------------------------------------------
# Flash Functions
#-------------------------------------------------------------------------------

flash_sd_card() {
    log_info "Flashing SD card..."
    
    # Detect SD card
    echo "Available block devices:"
    lsblk -d -o NAME,SIZE,MODEL | grep -E "sd|mmcblk"
    
    read -p "Enter SD card device (e.g., sdb or mmcblk0): " SD_DEVICE
    
    if [ -z "$SD_DEVICE" ]; then
        log_error "No device specified!"
        exit 1
    fi
    
    SD_PATH="/dev/${SD_DEVICE}"
    
    if [ ! -b "$SD_PATH" ]; then
        log_error "Device ${SD_PATH} not found!"
        exit 1
    fi
    
    log_warning "This will ERASE all data on ${SD_PATH}!"
    read -p "Are you sure? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        log_info "Aborted."
        exit 0
    fi
    
    # Unmount any mounted partitions
    sudo umount ${SD_PATH}* 2>/dev/null || true
    
    if [ "$USE_PYNQ" = true ]; then
        log_info "For PYNQ deployment:"
        echo "1. Download PYNQ v${PYNQ_VERSION} image from pynq.io"
        echo "2. Flash using Balena Etcher or:"
        echo "   sudo dd if=pynq_rfsoc4x2_v${PYNQ_VERSION}.img of=${SD_PATH} bs=4M status=progress"
        echo "3. Mount the rootfs partition and copy titan_radar files"
    else
        # Create partitions
        log_info "Creating partitions..."
        sudo parted ${SD_PATH} --script mklabel msdos
        sudo parted ${SD_PATH} --script mkpart primary fat32 1MiB 512MiB
        sudo parted ${SD_PATH} --script mkpart primary ext4 512MiB 100%
        
        # Format partitions
        log_info "Formatting partitions..."
        sudo mkfs.vfat -F 32 -n BOOT ${SD_PATH}1
        sudo mkfs.ext4 -L rootfs ${SD_PATH}2
        
        # Mount and copy files
        MOUNT_BOOT="/tmp/titan_boot"
        MOUNT_ROOT="/tmp/titan_root"
        
        mkdir -p ${MOUNT_BOOT} ${MOUNT_ROOT}
        sudo mount ${SD_PATH}1 ${MOUNT_BOOT}
        sudo mount ${SD_PATH}2 ${MOUNT_ROOT}
        
        sudo cp -r "${OUTPUT_DIR}/BOOT"/* ${MOUNT_BOOT}/
        sudo cp -r "${OUTPUT_DIR}/rootfs"/* ${MOUNT_ROOT}/
        
        sync
        sudo umount ${MOUNT_BOOT} ${MOUNT_ROOT}
    fi
    
    log_success "SD card preparation complete"
}

#-------------------------------------------------------------------------------
# Utility Functions
#-------------------------------------------------------------------------------

create_jupyter_notebook() {
    log_info "Creating Jupyter demo notebook..."
    
    mkdir -p "${BUILD_DIR}/overlay/notebooks"
    
    cat > "${BUILD_DIR}/overlay/notebooks/TITAN_Demo.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TITAN VHF Anti-Stealth Radar Demo\n",
    "## RFSoC 4x2 Platform\n",
    "\n",
    "Author: Dr. Mladen Mešter  \n",
    "Version: 2.0.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import TITAN modules\n",
    "import sys\n",
    "sys.path.insert(0, '/home/xilinx/titan_radar')\n",
    "\n",
    "from titan_signal_processor import TITANProcessor, TITANConfig\n",
    "from titan_display import TITANDisplay\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configure TITAN\n",
    "config = TITANConfig(\n",
    "    prbs_order=15,\n",
    "    num_range_bins=512,\n",
    "    num_doppler_bins=256,\n",
    "    cfar_pfa=1e-6\n",
    ")\n",
    "\n",
    "processor = TITANProcessor(config)\n",
    "print(f\"PRBS Length: {config.prbs_length:,}\")\n",
    "print(f\"Processing Gain: {config.processing_gain_db:.1f} dB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate test signal with targets\n",
    "print(\"Generating simulated radar data...\")\n",
    "\n",
    "# Simulate 256 CPIs\n",
    "for cpi in range(256):\n",
    "    # Simulated RX with noise + targets\n",
    "    rx = np.random.randn(config.prbs_length) + 1j * np.random.randn(config.prbs_length)\n",
    "    rx *= 0.1  # Noise level\n",
    "    \n",
    "    # Add target at range bin 100\n",
    "    delay = 100\n",
    "    doppler = 50  # Hz\n",
    "    t = np.arange(config.prbs_length) / 100e6\n",
    "    target_phase = np.exp(2j * np.pi * doppler * cpi * config.prbs_length / 100e6)\n",
    "    rx[delay:delay+100] += 10 * processor.prbs_bpsk[:100] * target_phase\n",
    "    \n",
    "    processor.process_cpi(rx)\n",
    "\n",
    "print(\"Processing complete!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Range-Doppler map\n",
    "rdmap = processor.generate_rdmap()\n",
    "\n",
    "plt.figure(figsize=(12, 8))\n",
    "plt.imshow(20*np.log10(rdmap + 1e-10), aspect='auto', cmap='jet')\n",
    "plt.colorbar(label='dB')\n",
    "plt.xlabel('Range Bin')\n",
    "plt.ylabel('Doppler Bin')\n",
    "plt.title('TITAN Range-Doppler Map')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detect targets\n",
    "detections = processor.detect_2d(rdmap)\n",
    "\n",
    "print(f\"\\nDetected {len(detections)} targets:\")\n",
    "for i, det in enumerate(detections[:10]):\n",
    "    print(f\"  {i+1}. Range bin: {det.range_bin}, Doppler bin: {det.doppler_bin}, SNR: {det.snr_db:.1f} dB\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

    log_success "Jupyter notebook created"
}

create_systemd_service() {
    log_info "Creating systemd service..."
    
    mkdir -p "${BUILD_DIR}/overlay/systemd"
    
    cat > "${BUILD_DIR}/overlay/systemd/titan-radar.service" << 'EOF'
[Unit]
Description=TITAN VHF Anti-Stealth Radar Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/xilinx/titan_radar
ExecStart=/usr/bin/python3 /home/xilinx/titan_radar/run_titan.py --mode radar
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    cat > "${BUILD_DIR}/overlay/systemd/install_service.sh" << 'EOF'
#!/bin/bash
# Install TITAN as systemd service

sudo cp titan-radar.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable titan-radar.service
echo "Service installed. Start with: sudo systemctl start titan-radar"
EOF
    chmod +x "${BUILD_DIR}/overlay/systemd/install_service.sh"
    
    log_success "Systemd service created"
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

print_banner() {
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║           TITAN RADAR - RFSoC 4x2 DEPLOYMENT SYSTEM                          ║"
    echo "║           Version: ${VERSION}                                                     ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build     - Build bitstream and PYNQ overlay"
    echo "  flash     - Flash SD card"
    echo "  package   - Create deployment package"
    echo "  all       - Full build + package pipeline"
    echo "  clean     - Clean build artifacts"
    echo ""
}

main() {
    print_banner
    
    case "$1" in
        build)
            check_prerequisites
            mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"
            build_bitstream
            build_pynq_overlay
            create_jupyter_notebook
            create_systemd_service
            build_petalinux
            ;;
        flash)
            flash_sd_card
            ;;
        package)
            create_sd_image
            create_deployment_package
            ;;
        all)
            check_prerequisites
            mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"
            build_bitstream
            build_pynq_overlay
            create_jupyter_notebook
            create_systemd_service
            build_petalinux
            create_sd_image
            create_deployment_package
            ;;
        clean)
            log_info "Cleaning build artifacts..."
            rm -rf "${BUILD_DIR}" "${OUTPUT_DIR}"
            log_success "Clean complete"
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
