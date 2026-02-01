# ============================================================================
# TITAN Radar - Vivado Block Design Build Script
# Complete radar overlay for RFSoC 4x2
#
# Author: Dr. Mladen Mešter
# Copyright (c) 2026 - All Rights Reserved
#
# Usage: vivado -mode batch -source build_titan_overlay.tcl
# ============================================================================

# Project settings
set PROJECT_NAME "titan_radar"
set PROJECT_DIR "./titan_radar_project"
set PART "xczu48dr-ffvg1517-2-e"
set BOARD "realdigital.org:rfsoc4x2:part0:1.0"

# IP Repository (HLS-generated cores)
set IP_REPO_PATH "../hls"

# ============================================================================
# Create Project
# ============================================================================

puts "=============================================="
puts "TITAN Radar - Vivado Build System"
puts "Part: $PART"
puts "=============================================="

# Create project
create_project $PROJECT_NAME $PROJECT_DIR -part $PART -force
set_property board_part $BOARD [current_project]

# Add IP repositories
set_property ip_repo_paths [list \
    "${IP_REPO_PATH}/waveform_generator_hls/solution1/impl/ip" \
    "${IP_REPO_PATH}/beamformer_hls/solution1/impl/ip" \
    "${IP_REPO_PATH}/correlator_hls/solution1/impl/ip" \
    "${IP_REPO_PATH}/doppler_fft_hls/solution1/impl/ip" \
    "${IP_REPO_PATH}/cfar_detector_hls/solution1/impl/ip" \
    "${IP_REPO_PATH}/track_processor_hls/solution1/impl/ip" \
] [current_project]
update_ip_catalog

# ============================================================================
# Create Block Design
# ============================================================================

create_bd_design "titan_radar_bd"

# ----------------------------------------------------------------------------
# Zynq UltraScale+ RFSoC
# ----------------------------------------------------------------------------

puts "Adding Zynq UltraScale+ RFSoC..."

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps

# Configure PS
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ {500} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE {1} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__USE__IRQ1 {1} \
] [get_bd_cells zynq_ps]

# ----------------------------------------------------------------------------
# RF Data Converter
# ----------------------------------------------------------------------------

puts "Adding RF Data Converter..."

create_bd_cell -type ip -vlnv xilinx.com:ip:usp_rf_data_converter:2.6 rf_data_converter

# Configure for RFSoC 4x2
set_property -dict [list \
    CONFIG.ADC0_Enable {1} \
    CONFIG.ADC0_Fabric_Freq {500.000} \
    CONFIG.ADC0_Outclk_Freq {500.000} \
    CONFIG.ADC0_PLL_Enable {true} \
    CONFIG.ADC0_Refclk_Freq {409.600} \
    CONFIG.ADC0_Sampling_Rate {4.9152} \
    CONFIG.ADC1_Enable {1} \
    CONFIG.ADC1_Fabric_Freq {500.000} \
    CONFIG.ADC1_Outclk_Freq {500.000} \
    CONFIG.ADC1_PLL_Enable {true} \
    CONFIG.ADC1_Refclk_Freq {409.600} \
    CONFIG.ADC1_Sampling_Rate {4.9152} \
    CONFIG.ADC2_Enable {1} \
    CONFIG.ADC2_Fabric_Freq {500.000} \
    CONFIG.ADC2_Outclk_Freq {500.000} \
    CONFIG.ADC2_PLL_Enable {true} \
    CONFIG.ADC2_Refclk_Freq {409.600} \
    CONFIG.ADC2_Sampling_Rate {4.9152} \
    CONFIG.ADC3_Enable {1} \
    CONFIG.ADC3_Fabric_Freq {500.000} \
    CONFIG.ADC3_Outclk_Freq {500.000} \
    CONFIG.ADC3_PLL_Enable {true} \
    CONFIG.ADC3_Refclk_Freq {409.600} \
    CONFIG.ADC3_Sampling_Rate {4.9152} \
    CONFIG.DAC0_Enable {1} \
    CONFIG.DAC0_Fabric_Freq {500.000} \
    CONFIG.DAC0_Outclk_Freq {500.000} \
    CONFIG.DAC0_PLL_Enable {true} \
    CONFIG.DAC0_Refclk_Freq {409.600} \
    CONFIG.DAC0_Sampling_Rate {9.8304} \
    CONFIG.DAC1_Enable {1} \
    CONFIG.DAC1_Fabric_Freq {500.000} \
    CONFIG.DAC1_Outclk_Freq {500.000} \
    CONFIG.DAC1_PLL_Enable {true} \
    CONFIG.DAC1_Refclk_Freq {409.600} \
    CONFIG.DAC1_Sampling_Rate {9.8304} \
    CONFIG.ADC_Decimation_Mode00 {1} \
    CONFIG.ADC_Decimation_Mode01 {1} \
    CONFIG.ADC_Decimation_Mode02 {1} \
    CONFIG.ADC_Decimation_Mode03 {1} \
    CONFIG.ADC_Mixer_Type00 {2} \
    CONFIG.ADC_Mixer_Type01 {2} \
    CONFIG.ADC_Mixer_Type02 {2} \
    CONFIG.ADC_Mixer_Type03 {2} \
    CONFIG.ADC_NCO_Freq00 {0.155} \
    CONFIG.ADC_NCO_Freq01 {0.155} \
    CONFIG.ADC_NCO_Freq02 {0.155} \
    CONFIG.ADC_NCO_Freq03 {0.155} \
    CONFIG.DAC_Interpolation_Mode00 {1} \
    CONFIG.DAC_Interpolation_Mode01 {1} \
    CONFIG.DAC_Mixer_Type00 {2} \
    CONFIG.DAC_Mixer_Type01 {2} \
    CONFIG.DAC_NCO_Freq00 {0.155} \
    CONFIG.DAC_NCO_Freq01 {0.155} \
] [get_bd_cells rf_data_converter]

# ----------------------------------------------------------------------------
# DMA Engines
# ----------------------------------------------------------------------------

puts "Adding DMA engines..."

# TX DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_tx
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_mm2s_burst_size {256} \
] [get_bd_cells dma_tx]

# RX DMAs (4 channels)
for {set i 0} {$i < 4} {incr i} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_rx_$i
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_sg_include_stscntrl_strm {0} \
        CONFIG.c_include_mm2s {0} \
        CONFIG.c_include_s2mm {1} \
        CONFIG.c_m_axi_s2mm_data_width {128} \
        CONFIG.c_s_axis_s2mm_tdata_width {128} \
        CONFIG.c_s2mm_burst_size {256} \
    ] [get_bd_cells dma_rx_$i]
}

# Results DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_results
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {0} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_s_axis_s2mm_tdata_width {64} \
] [get_bd_cells dma_results]

# ----------------------------------------------------------------------------
# HLS IP Cores
# ----------------------------------------------------------------------------

puts "Adding HLS IP cores..."

# Waveform Generator
create_bd_cell -type ip -vlnv titan.radar:hls:titan_waveform_generator:1.0 waveform_gen

# Beamformer
create_bd_cell -type ip -vlnv titan.radar:hls:titan_beamformer:1.0 beamformer

# Correlator
create_bd_cell -type ip -vlnv titan.radar:hls:titan_correlator:1.0 correlator

# Doppler FFT
create_bd_cell -type ip -vlnv titan.radar:hls:titan_doppler_fft:1.0 doppler_fft

# CFAR Detector
create_bd_cell -type ip -vlnv titan.radar:hls:titan_cfar_detector:1.0 cfar_detector

# Track Processor
create_bd_cell -type ip -vlnv titan.radar:hls:titan_track_processor:1.0 track_processor

# ----------------------------------------------------------------------------
# AXI Interconnects
# ----------------------------------------------------------------------------

puts "Adding AXI interconnects..."

# Control interconnect (PS -> IP cores)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ctrl
set_property -dict [list \
    CONFIG.NUM_MI {10} \
    CONFIG.NUM_SI {1} \
] [get_bd_cells axi_ctrl]

# DMA interconnect (DMAs -> PS DDR)
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_dma_ic
set_property -dict [list \
    CONFIG.NUM_SI {7} \
    CONFIG.NUM_MI {1} \
] [get_bd_cells axi_dma_ic]

# ----------------------------------------------------------------------------
# FIFOs
# ----------------------------------------------------------------------------

puts "Adding FIFOs..."

# TX FIFO
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_tx
set_property -dict [list \
    CONFIG.TDATA_NUM_BYTES {16} \
    CONFIG.FIFO_DEPTH {8192} \
    CONFIG.HAS_TKEEP {1} \
    CONFIG.HAS_TLAST {1} \
] [get_bd_cells fifo_tx]

# RX FIFOs
for {set i 0} {$i < 4} {incr i} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_rx_$i
    set_property -dict [list \
        CONFIG.TDATA_NUM_BYTES {16} \
        CONFIG.FIFO_DEPTH {16384} \
        CONFIG.HAS_TKEEP {1} \
        CONFIG.HAS_TLAST {1} \
    ] [get_bd_cells fifo_rx_$i]
}

# ----------------------------------------------------------------------------
# Clocking
# ----------------------------------------------------------------------------

puts "Adding clocking..."

create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz
set_property -dict [list \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {250} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {500} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.NUM_OUT_CLKS {2} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
] [get_bd_cells clk_wiz]

# ----------------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------------

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_250
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_500

# ----------------------------------------------------------------------------
# Connections
# ----------------------------------------------------------------------------

puts "Connecting blocks..."

# Clock connections
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]
connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins rst_250/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz/clk_out2] [get_bd_pins rst_500/slowest_sync_clk]

# Reset connections
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins clk_wiz/resetn]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins rst_250/ext_reset_in]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins rst_500/ext_reset_in]

# PS AXI connections
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_ctrl/S00_AXI]

# Control AXI connections to IP cores
set ctrl_slaves {
    rf_data_converter/s_axi
    waveform_gen/s_axi_control
    beamformer/s_axi_control
    correlator/s_axi_control
    doppler_fft/s_axi_control
    cfar_detector/s_axi_control
    track_processor/s_axi_control
    dma_tx/S_AXI_LITE
    dma_rx_0/S_AXI_LITE
    dma_results/S_AXI_LITE
}

set mi_idx 0
foreach slave $ctrl_slaves {
    set mi_port [format "M%02d_AXI" $mi_idx]
    connect_bd_intf_net [get_bd_intf_pins axi_ctrl/$mi_port] \
        [get_bd_intf_pins $slave]
    incr mi_idx
}

# Signal processing chain connections
# Waveform -> DAC
connect_bd_intf_net [get_bd_intf_pins waveform_gen/m_axis_dac] \
    [get_bd_intf_pins fifo_tx/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins fifo_tx/M_AXIS] \
    [get_bd_intf_pins rf_data_converter/s00_axis]

# ADC -> Beamformer
for {set i 0} {$i < 4} {incr i} {
    connect_bd_intf_net [get_bd_intf_pins rf_data_converter/m0${i}_axis] \
        [get_bd_intf_pins fifo_rx_$i/S_AXIS]
}

connect_bd_intf_net [get_bd_intf_pins fifo_rx_0/M_AXIS] \
    [get_bd_intf_pins beamformer/s_axis_ch0]
connect_bd_intf_net [get_bd_intf_pins fifo_rx_1/M_AXIS] \
    [get_bd_intf_pins beamformer/s_axis_ch1]
connect_bd_intf_net [get_bd_intf_pins fifo_rx_2/M_AXIS] \
    [get_bd_intf_pins beamformer/s_axis_ch2]
connect_bd_intf_net [get_bd_intf_pins fifo_rx_3/M_AXIS] \
    [get_bd_intf_pins beamformer/s_axis_ch3]

# Beamformer -> Correlator
connect_bd_intf_net [get_bd_intf_pins beamformer/m_axis_beam] \
    [get_bd_intf_pins correlator/s_axis_rx]

# Correlator -> Doppler FFT
connect_bd_intf_net [get_bd_intf_pins correlator/m_axis_range] \
    [get_bd_intf_pins doppler_fft/s_axis_range]

# Doppler FFT -> CFAR
connect_bd_intf_net [get_bd_intf_pins doppler_fft/m_axis_rdmap] \
    [get_bd_intf_pins cfar_detector/s_axis_rdmap]

# CFAR -> Tracker
connect_bd_intf_net [get_bd_intf_pins cfar_detector/m_axis_detections] \
    [get_bd_intf_pins track_processor/s_axis_detections]

# Tracker -> Results DMA
connect_bd_intf_net [get_bd_intf_pins track_processor/m_axis_tracks] \
    [get_bd_intf_pins dma_results/S_AXIS_S2MM]

# DMA to PS memory
connect_bd_intf_net [get_bd_intf_pins dma_tx/M_AXI_MM2S] \
    [get_bd_intf_pins axi_dma_ic/S00_AXI]
for {set i 0} {$i < 4} {incr i} {
    set si_port [format "S%02d_AXI" [expr $i + 1]]
    connect_bd_intf_net [get_bd_intf_pins dma_rx_$i/M_AXI_S2MM] \
        [get_bd_intf_pins axi_dma_ic/$si_port]
}
connect_bd_intf_net [get_bd_intf_pins dma_results/M_AXI_S2MM] \
    [get_bd_intf_pins axi_dma_ic/S05_AXI]

connect_bd_intf_net [get_bd_intf_pins axi_dma_ic/M00_AXI] \
    [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]

# ----------------------------------------------------------------------------
# Address Map
# ----------------------------------------------------------------------------

puts "Assigning addresses..."

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs rf_data_converter/s_axi/Reg] -range 256K -offset 0xA0000000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs waveform_gen/s_axi_control/Reg] -range 64K -offset 0xA0040000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs beamformer/s_axi_control/Reg] -range 64K -offset 0xA0050000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs correlator/s_axi_control/Reg] -range 64K -offset 0xA0060000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs doppler_fft/s_axi_control/Reg] -range 64K -offset 0xA0070000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs cfar_detector/s_axi_control/Reg] -range 64K -offset 0xA0080000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs track_processor/s_axi_control/Reg] -range 64K -offset 0xA0090000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs dma_tx/S_AXI_LITE/Reg] -range 64K -offset 0xA00A0000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs dma_rx_0/S_AXI_LITE/Reg] -range 64K -offset 0xA00B0000

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs dma_results/S_AXI_LITE/Reg] -range 64K -offset 0xA00C0000

# DMA memory access
assign_bd_address -target_address_space /dma_tx/Data_MM2S \
    [get_bd_addr_segs zynq_ps/SAXIGP0/HP0_DDR_LOW]

for {set i 0} {$i < 4} {incr i} {
    assign_bd_address -target_address_space /dma_rx_$i/Data_S2MM \
        [get_bd_addr_segs zynq_ps/SAXIGP0/HP0_DDR_LOW]
}

assign_bd_address -target_address_space /dma_results/Data_S2MM \
    [get_bd_addr_segs zynq_ps/SAXIGP0/HP0_DDR_LOW]

# ----------------------------------------------------------------------------
# Validate and Save
# ----------------------------------------------------------------------------

puts "Validating design..."
validate_bd_design

puts "Saving design..."
save_bd_design

# ----------------------------------------------------------------------------
# Generate Output Products
# ----------------------------------------------------------------------------

puts "Generating output products..."
generate_target all [get_files titan_radar_bd.bd]

# Create HDL wrapper
make_wrapper -files [get_files titan_radar_bd.bd] -top
add_files -norecurse $PROJECT_DIR/$PROJECT_NAME.srcs/sources_1/bd/titan_radar_bd/hdl/titan_radar_bd_wrapper.v

# ----------------------------------------------------------------------------
# Synthesis and Implementation
# ----------------------------------------------------------------------------

puts "Running synthesis..."
launch_runs synth_1 -jobs 8
wait_on_run synth_1

puts "Running implementation..."
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# ----------------------------------------------------------------------------
# Export Files
# ----------------------------------------------------------------------------

puts "Exporting files..."

# Copy bitstream
file copy -force $PROJECT_DIR/$PROJECT_NAME.runs/impl_1/titan_radar_bd_wrapper.bit \
    ../bitstreams/titan_radar.bit

# Copy hardware handoff
file copy -force $PROJECT_DIR/$PROJECT_NAME.gen/sources_1/bd/titan_radar_bd/hw_handoff/titan_radar_bd.hwh \
    ../bitstreams/titan_radar.hwh

puts "=============================================="
puts "Build complete!"
puts "=============================================="
puts ""
puts "Output files:"
puts "  Bitstream: ../bitstreams/titan_radar.bit"
puts "  HWH:       ../bitstreams/titan_radar.hwh"
puts ""
puts "Deploy to RFSoC 4x2:"
puts "  scp ../bitstreams/titan_radar.* xilinx@rfsoc4x2:/home/xilinx/"
puts ""

exit
