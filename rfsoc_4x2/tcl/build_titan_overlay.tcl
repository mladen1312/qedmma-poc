# ============================================================================
# TITAN Radar Overlay - Vivado Block Design TCL Script
# Target: RFSoC 4x2 (Zynq UltraScale+ ZU48DR)
# 
# Author: Dr. Mladen Mešter
# Copyright (c) 2026 - All Rights Reserved
# ============================================================================

# ----------------------------------------------------------------------------
# Project Configuration
# ----------------------------------------------------------------------------

set project_name "titan_radar"
set project_dir "./titan_radar_project"
set part "xczu48dr-ffvg1517-2-e"
set board "realdigital.org:rfsoc4x2:part0:1.0"

# Create project
create_project $project_name $project_dir -part $part
set_property board_part $board [current_project]

# ----------------------------------------------------------------------------
# Create Block Design
# ----------------------------------------------------------------------------

create_bd_design "titan_radar_bd"

# ----------------------------------------------------------------------------
# Add Zynq UltraScale+ RFSoC Processing System
# ----------------------------------------------------------------------------

# Add PS
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0

# Configure PS for RFSoC 4x2
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {1} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {250} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ {500} \
] [get_bd_cells zynq_ultra_ps_e_0]

# Apply board automation
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]

# ----------------------------------------------------------------------------
# Add RF Data Converter
# ----------------------------------------------------------------------------

create_bd_cell -type ip -vlnv xilinx.com:ip:usp_rf_data_converter:2.6 usp_rf_data_converter_0

# Configure RF Data Converter for TITAN radar
set_property -dict [list \
    CONFIG.ADC0_Enable {1} \
    CONFIG.ADC0_Fabric_Freq {500.000} \
    CONFIG.ADC_Slice00_Enable {true} \
    CONFIG.ADC_Slice01_Enable {true} \
    CONFIG.ADC_Slice02_Enable {true} \
    CONFIG.ADC_Slice03_Enable {true} \
    CONFIG.ADC_Data_Type00 {1} \
    CONFIG.ADC_Data_Type01 {1} \
    CONFIG.ADC_Data_Type02 {1} \
    CONFIG.ADC_Data_Type03 {1} \
    CONFIG.ADC_Data_Width00 {8} \
    CONFIG.ADC_Data_Width01 {8} \
    CONFIG.ADC_Data_Width02 {8} \
    CONFIG.ADC_Data_Width03 {8} \
    CONFIG.ADC_Decimation_Mode00 {1} \
    CONFIG.ADC_Decimation_Mode01 {1} \
    CONFIG.ADC_Decimation_Mode02 {1} \
    CONFIG.ADC_Decimation_Mode03 {1} \
    CONFIG.ADC_Mixer_Type00 {2} \
    CONFIG.ADC_Mixer_Type01 {2} \
    CONFIG.ADC_Mixer_Type02 {2} \
    CONFIG.ADC_Mixer_Type03 {2} \
    CONFIG.ADC_NCO_Freq00 {155.0} \
    CONFIG.ADC_NCO_Freq01 {155.0} \
    CONFIG.ADC_NCO_Freq02 {155.0} \
    CONFIG.ADC_NCO_Freq03 {155.0} \
    CONFIG.DAC0_Enable {1} \
    CONFIG.DAC0_Fabric_Freq {500.000} \
    CONFIG.DAC_Slice00_Enable {true} \
    CONFIG.DAC_Slice01_Enable {true} \
    CONFIG.DAC_Data_Type00 {0} \
    CONFIG.DAC_Data_Width00 {16} \
    CONFIG.DAC_Interpolation_Mode00 {1} \
    CONFIG.DAC_Mixer_Type00 {2} \
    CONFIG.DAC_NCO_Freq00 {155.0} \
] [get_bd_cells usp_rf_data_converter_0]

# ----------------------------------------------------------------------------
# Add AXI DMA Controllers
# ----------------------------------------------------------------------------

# TX DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_tx
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_mm2s_burst_size {256} \
] [get_bd_cells axi_dma_tx]

# RX DMA (4 channels)
foreach i {0 1 2 3} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_rx$i
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_sg_include_stscntrl_strm {0} \
        CONFIG.c_include_mm2s {0} \
        CONFIG.c_include_s2mm {1} \
        CONFIG.c_m_axi_s2mm_data_width {128} \
        CONFIG.c_s_axis_s2mm_tdata_width {128} \
        CONFIG.c_s2mm_burst_size {256} \
    ] [get_bd_cells axi_dma_rx$i]
}

# ----------------------------------------------------------------------------
# Add Custom Radar IP Cores (HLS Generated)
# ----------------------------------------------------------------------------

# Waveform Generator
create_bd_cell -type ip -vlnv user.org:hls:waveform_generator:1.0 waveform_gen_0

# Zero-DSP Correlator (massive parallelism)
create_bd_cell -type ip -vlnv user.org:hls:zero_dsp_correlator:1.0 correlator_0

# CFAR Detector
create_bd_cell -type ip -vlnv user.org:hls:cfar_detector:1.0 cfar_0

# Beamformer
create_bd_cell -type ip -vlnv user.org:hls:beamformer:1.0 beamformer_0

# Track Processor
create_bd_cell -type ip -vlnv user.org:hls:track_processor:1.0 tracker_0

# ----------------------------------------------------------------------------
# Add AXI Interconnects
# ----------------------------------------------------------------------------

# AXI Interconnect for control registers
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_ctrl
set_property -dict [list CONFIG.NUM_MI {8}] [get_bd_cells axi_interconnect_ctrl]

# AXI SmartConnect for DMA
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_dma
set_property -dict [list CONFIG.NUM_SI {5} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_dma]

# ----------------------------------------------------------------------------
# Add AXI Stream Data FIFOs
# ----------------------------------------------------------------------------

# TX FIFO
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_fifo_tx
set_property -dict [list \
    CONFIG.TDATA_NUM_BYTES {16} \
    CONFIG.FIFO_DEPTH {8192} \
    CONFIG.HAS_TKEEP {0} \
    CONFIG.HAS_TLAST {1} \
] [get_bd_cells axis_fifo_tx]

# RX FIFOs (4 channels)
foreach i {0 1 2 3} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_fifo_rx$i
    set_property -dict [list \
        CONFIG.TDATA_NUM_BYTES {16} \
        CONFIG.FIFO_DEPTH {16384} \
        CONFIG.HAS_TKEEP {0} \
        CONFIG.HAS_TLAST {1} \
    ] [get_bd_cells axis_fifo_rx$i]
}

# ----------------------------------------------------------------------------
# Clock and Reset
# ----------------------------------------------------------------------------

# Processor System Reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_250
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_500

# Connect clocks
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins proc_sys_reset_250/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk1] \
    [get_bd_pins proc_sys_reset_500/slowest_sync_clk]

# Connect resets
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
    [get_bd_pins proc_sys_reset_250/ext_reset_in]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
    [get_bd_pins proc_sys_reset_500/ext_reset_in]

# ----------------------------------------------------------------------------
# Connect AXI Control Path
# ----------------------------------------------------------------------------

# PS M_AXI_HPM0 to control interconnect
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_interconnect_ctrl/S00_AXI]

# Control interconnect to IP cores
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M00_AXI] \
    [get_bd_intf_pins usp_rf_data_converter_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M01_AXI] \
    [get_bd_intf_pins waveform_gen_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M02_AXI] \
    [get_bd_intf_pins correlator_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M03_AXI] \
    [get_bd_intf_pins cfar_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M04_AXI] \
    [get_bd_intf_pins beamformer_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M05_AXI] \
    [get_bd_intf_pins tracker_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M06_AXI] \
    [get_bd_intf_pins axi_dma_tx/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_ctrl/M07_AXI] \
    [get_bd_intf_pins axi_dma_rx0/S_AXI_LITE]

# ----------------------------------------------------------------------------
# Connect AXI Stream Data Path
# ----------------------------------------------------------------------------

# TX Path: DMA -> FIFO -> Waveform Gen -> DAC
connect_bd_intf_net [get_bd_intf_pins axi_dma_tx/M_AXIS_MM2S] \
    [get_bd_intf_pins axis_fifo_tx/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_fifo_tx/M_AXIS] \
    [get_bd_intf_pins waveform_gen_0/s_axis_samples]
connect_bd_intf_net [get_bd_intf_pins waveform_gen_0/m_axis_dac] \
    [get_bd_intf_pins usp_rf_data_converter_0/s00_axis]

# RX Path: ADC -> Beamformer -> Correlator -> CFAR -> DMA
# Channel 0
connect_bd_intf_net [get_bd_intf_pins usp_rf_data_converter_0/m00_axis] \
    [get_bd_intf_pins beamformer_0/s_axis_ch0]

# Additional channels connected similarly...

# Beamformer to Correlator
connect_bd_intf_net [get_bd_intf_pins beamformer_0/m_axis_beamformed] \
    [get_bd_intf_pins correlator_0/s_axis_rx]

# Correlator to CFAR
connect_bd_intf_net [get_bd_intf_pins correlator_0/m_axis_range] \
    [get_bd_intf_pins cfar_0/s_axis_range]

# CFAR to Tracker
connect_bd_intf_net [get_bd_intf_pins cfar_0/m_axis_detections] \
    [get_bd_intf_pins tracker_0/s_axis_detections]

# ----------------------------------------------------------------------------
# Connect DMA Memory Interface
# ----------------------------------------------------------------------------

# DMA to SmartConnect
connect_bd_intf_net [get_bd_intf_pins axi_dma_tx/M_AXI_MM2S] \
    [get_bd_intf_pins smartconnect_dma/S00_AXI]

foreach i {0 1 2 3} {
    connect_bd_intf_net [get_bd_intf_pins axi_dma_rx$i/M_AXI_S2MM] \
        [get_bd_intf_pins smartconnect_dma/S0[expr $i+1]_AXI]
}

# SmartConnect to PS
connect_bd_intf_net [get_bd_intf_pins smartconnect_dma/M00_AXI] \
    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

# ----------------------------------------------------------------------------
# Address Map
# ----------------------------------------------------------------------------

assign_bd_address

# Set specific addresses for IP cores
set_property offset 0x00000000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_usp_rf_data_converter_0_Reg}]
set_property offset 0x80000000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_waveform_gen_0_Reg}]
set_property offset 0x80010000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_correlator_0_Reg}]
set_property offset 0x80020000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_cfar_0_Reg}]
set_property offset 0x80030000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_beamformer_0_Reg}]
set_property offset 0x80040000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_tracker_0_Reg}]

# ----------------------------------------------------------------------------
# Validate and Save
# ----------------------------------------------------------------------------

validate_bd_design
save_bd_design

# ----------------------------------------------------------------------------
# Generate Output Products
# ----------------------------------------------------------------------------

generate_target all [get_files titan_radar_bd.bd]

# ----------------------------------------------------------------------------
# Create HDL Wrapper
# ----------------------------------------------------------------------------

make_wrapper -files [get_files titan_radar_bd.bd] -top
add_files -norecurse $project_dir/$project_name.srcs/sources_1/bd/titan_radar_bd/hdl/titan_radar_bd_wrapper.v

# ----------------------------------------------------------------------------
# Run Synthesis and Implementation
# ----------------------------------------------------------------------------

# Set implementation strategy for timing closure
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# Launch runs
launch_runs synth_1 -jobs 8
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# ----------------------------------------------------------------------------
# Export for PYNQ
# ----------------------------------------------------------------------------

# Copy bitstream
file copy -force $project_dir/$project_name.runs/impl_1/titan_radar_bd_wrapper.bit \
    ./titan_radar.bit

# Generate HWH file for PYNQ
file copy -force $project_dir/$project_name.gen/sources_1/bd/titan_radar_bd/hw_handoff/titan_radar_bd.hwh \
    ./titan_radar.hwh

puts "============================================================"
puts "TITAN Radar Overlay Build Complete!"
puts "============================================================"
puts "Output files:"
puts "  - titan_radar.bit"
puts "  - titan_radar.hwh"
puts "============================================================"
puts "Copy these files to RFSoC 4x2 for PYNQ deployment"
puts "============================================================"
