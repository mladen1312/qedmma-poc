# ============================================================================
# TITAN Radar - Vitis HLS Build Script
# Synthesizes all IP cores for RFSoC 4x2 overlay
#
# Author: Dr. Mladen Mešter
# Copyright (c) 2026 - All Rights Reserved
#
# Usage: vitis_hls -f run_hls.tcl
# ============================================================================

# Project settings
set PROJECT_NAME "titan_radar_ip"
set TOP_DIR [pwd]
set PART "xczu48dr-ffvg1517-2-e"
set CLOCK_PERIOD 4.0  ;# 250 MHz

# IP Core definitions
set IP_CORES {
    {waveform_generator  titan_waveform_generator  waveform_generator.cpp}
    {beamformer          titan_beamformer          beamformer.cpp}
    {correlator          titan_correlator          zero_dsp_correlator.cpp}
    {doppler_fft         titan_doppler_fft         doppler_fft.cpp}
    {cfar_detector       titan_cfar_detector       cfar_detector.cpp}
    {track_processor     titan_track_processor     track_processor.cpp}
}

# ============================================================================
# Build each IP core
# ============================================================================

proc build_ip {name top_func source_file} {
    global TOP_DIR PART CLOCK_PERIOD
    
    puts "=============================================="
    puts "Building IP: $name"
    puts "Top function: $top_func"
    puts "Source: $source_file"
    puts "=============================================="
    
    # Create project
    cd $TOP_DIR
    open_project -reset ${name}_hls
    
    # Set top function
    set_top $top_func
    
    # Add source files
    add_files $source_file -cflags "-I${TOP_DIR}/common"
    add_files common/types.hpp
    
    # Add testbench if exists
    if {[file exists "tb_${name}.cpp"]} {
        add_files -tb tb_${name}.cpp
    }
    
    # Create solution
    open_solution -reset "solution1" -flow_target vivado
    
    # Set part and clock
    set_part $PART
    create_clock -period $CLOCK_PERIOD -name default
    
    # Set config options
    config_interface -m_axi_alignment_byte_size 64
    config_interface -m_axi_max_widen_bitwidth 512
    config_compile -pipeline_loops 64
    config_schedule -effort high
    
    # Run synthesis
    csynth_design
    
    # Export IP
    export_design -rtl verilog -format ip_catalog \
        -description "TITAN Radar - ${name}" \
        -vendor "titan.radar" \
        -library "hls" \
        -version "1.0" \
        -display_name "TITAN ${name}"
    
    # Close project
    close_project
    
    puts "IP $name built successfully!"
    puts ""
}

# ============================================================================
# Main build loop
# ============================================================================

puts "=============================================="
puts "TITAN Radar - HLS IP Build System"
puts "Part: $PART"
puts "Clock: ${CLOCK_PERIOD}ns (250 MHz)"
puts "=============================================="
puts ""

# Build all IP cores
foreach ip $IP_CORES {
    set name [lindex $ip 0]
    set top_func [lindex $ip 1]
    set source_file [lindex $ip 2]
    
    if {[file exists $source_file]} {
        build_ip $name $top_func $source_file
    } else {
        puts "WARNING: Source file $source_file not found, skipping $name"
    }
}

# ============================================================================
# Generate summary
# ============================================================================

puts "=============================================="
puts "Build Complete!"
puts "=============================================="
puts ""
puts "IP Cores generated:"

foreach ip $IP_CORES {
    set name [lindex $ip 0]
    set ip_dir "${name}_hls/solution1/impl/ip"
    if {[file exists $ip_dir]} {
        puts "  ✓ $name -> $ip_dir"
    }
}

puts ""
puts "Next steps:"
puts "  1. Copy IP cores to Vivado IP repository"
puts "  2. Run: vivado -mode batch -source build_titan_overlay.tcl"
puts ""

exit
