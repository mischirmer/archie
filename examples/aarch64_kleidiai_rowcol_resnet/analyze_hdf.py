import h5py
import numpy as np
import argparse
import struct
import os
import csv
import sys

def find_address_in_objdump(fault_address, objdump_content):
    """
    Find the line in objdump.txt that corresponds to the given fault_address.
    Returns the matching line or None if not found.
    """
    if fault_address is None:
        return None
    
    # Convert fault_address to different hex string formats for searching
    addr_hex_lower = f"{fault_address:x}"          # e.g., "40002768"
    addr_hex_upper = f"{fault_address:X}"          # e.g., "40002768" 
    addr_hex_8char = f"{fault_address:08x}"        # e.g., "40002768"
    addr_hex_8char_upper = f"{fault_address:08X}"  # e.g., "40002768"
    
    # Search for lines containing this address
    for line in objdump_content:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Look for address at the beginning of the line (typical objdump format)
        # Format is usually: "address:" or "  address:"
        if ':' in line_stripped:
            # Extract the address part (before the first colon)
            addr_part = line_stripped.split(':')[0].strip()
            
            # Check if it matches our address in any format
            if (addr_part.lower() == addr_hex_lower.lower() or 
                addr_part.lower() == addr_hex_8char.lower() or
                addr_part == addr_hex_upper or 
                addr_part == addr_hex_8char_upper):
                return line
        
        # Also check for other possible formats where address might appear anywhere in the line
        line_lower = line_stripped.lower()
        if (addr_hex_lower in line_lower or 
            addr_hex_8char in line_lower):
            return line
    
    return None

def load_objdump_file(objdump_path="src/objdump.txt"):
    """
    Load the objdump.txt file and return its content as a list of lines.
    Returns None if file doesn't exist or can't be read.
    """
    if not os.path.exists(objdump_path):
        print(f"Warning: {objdump_path} not found")
        return None
    
    try:
        with open(objdump_path, 'r') as f:
            return f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {objdump_path}: {e}")
        return None

def compare_all_experiments(target_type, fault_type, analysis_type='abft', show_line=False, verbose=False):
    # Determine which HDF5 file to use based on analysis type
    if target_type == 'instructions' and fault_type == 'transient':
        hdf5_file = 'output/instructions/transient.hdf5'
        output_dir = 'output/instructions'
        analysis_name = 'instructions_transient'
    elif target_type == 'instructions' and fault_type == 'permanent':
        hdf5_file = 'output/instructions/permanent.hdf5'
        output_dir = 'output/instructions'
        analysis_name = 'instructions_permanent'
    elif target_type == 'fmla' and fault_type == 'transient':
        hdf5_file = 'output/instructions_fp/transient.hdf5'
        output_dir = 'output/instructions_fp'
        analysis_name = 'instructions_fp_transient'
    elif target_type == 'fmla' and fault_type == 'permanent':
        hdf5_file = 'output/instructions_fp/permanent.hdf5'
        output_dir = 'output/instructions_fp'
        analysis_name = 'instructions_fp_permanent'
    elif target_type == 'registers' and fault_type == 'transient':
        hdf5_file = 'output/registers/transient.hdf5'
        output_dir = 'output/registers'
        analysis_name = 'registers_transient'
    elif target_type == 'registers' and fault_type == 'permanent':
        hdf5_file = 'output/registers/permanent.hdf5'
        output_dir = 'output/registers'
        analysis_name = 'registers_permanent'
    else:
        raise ValueError(f"Invalid analysis type: target_type={target_type}, fault_type={fault_type}")
    
    # Create output filename for analysis results
    results_filename = f"{output_dir}/{analysis_name}_{analysis_type}_analysis_results.txt"
    
    print(f"Opening HDF5 file: {hdf5_file}")
    print(f"Analysis type: {target_type} + {fault_type} ({analysis_type})")

    # Always load objdump file for WARNING and FAILED cases
    objdump_content = load_objdump_file()
    if objdump_content:
        print(f"Loaded objdump.txt with {len(objdump_content)} lines")
    else:
        print("Warning: Could not load objdump.txt, objdump lines will not be shown")
    
    # Open the HDF5 file
    with h5py.File(hdf5_file, 'r') as f:
        # ABFT trigger location - 24 bytes at 0x40024e80
        # Byte 0: ABFT Hari triggered flag
        # Byte 1: ABFT Huang triggered flag
        # Byte 2: Hash triggered flag  
        # Bytes 4-7: Data to compare for ABFT trigger
        # Bytes 8-15: Data to compare for hash trigger
        abft_trigger_location = 'location_4014a6f0_24_1'

        # Extract trigger address from the location string
        # Format: 'location_ADDRESS_SIZE_COUNT'
        trigger_address = None
        if abft_trigger_location.startswith('location_'):
            parts = abft_trigger_location.split('_')
            if len(parts) >= 2:
                try:
                    trigger_address = int(parts[1], 16)  # Convert hex string to int
                except ValueError:
                    trigger_address = None
        
        print("Reading golden run ABFT trigger data...")
        
        # Read golden run ABFT trigger data
        golden_abft_path = f'/Goldenrun/memdumps/{abft_trigger_location}'
        try:
            golden_abft_data = f[golden_abft_path][:]
            print(f"  {abft_trigger_location}: shape {golden_abft_data.shape}, dtype {golden_abft_data.dtype}")
            
            # Handle both 1D and 2D arrays
            if len(golden_abft_data.shape) == 2:
                golden_abft_flat = golden_abft_data.flatten()
            else:
                golden_abft_flat = golden_abft_data
            
            if len(golden_abft_flat) >= 16:
                golden_abft_bytes = golden_abft_flat[4:8]   # Bytes 4-7 for ABFT trigger
                golden_hash_bytes = golden_abft_flat[8:16]  # Bytes 8-15 for hash trigger
                print(f"  Golden ABFT bytes (4-7): {[hex(b) for b in golden_abft_bytes]}")
                print(f"  Golden hash bytes (8-15): {[hex(b) for b in golden_hash_bytes]}")
            else:
                print(f"  ERROR: Golden ABFT data too short ({len(golden_abft_flat)} bytes, need at least 16)")
                return
        except KeyError:
            print(f"  ERROR: {abft_trigger_location} not found in golden run")
            return
        
        print("\nChecking all fault experiments...\n")
        
        # Get all experiments in the fault group
        fault_group = f['/fault']
        experiment_names = [name for name in fault_group.keys() if name.startswith('experiment')]
        experiment_names.sort(key=lambda x: int(x.replace('experiment', '')))
        
        # Analysis results for Hari detection
        correct_hari_trigger = []     # Hari triggered when bytes 4-7 differ
        incorrect_hari_trigger = []   # Hari triggered when bytes 4-7 are same
        correct_hari_no_trigger = []  # Hari not triggered when bytes 4-7 are same
        incorrect_hari_no_trigger = [] # Hari not triggered when bytes 4-7 differ
        
        # Analysis results for Huang detection  
        correct_huang_trigger = []     # Huang triggered when bytes 4-7 differ
        incorrect_huang_trigger = []   # Huang triggered when bytes 4-7 are same
        correct_huang_no_trigger = []  # Huang not triggered when bytes 4-7 are same
        incorrect_huang_no_trigger = [] # Huang not triggered when bytes 4-7 differ
        
        correct_hash_trigger = []     # Hash triggered when bytes 8-15 differ
        incorrect_hash_trigger = []   # Hash triggered when bytes 8-15 are same
        correct_hash_no_trigger = []  # Hash not triggered when bytes 8-15 are same
        incorrect_hash_no_trigger = [] # Hash not triggered when bytes 8-15 differ
        
        # Special ABFT classifications
        abft_timeouts = []            # ABFT experiments with timeouts (zero hash bytes or invalid flags)
        
        abft_data_errors = []         # Could not read ABFT data
        
        # Store max_diff values for failed ABFT cases
        failed_hari_max_diff = []     # List of (experiment_name, max_diff_value, fault_address) tuples for Hari
        failed_huang_max_diff = []    # List of (experiment_name, max_diff_value, fault_address) tuples for Huang
        
        for exp_name in experiment_names:
            abft_path = f'/fault/{exp_name}/memdumps/{abft_trigger_location}'
            
            try:
                # Read ABFT trigger data for this experiment
                exp_abft_data = f[abft_path][:]
                
                # Handle both 1D and 2D arrays
                if len(exp_abft_data.shape) == 2:
                    exp_abft_flat = exp_abft_data.flatten()
                else:
                    exp_abft_flat = exp_abft_data
                
                if len(exp_abft_flat) < 24:
                    print(f"ERROR: {exp_name} - ABFT data too short ({len(exp_abft_flat)} bytes, need at least 24)")
                    abft_data_errors.append(exp_name)
                    continue
                
                # Extract relevant bytes
                hari_triggered = exp_abft_flat[0]      # Byte 0: ABFT Hari triggered flag
                huang_triggered = exp_abft_flat[1]     # Byte 1: ABFT Huang triggered flag
                hash_triggered = exp_abft_flat[2]      # Byte 2: Hash triggered flag
                abft_bytes = exp_abft_flat[4:8]        # Bytes 4-7 for ABFT trigger
                hash_bytes = exp_abft_flat[8:16]       # Bytes 8-15 for hash trigger
                
                # Extract max_diff (float32) from bytes 20-23
                max_diff_bytes = exp_abft_flat[20:24]
                max_diff = struct.unpack('<f', bytes(max_diff_bytes))[0]  # Little-endian float32
                
                # Special ABFT classifications - Check for timeouts
                is_timeout = False
                
                # Check if ABFT hash bytes (4-7) are all 0
                if np.all(abft_bytes == 0):
                    is_timeout = True
                    """print(f"TIMEOUT: {exp_name} - Hash bytes (4-7) are all 0")
                    print(f"  ABFT bytes: {[hex(b) for b in abft_bytes]}")
                    print(f"  Hari flag: {hari_triggered}, Huang flag: {huang_triggered}, Hash flag: {hash_triggered}")"""
                
                # Check if bytes 0,1,2 are invalid (not 0 or 1)
                byte_0 = exp_abft_flat[0]
                byte_1 = exp_abft_flat[1]
                byte_2 = exp_abft_flat[2]
                if (byte_0 not in [0, 1]) or (byte_1 not in [0, 1]) or (byte_2 not in [0, 1]):
                    is_timeout = True
                    """print(f"TIMEOUT: {exp_name} - Invalid flag bytes")
                    print(f"  Byte 0 (Hari): {byte_0} (should be 0 or 1)")
                    print(f"  Byte 1 (Huang): {byte_1} (should be 0 or 1)")
                    print(f"  Byte 2 (Hash): {byte_2} (should be 0 or 1)")"""
                
                # Add to timeout list if either condition is met
                if is_timeout:
                    abft_timeouts.append(exp_name)
                    # print()
                    continue  # Skip normal analysis for timeout experiments
                
                # Check if bytes differ from golden run
                abft_bytes_differ = not np.array_equal(abft_bytes, golden_abft_bytes)
                hash_bytes_differ = not np.array_equal(hash_bytes, golden_hash_bytes)
                
                # Analyze ABFT trigger correctness (skip if hash bytes differ or if not analyzing ABFT)
                if (analysis_type == 'abft') and (not hash_bytes_differ):  # Only analyze ABFT when hash bytes are the same
                    if abft_bytes_differ:
                        # Bytes 4-7 are different, both Hari and Huang should be triggered (= 1)
                        
                        # Analyze Hari detection
                        if hari_triggered == 1:
                            correct_hari_trigger.append(exp_name)
                            """print(f"CORRECT HARI: {exp_name}")
                            print(f"  Bytes 4-7 differ, Hari correctly triggered")
                            print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                            print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                            print(f"  Hari flag: {hari_triggered}")"""
                        else:
                            # Read fault_address and trigger_address for this experiment
                            fault_address = None
                            trigger_address_from_hdf5 = None
                            faults_path = f'/fault/{exp_name}/faults'
                            try:
                                faults_data = f[faults_path][:]
                                if len(faults_data) > 0:
                                    # faults is a structured array with named fields
                                    fault_address = faults_data[0]['fault_address']  # Get fault_address field
                                    if 'trigger_address' in faults_data.dtype.names:
                                        trigger_address_from_hdf5 = faults_data[0]['trigger_address']  # Get trigger_address field
                            except KeyError:
                                fault_address = None
                                trigger_address_from_hdf5 = None
                            
                            incorrect_hari_no_trigger.append(exp_name)
                            failed_hari_max_diff.append((exp_name, max_diff, fault_address))
                            if verbose:
                                print(f"ERROR HARI: {exp_name} - CRITICAL!")
                                print(f"  Bytes 4-7 differ but Hari not triggered!")
                                print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                                print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                                print(f"  Hari flag: {hari_triggered} (should be 1)")
                                print(f"  L1 Norm: {max_diff}")
                                # For register faults, fault_address is directly the trigger_address
                                if target_type == 'registers':
                                    display_trigger_address = fault_address
                                else:
                                    # Use trigger_address from HDF5 if available, otherwise use extracted one
                                    display_trigger_address = trigger_address_from_hdf5 if trigger_address_from_hdf5 is not None else trigger_address
                                
                                if display_trigger_address is not None:
                                    print(f"  trigger_address: 0x{display_trigger_address:x}")
                                if fault_address is not None:
                                    print(f"  fault_address: 0x{fault_address:x}")
                                    # Show corresponding line from objdump for FAILED cases
                                    # For register faults, use trigger_address for objdump lookup
                                    # For instruction faults, use fault_address for objdump lookup
                                    if objdump_content:
                                        lookup_address = display_trigger_address if target_type == 'registers' else fault_address
                                        if lookup_address is not None:
                                            objdump_line = find_address_in_objdump(lookup_address, objdump_content)
                                            if objdump_line:
                                                print(f"  objdump_line: {objdump_line.strip()}")
                                            else:
                                                print(f"  objdump_line: not found for 0x{lookup_address:x}")
                                        else:
                                            print(f"  objdump_line: lookup address not available")
                                else:
                                    print(f"  fault_address: not found")
                                print()
                        
                        # Analyze Huang detection  
                        if huang_triggered == 1:
                            correct_huang_trigger.append(exp_name)
                            """print(f"CORRECT HUANG: {exp_name}")
                            print(f"  Bytes 4-7 differ, Huang correctly triggered")
                            print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                            print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                            print(f"  Huang flag: {huang_triggered}")"""
                        else:
                            # Read fault_address and trigger_address for this experiment (if not already read)
                            if 'fault_address' not in locals():
                                fault_address = None
                                trigger_address_from_hdf5 = None
                                faults_path = f'/fault/{exp_name}/faults'
                                try:
                                    faults_data = f[faults_path][:]
                                    if len(faults_data) > 0:
                                        # faults is a structured array with named fields
                                        fault_address = faults_data[0]['fault_address']  # Get fault_address field
                                        if 'trigger_address' in faults_data.dtype.names:
                                            trigger_address_from_hdf5 = faults_data[0]['trigger_address']  # Get trigger_address field
                                except KeyError:
                                    fault_address = None
                                    trigger_address_from_hdf5 = None
                            
                            incorrect_huang_no_trigger.append(exp_name)
                            failed_huang_max_diff.append((exp_name, max_diff, fault_address))
                            if verbose:
                                print(f"ERROR HUANG: {exp_name} - CRITICAL!")
                                print(f"  Bytes 4-7 differ but Huang not triggered!")
                                print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                                print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                                print(f"  Huang flag: {huang_triggered} (should be 1)")
                                print(f"  L1 Norm: {max_diff}")
                                # For register faults, fault_address is directly the trigger_address
                                if target_type == 'registers':
                                    display_trigger_address = fault_address
                                else:
                                    # Use trigger_address from HDF5 if available, otherwise use extracted one
                                    display_trigger_address = trigger_address_from_hdf5 if trigger_address_from_hdf5 is not None else trigger_address
                                
                                if display_trigger_address is not None:
                                    print(f"  trigger_address: 0x{display_trigger_address:x}")
                                if fault_address is not None:
                                    print(f"  fault_address: 0x{fault_address:x}")
                                    # Show corresponding line from objdump for FAILED cases
                                    if objdump_content:
                                        objdump_line = find_address_in_objdump(fault_address, objdump_content)
                                        if objdump_line:
                                            print(f"  objdump_line: {objdump_line.strip()}")
                                        else:
                                            print(f"  objdump_line: not found for 0x{fault_address:x}")
                                else:
                                    print(f"  fault_address: not found")
                                print()
                    else:
                        # Bytes 4-7 are same, both Hari and Huang should not be triggered (≠ 1)
                        
                        # Analyze Hari detection
                        if hari_triggered != 1:
                            correct_hari_no_trigger.append(exp_name)
                            """print(f"CORRECT HARI: {exp_name}")
                            print(f"  Bytes 4-7 same as golden, Hari correctly not triggered")
                            print(f"  Hari flag: {hari_triggered}")"""
                        else:
                            # Read fault_address and trigger_address for this experiment
                            fault_address = None
                            trigger_address_from_hdf5 = None
                            faults_path = f'/fault/{exp_name}/faults'
                            try:
                                faults_data = f[faults_path][:]
                                if len(faults_data) > 0:
                                    # faults is a structured array with named fields
                                    fault_address = faults_data[0]['fault_address']  # Get fault_address field
                                    if 'trigger_address' in faults_data.dtype.names:
                                        trigger_address_from_hdf5 = faults_data[0]['trigger_address']  # Get trigger_address field
                            except KeyError:
                                fault_address = None
                                trigger_address_from_hdf5 = None
                            
                            incorrect_hari_trigger.append(exp_name)
                            failed_hari_max_diff.append((exp_name, max_diff, fault_address))
                            if verbose:
                                print(f"WARNING HARI: {exp_name}")
                                print(f"  Bytes 4-7 same as golden but Hari triggered!")
                                print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                                print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                                print(f"  Hari flag: {hari_triggered} (should not be 1)")
                                print(f"  L1 Norm: {max_diff}")
                                # For register faults, fault_address is directly the trigger_address
                                if target_type == 'registers':
                                    display_trigger_address = fault_address
                                else:
                                    # Use trigger_address from HDF5 if available, otherwise use extracted one
                                    display_trigger_address = trigger_address_from_hdf5 if trigger_address_from_hdf5 is not None else trigger_address
                                
                                if display_trigger_address is not None:
                                    print(f"  trigger_address: 0x{display_trigger_address:x}")
                                if fault_address is not None:
                                    print(f"  fault_address: 0x{fault_address:x}")
                                    # Show corresponding line from objdump for WARNING cases
                                    if objdump_content:
                                        objdump_line = find_address_in_objdump(fault_address, objdump_content)
                                        if objdump_line:
                                            print(f"  objdump_line: {objdump_line.strip()}")
                                        else:
                                            print(f"  objdump_line: not found for 0x{fault_address:x}")
                                else:
                                    print(f"  fault_address: not found")
                                print()
                        
                        # Analyze Huang detection
                        if huang_triggered != 1:
                            correct_huang_no_trigger.append(exp_name)
                            """print(f"CORRECT HUANG: {exp_name}")
                            print(f"  Bytes 4-7 same as golden, Huang correctly not triggered")
                            print(f"  Huang flag: {huang_triggered}")"""
                        else:
                            # Read fault_address and trigger_address for this experiment (if not already read)
                            if 'fault_address' not in locals():
                                fault_address = None
                                trigger_address_from_hdf5 = None
                                faults_path = f'/fault/{exp_name}/faults'
                                try:
                                    faults_data = f[faults_path][:]
                                    if len(faults_data) > 0:
                                        # faults is a structured array with named fields
                                        fault_address = faults_data[0]['fault_address']  # Get fault_address field
                                        if 'trigger_address' in faults_data.dtype.names:
                                            trigger_address_from_hdf5 = faults_data[0]['trigger_address']  # Get trigger_address field
                                except KeyError:
                                    fault_address = None
                                    trigger_address_from_hdf5 = None
                            
                            incorrect_huang_trigger.append(exp_name)
                            failed_huang_max_diff.append((exp_name, max_diff, fault_address))
                            if verbose:
                                print(f"WARNING HUANG: {exp_name}")
                                print(f"  Bytes 4-7 same as golden but Huang triggered!")
                                print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                                print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                                print(f"  Huang flag: {huang_triggered} (should not be 1)")
                                print(f"  L1 Norm: {max_diff}")
                                # For register faults, fault_address is directly the trigger_address
                                if target_type == 'registers':
                                    display_trigger_address = fault_address
                                else:
                                    # Use trigger_address from HDF5 if available, otherwise use extracted one
                                    display_trigger_address = trigger_address_from_hdf5 if trigger_address_from_hdf5 is not None else trigger_address
                                
                                if display_trigger_address is not None:
                                    print(f"  trigger_address: 0x{display_trigger_address:x}")
                                if fault_address is not None:
                                    print(f"  fault_address: 0x{fault_address:x}")
                                    # Show corresponding line from objdump for WARNING cases
                                    if objdump_content:
                                        objdump_line = find_address_in_objdump(fault_address, objdump_content)
                                        if objdump_line:
                                            print(f"  objdump_line: {objdump_line.strip()}")
                                        else:
                                            print(f"  objdump_line: not found for 0x{fault_address:x}")
                                else:
                                    print(f"  fault_address: not found")
                                print()
                else:
                    if analysis_type == 'abft':
                        if verbose:
                            print(f"SKIPPED ABFT: {exp_name} - Hash bytes differ, ignoring Hari and Huang analysis")
                
                # Analyze Hash trigger correctness (only if analyzing hash)
                if analysis_type == 'hash':
                    if hash_bytes_differ:
                        # Bytes 8-15 are different, Hash should be triggered (= 1)
                        if hash_triggered == 1:
                            correct_hash_trigger.append(exp_name)
                            """print(f"CORRECT HASH: {exp_name}")
                            print(f"  Bytes 8-15 differ, Hash correctly triggered")
                            print(f"  Golden Hash:     {[hex(b) for b in golden_hash_bytes]}")
                            print(f"  Experiment Hash: {[hex(b) for b in hash_bytes]}")
                            print(f"  Hash flag: {hash_triggered}")"""
                        else:
                            incorrect_hash_no_trigger.append(exp_name)
                            if verbose:
                                print(f"ERROR HASH: {exp_name} - CRITICAL!")
                                print(f"  Bytes 8-15 differ but Hash not triggered!")
                                print(f"  Golden Hash:     {[hex(b) for b in golden_hash_bytes]}")
                                print(f"  Experiment Hash: {[hex(b) for b in hash_bytes]}")
                                print(f"  Hash flag: {hash_triggered} (should be 1)")
                                print()
                    else:
                        # Bytes 8-15 are same, Hash should not be triggered (≠ 1)
                        if hash_triggered != 1:
                            correct_hash_no_trigger.append(exp_name)
                            """print(f"CORRECT HASH: {exp_name}")
                            print(f"  Bytes 8-15 same as golden, Hash correctly not triggered")
                            print(f"  Hash flag: {hash_triggered}")"""
                        else:
                            incorrect_hash_trigger.append(exp_name)
                            if verbose:
                                print(f"WARNING HASH: {exp_name}")
                                print(f"  Bytes 8-15 same as golden but Hash triggered!")
                                print(f"  Golden Hash:     {[hex(b) for b in golden_hash_bytes]}")
                                print(f"  Experiment Hash: {[hex(b) for b in hash_bytes]}")
                                print(f"  Hash flag: {hash_triggered} (should not be 1)")
                                print()

                    
            except KeyError as e:
                if verbose:
                    print(f"SKIPPED: {exp_name} - ABFT trigger data not found: {e}\n")
                abft_data_errors.append(exp_name)
        
        # Calculate metrics
        total_experiments = len(experiment_names)
        
        # Hari metrics
        correct_hari_trig = len(correct_hari_trigger)
        incorrect_hari_trig = len(incorrect_hari_trigger)
        correct_hari_no_trig = len(correct_hari_no_trigger)
        incorrect_hari_no_trig = len(incorrect_hari_no_trigger)
        
        # Huang metrics
        correct_huang_trig = len(correct_huang_trigger)
        incorrect_huang_trig = len(incorrect_huang_trigger)
        correct_huang_no_trig = len(correct_huang_no_trigger)
        incorrect_huang_no_trig = len(incorrect_huang_no_trigger)
        
        # Hash metrics
        correct_hash_trig = len(correct_hash_trigger)
        incorrect_hash_trig = len(incorrect_hash_trigger)
        correct_hash_no_trig = len(correct_hash_no_trigger)
        incorrect_hash_no_trig = len(incorrect_hash_no_trigger)
        
        data_errors = len(abft_data_errors)
        timeouts = len(abft_timeouts)
        
        print(f"=== HARI & HUANG ABFT TRIGGER ANALYSIS SUMMARY ===")
        print(f"Analysis type: {target_type} + {fault_type} ({analysis_type})")
        print(f"Total experiments: {total_experiments}")
        print(f"Data errors: {data_errors}")
        print(f"Timeouts: {timeouts}")
        print()
        
        if analysis_type == 'abft':
            # Calculate total for percentage calculations
            hari_total = correct_hari_trig + incorrect_hari_trig + correct_hari_no_trig + incorrect_hari_no_trig
            huang_total = correct_huang_trig + incorrect_huang_trig + correct_huang_no_trig + incorrect_huang_no_trig
            
            print(f"=== HARI ABFT TRIGGER (Bytes 4-7) ===")
            print(f"Correct Hari triggers:     {correct_hari_trig} - Hari triggered when bytes 4-7 differ ({correct_hari_trig/hari_total*100:.2f}%)")
            print(f"Incorrect Hari triggers:   {incorrect_hari_trig} - Hari triggered when bytes 4-7 same ({incorrect_hari_trig/hari_total*100:.2f}%)")
            print(f"Correct Hari no-triggers:  {correct_hari_no_trig} - Hari not triggered when bytes 4-7 same ({correct_hari_no_trig/hari_total*100:.2f}%)")
            print(f"Incorrect Hari no-triggers: {incorrect_hari_no_trig} - Hari not triggered when bytes 4-7 differ ({incorrect_hari_no_trig/hari_total*100:.2f}%)")
            print()
            
            print(f"=== HUANG ABFT TRIGGER (Bytes 4-7) ===")
            print(f"Correct Huang triggers:     {correct_huang_trig} - Huang triggered when bytes 4-7 differ ({correct_huang_trig/huang_total*100:.2f}%)")
            print(f"Incorrect Huang triggers:   {incorrect_huang_trig} - Huang triggered when bytes 4-7 same ({incorrect_huang_trig/huang_total*100:.2f}%)")
            print(f"Correct Huang no-triggers:  {correct_huang_no_trig} - Huang not triggered when bytes 4-7 same ({correct_huang_no_trig/huang_total*100:.2f}%)")
            print(f"Incorrect Huang no-triggers: {incorrect_huang_no_trig} - Huang not triggered when bytes 4-7 differ ({incorrect_huang_no_trig/huang_total*100:.2f}%)")
        
        if analysis_type == 'hash':
            # Calculate total for percentage calculations
            hash_total = correct_hash_trig + incorrect_hash_trig + correct_hash_no_trig + incorrect_hash_no_trig
            
            print(f"=== HASH TRIGGER (Bytes 8-15) ===")
            print(f"Correct Hash triggers:     {correct_hash_trig} - Hash triggered when bytes 8-15 differ ({correct_hash_trig/hash_total*100:.2f}%)")
            print(f"Incorrect Hash triggers:   {incorrect_hash_trig} - Hash triggered when bytes 8-15 same ({incorrect_hash_trig/hash_total*100:.2f}%)")
            print(f"Correct Hash no-triggers:  {correct_hash_no_trig} - Hash not triggered when bytes 8-15 same ({correct_hash_no_trig/hash_total*100:.2f}%)")
            print(f"Incorrect Hash no-triggers: {incorrect_hash_no_trig} - Hash not triggered when bytes 8-15 differ ({incorrect_hash_no_trig/hash_total*100:.2f}%)")
        
        # Calculate performance metrics
        valid_experiments = total_experiments - data_errors
        if valid_experiments > 0:
            print(f"\n=== PERFORMANCE METRICS ===")
            if analysis_type == 'abft':
                hari_accuracy = (correct_hari_trig + correct_hari_no_trig) / valid_experiments
                huang_accuracy = (correct_huang_trig + correct_huang_no_trig) / valid_experiments
                print(f"Hari Overall Accuracy: {hari_accuracy:.4f} ({hari_accuracy*100:.2f}%)")
                print(f"Huang Overall Accuracy: {huang_accuracy:.4f} ({huang_accuracy*100:.2f}%)")
            if analysis_type == 'hash':
                hash_accuracy = (correct_hash_trig + correct_hash_no_trig) / valid_experiments
                print(f"Hash Overall Accuracy: {hash_accuracy:.4f} ({hash_accuracy*100:.2f}%)")
        
        # Calculate ABFT detailed metrics using confusion matrix approach (only if analyzing ABFT)
        if analysis_type == 'abft':
            # Hari metrics
            # True Positives (TP): correctly detected changes (correct_hari_trig)
            # False Positives (FP): incorrectly detected changes (incorrect_hari_trig)
            # True Negatives (TN): correctly detected no changes (correct_hari_no_trig)
            # False Negatives (FN): missed changes (incorrect_hari_no_trig)
            
            hari_tp = correct_hari_trig
            hari_fp = incorrect_hari_trig
            hari_tn = correct_hari_no_trig
            hari_fn = incorrect_hari_no_trig
            
            print(f"\n=== HARI DETAILED METRICS ===")
            print(f"True Positives (TP):  {hari_tp} - Correctly detected changes")
            print(f"False Positives (FP): {hari_fp} - Incorrectly detected changes")
            print(f"True Negatives (TN):  {hari_tn} - Correctly detected no changes")
            print(f"False Negatives (FN): {hari_fn} - Missed changes")
            
            # Calculate Hari metrics
            if (hari_tp + hari_fp) > 0:
                hari_precision = hari_tp / (hari_tp + hari_fp)
                print(f"Hari Precision: {hari_precision:.4f} ({hari_precision*100:.2f}%)")
            else:
                print(f"Hari Precision: N/A (no positive predictions)")
            
            if (hari_tp + hari_fn) > 0:
                hari_recall = hari_tp / (hari_tp + hari_fn)
                print(f"Hari Recall (Sensitivity): {hari_recall:.4f} ({hari_recall*100:.2f}%)")
            else:
                print(f"Hari Recall: N/A (no actual positives)")
            
            if (hari_tp + hari_fp + hari_tn + hari_fn) > 0:
                hari_accuracy_detailed = (hari_tp + hari_tn) / (hari_tp + hari_fp + hari_tn + hari_fn)
                print(f"Hari Accuracy: {hari_accuracy_detailed:.4f} ({hari_accuracy_detailed*100:.2f}%)")
            else:
                print(f"Hari Accuracy: N/A (no valid experiments)")
            
            # Calculate Hari F1 Score
            if (hari_tp + hari_fp) > 0 and (hari_tp + hari_fn) > 0:
                hari_precision_calc = hari_tp / (hari_tp + hari_fp)
                hari_recall_calc = hari_tp / (hari_tp + hari_fn)
                if (hari_precision_calc + hari_recall_calc) > 0:
                    hari_f1 = 2 * (hari_precision_calc * hari_recall_calc) / (hari_precision_calc + hari_recall_calc)
                    print(f"Hari F1 Score: {hari_f1:.4f} ({hari_f1*100:.2f}%)")
                else:
                    print(f"Hari F1 Score: 0.0000 (0.00%)")
            else:
                print(f"Hari F1 Score: N/A (cannot calculate)")
            
            # Huang metrics
            # True Positives (TP): correctly detected changes (correct_huang_trig)
            # False Positives (FP): incorrectly detected changes (incorrect_huang_trig)
            # True Negatives (TN): correctly detected no changes (correct_huang_no_trig)
            # False Negatives (FN): missed changes (incorrect_huang_no_trig)
            
            huang_tp = correct_huang_trig
            huang_fp = incorrect_huang_trig
            huang_tn = correct_huang_no_trig
            huang_fn = incorrect_huang_no_trig
            
            print(f"\n=== HUANG DETAILED METRICS ===")
            print(f"True Positives (TP):  {huang_tp} - Correctly detected changes")
            print(f"False Positives (FP): {huang_fp} - Incorrectly detected changes")
            print(f"True Negatives (TN):  {huang_tn} - Correctly detected no changes")
            print(f"False Negatives (FN): {huang_fn} - Missed changes")
            
            # Calculate Huang metrics
            if (huang_tp + huang_fp) > 0:
                huang_precision = huang_tp / (huang_tp + huang_fp)
                print(f"Huang Precision: {huang_precision:.4f} ({huang_precision*100:.2f}%)")
            else:
                print(f"Huang Precision: N/A (no positive predictions)")
            
            if (huang_tp + huang_fn) > 0:
                huang_recall = huang_tp / (huang_tp + huang_fn)
                print(f"Huang Recall (Sensitivity): {huang_recall:.4f} ({huang_recall*100:.2f}%)")
            else:
                print(f"Huang Recall: N/A (no actual positives)")
            
            if (huang_tp + huang_fp + huang_tn + huang_fn) > 0:
                huang_accuracy_detailed = (huang_tp + huang_tn) / (huang_tp + huang_fp + huang_tn + huang_fn)
                print(f"Huang Accuracy: {huang_accuracy_detailed:.4f} ({huang_accuracy_detailed*100:.2f}%)")
            else:
                print(f"Huang Accuracy: N/A (no valid experiments)")
            
            # Calculate Huang F1 Score
            if (huang_tp + huang_fp) > 0 and (huang_tp + huang_fn) > 0:
                huang_precision_calc = huang_tp / (huang_tp + huang_fp)
                huang_recall_calc = huang_tp / (huang_tp + huang_fn)
                if (huang_precision_calc + huang_recall_calc) > 0:
                    huang_f1 = 2 * (huang_precision_calc * huang_recall_calc) / (huang_precision_calc + huang_recall_calc)
                    print(f"Huang F1 Score: {huang_f1:.4f} ({huang_f1*100:.2f}%)")
                else:
                    print(f"Huang F1 Score: 0.0000 (0.00%)")
            else:
                print(f"Huang F1 Score: N/A (cannot calculate)")
            
            # Calculate Hari detection metrics
            hari_experiments_with_changes = correct_hari_trig + incorrect_hari_no_trig
            if hari_experiments_with_changes > 0:
                hari_detection_rate = correct_hari_trig / hari_experiments_with_changes
                hari_miss_rate = incorrect_hari_no_trig / hari_experiments_with_changes
                print(f"Hari Detection Rate (when bytes 4-7 differ): {hari_detection_rate:.4f} ({hari_detection_rate*100:.2f}%)")
                print(f"Hari Miss Rate (when bytes 4-7 differ): {hari_miss_rate:.4f} ({hari_miss_rate*100:.2f}%)")
            
            # Calculate Hari false positive metrics  
            hari_experiments_without_changes = correct_hari_no_trig + incorrect_hari_trig
            if hari_experiments_without_changes > 0:
                hari_false_positive_rate = incorrect_hari_trig / hari_experiments_without_changes
                hari_specificity = correct_hari_no_trig / hari_experiments_without_changes
                print(f"Hari False Positive Rate (when bytes 4-7 same): {hari_false_positive_rate:.4f} ({hari_false_positive_rate*100:.2f}%)")
                print(f"Hari Specificity (when bytes 4-7 same): {hari_specificity:.4f} ({hari_specificity*100:.2f}%)")
            
            # Calculate Huang detection metrics
            huang_experiments_with_changes = correct_huang_trig + incorrect_huang_no_trig
            if huang_experiments_with_changes > 0:
                huang_detection_rate = correct_huang_trig / huang_experiments_with_changes
                huang_miss_rate = incorrect_huang_no_trig / huang_experiments_with_changes
                print(f"Huang Detection Rate (when bytes 4-7 differ): {huang_detection_rate:.4f} ({huang_detection_rate*100:.2f}%)")
                print(f"Huang Miss Rate (when bytes 4-7 differ): {huang_miss_rate:.4f} ({huang_miss_rate*100:.2f}%)")
            
            # Calculate Huang false positive metrics  
            huang_experiments_without_changes = correct_huang_no_trig + incorrect_huang_trig
            if huang_experiments_without_changes > 0:
                huang_false_positive_rate = incorrect_huang_trig / huang_experiments_without_changes
                huang_specificity = correct_huang_no_trig / huang_experiments_without_changes
                print(f"Huang False Positive Rate (when bytes 4-7 same): {huang_false_positive_rate:.4f} ({huang_false_positive_rate*100:.2f}%)")
                print(f"Huang Specificity (when bytes 4-7 same): {huang_specificity:.4f} ({huang_specificity*100:.2f}%)")
        
        # Calculate hash detection metrics (only if analyzing hash)
        if analysis_type == 'hash':
            hash_experiments_with_changes = correct_hash_trig + incorrect_hash_no_trig
            if hash_experiments_with_changes > 0:
                hash_detection_rate = correct_hash_trig / hash_experiments_with_changes
                hash_miss_rate = incorrect_hash_no_trig / hash_experiments_with_changes
                print(f"Hash Detection Rate (when bytes 8-12 differ): {hash_detection_rate:.4f} ({hash_detection_rate*100:.2f}%)")
                print(f"Hash Miss Rate (when bytes 8-12 differ): {hash_miss_rate:.4f} ({hash_miss_rate*100:.2f}%)")
            
            hash_experiments_without_changes = correct_hash_no_trig + incorrect_hash_trig
            if hash_experiments_without_changes > 0:
                hash_false_positive_rate = incorrect_hash_trig / hash_experiments_without_changes
                hash_specificity = correct_hash_no_trig / hash_experiments_without_changes
                print(f"Hash False Positive Rate (when bytes 8-12 same): {hash_false_positive_rate:.4f} ({hash_false_positive_rate*100:.2f}%)")
                print(f"Hash Specificity (when bytes 8-12 same): {hash_specificity:.4f} ({hash_specificity*100:.2f}%)")
        
        # List critical cases (only if analyzing the respective type)
        if analysis_type == 'abft' and incorrect_hari_no_trigger:
            print(f"\n=== CRITICAL: MISSED HARI DETECTIONS ===")
            print("These experiments had different bytes 4-7 but Hari was not triggered:")
            print(f"Total count: {len(incorrect_hari_no_trigger)}")
            # Print first 10 as sample
            for i, exp in enumerate(incorrect_hari_no_trigger[:10]):
                print(f"  {exp}")
            if len(incorrect_hari_no_trigger) > 10:
                print(f"  ... and {len(incorrect_hari_no_trigger) - 10} more")
        
        if analysis_type == 'abft' and incorrect_huang_no_trigger:
            print(f"\n=== CRITICAL: MISSED HUANG DETECTIONS ===")
            print("These experiments had different bytes 4-7 but Huang was not triggered:")
            print(f"Total count: {len(incorrect_huang_no_trigger)}")
            # Print first 10 as sample
            for i, exp in enumerate(incorrect_huang_no_trigger[:10]):
                print(f"  {exp}")
            if len(incorrect_huang_no_trigger) > 10:
                print(f"  ... and {len(incorrect_huang_no_trigger) - 10} more")
        
        if analysis_type == 'hash' and incorrect_hash_no_trigger:
            print(f"\n=== CRITICAL: MISSED HASH DETECTIONS ===")
            print("These experiments had different bytes 8-12 but Hash was not triggered:")
            print(f"Total count: {len(incorrect_hash_no_trigger)}")
            # Print first 10 as sample
            for i, exp in enumerate(incorrect_hash_no_trigger[:10]):
                print(f"  {exp}")
            if len(incorrect_hash_no_trigger) > 10:
                print(f"  ... and {len(incorrect_hash_no_trigger) - 10} more")
        
        if analysis_type == 'abft' and incorrect_hari_trigger:
            print(f"\n=== FALSE HARI TRIGGERS ===")
            print("These experiments had same bytes 4-7 but Hari was triggered:")
            if verbose:
                for exp in incorrect_hari_trigger:
                    print(f"  {exp}")
            else:
                for i, exp in enumerate(incorrect_hari_trigger[:10]):
                    print(f"  {exp}")
                if len(incorrect_hari_trigger) > 10:
                    print(f"  ... and {len(incorrect_hari_trigger) - 10} more")
        
        if analysis_type == 'abft' and incorrect_huang_trigger:
            print(f"\n=== FALSE HUANG TRIGGERS ===")
            print("These experiments had same bytes 4-7 but Huang was triggered:")
            if verbose:
                for exp in incorrect_huang_trigger:
                    print(f"  {exp}")
            else:
                for i, exp in enumerate(incorrect_huang_trigger[:10]):
                    print(f"  {exp}")
                if len(incorrect_huang_trigger) > 10:
                    print(f"  ... and {len(incorrect_huang_trigger) - 10} more")
        
        if analysis_type == 'hash' and incorrect_hash_trigger:
            print(f"\n=== FALSE HASH TRIGGERS ===")
            print("These experiments had same bytes 8-12 but Hash was triggered:")
            for exp in incorrect_hash_trigger:
                print(f"  {exp}")
        
        if abft_data_errors:
            print(f"\n=== DATA ERRORS ===")
            print("These experiments could not be analyzed due to missing ABFT data:")
            for exp in abft_data_errors:
                print(f"  {exp}")
        
        # Export CSV files for each ABFT class
        if analysis_type == 'abft':
            print(f"\n=== EXPORTING CSV FILES ===")
            
            def export_class_to_csv(experiment_list, class_name, filename):
                """Export experiment data for a class to CSV file"""
                if not experiment_list:
                    print(f"No data to export for {class_name}")
                    return
                
                csv_data = []
                for exp_name in experiment_list:
                    faults_path = f'/fault/{exp_name}/faults'
                    try:
                        faults_data = f[faults_path][:]
                        if len(faults_data) > 0:
                            fault_address = faults_data[0]['fault_address']
                            
                            # For register analysis, customize the CSV output
                            if target_type == 'registers':
                                # Extract trigger address and bitmask for register analysis
                                trigger_address = None
                                bitmask = None
                                
                                # For register faults, fault_address is the register address (trigger_address)
                                trigger_address = fault_address
                                
                                # Try to get trigger_address from HDF5 if available
                                if 'trigger_address' in faults_data.dtype.names:
                                    trigger_address_from_hdf5 = faults_data[0]['trigger_address']
                                    if trigger_address_from_hdf5 is not None:
                                        trigger_address = trigger_address_from_hdf5
                                
                                # Try to get bitmask if available
                                if 'bitmask' in faults_data.dtype.names:
                                    bitmask = faults_data[0]['bitmask']
                                elif 'mask' in faults_data.dtype.names:
                                    bitmask = faults_data[0]['mask']
                                elif 'fault_mask' in faults_data.dtype.names:
                                    bitmask = faults_data[0]['fault_mask']
                                
                                csv_data.append({
                                    'experiment': exp_name,
                                    'fault_address_register': f"0x{fault_address:x}" if fault_address is not None else "not found",
                                    'trigger_address': f"0x{trigger_address:x}" if trigger_address is not None else "not found",
                                    'bitmask': f"0x{bitmask:x}" if bitmask is not None else "not found"
                                })
                            else:
                                # For instruction analysis, keep the original format
                                objdump_line = ""
                                instruction = ""
                                
                                if objdump_content and fault_address is not None:
                                    objdump_result = find_address_in_objdump(fault_address, objdump_content)
                                    if objdump_result:
                                        objdump_line = objdump_result.strip()
                                        # Extract instruction part
                                        line_parts = objdump_line.split('\t')
                                        if len(line_parts) >= 3:
                                            instruction = line_parts[2].strip()
                                
                                csv_data.append({
                                    'experiment': exp_name,
                                    'fault_address': f"0x{fault_address:x}" if fault_address is not None else "not found",
                                    'fault_address_dec': fault_address if fault_address is not None else "",
                                    'instruction': instruction,
                                    'objdump_line': objdump_line
                                })
                        else:
                            if target_type == 'registers':
                                csv_data.append({
                                    'experiment': exp_name,
                                    'fault_address_register': "no fault data",
                                    'trigger_address': "no fault data",
                                    'bitmask': "no fault data"
                                })
                            else:
                                csv_data.append({
                                    'experiment': exp_name,
                                    'fault_address': "no fault data",
                                    'fault_address_dec': "",
                                    'instruction': "",
                                    'objdump_line': ""
                                })
                    except KeyError:
                        if target_type == 'registers':
                            csv_data.append({
                                'experiment': exp_name,
                                'fault_address_register': "no fault data",
                                'trigger_address': "no fault data",
                                'bitmask': "no fault data"
                            })
                        else:
                            csv_data.append({
                                'experiment': exp_name,
                                'fault_address': "no fault data",
                                'fault_address_dec': "",
                                'instruction': "",
                                'objdump_line': ""
                            })
                
                # Write to CSV
                print(filename)
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    if target_type == 'registers':
                        fieldnames = ['experiment', 'fault_address_register', 'trigger_address', 'bitmask']
                    else:
                        fieldnames = ['experiment', 'fault_address', 'fault_address_dec', 'instruction', 'objdump_line']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)
                
                print(f"Exported {len(csv_data)} entries to {filename}")
            
            # Export each class to its own CSV file for Hari
            export_class_to_csv(correct_hari_trigger, "Correct Hari triggers", f"{output_dir}/{analysis_name}_{analysis_type}_correct_hari_triggers.csv")
            export_class_to_csv(incorrect_hari_trigger, "Incorrect Hari triggers (False Positives)", f"{output_dir}/{analysis_name}_{analysis_type}_incorrect_hari_triggers.csv")
            export_class_to_csv(correct_hari_no_trigger, "Correct Hari no-triggers", f"{output_dir}/{analysis_name}_{analysis_type}_correct_hari_no_triggers.csv")
            export_class_to_csv(incorrect_hari_no_trigger, "Incorrect Hari no-triggers (False Negatives)", f"{output_dir}/{analysis_name}_{analysis_type}_incorrect_hari_no_triggers.csv")
            
            # Export each class to its own CSV file for Huang
            export_class_to_csv(correct_huang_trigger, "Correct Huang triggers", f"{output_dir}/{analysis_name}_{analysis_type}_correct_huang_triggers.csv")
            export_class_to_csv(incorrect_huang_trigger, "Incorrect Huang triggers (False Positives)", f"{output_dir}/{analysis_name}_{analysis_type}_incorrect_huang_triggers.csv")
            export_class_to_csv(correct_huang_no_trigger, "Correct Huang no-triggers", f"{output_dir}/{analysis_name}_{analysis_type}_correct_huang_no_triggers.csv")
            export_class_to_csv(incorrect_huang_no_trigger, "Incorrect Huang no-triggers (False Negatives)", f"{output_dir}/{analysis_name}_{analysis_type}_incorrect_huang_no_triggers.csv")

        # Print max_diff summary for failed Hari cases
        if analysis_type == 'abft' and failed_hari_max_diff:
            print(f"\n=== L1 Norm VALUES FOR FAILED HARI CASES ===")
            print(f"Total failed Hari cases: {len(failed_hari_max_diff)}")
            
            # Find maximum max_diff value
            max_diff_values = [max_diff for _, max_diff, _ in failed_hari_max_diff]
            overall_max_diff = max(max_diff_values)
            
            print(f"Maximum L1 Norm across all failed Hari cases: {overall_max_diff}")
            
            if verbose:
                print(f"\nFailed Hari cases with their L1 Norm and fault_address values:")
                
                # Sort by max_diff value in descending order
                sorted_failed = sorted(failed_hari_max_diff, key=lambda x: x[1], reverse=True)
                for exp_name, max_diff, fault_address in sorted_failed:
                    fault_addr_str = f"0x{fault_address:x}" if fault_address is not None else "not found"
                    if objdump_content and fault_address is not None:
                        objdump_line = find_address_in_objdump(fault_address, objdump_content)
                        if objdump_line:
                            print(f"  {exp_name}: L1 Norm={max_diff}, fault_address={fault_addr_str}, objdump_line={objdump_line.strip()}")
                        else:
                            print(f"  {exp_name}: L1 Norm={max_diff}, fault_address={fault_addr_str}, objdump_line=not found")
                    else:
                        print(f"  {exp_name}: L1 Norm={max_diff}, fault_address={fault_addr_str}")
        
        # Print max_diff summary for failed Huang cases
        if analysis_type == 'abft' and failed_huang_max_diff:
            print(f"\n=== L1 Norm VALUES FOR FAILED HUANG CASES ===")
            print(f"Total failed Huang cases: {len(failed_huang_max_diff)}")
            
            # Find maximum max_diff value
            max_diff_values = [max_diff for _, max_diff, _ in failed_huang_max_diff]
            overall_max_diff = max(max_diff_values)
            
            print(f"Maximum L1 Norm across all failed Huang cases: {overall_max_diff}")
            
            if verbose:
                print(f"\nFailed Huang cases with their L1 Norm and fault_address values:")
                
                # Sort by max_diff value in descending order
                sorted_failed = sorted(failed_huang_max_diff, key=lambda x: x[1], reverse=True)
                for exp_name, max_diff, fault_address in sorted_failed:
                    fault_addr_str = f"0x{fault_address:x}" if fault_address is not None else "not found"
                    if objdump_content and fault_address is not None:
                        objdump_line = find_address_in_objdump(fault_address, objdump_content)
                        if objdump_line:
                            print(f"  {exp_name}: L1 Norm={max_diff}, fault_address={fault_addr_str}, objdump_line={objdump_line.strip()}")
                        else:
                            print(f"  {exp_name}: L1 Norm={max_diff}, fault_address={fault_addr_str}, objdump_line=not found")
                    else:
                        print(f"  {exp_name}: L1 Norm={max_diff}, fault_address={fault_addr_str}")
        
        # Write only the ABFT trigger section to file for both Hari and Huang
        if analysis_type == 'abft':
            os.makedirs(output_dir, exist_ok=True)
            with open(results_filename, 'w') as f:
                hari_total = correct_hari_trig + incorrect_hari_trig + correct_hari_no_trig + incorrect_hari_no_trig
                huang_total = correct_huang_trig + incorrect_huang_trig + correct_huang_no_trig + incorrect_huang_no_trig
                
                f.write(f"=== HARI ABFT TRIGGER (Bytes 4-7) ===\n")
                f.write(f"Correct Hari triggers:     {correct_hari_trig} - Hari triggered when bytes 4-7 differ ({correct_hari_trig/hari_total*100:.2f}%)\n")
                f.write(f"Incorrect Hari triggers:   {incorrect_hari_trig} - Hari triggered when bytes 4-7 same ({incorrect_hari_trig/hari_total*100:.2f}%)\n")
                f.write(f"Correct Hari no-triggers:  {correct_hari_no_trig} - Hari not triggered when bytes 4-7 same ({correct_hari_no_trig/hari_total*100:.2f}%)\n")
                f.write(f"Incorrect Hari no-triggers: {incorrect_hari_no_trig} - Hari not triggered when bytes 4-7 differ ({incorrect_hari_no_trig/hari_total*100:.2f}%)\n")
                f.write(f"\n")
                
                f.write(f"=== HUANG ABFT TRIGGER (Bytes 4-7) ===\n")
                f.write(f"Correct Huang triggers:     {correct_huang_trig} - Huang triggered when bytes 4-7 differ ({correct_huang_trig/huang_total*100:.2f}%)\n")
                f.write(f"Incorrect Huang triggers:   {incorrect_huang_trig} - Huang triggered when bytes 4-7 same ({incorrect_huang_trig/huang_total*100:.2f}%)\n")
                f.write(f"Correct Huang no-triggers:  {correct_huang_no_trig} - Huang not triggered when bytes 4-7 same ({correct_huang_no_trig/huang_total*100:.2f}%)\n")
                f.write(f"Incorrect Huang no-triggers: {incorrect_huang_no_trig} - Huang not triggered when bytes 4-7 differ ({incorrect_huang_no_trig/huang_total*100:.2f}%)\n")
            print(f"\nHari and Huang ABFT trigger analysis saved to: {results_filename}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze HDF5 file for Hari, Huang ABFT and Hash trigger correctness')
    parser.add_argument('--target', choices=['instructions', 'registers', 'fmla'], default='instructions',
                        help='Target type for analysis: instructions or registers (default: instructions)')
    parser.add_argument('--fault', choices=['transient', 'permanent'], default='transient',
                        help='Fault type for analysis: transient or permanent (default: transient)')
    parser.add_argument('--analysis', choices=['abft', 'hash'], default='abft',
                        help='Analysis type: abft (includes both Hari and Huang) or hash (default: abft)')
    parser.add_argument('--show-line', action='store_true',
                        help='Show corresponding lines from objdump.txt for fault addresses')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed WARNING and ERROR blocks for each failed experiment')
    
    args = parser.parse_args()
    
    try:
        compare_all_experiments(args.target, args.fault, args.analysis, args.show_line, args.verbose)
    except FileNotFoundError:
        print("Error: Required HDF5 file not found")
        print("Expected files:")
        print("  output/instructions/transient.hdf5")
        print("  output/instructions/permanent.hdf5")
        print("  output/registers/transient.hdf5")
        print("  output/registers/permanent.hdf5")
    except KeyError as e:
        print(f"Error: Path not found in HDF5 file: {e}")
    except Exception as e:
        print(f"Error: {e}")
