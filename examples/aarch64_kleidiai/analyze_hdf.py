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
        # ABFT trigger location - 16 bytes at 0x40020cb8
        # Byte 0: ABFT triggered flag
        # Byte 2: Hash triggered flag  
        # Bytes 4-7: Data to compare for ABFT trigger
        # Bytes 8-15: Data to compare for hash trigger
        abft_trigger_location = 'location_40024c80_24_1'
        
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
        
        # Analysis results
        correct_abft_trigger = []     # ABFT triggered when bytes 4-6 differ
        incorrect_abft_trigger = []   # ABFT triggered when bytes 4-6 are same
        correct_abft_no_trigger = []  # ABFT not triggered when bytes 4-6 are same
        incorrect_abft_no_trigger = [] # ABFT not triggered when bytes 4-6 differ
        
        correct_hash_trigger = []     # Hash triggered when bytes 8-12 differ
        incorrect_hash_trigger = []   # Hash triggered when bytes 8-12 are same
        correct_hash_no_trigger = []  # Hash not triggered when bytes 8-12 are same
        incorrect_hash_no_trigger = [] # Hash not triggered when bytes 8-12 differ
        
        # Special ABFT classifications
        abft_timeouts = []            # ABFT experiments with timeouts (zero hash bytes or invalid flags)
        
        abft_data_errors = []         # Could not read ABFT data
        
        # Store max_diff values for failed ABFT cases
        failed_abft_max_diff = []     # List of (experiment_name, max_diff_value, fault_address) tuples
        
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
                abft_triggered = exp_abft_flat[0]      # Byte 0: ABFT triggered flag
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
                    print(f"  ABFT flag: {abft_triggered}, Hash flag: {hash_triggered}")"""
                
                # Check if bytes 0,1 are invalid (not 0 or 1)
                byte_0 = exp_abft_flat[0]
                byte_1 = exp_abft_flat[1]
                if (byte_0 not in [0, 1]) or (byte_1 not in [0, 1]):
                    is_timeout = True
                    """print(f"TIMEOUT: {exp_name} - Invalid flag bytes")
                    print(f"  Byte 0: {byte_0} (should be 0 or 1)")
                    print(f"  Byte 1: {byte_1} (should be 0 or 1)")"""
                
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
                        # Bytes 4-7 are different, ABFT should be triggered (= 1)
                        if abft_triggered == 1:
                            correct_abft_trigger.append(exp_name)
                            """print(f"CORRECT ABFT: {exp_name}")
                            print(f"  Bytes 4-7 differ, ABFT correctly triggered")
                            print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                            print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                            print(f"  ABFT flag: {abft_triggered}")"""
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
                                        trigger_address_from_hdf5 = faults_data[0]['fault_address']  # Get trigger_address field
                            except KeyError:
                                fault_address = None
                                trigger_address_from_hdf5 = None
                            
                            incorrect_abft_no_trigger.append(exp_name)
                            failed_abft_max_diff.append((exp_name, max_diff, fault_address))
                            if verbose:
                                print(f"ERROR ABFT: {exp_name} - CRITICAL!")
                                print(f"  Bytes 4-7 differ but ABFT not triggered!")
                                print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                                print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                                print(f"  ABFT flag: {abft_triggered} (should be 1)")
                                print(f"  L1 Norm: {max_diff}")
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
                        # Bytes 4-7 are same, ABFT should not be triggered (≠ 1)
                        if abft_triggered != 1:
                            correct_abft_no_trigger.append(exp_name)
                            """print(f"CORRECT ABFT: {exp_name}")
                            print(f"  Bytes 4-7 same as golden, ABFT correctly not triggered")
                            print(f"  ABFT flag: {abft_triggered}")"""
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
                                        trigger_address_from_hdf5 = faults_data[0]['fault_address']  # Get trigger_address field
                            except KeyError:
                                fault_address = None
                                trigger_address_from_hdf5 = None
                            
                            incorrect_abft_trigger.append(exp_name)
                            failed_abft_max_diff.append((exp_name, max_diff, fault_address))
                            if verbose:
                                print(f"WARNING ABFT: {exp_name}")
                                print(f"  Bytes 4-7 same as golden but ABFT triggered!")
                                print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                                print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                                print(f"  ABFT flag: {abft_triggered} (should not be 1)")
                                print(f"  L1 Norm: {max_diff}")
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
                            print(f"SKIPPED ABFT: {exp_name} - Hash bytes differ, ignoring ABFT analysis")
                
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
        
        # ABFT metrics
        correct_abft_trig = len(correct_abft_trigger)
        incorrect_abft_trig = len(incorrect_abft_trigger)
        correct_abft_no_trig = len(correct_abft_no_trigger)
        incorrect_abft_no_trig = len(incorrect_abft_no_trigger)
        
        # Hash metrics
        correct_hash_trig = len(correct_hash_trigger)
        incorrect_hash_trig = len(incorrect_hash_trigger)
        correct_hash_no_trig = len(correct_hash_no_trigger)
        incorrect_hash_no_trig = len(incorrect_hash_no_trigger)
        
        data_errors = len(abft_data_errors)
        timeouts = len(abft_timeouts)
        
        print(f"=== ABFT & HASH TRIGGER ANALYSIS SUMMARY ===")
        print(f"Analysis type: {target_type} + {fault_type} ({analysis_type})")
        print(f"Total experiments: {total_experiments}")
        print(f"Data errors: {data_errors}")
        print(f"Timeouts: {timeouts}")
        print()
        
        if analysis_type == 'abft':
            # Calculate total for percentage calculations
            abft_total = correct_abft_trig + incorrect_abft_trig + correct_abft_no_trig + incorrect_abft_no_trig
            
            print(f"=== ABFT TRIGGER (Bytes 4-7) ===")
            print(f"Correct ABFT triggers:     {correct_abft_trig} - ABFT triggered when bytes 4-7 differ ({correct_abft_trig/abft_total*100:.2f}%)")
            print(f"Incorrect ABFT triggers:   {incorrect_abft_trig} - ABFT triggered when bytes 4-7 same ({incorrect_abft_trig/abft_total*100:.2f}%)")
            print(f"Correct ABFT no-triggers:  {correct_abft_no_trig} - ABFT not triggered when bytes 4-7 same ({correct_abft_no_trig/abft_total*100:.2f}%)")
            print(f"Incorrect ABFT no-triggers: {incorrect_abft_no_trig} - ABFT not triggered when bytes 4-7 differ ({incorrect_abft_no_trig/abft_total*100:.2f}%)")
            print()
        
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
                abft_accuracy = (correct_abft_trig + correct_abft_no_trig) / valid_experiments
                print(f"ABFT Overall Accuracy: {abft_accuracy:.4f} ({abft_accuracy*100:.2f}%)")
            if analysis_type == 'hash':
                hash_accuracy = (correct_hash_trig + correct_hash_no_trig) / valid_experiments
                print(f"Hash Overall Accuracy: {hash_accuracy:.4f} ({hash_accuracy*100:.2f}%)")
        
        # Calculate ABFT detailed metrics using confusion matrix approach (only if analyzing ABFT)
        if analysis_type == 'abft':
            # True Positives (TP): correctly detected changes (correct_abft_trig)
            # False Positives (FP): incorrectly detected changes (incorrect_abft_trig)
            # True Negatives (TN): correctly detected no changes (correct_abft_no_trig)
            # False Negatives (FN): missed changes (incorrect_abft_no_trig)
            
            abft_tp = correct_abft_trig
            abft_fp = incorrect_abft_trig
            abft_tn = correct_abft_no_trig
            abft_fn = incorrect_abft_no_trig
            
            print(f"\n=== ABFT DETAILED METRICS ===")
            print(f"True Positives (TP):  {abft_tp} - Correctly detected changes")
            print(f"False Positives (FP): {abft_fp} - Incorrectly detected changes")
            print(f"True Negatives (TN):  {abft_tn} - Correctly detected no changes")
            print(f"False Negatives (FN): {abft_fn} - Missed changes")
            
            # Calculate metrics
            if (abft_tp + abft_fp) > 0:
                abft_precision = abft_tp / (abft_tp + abft_fp)
                print(f"ABFT Precision: {abft_precision:.4f} ({abft_precision*100:.2f}%)")
            else:
                print(f"ABFT Precision: N/A (no positive predictions)")
            
            if (abft_tp + abft_fn) > 0:
                abft_recall = abft_tp / (abft_tp + abft_fn)
                print(f"ABFT Recall (Sensitivity): {abft_recall:.4f} ({abft_recall*100:.2f}%)")
            else:
                print(f"ABFT Recall: N/A (no actual positives)")
            
            if (abft_tp + abft_fp + abft_tn + abft_fn) > 0:
                abft_accuracy_detailed = (abft_tp + abft_tn) / (abft_tp + abft_fp + abft_tn + abft_fn)
                print(f"ABFT Accuracy: {abft_accuracy_detailed:.4f} ({abft_accuracy_detailed*100:.2f}%)")
            else:
                print(f"ABFT Accuracy: N/A (no valid experiments)")
            
            # Calculate F1 Score
            if (abft_tp + abft_fp) > 0 and (abft_tp + abft_fn) > 0:
                abft_precision_calc = abft_tp / (abft_tp + abft_fp)
                abft_recall_calc = abft_tp / (abft_tp + abft_fn)
                if (abft_precision_calc + abft_recall_calc) > 0:
                    abft_f1 = 2 * (abft_precision_calc * abft_recall_calc) / (abft_precision_calc + abft_recall_calc)
                    print(f"ABFT F1 Score: {abft_f1:.4f} ({abft_f1*100:.2f}%)")
                else:
                    print(f"ABFT F1 Score: 0.0000 (0.00%)")
            else:
                print(f"ABFT F1 Score: N/A (cannot calculate)")
            
            # Calculate ABFT detection metrics
            abft_experiments_with_changes = correct_abft_trig + incorrect_abft_no_trig
            if abft_experiments_with_changes > 0:
                abft_detection_rate = correct_abft_trig / abft_experiments_with_changes
                abft_miss_rate = incorrect_abft_no_trig / abft_experiments_with_changes
                print(f"ABFT Detection Rate (when bytes 4-6 differ): {abft_detection_rate:.4f} ({abft_detection_rate*100:.2f}%)")
                print(f"ABFT Miss Rate (when bytes 4-6 differ): {abft_miss_rate:.4f} ({abft_miss_rate*100:.2f}%)")
            
            # Calculate false positive metrics  
            abft_experiments_without_changes = correct_abft_no_trig + incorrect_abft_trig
            if abft_experiments_without_changes > 0:
                abft_false_positive_rate = incorrect_abft_trig / abft_experiments_without_changes
                abft_specificity = correct_abft_no_trig / abft_experiments_without_changes
                print(f"ABFT False Positive Rate (when bytes 4-6 same): {abft_false_positive_rate:.4f} ({abft_false_positive_rate*100:.2f}%)")
                print(f"ABFT Specificity (when bytes 4-6 same): {abft_specificity:.4f} ({abft_specificity*100:.2f}%)")
        
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
        if analysis_type == 'abft' and incorrect_abft_no_trigger:
            print(f"\n=== CRITICAL: MISSED ABFT DETECTIONS ===")
            print("These experiments had different bytes 4-6 but ABFT was not triggered:")
            print(f"Total count: {len(incorrect_abft_no_trigger)}")
            # Print first 10 as sample
            for i, exp in enumerate(incorrect_abft_no_trigger[:10]):
                print(f"  {exp}")
            if len(incorrect_abft_no_trigger) > 10:
                print(f"  ... and {len(incorrect_abft_no_trigger) - 10} more")
        
        if analysis_type == 'hash' and incorrect_hash_no_trigger:
            print(f"\n=== CRITICAL: MISSED HASH DETECTIONS ===")
            print("These experiments had different bytes 8-12 but Hash was not triggered:")
            print(f"Total count: {len(incorrect_hash_no_trigger)}")
            # Print first 10 as sample
            for i, exp in enumerate(incorrect_hash_no_trigger[:10]):
                print(f"  {exp}")
            if len(incorrect_hash_no_trigger) > 10:
                print(f"  ... and {len(incorrect_hash_no_trigger) - 10} more")
        
        if analysis_type == 'abft' and incorrect_abft_trigger:
            print(f"\n=== FALSE ABFT TRIGGERS ===")
            print("These experiments had same bytes 4-6 but ABFT was triggered:")
            if verbose:
                for exp in incorrect_abft_trigger:
                    print(f"  {exp}")
            else:
                for i, exp in enumerate(incorrect_abft_trigger[:10]):
                    print(f"  {exp}")
                if len(incorrect_abft_trigger) > 10:
                    print(f"  ... and {len(incorrect_abft_trigger) - 10} more")
        
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
                            csv_data.append({
                                'experiment': exp_name,
                                'fault_address': "no fault data",
                                'fault_address_dec': "",
                                'instruction': "",
                                'objdump_line': ""
                            })
                    except KeyError:
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
                    fieldnames = ['experiment', 'fault_address', 'fault_address_dec', 'instruction', 'objdump_line']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)
                
                print(f"Exported {len(csv_data)} entries to {filename}")
            
            # Export each class to its own CSV file
            export_class_to_csv(correct_abft_trigger, "Correct ABFT triggers", f"{output_dir}/{analysis_name}_{analysis_type}_correct_abft_triggers.csv")
            export_class_to_csv(incorrect_abft_trigger, "Incorrect ABFT triggers (False Positives)", f"{output_dir}/{analysis_name}_{analysis_type}_incorrect_abft_triggers.csv")
            export_class_to_csv(correct_abft_no_trigger, "Correct ABFT no-triggers", f"{output_dir}/{analysis_name}_{analysis_type}_correct_abft_no_triggers.csv")
            export_class_to_csv(incorrect_abft_no_trigger, "Incorrect ABFT no-triggers (False Negatives)", f"{output_dir}/{analysis_name}_{analysis_type}_incorrect_abft_no_triggers.csv")

        # Print max_diff summary for failed ABFT cases
        if analysis_type == 'abft' and failed_abft_max_diff:
            print(f"\n=== L1 Norm VALUES FOR FAILED ABFT CASES ===")
            print(f"Total failed ABFT cases: {len(failed_abft_max_diff)}")
            
            # Find maximum max_diff value
            max_diff_values = [max_diff for _, max_diff, _ in failed_abft_max_diff]
            overall_max_diff = max(max_diff_values)
            
            print(f"Maximum L1 Norm across all failed cases: {overall_max_diff}")
            
            if verbose:
                print(f"\nFailed cases with their L1 Norm and fault_address values:")
                
                # Sort by max_diff value in descending order
                sorted_failed = sorted(failed_abft_max_diff, key=lambda x: x[1], reverse=True)
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
        
        # Write only the ABFT trigger section to file
        if analysis_type == 'abft':
            os.makedirs(output_dir, exist_ok=True)
            with open(results_filename, 'w') as f:
                abft_total = correct_abft_trig + incorrect_abft_trig + correct_abft_no_trig + incorrect_abft_no_trig
                f.write(f"=== ABFT TRIGGER (Bytes 4-7) ===\n")
                f.write(f"Correct ABFT triggers:     {correct_abft_trig} - ABFT triggered when bytes 4-7 differ ({correct_abft_trig/abft_total*100:.2f}%)\n")
                f.write(f"Incorrect ABFT triggers:   {incorrect_abft_trig} - ABFT triggered when bytes 4-7 same ({incorrect_abft_trig/abft_total*100:.2f}%)\n")
                f.write(f"Correct ABFT no-triggers:  {correct_abft_no_trig} - ABFT not triggered when bytes 4-7 same ({correct_abft_no_trig/abft_total*100:.2f}%)\n")
                f.write(f"Incorrect ABFT no-triggers: {incorrect_abft_no_trig} - ABFT not triggered when bytes 4-7 differ ({incorrect_abft_no_trig/abft_total*100:.2f}%)\n")
            print(f"\nABFT trigger analysis saved to: {results_filename}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze HDF5 file for ABFT and Hash trigger correctness')
    parser.add_argument('--target', choices=['instructions', 'registers'], default='instructions',
                        help='Target type for analysis: instructions or registers (default: instructions)')
    parser.add_argument('--fault', choices=['transient', 'permanent'], default='transient',
                        help='Fault type for analysis: transient or permanent (default: transient)')
    parser.add_argument('--analysis', choices=['abft', 'hash'], default='abft',
                        help='Analysis type: abft or hash (default: abft)')
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
