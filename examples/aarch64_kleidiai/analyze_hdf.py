import h5py
import numpy as np
import argparse

def compare_all_experiments(analysis_type):
    # Determine which HDF5 file to use based on analysis type
    if analysis_type == 'abft':
        hdf5_file = 'output_instruction_skip_kernel.hdf5'
    elif analysis_type == 'hash':
        hdf5_file = 'output_weight_tampering.hdf5'
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type}")
    
    print(f"Opening HDF5 file: {hdf5_file}")
    
    # Open the HDF5 file
    with h5py.File(hdf5_file, 'r') as f:
        # ABFT trigger location - 16 bytes at 0x40020cb8
        # Byte 0: ABFT triggered flag
        # Byte 2: Hash triggered flag  
        # Bytes 4-7: Data to compare for ABFT trigger
        # Bytes 8-15: Data to compare for hash trigger
        abft_trigger_location = 'location_40020cb8_16_1'
        
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
                
                if len(exp_abft_flat) < 16:
                    print(f"ERROR: {exp_name} - ABFT data too short ({len(exp_abft_flat)} bytes)")
                    abft_data_errors.append(exp_name)
                    continue
                
                # Extract relevant bytes
                abft_triggered = exp_abft_flat[0]      # Byte 0: ABFT triggered flag
                hash_triggered = exp_abft_flat[2]      # Byte 2: Hash triggered flag
                abft_bytes = exp_abft_flat[4:8]        # Bytes 4-7 for ABFT trigger
                hash_bytes = exp_abft_flat[8:16]       # Bytes 8-15 for hash trigger
                
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
                    print()
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
                            incorrect_abft_no_trigger.append(exp_name)
                            print(f"ERROR ABFT: {exp_name} - CRITICAL!")
                            print(f"  Bytes 4-7 differ but ABFT not triggered!")
                            print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                            print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                            print(f"  ABFT flag: {abft_triggered} (should be 1)")
                            print()
                    else:
                        # Bytes 4-7 are same, ABFT should not be triggered (≠ 1)
                        if abft_triggered != 1:
                            correct_abft_no_trigger.append(exp_name)
                            """print(f"CORRECT ABFT: {exp_name}")
                            print(f"  Bytes 4-7 same as golden, ABFT correctly not triggered")
                            print(f"  ABFT flag: {abft_triggered}")"""
                        else:
                            incorrect_abft_trigger.append(exp_name)
                            print(f"WARNING ABFT: {exp_name}")
                            print(f"  Bytes 4-7 same as golden but ABFT triggered!")
                            print(f"  Golden ABFT:     {[hex(b) for b in golden_abft_bytes]}")
                            print(f"  Experiment ABFT: {[hex(b) for b in abft_bytes]}")
                            print(f"  ABFT flag: {abft_triggered} (should not be 1)")
                            print()
                else:
                    if analysis_type == 'abft':
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
                            print(f"WARNING HASH: {exp_name}")
                            print(f"  Bytes 8-15 same as golden but Hash triggered!")
                            print(f"  Golden Hash:     {[hex(b) for b in golden_hash_bytes]}")
                            print(f"  Experiment Hash: {[hex(b) for b in hash_bytes]}")
                            print(f"  Hash flag: {hash_triggered} (should not be 1)")
                            print()

                    
            except KeyError as e:
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
        print(f"Analysis type: {analysis_type}")
        print(f"Total experiments: {total_experiments}")
        print(f"Data errors: {data_errors}")
        print(f"Timeouts: {timeouts}")
        print()
        
        if analysis_type == 'abft':
            print(f"=== ABFT TRIGGER (Bytes 4-7) ===")
            print(f"Correct ABFT triggers:     {correct_abft_trig} - ABFT triggered when bytes 4-7 differ")
            print(f"Incorrect ABFT triggers:   {incorrect_abft_trig} - ABFT triggered when bytes 4-7 same")
            print(f"Correct ABFT no-triggers:  {correct_abft_no_trig} - ABFT not triggered when bytes 4-7 same")
            print(f"Incorrect ABFT no-triggers: {incorrect_abft_no_trig} - ABFT not triggered when bytes 4-7 differ")
            print()
        
        if analysis_type == 'hash':
            print(f"=== HASH TRIGGER (Bytes 8-15) ===")
            print(f"Correct Hash triggers:     {correct_hash_trig} - Hash triggered when bytes 8-15 differ")
            print(f"Incorrect Hash triggers:   {incorrect_hash_trig} - Hash triggered when bytes 8-15 same")
            print(f"Correct Hash no-triggers:  {correct_hash_no_trig} - Hash not triggered when bytes 8-15 same")
            print(f"Incorrect Hash no-triggers: {incorrect_hash_no_trig} - Hash not triggered when bytes 8-15 differ")
        
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
            for exp in incorrect_abft_trigger:
                print(f"  {exp}")
        
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
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze HDF5 file for ABFT and Hash trigger correctness')
    parser.add_argument('--type', choices=['abft', 'hash'], default='abft',
                        help='Type of analysis to perform: abft (uses output_instruction_skip_kernel.hdf5) or hash (uses output_weight_tampering.hdf5, default: abft)')
    
    args = parser.parse_args()
    
    try:
        compare_all_experiments(args.type)
    except FileNotFoundError:
        print("Error: Required HDF5 file not found in current directory")
        print("  For ABFT analysis: output_instruction_skip_kernel.hdf5")
        print("  For Hash analysis: output_weight_tampering.hdf5")
    except KeyError as e:
        print(f"Error: Path not found in HDF5 file: {e}")
    except Exception as e:
        print(f"Error: {e}")
