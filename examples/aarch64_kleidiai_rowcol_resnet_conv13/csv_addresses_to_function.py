#!/usr/bin/env python3
"""
Script to analyze fault addresses from CSV and map them to functions from objdump.
"""

import csv
import re
import argparse
from collections import defaultdict, Counter

def get_csv_filename(target_type, fault_type, classification):
    """Generate the appropriate CSV filename based on parameters."""
    # Base directory structure
    if target_type == 'instructions' and fault_type == 'transient':
        base_dir = 'output/instructions'
        analysis_name = 'instructions_transient'
    elif target_type == 'instructions' and fault_type == 'permanent':
        base_dir = 'output/instructions'
        analysis_name = 'instructions_permanent'
    elif target_type == 'registers' and fault_type == 'transient':
        base_dir = 'output/registers'
        analysis_name = 'registers_transient'
    elif target_type == 'registers' and fault_type == 'permanent':
        base_dir = 'output/registers'
        analysis_name = 'registers_permanent'
    else:
        raise ValueError(f"Invalid combination: target_type={target_type}, fault_type={fault_type}")
    
    # Map classification to CSV filename suffix
    classification_map = {
        'TP': 'correct_abft_triggers',      # True Positives: correctly detected changes
        'TN': 'correct_abft_no_triggers',   # True Negatives: correctly detected no changes  
        'FP': 'incorrect_abft_triggers',    # False Positives: incorrectly detected changes
        'FN': 'incorrect_abft_no_triggers'  # False Negatives: missed changes
    }
    
    if classification not in classification_map:
        raise ValueError(f"Invalid classification: {classification}. Must be one of: {list(classification_map.keys())}")
    
    suffix = classification_map[classification]
    filename = f"{base_dir}/{analysis_name}_abft_{suffix}.csv"
    
    return filename

def parse_objdump_functions(objdump_file):
    """Parse objdump.txt to extract function names and their address ranges."""
    functions = []
    
    with open(objdump_file, 'r') as f:
        content = f.read()
    
    # Find all function definitions using regex
    # Pattern matches lines like: "0000000040000028 <function_name>:"
    function_pattern = r'^([0-9a-f]+) <([^>]+)>:'
    
    lines = content.split('\n')
    function_starts = []
    
    # First pass: collect all function start addresses and names
    for i, line in enumerate(lines):
        match = re.match(function_pattern, line)
        if match:
            start_addr = int(match.group(1), 16)
            function_name = match.group(2)
            function_starts.append((start_addr, function_name, i))
    
    # Sort by start address to ensure proper ordering
    function_starts.sort(key=lambda x: x[0])
    
    # Second pass: calculate end addresses based on next function start
    for i, (start_addr, function_name, line_num) in enumerate(function_starts):
        if i < len(function_starts) - 1:
            # End address is just before the next function starts
            next_start = function_starts[i + 1][0]
            end_addr = next_start - 1
        else:
            # For the last function, set a large end address
            end_addr = 0xFFFFFFFFFFFFFFFF
        
        functions.append({
            'name': function_name,
            'start': start_addr,
            'end': end_addr
        })
    
    return functions

def find_function_for_address(address, functions):
    """Find which function contains the given address."""
    for func in functions:
        if func['start'] <= address <= func['end']:
            return func['name']
    return 'Unknown'

def analyze_csv_addresses(csv_file, functions):
    """Analyze the CSV file and map addresses to functions."""
    function_counts = Counter()
    total_addresses = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get the fault address (decimal)
            fault_addr_dec = int(row['fault_address_dec'])
            total_addresses += 1
            
            # Find which function this address belongs to
            function_name = find_function_for_address(fault_addr_dec, functions)
            function_counts[function_name] += 1
    
    return function_counts, total_addresses

def print_addresses_for_function(csv_file, functions, target_function):
    """Print all addresses that belong to a specific function."""
    target_addresses = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get the fault address (decimal)
            fault_addr_dec = int(row['fault_address_dec'])
            
            # Find which function this address belongs to
            function_name = find_function_for_address(fault_addr_dec, functions)
            
            # Check if this address belongs to the target function
            if function_name == target_function:
                fault_addr_hex = row['fault_address']  # This is the hex address
                experiment = row['experiment']
                instruction = row['instruction']
                target_addresses.append((fault_addr_dec, fault_addr_hex, experiment, instruction))
    
    return target_addresses

def demangle_function_name(mangled_name):
    """Basic demangling for C++ function names."""
    if not mangled_name.startswith('_Z'):
        return mangled_name
    
    # Handle common patterns found in the objdump
    if mangled_name == '_Z18calculate_hash_f16PKDhm':
        return 'calculate_hash_f16'
    elif mangled_name == '_Z13run_test_caseRK8TestCase':
        return 'run_test_case'
    elif mangled_name.startswith('_ZSt3minImERKT_S2_S2_'):
        return 'std::min<size_t>'
    elif mangled_name.startswith('_ZStL19piecewise_construct'):
        return 'std::piecewise_construct'
    elif '_GLOBAL__N_1' in mangled_name:
        # Anonymous namespace functions
        if '16fill_zero_matrix' in mangled_name:
            return 'fill_zero_matrix (anonymous namespace)'
        elif '15neon_oc_f16_f16' in mangled_name:
            return 'neon_oc_f16_f16 (anonymous namespace)'
        elif '15neon_ic_f16_f16' in mangled_name:
            return 'neon_ic_f16_f16 (anonymous namespace)'
        elif '15neon_ic_f16_f32' in mangled_name:
            return 'neon_ic_f16_f32 (anonymous namespace)'
        elif '21neon_checksum_row_f16' in mangled_name:
            return 'neon_checksum_row_f16 (anonymous namespace)'
        elif '21neon_checksum_col_f16' in mangled_name:
            return 'neon_checksum_col_f16 (anonymous namespace)'
        elif '7ukernelE' in mangled_name:
            return 'ukernel (anonymous namespace)'
        else:
            return f'{mangled_name} (anonymous namespace)'
    elif mangled_name.startswith('_ZL'):
        # Static/local functions
        if 'gemm_weights_4_model_conv2d_3_Conv2D' in mangled_name:
            return 'gemm_weights_4_model_conv2d_3_Conv2D (static)'
        elif 'checksum_4_model_conv2d_3_Conv2D' in mangled_name:
            return 'checksum_4_model_conv2d_3_Conv2D (static)'
        elif 'example_input_conv_4_im2col' in mangled_name:
            return 'example_input_conv_4_im2col (static)'
        elif 'hash_4_model_conv2d_3_Conv2D' in mangled_name:
            return 'hash_4_model_conv2d_3_Conv2D (static)'
        elif 'test_cases' in mangled_name:
            return 'test_cases (static)'
        else:
            return f'{mangled_name} (static)'
    
    # Fallback: return the mangled name as-is
    return mangled_name

def debug_analyze_csv_addresses(csv_file, functions):
    """Debug version that tracks all addresses and unknown functions."""
    function_counts = Counter()
    total_addresses = 0
    unknown_addresses = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Get the fault address (decimal)
            fault_addr_dec = int(row['fault_address_dec'])
            total_addresses += 1
            
            # Find which function this address belongs to
            function_name = find_function_for_address(fault_addr_dec, functions)
            function_counts[function_name] += 1
            
            # Track unknown addresses for debugging
            if function_name == 'Unknown':
                unknown_addresses.append({
                    'dec': fault_addr_dec,
                    'hex': row['fault_address'],
                    'experiment': row['experiment'],
                    'instruction': row['instruction']
                })
    
    return function_counts, total_addresses, unknown_addresses

def main():
    parser = argparse.ArgumentParser(
        description='Analyze fault addresses from CSV and map them to functions from objdump',
        epilog="""
Classification Types:
  TP (True Positives):  ABFT correctly triggered when bytes differ - no problem
  TN (True Negatives):  ABFT correctly NOT triggered when bytes same - no problem
  FP (False Positives): ABFT incorrectly triggered when bytes same - false alarm
  FN (False Negatives): ABFT missed triggering when bytes differ - CRITICAL failure
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--target', choices=['instructions', 'registers'], default='instructions',
                        help='Target type for analysis: instructions or registers (default: instructions)')
    parser.add_argument('--fault', choices=['transient', 'permanent'], default='transient',
                        help='Fault type for analysis: transient or permanent (default: transient)')
    parser.add_argument('--classification', choices=['TP', 'TN', 'FP', 'FN'], default='FN',
                        help='Classification type: TP (True Positives), TN (True Negatives), FP (False Positives), FN (False Negatives) (default: FN)')
    parser.add_argument('--objdump', default='src/objdump.txt',
                        help='Path to objdump.txt file (default: src/objdump.txt)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed analysis and individual address listings')
    
    args = parser.parse_args()
    
    # Generate the appropriate CSV filename
    try:
        csv_file = get_csv_filename(args.target, args.fault, args.classification)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    objdump_file = args.objdump
    
    # Map classification codes to descriptions
    classification_descriptions = {
        'TP': 'True Positives (Correctly detected changes)',
        'TN': 'True Negatives (Correctly detected no changes)',
        'FP': 'False Positives (Incorrectly detected changes)',
        'FN': 'False Negatives (Missed changes)'
    }
    
    print(f"Analysis Configuration:")
    print(f"  Target: {args.target}")
    print(f"  Fault Type: {args.fault}")
    print(f"  Classification: {args.classification} - {classification_descriptions[args.classification]}")
    print(f"  CSV File: {csv_file}")
    print(f"  Objdump File: {objdump_file}")
    print()
    
    try:
        print("Parsing objdump to extract functions...")
        functions = parse_objdump_functions(objdump_file)
        print(f"Found {len(functions)} functions")
    except FileNotFoundError:
        print(f"Error: Objdump file not found: {objdump_file}")
        return
    except Exception as e:
        print(f"Error reading objdump file: {e}")
        return
    
    print("\nAnalyzing CSV addresses...")
    try:
        function_counts, total_addresses, unknown_addresses = debug_analyze_csv_addresses(csv_file, functions)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file}")
        print("Make sure you have run the analyze_hdf.py script first to generate the CSV files.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    if total_addresses == 0:
        print("No fault addresses found in the CSV file.")
        return
    
    # Calculate sum of all function counts to verify
    sum_of_function_counts = sum(function_counts.values())
    
    print(f"\nDEBUG INFORMATION:")
    print(f"Total addresses read from CSV: {total_addresses}")
    print(f"Sum of all function counts: {sum_of_function_counts}")
    print(f"Unknown addresses: {function_counts.get('Unknown', 0)}")
    print(f"Difference: {total_addresses - sum_of_function_counts}")
    
    if unknown_addresses:
        print(f"\nFirst 10 unknown addresses:")
        for i, addr in enumerate(unknown_addresses[:10]):
            print(f"  {addr['dec']:>12} (0x{addr['dec']:08x}) - {addr['hex']} - {addr['experiment']}")
    
    print(f"\nBreakdown of {total_addresses} fault addresses by function:")
    print(f"Analysis: {args.target} + {args.fault} ({args.classification})")
    print("=" * 100)
    print(f"{'Function Name':<60} {'Demangled Name':<25} {'Count':<8} {'%'}")
    print("=" * 100)
    
    # Sort by count (descending)
    for function_name, count in function_counts.most_common():
        percentage = (count / total_addresses) * 100
        demangled = demangle_function_name(function_name)
        # Truncate long names for display
        display_name = function_name[:58] + ".." if len(function_name) > 60 else function_name
        demangled_display = demangled[:23] + ".." if len(demangled) > 25 else demangled
        print(f"{display_name:<60} {demangled_display:<25} {count:<8} {percentage:5.2f}")
    
    print("=" * 100)
    print(f"{'Total':<60} {'':<25} {total_addresses:<8} 100.00")
    
    # Additional verification - show top functions with actual counts
    print(f"\nTOP FUNCTIONS BY TRIGGER COUNT:")
    print("-" * 80)
    running_total = 0
    for function_name, count in function_counts.most_common(20):  # Show top 20
        percentage = (count / total_addresses) * 100
        demangled = demangle_function_name(function_name)
        running_total += count
        # Truncate names for display
        demangled_short = demangled[:40] + ".." if len(demangled) > 42 else demangled
        print(f"{demangled_short:<45} {count:>6} ({percentage:5.2f}%)")
    
    print(f"\nRunning total of top functions: {running_total}")
    print(f"Remaining addresses: {total_addresses - running_total}")
    
    # Show detailed analysis only in verbose mode
    if args.verbose:
        print(f"\nDetailed Analysis:")
        print("-" * 60)
        for function_name, count in function_counts.most_common():
            percentage = (count / total_addresses) * 100
            demangled = demangle_function_name(function_name)
            print(f"\n{demangled}:")
            print(f"  Mangled name: {function_name}")
            print(f"  Fault count: {count}")
            print(f"  Percentage: {percentage:.2f}%")
    
    # Print all addresses for the top function with most faults (only in verbose mode)
    if args.verbose and function_counts:
        top_function, top_count = function_counts.most_common(1)[0]
        top_demangled = demangle_function_name(top_function)
        
        print(f"\n{'='*80}")
        print(f"ALL ADDRESSES FOR TOP FUNCTION: {top_demangled}")
        print(f"{'='*80}")
        
        top_function_addresses = print_addresses_for_function(csv_file, functions, top_function)
        
        if top_function_addresses:
            print(f"Found {len(top_function_addresses)} addresses in {top_demangled} function:")
            print(f"{'Index':<6} {'Decimal Address':<15} {'Hex Address':<12} {'Experiment':<20} {'Instruction'}")
            print("-" * 80)
            for i, (addr_dec, addr_hex, experiment, instruction) in enumerate(top_function_addresses, 1):
                # Truncate long instructions for display
                instruction_display = instruction[:40] + ".." if len(instruction) > 42 else instruction
                print(f"{i:<6} {addr_dec:<15} {addr_hex:<12} {experiment:<20} {instruction_display}")
        else:
            print(f"No addresses found for {top_demangled} function")
    elif not args.verbose and function_counts:
        print(f"\nUse --verbose to see detailed analysis and individual address listings.")
    else:
        print("\nNo functions found with faults.")

if __name__ == "__main__":
    main()