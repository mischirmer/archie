#!/usr/bin/env python3
"""
Enhanced script to map trigger addresses from tbinfo_based_faultconfig.json to their corresponding functions in objdump.txt
Also provides detailed analysis and saves results to a CSV file.
"""

import json
import re
import sys
import csv
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def parse_objdump_functions(objdump_file: str) -> Dict[int, str]:
    """
    Parse the objdump file and extract function definitions with their address ranges.
    Returns a dictionary mapping address ranges to function names.
    """
    functions = {}
    current_function = None
    current_start_addr = None
    
    with open(objdump_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Look for function definitions (lines ending with >:)
            if line.endswith('>:'):
                # Extract address and function name
                match = re.match(r'^([0-9a-fA-F]+)\s+<([^>]+)>:$', line)
                if match:
                    addr_str, func_name = match.groups()
                    addr = int(addr_str, 16)
                    
                    # Store the previous function's range if we had one
                    if current_function and current_start_addr is not None:
                        functions[current_start_addr] = current_function
                    
                    current_function = func_name
                    current_start_addr = addr
            
            # Look for instruction lines to determine function boundaries
            elif line and ':' in line and current_start_addr is not None:
                # Check if this is an instruction line
                match = re.match(r'^([0-9a-fA-F]+):', line)
                if match:
                    addr = int(match.group(1), 16)
                    # Update the function mapping for each address
                    functions[addr] = current_function
    
    return functions


def find_function_for_address(address: int, functions: Dict[int, str]) -> Optional[str]:
    """
    Find which function contains the given address.
    """
    # Find the largest function start address that is <= our target address
    candidates = [(addr, func) for addr, func in functions.items() if addr <= address]
    
    if not candidates:
        return None
    
    # Sort by address and get the closest one
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Get the function that starts at the highest address <= our target
    return candidates[0][1]


def parse_fault_config(json_file: str) -> List[Tuple[int, int, dict]]:
    """
    Parse the fault configuration JSON and extract all trigger addresses with fault information.
    Returns list of tuples: (trigger_address, trigger_counter, fault_info)
    """
    trigger_data = []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract trigger addresses from all faults
    if 'faults' in data:
        for fault_group_idx, fault_group in enumerate(data['faults']):
            for fault_idx, fault in enumerate(fault_group):
                if 'trigger_address' in fault and 'trigger_counter' in fault:
                    # trigger_address and trigger_counter are lists
                    trigger_addresses = fault['fault_address']
                    trigger_counters = fault['trigger_counter']
                    
                    # Pair each address with its counter
                    for addr, counter in zip(trigger_addresses, trigger_counters):
                        fault_info = {
                            'fault_group': fault_group_idx,
                            'fault_index': fault_idx,
                            'fault_type': fault.get('fault_type', 'unknown'),
                            'fault_model': fault.get('fault_model', 'unknown'),
                            'fault_address': fault.get('fault_address', []),
                            'fault_mask': fault.get('fault_mask', []),
                            'fault_lifespan': fault.get('fault_lifespan', [])
                        }
                        trigger_data.append((addr, counter, fault_info))
    
    return trigger_data


def demangle_function_name(mangled_name: str) -> str:
    """
    Attempt to demangle C++ function names for better readability.
    This is a simple demangling - for more complex names, use c++filt.
    """
    # Common C++ mangled prefixes
    if mangled_name.startswith('_ZN'):
        # This is a very basic demangling attempt
        # For full demangling, you'd want to use binutils c++filt
        if 'neon_oc_f16_f16' in mangled_name:
            return 'neon_oc_f16_f16 (output channel convolution)'
        elif 'neon_ic_f16_f16' in mangled_name:
            return 'neon_ic_f16_f16 (input channel convolution)'
        elif 'fill_zero_matrix' in mangled_name:
            return 'fill_zero_matrix'
        else:
            return mangled_name + ' (C++ function)'
    elif mangled_name.startswith('_Z'):
        if 'run_test_case' in mangled_name:
            return 'run_test_case'
        elif 'calculate_hash_f16' in mangled_name:
            return 'calculate_hash_f16'
        else:
            return mangled_name + ' (C++ function)'
    else:
        return mangled_name


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 address_to_function_mapper.py <objdump.txt> <tbinfo_based_faultconfig.json> [output.csv]")
        sys.exit(1)
    
    objdump_file = sys.argv[1]
    json_file = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else 'trigger_address_mapping.csv'
    
    try:
        print("Parsing objdump file...")
        functions = parse_objdump_functions(objdump_file)
        print(f"Found {len(set(functions.values()))} unique functions")
        
        print("Parsing fault configuration...")
        trigger_data = parse_fault_config(json_file)
        print(f"Found {len(trigger_data)} trigger address entries")
        
        print("\nMapping trigger addresses to functions...")
        print("-" * 100)
        print(f"{'Address (Hex)':<15} {'Address (Dec)':<15} {'Counter':<10} {'Function'}")
        print("-" * 100)
        
        # Track mappings for analysis
        address_to_function = {}
        function_counters = defaultdict(list)
        
        # Prepare CSV data
        csv_data = []
        
        for addr, counter, fault_info in trigger_data:
            func_name = find_function_for_address(addr, functions)
            address_to_function[addr] = func_name
            
            if func_name:
                function_counters[func_name].append((addr, counter))
                demangled_name = demangle_function_name(func_name)
                print(f"0x{addr:08x}      {addr:<15} {counter:<10} {demangled_name}")
                
                # Add to CSV data
                csv_data.append({
                    'address_hex': f"0x{addr:08x}",
                    'address_dec': addr,
                    'trigger_counter': counter,
                    'function_name': func_name,
                    'demangled_name': demangled_name,
                    'fault_group': fault_info['fault_group'],
                    'fault_index': fault_info['fault_index'],
                    'fault_type': fault_info['fault_type'],
                    'fault_model': fault_info['fault_model'],
                    'fault_address': ','.join(map(str, fault_info['fault_address'])),
                    'fault_mask': ','.join(map(str, fault_info['fault_mask'])),
                    'fault_lifespan': ','.join(map(str, fault_info['fault_lifespan']))
                })
            else:
                print(f"0x{addr:08x}      {addr:<15} {counter:<10} <UNKNOWN>")
                csv_data.append({
                    'address_hex': f"0x{addr:08x}",
                    'address_dec': addr,
                    'trigger_counter': counter,
                    'function_name': '<UNKNOWN>',
                    'demangled_name': '<UNKNOWN>',
                    'fault_group': fault_info['fault_group'],
                    'fault_index': fault_info['fault_index'],
                    'fault_type': fault_info['fault_type'],
                    'fault_model': fault_info['fault_model'],
                    'fault_address': ','.join(map(str, fault_info['fault_address'])),
                    'fault_mask': ','.join(map(str, fault_info['fault_mask'])),
                    'fault_lifespan': ','.join(map(str, fault_info['fault_lifespan']))
                })
        
        # Save to CSV
        print(f"\nSaving detailed results to {output_csv}...")
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['address_hex', 'address_dec', 'trigger_counter', 'function_name', 
                         'demangled_name', 'fault_group', 'fault_index', 'fault_type', 
                         'fault_model', 'fault_address', 'fault_mask', 'fault_lifespan']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        # Summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        
        function_counts = {}
        unknown_count = 0
        
        for addr, func in address_to_function.items():
            if func:
                function_counts[func] = function_counts.get(func, 0) + 1
            else:
                unknown_count += 1
        
        print(f"Total trigger addresses: {len(trigger_data)}")
        print(f"Unknown functions: {unknown_count}")
        print(f"Functions with trigger addresses:")
        
        for func, count in sorted(function_counts.items(), key=lambda x: x[1], reverse=True):
            demangled = demangle_function_name(func)
            print(f"  {demangled}: {count} trigger(s)")
        
        # Additional analysis
        print(f"\nDetailed analysis saved to: {output_csv}")
        print(f"You can analyze the CSV file for more detailed fault injection patterns.")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file} - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
