import random
import json
import tables
import pandas as pd

def read_tbinfo_from_hdf5(hdf5_file_path):
    """
    Read the tbinfo table from the HDF5 file and return as pandas DataFrame
    """
    with tables.open_file(hdf5_file_path, mode='r') as f:
        # Navigate to the tbinfo table in the Goldenrun group
        tbinfo_table = f.root.Goldenrun.tbinfo
        
        # Read all data into a pandas DataFrame
        tbinfo_data = []
        for row in tbinfo_table.iterrows():
            tbinfo_data.append({
                'identity': row['identity'],
                'size': row['size'], 
                'ins_count': row['ins_count'],
                'num_exec': row['num_exec'],
                'assembler': row['assembler'].decode('utf-8') if isinstance(row['assembler'], bytes) else row['assembler']
            })
    
    return pd.DataFrame(tbinfo_data)

def is_address_excluded(address, exclude_ranges):
    """Check if an address falls within any of the excluded ranges"""
    if exclude_ranges is None:
        return False
    
    for start_exclude, end_exclude in exclude_ranges:
        if start_exclude <= address <= end_exclude:
            return True
    return False

def generate_fault_config_from_tbinfo(
    hdf5_file_path, start_addr, end_addr, sample_count=None, reg_min=0, reg_max=31, word_size=64, exclude_ranges=None
):
    instruction_size = 4  # assuming 4-byte instructions
    fault_lifespan = [1]
    fault_type = "instruction"
    fault_model = "overwrite"
    nop_instruction = 0xD503201F  # AArch64 NOP instruction
    
    # Read tbinfo data from HDF5 file
    tbinfo_df = read_tbinfo_from_hdf5(hdf5_file_path)
    
    # First, calculate the total fault space size and build instruction execution list
    instruction_executions = []
    total_instructions = 0
    excluded_instructions = 0
    
    for _, row in tbinfo_df.iterrows():
        block_start_addr = row['identity']
        ins_count = row['ins_count']
        num_exec = row['num_exec']
        
        # Generate instruction execution entries
        for ins_offset in range(ins_count):
            instruction_addr = block_start_addr + (ins_offset * instruction_size)
            
            # Check if this instruction address should be excluded
            if is_address_excluded(instruction_addr, exclude_ranges):
                excluded_instructions += num_exec  # Count all executions of this instruction as excluded
                continue
            
            for exec_count in range(1, num_exec + 1):
                instruction_executions.append({
                    'instruction_addr': instruction_addr,
                    'exec_count': exec_count
                })
                total_instructions += 1
    
    total_fault_space = total_instructions  # One fault per instruction execution
    
    print(f"Total instruction executions: {total_instructions}")
    if excluded_instructions > 0:
        print(f"Excluded instruction executions: {excluded_instructions}")
        if exclude_ranges:
            print(f"Excluded address ranges: {[hex(x[0]) + '-' + hex(x[1]) for x in exclude_ranges]}")
    print(f"Total fault space size (instruction executions): {total_fault_space}")
    print(f"Each instruction execution will be overwritten with NOP (0x{nop_instruction:08X})")
    
    # Generate faults efficiently
    if sample_count is not None and sample_count < total_fault_space:
        print(f"Randomly sampling {sample_count} faults from the complete space...")
        faults = generate_random_faults(instruction_executions, sample_count, fault_type, fault_model, fault_lifespan, nop_instruction)
    else:
        print(f"Generating all {total_fault_space} possible faults...")
        faults = generate_all_faults(instruction_executions, fault_type, fault_model, fault_lifespan, nop_instruction)
    
    return faults, total_instructions, total_fault_space


def generate_random_faults(instruction_executions, sample_count, fault_type, fault_model, fault_lifespan, nop_instruction):
    """Generate a random sample of instruction faults without duplicates"""
    faults = []
    used_faults = set()
    attempts = 0
    max_attempts = sample_count * 10  # Prevent infinite loop
    
    print("Generating random instruction faults (no duplicates)...")
    while len(faults) < sample_count and attempts < max_attempts:
        # Randomly select an instruction execution
        instruction_exec = random.choice(instruction_executions)
        
        # Create unique identifier for this fault (address + execution count)
        fault_id = (instruction_exec['instruction_addr'], instruction_exec['exec_count'])
        
        if fault_id not in used_faults:
            used_faults.add(fault_id)
            
            fault_entry = [{
                "fault_address": [instruction_exec['instruction_addr']],
                "fault_type": fault_type,
                "fault_model": fault_model,
                "num_bytes": 4,
                "fault_lifespan": fault_lifespan,
                "fault_mask": [nop_instruction],
                "trigger_address": [-1],
                "trigger_counter": [instruction_exec['exec_count']]
            }]
            faults.append(fault_entry)
        
        attempts += 1
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts. Generated {len(faults)} unique faults out of {sample_count} requested.")
    
    return faults


def generate_all_faults(instruction_executions, fault_type, fault_model, fault_lifespan, nop_instruction):
    """Generate all possible instruction faults systematically"""
    faults = []
    
    print("Generating all instruction faults systematically...")
    for i, instruction_exec in enumerate(instruction_executions):
        if i % 10000 == 0:
            print(f"Processing instruction execution {i}/{len(instruction_executions)}")
        
        fault_entry = [{
            "fault_address": [instruction_exec['instruction_addr']],
            "fault_type": fault_type,
            "fault_model": fault_model,
            "num_bytes": 4,
            "fault_lifespan": fault_lifespan,
            "fault_mask": [nop_instruction],
            "trigger_address": [-1],
            "trigger_counter": [instruction_exec['exec_count']]
        }]
        faults.append(fault_entry)
    
    return faults


def create_fault_config(faults, start_addr, end_addr):
    """Create the complete fault configuration"""
    config = {
        "max_instruction_count": 10000,
        "start": {
            "address": start_addr,
            "counter": 1
        },
        "end": {
            "address": end_addr,
            "counter": 1
        },
        "faults": faults,
        "memorydump": [
            {
                "address": 0x40103a58,
                "length": 24
            }
        ],
        "mem_info": True
    }
    return config

# Example usage
hdf5_file_path = "output/output_goldenrun.hdf5"
start_address = 0x40007ee4
end_address = 0x40008740
sample_count = 17000 # 7000 is enough for statistical analysis

# Define address ranges to exclude from fault injection
# Format: list of tuples (start_addr, end_addr)
exclude_ranges = [
    (0x40007ee4, 0x40007f54),  # Exclude before loop start
    (0x40008600, 0x40008740), # Exclude after loop end,
    
    #(0x40007ee4, 0x400084a8),  # Exclude TestCase Function

    (0x400070dc, 0x4000713c), # Exclude the Bias Filling Function
    
    (0x4000314c, 0x40006e30), # Exclude Hash Library
    (0x40007034, 0x40007068), # Calculate Hash Function
    #(0x40007140, 0x40007574), # OC
    #(0x40007578, 0x40007770), # IC
]

faults, total_instructions, total_fault_space = generate_fault_config_from_tbinfo(
    hdf5_file_path, start_address, end_address, sample_count, exclude_ranges=exclude_ranges
)

config = create_fault_config(faults, start_address, end_address)

# Save to JSON file
with open("configs/instruction_transient_.json", "w") as f:
    json.dump(config, f, indent=4)

print(f"\nSUMMARY:")
print(f"Total instruction executions: {total_instructions}")
print(f"Total fault space size: {total_fault_space:,}")
print(f"Each fault overwrites one instruction execution with NOP (0xD503201F)")
print(f"Generated fault config with {len(config['faults']):,} faults with no duplicates (ensured during generation)")
print(f"Saved to: output/instruction_transient_.json")
