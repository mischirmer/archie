# ARCHIE AArch64 Kleidi AI Example

This example demonstrates fault injection analysis for AArch64 Kleidi AI workloads.

## Build Instructions

To build the Docker container, navigate to the root level of the repository and run:

```bash
docker build -t qemu-archie .
```

## Running the Example

Once the container is built, you can run different fault injection experiments inside the container:

### Instruction Skip Transient Analysis
```bash
./run_instruction_skip_transient.sh
```
This script performs transient instruction error analysis by injecting faults that cause instructions to be skipped temporarily.

### Other Available Scripts
- `./run_instruction_skip.sh` - Standard instruction skip analysis
- `./run_weight_tampering.sh` - Weight tampering analysis  
- `./run_minimal.sh` - Minimal fault injection test

These scripts will execute the fault injection experiments and generate the analysis results in HDF5 format.

## Analysis

After running the experiments, you can analyze the results using the provided Python script:

```bash
# For ABFT analysis (uses output_instruction_skip_kernel.hdf5)
python3 analyze_hdf.py --type abft

# For Hash analysis (uses output_weight_tampering.hdf5)  
python3 analyze_hdf.py --type hash
```

The analysis script will provide detailed metrics about trigger correctness, detection rates, and identify critical cases where triggers were missed.