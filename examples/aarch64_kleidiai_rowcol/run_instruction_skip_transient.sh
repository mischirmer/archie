#!/bin/sh
python3 ../../controller.py --fault configs/fault_instruction_skip_transient.json  --qemu qemuconf.json output/output_instruction_skip_transient.hdf5 --overwrite --worker 40
rm log_*
