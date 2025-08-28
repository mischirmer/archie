#!/bin/sh
python3 ../../controller.py --fault configs/fault_instruction_skip.json  --qemu qemuconf.json output/output_instruction_skip_goldenrun.hdf5 --overwrite --worker 40
