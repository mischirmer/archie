#!/bin/sh
python3 ../../controller.py --debug --fault fault_instruction_skip.json  --qemu qemuconf.json output_instruction_skip.hdf5 -o
rm log_*
