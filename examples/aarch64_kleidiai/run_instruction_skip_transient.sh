#!/bin/sh
python3 ../../controller.py --debug --fault fault_instruction_skip_transient.json  --qemu qemuconf.json output_instruction_skip_transient.hdf5 -o
rm log_*
