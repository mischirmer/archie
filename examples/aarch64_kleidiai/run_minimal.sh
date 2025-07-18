#!/bin/sh
python3 ../../controller.py --debug --fault fault_minimal.json  --qemu qemuconf.json output_minimal.hdf5 -o
rm log_*
