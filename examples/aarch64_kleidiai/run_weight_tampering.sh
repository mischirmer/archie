#!/bin/sh
python3 ../../controller.py --debug --fault fault_weight_tampering.json  --qemu qemuconf.json output_weight_tampering.hdf5 -o
rm log_*
