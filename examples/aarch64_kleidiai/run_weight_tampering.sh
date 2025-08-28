#!/bin/sh
python3 ../../controller.py --fault configs/fault_weight_tampering.json  --qemu qemuconf.json output/output_weight_tampering.hdf5 -o
