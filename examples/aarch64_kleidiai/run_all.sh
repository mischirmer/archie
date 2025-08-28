#!/bin/sh
# Instructions
python3 ../../controller.py --fault configs/instruction_permanent.json  --qemu qemuconf.json output/instructions/permanent.hdf5 -o --worker 40
python3 ../../controller.py --fault configs/instruction_transient.json  --qemu qemuconf.json output/instructions/transient.hdf5 -o --worker 40

# Registers
python3 ../../controller.py --fault configs/registers_permanent.json  --qemu qemuconf.json output/registers/permanent.hdf5 -o --worker 40
python3 ../../controller.py --fault configs/registers_transient.json  --qemu qemuconf.json output/registers/transient.hdf5 -o --worker 40
