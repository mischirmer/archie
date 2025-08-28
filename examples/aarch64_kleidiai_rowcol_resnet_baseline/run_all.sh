#!/bin/sh
# Instructions
python3 ../../controller.py --fault configs/instruction_permanent.json  --qemu qemuconf.json output/instructions/permanent.hdf5 -o --worker 64
python3 ../../controller.py --fault configs/instruction_transient.json  --qemu qemuconf.json output/instructions/transient.hdf5 -o --worker 64

# Bitflip in FMLA instruction
python3 ../../controller.py --fault configs/instruction_fmla_permanent.json  --qemu qemuconf.json output/instructions_fp/permanent.hdf5 -o --worker 64
python3 ../../controller.py --fault configs/instruction_fmla_transient.json  --qemu qemuconf.json output/instructions_fp/transient.hdf5 -o --worker 64


# Registers
python3 ../../controller.py --fault configs/registers_permanent.json  --qemu qemuconf.json output/registers/permanent.hdf5 -o --worker 64
python3 ../../controller.py --fault configs/registers_transient.json  --qemu qemuconf.json output/registers/transient.hdf5 -o --worker 64
