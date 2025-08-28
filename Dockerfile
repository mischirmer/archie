FROM ubuntu:22.04

# Install necessary tools
RUN apt-get update && apt-get install -y \
    git \
    bash \
    build-essential \
    ninja-build \
    libglib2.0-dev \
    libfdt-dev \
    libpixman-1-dev \
    zlib1g-dev \
    libprotobuf-c-dev \
    python3 \
    python3-pip \
    libcap-dev \
    && rm -rf /var/lib/apt/lists/*


# Install aarch64-none-elf toolchain
WORKDIR /archie
RUN apt-get update && apt-get install -y wget nano && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.arm.com/-/media/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-x86_64-aarch64-none-elf.tar.xz -O /tmp/toolchain.tar.xz
RUN tar -xf /tmp/toolchain.tar.xz -C /opt/
RUN rm /tmp/toolchain.tar.xz
ENV PATH="/opt/arm-gnu-toolchain-14.3.rel1-x86_64-aarch64-none-elf/bin:${PATH}"

# Install Requirements before copying the rest of the code
# This allows Docker to cache the layer if requirements don't change
WORKDIR /archie
COPY requirements.txt /archie/requirements.txt
RUN pip3 install -r requirements.txt

# Copy everything except examples/aarch64_kleidiai to cache layers
COPY build.sh calculate_trigger.py controller.py fault-readme.md faultclass.py goldenrun.py hdf5-readme.md hdf5logger.py LICENSE README.md requirements.txt util.py /archie/
COPY .git/ /archie/.git/
COPY analysis/ /archie/analysis/
COPY examples/aarch64/ /archie/examples/aarch64/
COPY examples/aarch64-softmmu/ /archie/examples/aarch64-softmmu/
COPY examples/riscv64/ /archie/examples/riscv64/
COPY examples/stm32/ /archie/examples/stm32/
COPY examples/stm32-timeout-wfi/ /archie/examples/stm32-timeout-wfi/
COPY faultplugin/ /archie/faultplugin/
COPY protobuf/ /archie/protobuf/
COPY qemu/ /archie/qemu/
RUN git submodule update --init

# Build QEMU
RUN ./build.sh
WORKDIR /archie/qemu
RUN git checkout 1ab3c799b6fd23da29eb41a3accf8d053ee2d9cc
RUN mkdir -p qemu/build/debug
WORKDIR /archie/qemu/build/debug
RUN ./../../configure --target-list=arm-softmmu,aarch64-softmmu,riscv64-softmmu --enable-debug --enable-plugins --disable-sdl --disable-gtk --disable-curses --disable-vnc
RUN make -j $(nproc)

# Build fault plugin
WORKDIR /archie/faultplugin
RUN make clean && make


# Copy the aarch64_kleidiai example folder
COPY examples/aarch64_kleidiai/ /archie/examples/aarch64_kleidiai/
COPY examples/aarch64_kleidiai_rowcol/ /archie/examples/aarch64_kleidiai_rowcol/
COPY examples/aarch64_kleidiai_rowcol_resnet/ /archie/examples/aarch64_kleidiai_rowcol_resnet/
COPY examples/aarch64_kleidiai_rowcol_resnet_baseline/ /archie/examples/aarch64_kleidiai_rowcol_resnet_baseline/
COPY examples/aarch64_kleidiai_rowcol_resnet_conv13/ /archie/examples/aarch64_kleidiai_rowcol_resnet_conv13/
COPY examples/aarch64_kleidiai_rowcol_resnet_conv13_baseline/ /archie/examples/aarch64_kleidiai_rowcol_resnet_conv13_baseline/

WORKDIR /archie/examples/aarch64_kleidiai/src
# RUN make
# Copy the Precompiled ELF to ensure correct addresses
COPY examples/aarch64_kleidiai/src/kleidiai_test.elf examples/aarch64_kleidiai/src/

WORKDIR /archie/examples/aarch64_kleidiai/
RUN chmod +x run_instruction_skip.sh run_weight_tampering.sh

# Test Case
RUN chmod +x run_minimal.sh && ./run_minimal.sh

RUN pip install h5py
WORKDIR /archie/examples/aarch64_kleidiai_rowcol_resnet_conv13/

ENTRYPOINT ["/bin/bash"]
