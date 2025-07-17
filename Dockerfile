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


RUN git clone https://github.com/JoGei/archie.git /archie

WORKDIR /archie

RUN git submodule update --init
RUN pip3 install -r requirements.txt
RUN chmod +x build.sh && ./build.sh

# Build QEMU
RUN mkdir -p qemu/build/debug
WORKDIR /archie/qemu/build/debug
RUN ./../../configure --target-list=arm-softmmu,aarch64-softmmu,riscv64-softmmu --enable-debug --enable-plugins --disable-sdl --disable-gtk --disable-curses --disable-vnc
RUN make -j $(nproc)

# Build fault plugin
WORKDIR /archie/faultplugin
RUN make

WORKDIR /archie



# Install aarch64-none-elf toolchain
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.arm.com/-/media/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-x86_64-aarch64-none-elf.tar.xz -O /tmp/toolchain.tar.xz
RUN tar -xf /tmp/toolchain.tar.xz -C /opt/
RUN rm /tmp/toolchain.tar.xz
ENV PATH="/opt/arm-gnu-toolchain-14.3.rel1-x86_64-aarch64-none-elf/bin:${PATH}"

# Copy the build source of the kleidi example
RUN mkdir /archie/examples/aarch64_kleidiai
RUN mkdir /archie/examples/aarch64_kleidiai/src
COPY examples/aarch64_kleidiai/src/kleidi examples/aarch64_kleidiai/src/kleidi
COPY examples/aarch64_kleidiai/src/link_script.ld examples/aarch64_kleidiai/src/
COPY examples/aarch64_kleidiai/src/Makefile examples/aarch64_kleidiai/src/
COPY examples/aarch64_kleidiai/src/startup.s examples/aarch64_kleidiai/src/
COPY examples/aarch64_kleidiai/fault.json examples/aarch64_kleidiai/
COPY examples/aarch64_kleidiai/qemuconf.json examples/aarch64_kleidiai/
COPY examples/aarch64_kleidiai/run.sh examples/aarch64_kleidiai/

WORKDIR /archie/examples/aarch64_kleidiai/src
RUN make

WORKDIR /archie/examples/aarch64_kleidiai/
RUN chmod +x run.sh