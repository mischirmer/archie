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
RUN mkdir /archie/examples/aarch64_kleidiai/src/build

COPY examples/aarch64_kleidiai/src/. examples/aarch64_kleidiai/src/
RUN rm -rf examples/aarch64_kleidiai/src/build

COPY examples/aarch64_kleidiai/*.json examples/aarch64_kleidiai/
COPY examples/aarch64_kleidiai/run.sh examples/aarch64_kleidiai/

WORKDIR /archie/examples/aarch64_kleidiai/src
RUN make

# Copy the Precompiled ELF to ensure correct addresses
COPY examples/aarch64_kleidiai/src/kleidiai_test.elf examples/aarch64_kleidiai/src/build/

WORKDIR /archie/examples/aarch64_kleidiai/
RUN chmod +x run.sh

# Test Case
WORKDIR /archie/examples/stm32
RUN ./run.sh