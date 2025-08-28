# Cross-compilation toolchain for aarch64
CC = aarch64-none-elf-gcc
CXX = aarch64-none-elf-g++
AS = aarch64-none-elf-as
LD = aarch64-none-elf-ld

# Build directory
BUILD_DIR = build

# Compiler flags for bare metal (without newlib)
CFLAGS = -mcpu=cortex-a53+fp16 -nostdlib -ffreestanding -IxxHash -DXXH_NO_MALLOC -DXXH_STATIC_LINKING_ONLY -DXXH_NO_STREAM
CXXFLAGS = -mcpu=cortex-a53+fp16 -nostdlib -ffreestanding -fno-exceptions -fno-rtti -std=c++11 -IxxHash -DXXH_NO_MALLOC -DXXH_STATIC_LINKING_ONLY -DXXH_NO_STREAM

# Source files
KLEIDI_DIR = kleidi
KLEIDI_C_SOURCES = $(KLEIDI_DIR)/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c \
                   $(KLEIDI_DIR)/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c \
                   xxHash/xxhash.c \
                   baremetal_libc.c

KLEIDI_CPP_SOURCES = $(KLEIDI_DIR)/matmul_clamp_f16_f16_f16p_tiled.cpp

# Assembly startup file
STARTUP_ASM = startup.s

# Map source files to build directory object files
C_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(KLEIDI_C_SOURCES))
CPP_OBJECTS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(KLEIDI_CPP_SOURCES))
STARTUP_OBJECT = $(BUILD_DIR)/startup.o
OBJECTS = $(STARTUP_OBJECT) $(C_OBJECTS) $(CPP_OBJECTS)

# Default target
all: kleidiai_test.elf

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(KLEIDI_DIR)
	mkdir -p $(BUILD_DIR)/xxHash

kleidiai_test.elf: link_script.ld $(OBJECTS)
	$(LD) -T link_script.ld $(OBJECTS) -o kleidiai_test.elf

# Rule for assembly files
$(BUILD_DIR)/startup.o: startup.s | $(BUILD_DIR)
	$(AS) -o $@ $<

# Rule for C files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Rule for C++ files
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) kleidiai_test.elf

.PHONY: all clean
