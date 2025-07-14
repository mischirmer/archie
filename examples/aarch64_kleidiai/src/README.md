# Bare Metal AArch64 Project

This project demonstrates a bare metal AArch64 application that uses the main function from `matmul_clamp_f16_f16_f16p_tiled.cpp` as the entry point. It's designed to run in QEMU system aarch64.

## Project Structure

- `kleidi/matmul_clamp_f16_f16_f16p_tiled.cpp` - Main C++ file containing the entry point
- `kleidi/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c` - Matrix multiplication implementation
- `kleidi/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c` - Matrix packing implementation
- `kleidi/kai/kai_common.h` - Common header with bare metal compatibility
- `startup.s` - Custom assembly startup code that calls main and enables NEON FP16
- `link_script_custom.ld` - Custom linker script for bare metal AArch64
- `Makefile` - Build configuration
- `build/` - Build directory containing all object files

## Building

The project uses the ARM GNU Toolchain for bare metal compilation:

```bash
make clean
make
```

This produces `test64.elf` which is suitable for QEMU system aarch64. All object files are placed in the `build/` directory to keep the source directories clean.

## Running

### Using QEMU System AArch64

```bash
# Using the provided script (recommended)
./run_qemu.sh [timeout_seconds]

# Or manually
qemu-system-aarch64 -machine virt -cpu cortex-a53 -kernel test64.elf -nographic
```

### Exit QEMU
- Press `Ctrl+A` then `X` to exit QEMU manually
- Or the program will exit automatically when complete

## Key Features

- **Bare Metal Compatible**: Uses custom startup assembly code, no standard library dependencies
- **QEMU Ready**: Entry point at 0x40000000, compatible with QEMU virt machine
- **Custom Memory Layout**: Linker script defines appropriate memory sections
- **Simple Startup**: Custom `startup.s` directly calls main without CRT overhead
- **Error Handling**: Custom error handling suitable for bare metal environments
- **Clean Build System**: All object files are organized in a separate `build/` directory

## Technical Details

- **Target**: AArch64 (Cortex-A53 with FP16 support)
- **Entry Point**: `_Reset` (custom assembly startup)
- **Main Function**: Located in `matmul_clamp_f16_f16_f16p_tiled.cpp`
- **Memory Base**: 0x40000000 (QEMU virt machine RAM)
- **Stack Size**: 64KB
- **Compiler**: ARM GCC 14.2.1 with `-nostdlib -ffreestanding`
- **Linker**: Direct LD linking without CRT

## Dependencies

- ARM GNU Toolchain (aarch64-none-elf-*)
- QEMU system aarch64 (for testing)

## Notes

- The project removes all dependencies on standard C library functions
- Custom implementations are provided for necessary functions like `memcpy`
- All `printf` calls are commented out for bare metal compatibility
- The program runs matrix multiplication tests and returns success/failure status
- Uses custom assembly startup instead of CRT for minimal overhead
- When main returns, the program enters an infinite loop (typical for bare metal)
