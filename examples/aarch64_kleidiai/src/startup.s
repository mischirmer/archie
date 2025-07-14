.global _Reset
_Reset:
    // Enable FP/NEON by setting CPACR_EL1.FPEN bits
    mrs x0, CPACR_EL1
    orr x0, x0, #(3 << 20)  // Set FPEN bits (20-21) to 11
    msr CPACR_EL1, x0
    isb  // Instruction synchronization barrier
    
    ldr x30, =stack_top	// setup stack
    mov sp, x30
    bl main
    b .
    