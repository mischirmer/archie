ENTRY(_start)

/* Memory layout for QEMU virt machine (default 128MB RAM) */
MEMORY {
    RAM : ORIGIN = 0x40000000, LENGTH = 128M
}

SECTIONS
{
	. = 0x40000000;  /* Standard QEMU virt machine RAM base */
	PROVIDE(_start = .);  /* Provide fallback _start symbol */
	.text : { 
		KEEP(*(.text.start))  /* Ensure _start symbol comes first and isn't removed */
		*(.text) 
	} > RAM
	.rodata : { *(.rodata*) } > RAM
	.data : { *(.data) } > RAM
	.bss : { 
		__bss_start__ = .;
		*(.bss COMMON) 
		__bss_end__ = .;
	} > RAM
	. = ALIGN(8);
	. = . + 0x800000; /* 1MB of stack memory (increased for large data) */
	stack_top = .;
	end = .; /* Symbol required by newlib for sbrk */
	
	/* Ensure we don't exceed available RAM */
	ASSERT(. <= 0x40000000 + 128M, "Program too large for available RAM")
}