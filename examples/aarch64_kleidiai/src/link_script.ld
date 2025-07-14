ENTRY(_start)
SECTIONS
{
	. = 0x40000000;  /* Standard QEMU virt machine RAM base */
	PROVIDE(_start = .);  /* Provide fallback _start symbol */
	.text : { 
		KEEP(*(.text.start))  /* Ensure _start symbol comes first and isn't removed */
		*(.text) 
	}
	.rodata : { *(.rodata*) }
	.data : { *(.data) }
	.bss : { 
		__bss_start__ = .;
		*(.bss COMMON) 
		__bss_end__ = .;
	}
	. = ALIGN(8);
	. = . + 0x10000; /* 64kB of stack memory for QEMU */
	stack_top = .;
	end = .; /* Symbol required by newlib for sbrk */
}