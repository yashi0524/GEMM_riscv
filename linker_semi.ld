/* linker.ld — bare-metal linker script for dgemm_riscv on gem5 RISC-V M-mode
 *
 * Memory map (matching existing binary layout observed from dgemm_riscv.dis):
 *
 *   0x00010000  FLASH/ROM  — code + rodata            (existing binary base)
 *   0x00020490  RAM start  — data, BSS                (just above existing BSS end 0x20488)
 *   0x00020490  heap start — grows upward
 *   0x00120490  stack top  — grows downward (1 MB stack)
 *
 * gem5 RiscvBareMetal ELF loader maps segments at their linked addresses,
 * so RAM must start at 0x0 in gem5 config (or use --mem-start=0x0).
 * Alternatively, relink at 0x80000000; see note at bottom.
 *
 * HTIF tohost/fromhost must be 64-byte aligned for gem5 to detect them.
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    /* Code + read-only data — loaded by gem5 ELF loader */
    FLASH (rx)  : ORIGIN = 0x00010000, LENGTH = 0x00020000   /* 128 KB */

    /* Read-write data, BSS, heap, and stack */
    RAM   (rwx) : ORIGIN = 0x00030000, LENGTH = 0x00200000   /* 2 MB   */
}

/*
 * Stack configuration
 *   _STACK_SIZE : size of the stack in bytes (1 MB default)
 *   Grows downward from _stack_top toward _stack_bottom.
 */
_STACK_SIZE = 0x100000;   /* 1 MB */

/*
 * Heap configuration
 *   Grows upward from _heap_start toward _heap_end.
 *   _HEAP_SIZE : maximum heap size (512 KB default; tune for your workload)
 */
_HEAP_SIZE  = 0x080000;   /* 512 KB */

SECTIONS
{
    /* ------------------------------------------------------------------ */
    /* .text.init MUST come first so _start lands at the ELF entry point.  */
    /* ------------------------------------------------------------------ */
    .text.init :
    {
        KEEP(*(.text.init))
    } > FLASH

    /* ------------------------------------------------------------------ */
    /* Remaining code                                                       */
    /* ------------------------------------------------------------------ */
    .text :
    {
        *(.text)
        *(.text.*)
        *(.gnu.linkonce.t.*)
    } > FLASH

    /* ------------------------------------------------------------------ */
    /* Read-only data                                                       */
    /* ------------------------------------------------------------------ */
    .rodata :
    {
        *(.rodata)
        *(.rodata.*)
        *(.gnu.linkonce.r.*)
    } > FLASH

    /* ------------------------------------------------------------------ */
    /* HTIF communication region — must be 64-byte aligned.                */
    /* gem5 scans the ELF symbol table for "tohost" and "fromhost".        */
    /* ------------------------------------------------------------------ */
    .tohost :
    {
        . = ALIGN(64);
        KEEP(*(.tohost))
    } > FLASH

    /* ------------------------------------------------------------------ */
    /* C++ init / fini arrays                                               */
    /* ------------------------------------------------------------------ */
    .preinit_array :
    {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } > FLASH

    .init_array :
    {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN(__init_array_end = .);
    } > FLASH

    .fini_array :
    {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > FLASH

    /* ------------------------------------------------------------------ */
    /* Initialized data — loaded from FLASH, copied to RAM at startup.     */
    /* (start.S does NOT copy .data; add a copy loop if you use globals.)  */
    /* ------------------------------------------------------------------ */
    .data :
    {
        . = ALIGN(8);
        _data_start = .;
        *(.data)
        *(.data.*)
        *(.gnu.linkonce.d.*)

        /* RISC-V small-data: without these, the linker orphans any
         * *(.sdata*) input section (gcc's gp-relative optimization for
         * small globals — this is where newlib puts _impure_ptr,
         * __malloc_*, __lock_* etc.) into its own PROGBITS section
         * placed right after .data but OUTSIDE [_data_start,_data_end),
         * so start.S's copy loop below never initializes it: it silently
         * reads back as zero at runtime instead of its real FLASH value,
         * which nulls out _impure_ptr and breaks all buffered stdio
         * (printf/puts silently produce no output, no crash) without
         * ever touching .text/.rodata/ordinary .data. */
        *(.sdata)
        *(.sdata.*)
        *(.gnu.linkonce.s.*)

        /* Global pointer anchor — must be here for linker relaxation. */
        . = ALIGN(8);
        PROVIDE(__global_pointer$ = . + 0x800);

        _data_end = .;
    } > RAM AT > FLASH

    _data_load = LOADADDR(.data);   /* source address in FLASH */

    /* ------------------------------------------------------------------ */
    /* BSS — zero-initialized; start.S zeroes this region at boot.         */
    /* ------------------------------------------------------------------ */
    .bss :
    {
        . = ALIGN(8);
        _bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)

        /* RISC-V small-bss counterpart to .sdata above (see comment
         * there) — errno, __lock_* mutexes, __stdio init flag, malloc
         * bookkeeping all live here. Must be inside [_bss_start,_bss_end)
         * so start.S's zeroing loop actually covers them. */
        *(.sbss)
        *(.sbss.*)
        *(.gnu.linkonce.sb.*)
        *(.scommon)

        . = ALIGN(8);
        _bss_end = .;

        /* newlib reent symbols — not exported by all newlib builds */
        PROVIDE(_global_impure_ptr = 0x1fba0);
        PROVIDE(_impure_ptr        = 0x1fba0);
    } > RAM

    /* ------------------------------------------------------------------ */
    /* Heap — grows upward from _heap_start.                               */
    /* ------------------------------------------------------------------ */
    .heap :
    {
        . = ALIGN(16);
        _heap_start = .;
        . += _HEAP_SIZE;
        _heap_end = .;
    } > RAM

    /* ------------------------------------------------------------------ */
    /* Stack — grows downward from _stack_top.                             */
    /* _stack_top is loaded into sp as the very first instruction.         */
    /* ------------------------------------------------------------------ */
    .stack :
    {
        . = ALIGN(16);
        _stack_bottom = .;
        . += _STACK_SIZE;
        . = ALIGN(16);
        _stack_top = .;             /* sp is initialized to this address */
    } > RAM

    /* ------------------------------------------------------------------ */
    /* Discard unwanted sections                                            */
    /* ------------------------------------------------------------------ */
    /DISCARD/ :
    {
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note.*)
        *(.comment)
        *(.debug*)
    }
}

/*
 * NOTE — relinking at 0x80000000 (gem5 default reset target):
 *
 * If you want gem5 to boot directly to _start without a boot ROM,
 * change FLASH ORIGIN to 0x80000000 and update gem5 mem config:
 *
 *   system.mem_ranges = [AddrRange("2GB")]
 *
 * Then compile with:
 *   riscv64-unknown-elf-gcc ... -T linker.ld -Wl,--no-relax ...
 *
 * The current layout (0x10000 base) matches the existing dgemm_riscv binary.
 */
