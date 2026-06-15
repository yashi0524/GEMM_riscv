# Makefile — dgemm_riscv bare-metal for gem5 RISC-V M-mode
# Output via NS16550A UART at 0x10000000, viewed with m5term

# --- Compiler and Tools ---
CC  := clang-18
LD  := lld-18

# --- Paths (set in environment or override here) ---
# TOOLCHAIN    := /path/to/riscv/toolchain
# LIBC_DIR     := /path/to/newlib/lib
# GCC_LIB_DIR  := /path/to/gcc/lib

SRC_DIR  := src
INC_DIR  := inc
STARTUP  := start_semi.S
LDSCRIPT := linker_semi.ld

# --- Target Architecture ---
TARGET_FLAGS := --target=riscv64-unknown-elf \
                -march=rv64gcv \
                -mabi=lp64d

# --- Toolchain Paths ---
SYSROOT_FLAGS := --sysroot=$(TOOLCHAIN)/riscv-none-elf \
                 --gcc-toolchain=$(TOOLCHAIN) \
                 -B$(LIBC_DIR) \
                 -B$(GCC_LIB_DIR)

# --- Compiler Flags ---
TUNING_FLAGS := -O3 \
                -mllvm -force-vector-width=8 \
                -Rpass=loop-vectorize \
                -fno-asynchronous-unwind-tables \
                -fno-unwind-tables

# Matrix dimensions args:
# by default M=N=4, user can use -DM to set M=N=K, or use -DM -DN -DK to set seprately
M :=4
TUNING_FLAGS += -DM=$(M)

#TUNING_FLAGS += -g -mllvm -force-vector-interleave=4

# --- Linker Flags ---
# -nostdlib        : suppress all default libs/crt0
# start.S first    : our _write/_exit win over newlib's
# -lc -lm -lgcc   : re-add only what we need, after our objects
# --icf=none       : prevent lld folding identical code sequences
# --no-relax       : prevent linker relaxation breaking .option norvc sections
LDFLAGS := -L$(LIBC_DIR) \
           -L$(GCC_LIB_DIR) \
           -fuse-ld=$(LD) \
           -static \
           -nostdlib \
           -Wl,--icf=none \
           -Wl,--no-relax \
           -T $(LDSCRIPT)

CFLAGS := $(TARGET_FLAGS) $(SYSROOT_FLAGS) $(TUNING_FLAGS) -I$(INC_DIR)

# --- Build ---
TARGET := dgemm_riscv
SRC    := $(SRC_DIR)/dgemm.c

all: $(TARGET) inject_flags

$(TARGET): $(SRC) $(STARTUP) $(LDSCRIPT)
	$(CC) $(CFLAGS) $(LDFLAGS) \
	    -o $@ $(STARTUP) $(SRC) \
    -lc -lm -lgcc

clean: clean_flags
	rm -f $(TARGET) $(TARGET).dis

# Disassemble and verify _write uses UART (sb to 0x10000000), not ecall
dis: $(TARGET)
	riscv64-unknown-elf-objdump -d -M no-aliases $(TARGET) > $(TARGET).dis
	@echo "=== _write (should show UART polling loop, no ecall) ==="
	@grep -A 25 "<_write>:" $(TARGET).dis | head -30
	@echo "=== _exit (should show 0x4200007b) ==="
	@grep -A 5 "<_exit>:" $(TARGET).dis | head -8

inject_flags:
	echo -n "$(CFLAGS)" > build_flags.tmp
	riscv64-unknown-elf-objcopy --add-section .build_flags=build_flags.tmp \
		--set-section-flags .build_flags=readonly,data \
		$(TARGET) $(TARGET)_flags
	rm -f build_flags.tmp

clean_flags:
	rm $(TARGET)_flags


.PHONY: all clean dis
