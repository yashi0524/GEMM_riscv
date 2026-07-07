# Makefile — dgemm_riscv bare-metal for gem5 RISC-V M-mode
# Output via NS16550A UART at 0x10000000, viewed with m5term

# --- Compiler and Tools ---
CC := clang-18
LD := lld-18

# --- Paths (TOOLCHAIN / LIBC_DIR / GCC_LIB_DIR from environment) ---
SRC_DIR  := src
INC_DIR  := inc
STARTUP  := start_semi.S
LDSCRIPT := linker_semi.ld
OUT_DIR  := test

# --- Target Architecture ---
# MARCH: override on the command line to add extensions
#   double : rv64gcv  (default)
#   __bf16 : rv64gcv_zvfbfmin1p0_zvfbfwma1p0
MARCH        := rv64gcv

TARGET_FLAGS := --target=riscv64-unknown-elf \
                -march=$(MARCH) \
                -menable-experimental-extensions \
                -mabi=lp64d

# --- Toolchain Paths ---
SYSROOT_FLAGS := --sysroot=$(TOOLCHAIN)/riscv-none-elf \
                 --gcc-toolchain=$(TOOLCHAIN) \
                 -B$(LIBC_DIR) \
                 -B$(GCC_LIB_DIR)

# --- Base flags shared by every pattern ---
BASE_FLAGS := $(TARGET_FLAGS) $(SYSROOT_FLAGS) -I$(INC_DIR) \
              -O3 \
              -Rpass=loop-vectorize \
              -fno-asynchronous-unwind-tables \
              -fno-unwind-tables

# --- Per-benchmark extras; set via target-specific vars below or command line ---
BENCH_EXTRA_FLAGS :=

# --- Linker Flags ---
# -nostdlib      : suppress all default libs/crt0
# start.S first  : our _write/_exit win over newlib's
# -lc -lm -lgcc  : re-add only what we need
# --icf=none     : prevent lld folding identical code sequences
# --no-relax     : prevent linker relaxation breaking .option norvc sections
LDFLAGS := -L$(LIBC_DIR) \
           -L$(GCC_LIB_DIR) \
           -fuse-ld=$(LD) \
           -static \
           -nostdlib \
           -Wl,--icf=none \
           -Wl,--no-relax \
           -T $(LDSCRIPT)

# =============================================================
# Generic pattern rule  src/%.c  →  test/%_riscv  (+  _flags)
# Adding a new pattern: drop src/<name>.c, add 2 lines below
# =============================================================
$(OUT_DIR)/%_riscv: $(SRC_DIR)/%.c $(STARTUP) $(LDSCRIPT)
	$(CC) $(BASE_FLAGS) $(BENCH_EXTRA_FLAGS) $(LDFLAGS) \
	    -o $@ $(STARTUP) $< \
	    -lc -lm -lgcc
	echo -n "$(BASE_FLAGS) $(BENCH_EXTRA_FLAGS)" > build_flags.tmp
	riscv64-unknown-elf-objcopy --add-section .build_flags=build_flags.tmp \
		--set-section-flags .build_flags=readonly,data \
		$@ $@_flags
	rm -f build_flags.tmp

# =============================================================
# Per-benchmark flag overrides (target-specific variables)
# =============================================================
M            := 4
ITERS        := 10000
TARGET_FLOAT := double

$(OUT_DIR)/dgemm_riscv: BENCH_EXTRA_FLAGS = -DM=$(M) -mllvm -force-vector-width=8
$(OUT_DIR)/gemm_riscv:  BENCH_EXTRA_FLAGS = -DM=$(M) -Dtarget_float=$(TARGET_FLOAT) -mllvm -force-vector-width=8
$(OUT_DIR)/fmacc_riscv: BENCH_EXTRA_FLAGS = -DITERS=$(ITERS)
# fmacc_fp16 needs the zvfh extension for vector half-precision FMA; a second
# -march= wins over the base one (clang takes the last -march on the line).
$(OUT_DIR)/fmacc_fp16_riscv: BENCH_EXTRA_FLAGS = -march=rv64gcv_zvfh -DITERS=$(ITERS)

# =============================================================
# Convenience aliases
# =============================================================
all:        $(OUT_DIR)/dgemm_riscv
dgemm:      $(OUT_DIR)/dgemm_riscv
gemm:       $(OUT_DIR)/gemm_riscv
fmacc:      $(OUT_DIR)/fmacc_riscv
fmacc_fp16: $(OUT_DIR)/fmacc_fp16_riscv

# =============================================================
# Utility targets
# =============================================================

# Disassemble + verify _write/_exit; override with: make dis BIN=test/fmacc_riscv
BIN ?= $(OUT_DIR)/dgemm_riscv
dis: $(BIN)
	riscv64-unknown-elf-objdump -d -M no-aliases $(BIN) > $(BIN).dis
	@echo "=== _write (should show UART polling loop, no ecall) ==="
	@grep -A 25 "<_write>:" $(BIN).dis | head -30
	@echo "=== _exit (should show 0x4200007b) ==="
	@grep -A 5 "<_exit>:" $(BIN).dis | head -8

# Read embedded build flags; override with: make dump_flags BIN=test/fmacc_riscv
dump_flags:
	riscv64-unknown-elf-readelf -p .build_flags $(BIN)_flags

clean:
	rm -f $(OUT_DIR)/*_riscv $(OUT_DIR)/*_riscv_flags $(OUT_DIR)/*_riscv.dis

.PHONY: all dgemm fmacc fmacc_fp16 dis dump_flags clean
