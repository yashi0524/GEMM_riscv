#!/usr/bin/env python3
"""Generate the hierarchical (DRAM + L1) roofline SVG for the FP16 / O3CPU
GEMM kernels (scalar_gemm, opt_gemm, opt_gemm_blocked), for
doc/gemm_analysis.md ("How to generate the roofline").

Reads every number from the run outputs -- nothing is hand-copied:

  per kernel (RUNS below), built with -DROOFLINE_KERNEL and run on gem5 O3
  with --debug-flags=O3PipeView:
    run log       "roofline <kernel> cold|warm: mcycle = N"  -> time per call
    stats.txt     dump 1 (cold call)  dcache overallMshrMisses -> DRAM line fills
                  dump 2 (evict)      dcache writebacks        -> DRAM write-backs
                  dump 3 (warm call)  dcache overallMshrMisses -> DRAM line fills
    pipeview      committed load/store micro-ops inside the kernel x access
                  width (vector micro-op = one 64 B register, flh = 2 B, ...)
                  -> L1 bytes per call
  ceilings:
    compute roof  fmacc_fp16 x16 run log (first block): total_ops x vl x 2 / mcycle
    L1            l1_bw run log: bytes loaded / mcycle
    DRAM          DDR3-1600 8x8 theoretical peak (same as test/sweep.py PEAK_BW)

Pure stdlib apart from script/analyze_o3_pipeview.py (trace parsing,
llvm-nm-18 symbol lookup). Static SVG, light surface, <title> tooltips on
points -- same conventions as gen_roofline_svg.py.

Usage:
  python3 script/gen_hroofline_svg.py   # prints the data table, writes the SVG
"""
import math
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "script"))
import analyze_o3_pipeview as pv  # noqa: E402

T = os.path.join(ROOT, "test")
M, N, K = 16, 64, 16                # FP16 shape, -DM=16 -DN=64 -DK=16
FLOPS = 2 * M * N * K
CLOCK_GHZ = 1.0
LINE = 64                           # cache line / one VLEN=512 vector register
DRAM_BW = 12.8                      # GB/s

# (label, symbol, binary, m5out dir, run log)
RUNS = [
    ("scalar_gemm",      "scalar_gemm",      "gemm_rl_fp16_scalar_riscv",  "m5out_rl_fp16_o3_scalar",  "rl_fp16_o3_scalar_run.log"),
    ("opt_gemm",         "opt_gemm",         "gemm_rl_fp16_opt_riscv",     "m5out_rl_fp16_o3_opt",     "rl_fp16_o3_opt_run.log"),
    ("opt_gemm_blocked", "opt_gemm_blocked", "gemm_rl_fp16_blocked_riscv", "m5out_rl_fp16_o3_blocked", "rl_fp16_o3_blocked_run.log"),
]
FMACC_LOG = os.path.join(T, "fmacc_fp16_o3_oplat6_run.log")
L1BW_LOG = os.path.join(T, "l1_bw_o3_run.log")
OUT_SVG = os.path.join(ROOT, "doc", "hroofline_fp16_o3.svg")


# ---- data collection --------------------------------------------------------
def first_int(pattern, text):
    m = re.search(pattern, text)
    if not m:
        sys.exit(f"pattern not found: {pattern}")
    return int(m.group(1))


def ceilings():
    fm = open(FMACC_LOG).read().split("--- FMACC")[1]          # first block = x16
    peak = (first_int(r"total_ops = (\d+)", fm) * first_int(r"vl = (\d+)", fm) * 2
            / first_int(r"mcycle = (\d+)", fm) * CLOCK_GHZ)
    l1 = open(L1BW_LOG).read()
    l1_bw = first_int(r"bytes loaded = (\d+)", l1) / first_int(r"mcycle = (\d+)", l1) * CLOCK_GHZ
    return peak, l1_bw


def stat(section, name):
    m = re.search(r"^system\.cpu\.dcache\." + re.escape(name) + r"\s+(\d+)", section, re.M)
    return int(m.group(1)) if m else 0       # gem5 omits zero-valued stats


MEM_W = {"b": 1, "h": 2, "w": 4, "d": 8}


def mem_bytes(mnemonic):
    m = mnemonic.replace("c_", "")
    if re.match(r"v[ls](e|se|ue|oxe|uxe)\d+_v", m):
        return LINE                          # one micro-op = one VLEN register
    s = re.match(r"f?[ls]([bhwd])(u|sp)?$", m)
    return MEM_W[s.group(1)] if s else 0


def l1_bytes(binary, symbol, trace):
    lo, hi = pv.kernel_ranges(binary, [symbol])[symbol]
    recs = sorted((r for r in pv.parse_trace(trace)
                   if lo <= r["pc"] < hi and r.get("retire", 0) > 0), key=lambda r: r["sn"])
    entries = [i for i, r in enumerate(recs) if r["pc"] == lo]
    # cold, warm-up, warm (+ the correctness-check call when symbol is scalar_gemm)
    if len(entries) < 3:
        sys.exit(f"{symbol}: expected >= 3 calls in trace, found {len(entries)}")
    entries.append(len(recs))
    per_call = []
    for a, b in ((entries[0], entries[1]), (entries[2], entries[3])):
        per_call.append(sum(mem_bytes(r["dis"].split()[0]) for r in recs[a:b]))
    return per_call                          # [cold, warm]


def collect():
    rows = []
    for label, symbol, binary, m5out, runlog in RUNS:
        log = open(os.path.join(T, runlog)).read()
        if "PASS" not in log:
            sys.exit(f"{label}: correctness check did not PASS")
        cyc = {s: first_int(rf"roofline {symbol} {s}: mcycle = (\d+)", log) for s in ("cold", "warm")}
        secs = open(os.path.join(T, m5out, "stats.txt")).read().split(
            "---------- Begin Simulation Statistics ----------")[1:]
        fills_cold = stat(secs[0], "overallMshrMisses::total")
        wb = stat(secs[1], "writebacks::total")
        fills_warm = stat(secs[2], "overallMshrMisses::total")
        l1c, l1w = l1_bytes(os.path.join(T, binary), symbol, os.path.join(T, m5out, "pipeview.trace"))
        q = {("dram", "cold"): (fills_cold + wb) * LINE, ("dram", "warm"): fills_warm * LINE,
             ("l1", "cold"): l1c, ("l1", "warm"): l1w}
        for (level, state), qb in q.items():
            rows.append({"kernel": label, "level": level, "state": state, "q": qb,
                         "ai": FLOPS / qb if qb else float("inf"),
                         "gflops": FLOPS / cyc[state] * CLOCK_GHZ, "mcycle": cyc[state]})
    return rows


# ---- plot -------------------------------------------------------------------
SURFACE    = "#fcfcfb"
TEXT_PRI   = "#0b0b0b"
TEXT_SEC   = "#52514e"
TEXT_MUTED = "#767671"
GRID       = "#e4e3dd"
AXIS       = "#b9b8b1"
ROOF       = "#52514e"
KERNEL_COLOR = {"scalar_gemm": "#2a78d6",        # dataviz categorical slot 1 (blue)
                "opt_gemm": "#eb6834",           # slot 2 (orange)
                "opt_gemm_blocked": "#1baf7a"}   # slot 3 (aqua)
LEVEL_NAME = {"dram": "DRAM", "l1": "L1"}

W = 900
ML, MR = 78, 40
PANEL_H, PANEL_GAP, TOP, BOTTOM = 330, 70, 70, 96
plotW = W - ML - MR
H = TOP + 2 * PANEL_H + PANEL_GAP + BOTTOM
X0, X1 = 0.1, 1000.0
Y0, Y1 = 1.0, 400.0
XT = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
YT = (1, 2, 5, 10, 20, 50, 100, 200, 400)


def px(ai):
    return ML + (math.log10(ai) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * plotW


def py(g, top):
    return top + PANEL_H - (math.log10(g) - math.log10(Y0)) / (math.log10(Y1) - math.log10(Y0)) * PANEL_H


def mark(level, x, y, color):
    """square = DRAM, circle = L1; a 2 px surface ring keeps overlaps separable"""
    if level == "dram":
        return (f'<rect x="{x-8:.1f}" y="{y-8:.1f}" width="16" height="16" rx="3" fill="{SURFACE}"/>'
                f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" rx="2" fill="{color}"/>')
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8.5" fill="{SURFACE}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}"/>')


def panel(svg, rows, state, top, peak, l1_bw, label_pos):
    bw = {"dram": DRAM_BW, "l1": l1_bw}
    title = {"cold": "Cold: first call (A, B, C evicted from the D-cache)",
             "warm": "Warm: second call (data and code already cached)"}[state]
    svg.append(f'<text x="{ML}" y="{top-12}" font-size="13" fill="{TEXT_PRI}" font-weight="700">{title}</text>')
    for v in XT:
        x = px(v)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+PANEL_H}" stroke="{GRID}"/>')
        svg.append(f'<text x="{x:.1f}" y="{top+PANEL_H+17}" font-size="11.5" fill="{TEXT_MUTED}" '
                   f'text-anchor="middle">{v:g}</text>')
    for v in YT:
        y = py(v, top)
        svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plotW}" y2="{y:.1f}" stroke="{GRID}"/>')
        svg.append(f'<text x="{ML-10}" y="{y+4:.1f}" font-size="11.5" fill="{TEXT_MUTED}" text-anchor="end">{v:g}</text>')
    svg.append(f'<rect x="{ML}" y="{top}" width="{plotW}" height="{PANEL_H}" fill="none" stroke="{AXIS}"/>')
    svg.append(f'<text x="22" y="{top+PANEL_H/2:.1f}" font-size="12.5" fill="{TEXT_SEC}" text-anchor="middle" '
               f'transform="rotate(-90 22 {top+PANEL_H/2:.1f})">GFLOP/s (log)</text>')

    # ceilings: compute roof (solid), DRAM slope (solid), L1 slope (dashed)
    svg.append(f'<line x1="{px(X0):.1f}" y1="{py(peak, top):.1f}" x2="{px(X1):.1f}" y2="{py(peak, top):.1f}" '
               f'stroke="{ROOF}" stroke-width="2"/>')
    svg.append(f'<text x="{px(X1)-6:.1f}" y="{py(peak, top)-7:.1f}" font-size="11.5" fill="{TEXT_SEC}" '
               f'text-anchor="end">compute {peak:.1f} GFLOP/s</text>')
    ang = -math.degrees(math.atan2(py(1, top) - py(10, top), px(10) - px(1)))
    for level, dash, off, note in (("dram", "", -7, "peak"), ("l1", ' stroke-dasharray="6 4"', -7, "measured")):
        ridge = peak / bw[level]
        xa = max(X0, Y0 / bw[level])
        svg.append(f'<line x1="{px(xa):.1f}" y1="{py(bw[level]*xa, top):.1f}" x2="{px(ridge):.1f}" '
                   f'y2="{py(peak, top):.1f}" stroke="{ROOF}" stroke-width="1.5"{dash}/>')
        xl = ridge / 6 if level == "dram" else X0 * 1.3
        lx, ly = px(xl), py(bw[level] * xl, top) + off
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" fill="{TEXT_SEC}" '
                   f'transform="rotate({ang:.1f} {lx:.1f} {ly:.1f})">'
                   f'{LEVEL_NAME[level]} {bw[level]:.1f} GB/s ({note})</text>')

    # per kernel: thin link DRAM <-> L1 (same FLOPs/time, different bytes), then marks
    for kernel, color in KERNEL_COLOR.items():
        pts = {r["level"]: r for r in rows if r["kernel"] == kernel and r["state"] == state}
        y = py(pts["l1"]["gflops"], top)
        svg.append(f'<line x1="{px(pts["l1"]["ai"]):.1f}" y1="{y:.1f}" x2="{px(pts["dram"]["ai"]):.1f}" '
                   f'y2="{y:.1f}" stroke="{color}" stroke-width="2" opacity="0.55"/>')
        for level in ("dram", "l1"):
            r = pts[level]
            ceil = min(peak, bw[level] * r["ai"])
            tip = (f'{kernel} · {LEVEL_NAME[level]} · {state}: {r["q"]:,} B, AI {r["ai"]:.2f} FLOP/B, '
                   f'{r["gflops"]:.2f} GFLOP/s ({r["mcycle"]:,} cycles), {100*r["gflops"]/ceil:.0f}% of {ceil:.1f}')
            svg.append(f'<g><title>{tip}</title>{mark(level, px(r["ai"]), y, color)}</g>')
        at_ai, dy = label_pos[(kernel, state)]
        lx = px(at_ai) if at_ai else px(pts["dram"]["ai"]) + 14
        svg.append(f'<text x="{lx:.1f}" y="{y+dy:.1f}" font-size="12" fill="{TEXT_PRI}">'
                   f'{kernel} · {pts["l1"]["gflops"]:.1f} GFLOP/s</text>')


def render(rows, peak, l1_bw):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect x="0" y="0" width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="{ML}" y="24" font-size="15" fill="{TEXT_PRI}" font-weight="700">'
           f'Hierarchical roofline: FP16 GEMM on O3CPU (DRAM and L1 traffic per kernel)</text>',
           f'<text x="{ML}" y="43" font-size="12" fill="{TEXT_SEC}">M=16 N=64 K=16 ({FLOPS:,} FLOP per call) · '
           f'FMA opLat=6 · each kernel: square = DRAM, circle = L1 (same call, same height)</text>']
    # label placement: (x as AI, or None = just right of the DRAM point; dy in px)
    # cold: right of the DRAM squares, below the DRAM slope; opt/blocked are
    #       ~10 px apart in y, so blocked's label goes up and opt's down
    # warm: on the links, right of the DRAM ridge where only the roof is above
    pos = {("scalar_gemm", "cold"): (None, 4), ("opt_gemm", "cold"): (None, 13),
           ("opt_gemm_blocked", "cold"): (None, -5),
           ("scalar_gemm", "warm"): (20, -7), ("opt_gemm", "warm"): (20, 16),
           ("opt_gemm_blocked", "warm"): (20, -7)}
    top_cold = TOP
    top_warm = TOP + PANEL_H + PANEL_GAP
    panel(svg, rows, "cold", top_cold, peak, l1_bw, pos)
    panel(svg, rows, "warm", top_warm, peak, l1_bw, pos)
    svg.append(f'<text x="{ML+plotW/2:.1f}" y="{top_warm+PANEL_H+38}" font-size="12.5" fill="{TEXT_SEC}" '
               f'text-anchor="middle">Arithmetic intensity (FLOP per byte moved at that level, log)</text>')
    # legend
    ly, lx = H - 26, ML
    for kernel, color in KERNEL_COLOR.items():
        svg.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{lx+18}" y="{ly}" font-size="12" fill="{TEXT_SEC}">{kernel}</text>')
        lx += 160
    svg.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" rx="2" fill="{TEXT_MUTED}"/>')
    svg.append(f'<text x="{lx+18}" y="{ly}" font-size="12" fill="{TEXT_SEC}">DRAM</text>')
    svg.append(f'<circle cx="{lx+86}" cy="{ly-4}" r="6.5" fill="{TEXT_MUTED}"/>')
    svg.append(f'<text x="{lx+98}" y="{ly}" font-size="12" fill="{TEXT_SEC}">L1</text>')
    svg.append('</svg>')
    with open(OUT_SVG, "w") as fh:
        fh.write("\n".join(svg) + "\n")


def main():
    peak, l1_bw = ceilings()
    rows = collect()
    print(f"compute roof {peak:.1f} GFLOP/s · L1 {l1_bw:.1f} GB/s (ridge {peak/l1_bw:.2f}) · "
          f"DRAM {DRAM_BW} GB/s (ridge {peak/DRAM_BW:.2f})")
    print("| kernel | state | level | bytes | AI (FLOP/B) | mcycle | GFLOP/s | ceiling | % of ceiling |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ceil = min(peak, (DRAM_BW if r["level"] == "dram" else l1_bw) * r["ai"])
        print(f'| {r["kernel"]} | {r["state"]} | {LEVEL_NAME[r["level"]]} | {r["q"]:,} | {r["ai"]:.2f} | '
              f'{r["mcycle"]:,} | {r["gflops"]:.2f} | {ceil:.1f} | {100*r["gflops"]/ceil:.0f}% |')
    render(rows, peak, l1_bw)
    print(f"wrote {os.path.relpath(OUT_SVG, ROOT)}")


if __name__ == "__main__":
    main()
