#!/usr/bin/env python3
"""Generate the static roofline SVG for doc/gemm_analysis.md.

No external deps (pure stdlib `math` + string building) so it renders
inline on GitHub as a plain image — no matplotlib/cairosvg available in
this environment, and even if there were, a hand-rolled SVG keeps the
output dependency-free for anyone re-running this later.

Usage:
  python3 script/gen_roofline_svg.py

To refresh after a new test/sweep.py run: copy the relevant mcycle/AI/
GFLOP/s values from test/output/sweep_summary.txt (or the printed table)
into SERIES (compute-roof ceilings, from doc/microbenchmark.md) and POINTS
(one entry per swept row) below, then re-run. il=4 (scalar-fallback) rows
have no defined AI — give them ai=None and they'll be skipped in the plot
but should still appear in gemm_analysis.md's table.
"""
import math

SERIES = {
    "fp64_minor": {"label": "FP64 · MinorCPU", "color": "#2a78d6", "ceiling": 15.73},
    "fp64_o3":    {"label": "FP64 · O3CPU",    "color": "#1baf7a", "ceiling": 49.97},
    "fp16_minor": {"label": "FP16 · MinorCPU", "color": "#c98500", "ceiling": 62.63},
    "fp16_o3":    {"label": "FP16 · O3CPU",    "color": "#008300", "ceiling": 200.1},
}

# (series, kernel, w, il, ai, gflops) — ai=None rows (il=4 scalar fallback) are omitted from the plot
POINTS = [
    ("fp64_minor", "scalar", 4, 1, 0.080, 0.4935), ("fp64_minor", "scalar", 4, 2, 0.080, 0.5985),
    ("fp64_minor", "scalar", 4, 4, 0.080, 0.7244), ("fp64_minor", "scalar", 8, 1, 0.080, 0.7299),
    ("fp64_minor", "scalar", 8, 2, 0.080, 0.8575), ("fp64_minor", "scalar", 8, 4, None,  0.1452),

    ("fp64_o3", "scalar", 4, 1, 0.080, 1.2917), ("fp64_o3", "scalar", 4, 2, 0.080, 1.9100),
    ("fp64_o3", "scalar", 4, 4, 0.080, 2.0531), ("fp64_o3", "scalar", 8, 1, 0.080, 1.9389),
    ("fp64_o3", "scalar", 8, 2, 0.080, 2.8435), ("fp64_o3", "scalar", 8, 4, None,  0.3537),

    ("fp16_minor", "scalar", 32, 1, 0.320, 1.5328), ("fp16_minor", "scalar", 32, 2, 0.195, 0.7831),
    ("fp16_minor", "scalar", 32, 4, None,  0.0916),

    ("fp16_o3", "scalar", 32, 1, 0.320, 5.5398), ("fp16_o3", "scalar", 32, 2, 0.195, 2.0195),
    ("fp16_o3", "scalar", 32, 4, None,  0.1477),

    ("fp64_minor", "opt", 8, 1, 0.222, 1.4412),
    ("fp64_o3",    "opt", 8, 1, 0.222, 4.7053),
    ("fp16_minor", "opt", 32, 1, 0.889, 3.5072),
    ("fp16_o3",    "opt", 32, 1, 0.889, 15.0381),
]

PEAK_BW = 12.8   # GB/s (DDR3-1600 8x8), matches test/sweep.py's PEAK_BW

W = 900
ML, MR, MT = 78, 190, 40
plotW, plotH = W - ML - MR, 480
axisBottom = MT + plotH        # 520
tickLabelY  = axisBottom + 20  # 540
axisTitleY  = axisBottom + 40  # 560
legend1Y    = axisBottom + 65  # 585
legend2Y    = axisBottom + 90  # 610
footnoteY   = axisBottom + 114 # 634
H = footnoteY + 20             # bottom padding

x_dom = (0.05, 30)
y_dom = (0.3, 300)
lx, rx = math.log10(x_dom[0]), math.log10(x_dom[1])
ly, ry = math.log10(y_dom[0]), math.log10(y_dom[1])

def px(ai): return ML + (math.log10(ai) - lx) / (rx - lx) * plotW
def py(g):  return MT + plotH - (math.log10(g) - ly) / (ry - ly) * plotH

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

TEXT_MUTED = "#767671"
TEXT_SEC   = "#3d3d3a"
TEXT_PRI   = "#111111"
GRID       = "#e4e3dd"
AXIS       = "#b9b8b1"

svg.append(f'<text x="{ML}" y="22" font-size="15" fill="{TEXT_PRI}" font-weight="700">GEMM roofline: FP64/FP16 &#215; MinorCPU/O3CPU vs. fmacc peak compute</text>')

# grid + axis ticks
x_ticks = [0.05, 0.1, 0.3, 1, 3, 10, 30]
y_ticks = [0.3, 1, 3, 10, 30, 100, 300]
for v in x_ticks:
    x = px(v)
    svg.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+plotH}" stroke="{GRID}" stroke-width="1"/>')
    svg.append(f'<text x="{x:.1f}" y="{tickLabelY}" font-size="12" fill="{TEXT_MUTED}" text-anchor="middle">{v}</text>')
for v in y_ticks:
    y = py(v)
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plotW}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
    svg.append(f'<text x="{ML-10}" y="{y+4:.1f}" font-size="12" fill="{TEXT_MUTED}" text-anchor="end">{v}</text>')

svg.append(f'<line x1="{ML}" y1="{MT+plotH}" x2="{ML+plotW}" y2="{MT+plotH}" stroke="{AXIS}" stroke-width="1"/>')
svg.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+plotH}" stroke="{AXIS}" stroke-width="1"/>')

svg.append(f'<text x="{ML+plotW/2:.1f}" y="{axisTitleY}" font-size="13" fill="{TEXT_SEC}" text-anchor="middle">Arithmetic intensity (FLOP / byte, log scale)</text>')
svg.append(f'<text x="20" y="{MT+plotH/2:.1f}" font-size="13" fill="{TEXT_SEC}" text-anchor="middle" transform="rotate(-90 20 {MT+plotH/2:.1f})">Achieved performance (GFLOP/s, log scale)</text>')

# memory-bound roof: g = PEAK_BW * ai
x1 = x_dom[0]
x2 = min(x_dom[1], y_dom[1] / PEAK_BW)
svg.append(f'<line x1="{px(x1):.1f}" y1="{py(PEAK_BW*x1):.1f}" x2="{px(x2):.1f}" y2="{py(PEAK_BW*x2):.1f}" '
           f'stroke="{TEXT_MUTED}" stroke-width="2" stroke-dasharray="5 4"/>')
svg.append(f'<text x="{px(x2)-6:.1f}" y="{py(PEAK_BW*x2)-8:.1f}" font-size="11.5" fill="{TEXT_SEC}" text-anchor="end">memory roof ({PEAK_BW} GB/s)</text>')

# compute-roof ceilings (right-margin direct labels double as identity + value)
for key, s in SERIES.items():
    y = py(s["ceiling"])
    svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+plotW}" y2="{y:.1f}" stroke="{s["color"]}" '
               f'stroke-width="2" stroke-dasharray="2 4" opacity="0.85"/>')
    svg.append(f'<text x="{ML+plotW+8}" y="{y+4:.1f}" font-size="11" fill="{s["color"]}" font-weight="700">{s["ceiling"]} GFLOP/s</text>')

# scatter points: circle = scalar_gemm, diamond = opt_gemm (same color = same series identity)
for key, kernel, w, il, ai, g in POINTS:
    if ai is None:
        continue
    s = SERIES[key]
    cx, cy = px(ai), py(g)
    if kernel == "opt":
        r = 7.5
        pts = f"{cx:.1f},{cy-r:.1f} {cx+r:.1f},{cy:.1f} {cx:.1f},{cy+r:.1f} {cx-r:.1f},{cy:.1f}"
        svg.append(f'<polygon points="{pts}" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.5"/>')
    else:
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.5" opacity="0.7"/>')

# legend row(s) below the chart
leg_y = legend1Y
leg_x = ML
for key, s in SERIES.items():
    svg.append(f'<rect x="{leg_x}" y="{leg_y-10}" width="12" height="12" rx="3" fill="{s["color"]}"/>')
    svg.append(f'<text x="{leg_x+17}" y="{leg_y}" font-size="12" fill="{TEXT_SEC}">{s["label"]}</text>')
    leg_x += 17 + 8 * len(s["label"]) + 26
svg.append(f'<line x1="{leg_x}" y1="{leg_y-5}" x2="{leg_x+20}" y2="{leg_y-5}" stroke="{TEXT_MUTED}" stroke-width="2" stroke-dasharray="4 3"/>')
svg.append(f'<text x="{leg_x+26}" y="{leg_y}" font-size="12" fill="{TEXT_SEC}">Memory roof</text>')

leg_y2 = legend2Y
leg_x2 = ML
svg.append(f'<circle cx="{leg_x2+6}" cy="{leg_y2-4}" r="6.5" fill="{TEXT_MUTED}" stroke="#ffffff" stroke-width="1.5" opacity="0.7"/>')
svg.append(f'<text x="{leg_x2+20}" y="{leg_y2}" font-size="12" fill="{TEXT_SEC}">scalar_gemm (circle)</text>')
leg_x2 += 20 + 8 * len("scalar_gemm (circle)") + 22
r = 7.5
pts = f"{leg_x2+6:.1f},{leg_y2-4-r:.1f} {leg_x2+6+r:.1f},{leg_y2-4:.1f} {leg_x2+6:.1f},{leg_y2-4+r:.1f} {leg_x2+6-r:.1f},{leg_y2-4:.1f}"
svg.append(f'<polygon points="{pts}" fill="{TEXT_MUTED}" stroke="#ffffff" stroke-width="1.5"/>')
svg.append(f'<text x="{leg_x2+22}" y="{leg_y2}" font-size="12" fill="{TEXT_SEC}">opt_gemm (diamond)</text>')

svg.append(f'<text x="{ML}" y="{footnoteY}" font-size="10.5" fill="{TEXT_MUTED}">il=4 rows (scalar fallback) omitted — see table for values. opt_gemm plots one (w,il); it ignores those flags.</text>')

svg.append('</svg>')

pattern_root = "/home/ajno5/work/2_pattern/gemm"
out_path = f"{pattern_root}/doc/gemm_roofline.svg"
with open(out_path, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {out_path}")
