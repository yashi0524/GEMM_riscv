#!/usr/bin/env python3
"""Generate the opt_gemm per-tile dataflow diagram for doc/opt_gemm_dataflow.svg.

No external deps (pure stdlib string building) so it renders inline on
GitHub as a plain image, same convention as gen_roofline_svg.py.

Diagrams the innermost tile computation of src/gemm.c's opt_gemm(): one
(row i, column-tile j:j+vl) iteration -- 8 independent vfmacc accumulator
chains (the OPT_GEMM_UNROLL=8 unroll) running in parallel over the
l-reduction, then a 3-level binary reduction tree merging them into the
single vector that gets stored back to C. See the docstring on opt_gemm()
in src/gemm.c for the "why" (breaking the vc->vc serial FMA dependency
chain so MinorCPU's dual-issue pipeline can overlap FMA latency across
lanes instead of stalling on one chain).

Usage:
  python3 script/gen_opt_gemm_dataflow_svg.py
"""

W, H = 1720, 900

# ---- lane geometry -----------------------------------------------------
N_LANES = 8
LANE_Y0 = 130       # center-y of lane 0
LANE_DY = 80         # vertical spacing between lane centers
LANE_H = 52          # box height within a lane row
lane_y = [LANE_Y0 + i * LANE_DY for i in range(N_LANES)]

# ---- column x-ranges -----------------------------------------------------
INIT_X, INIT_W = 96, 172
A_X, A_W = 300, 150
B_X, B_W = 486, 190
FMA_X, FMA_W = 712, 190

L1_X, L1_W = 986, 116
L2_X, L2_W = 1156, 116
L3_X, L3_W = 1326, 116
STORE_X, STORE_W = 1512, 168

def lane_pair_y(ys):
    return (ys[0] + ys[1]) / 2

l1_y = [lane_pair_y((lane_y[2 * i], lane_y[2 * i + 1])) for i in range(4)]
l2_y = [lane_pair_y((l1_y[2 * i], l1_y[2 * i + 1])) for i in range(2)]
l3_y = lane_pair_y((l2_y[0], l2_y[1]))

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
          f'font-family="ui-sans-serif,-apple-system,\'Segoe UI\',Helvetica,Arial,sans-serif">')

# ---- styles (self-contained, theme-aware) -------------------------------
svg.append('''
<style>
  :root {
    --ink: #14171c; --paper: #f1eee6; --graphite: #5b6472;
    --amber: #c97a1f; --amber-fill: #f3ddb9; --amber-line: #c97a1f;
    --teal: #2f7d74; --teal-fill: #cfe9e5; --teal-line: #2f7d74;
    --rule: #d8d3c6; --panel: #ffffff; --muted: #7b8291;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ink: #f1eee6; --paper: #14171c; --graphite: #a7afbd;
      --amber: #e7a24a; --amber-fill: #4a3818; --amber-line: #e7a24a;
      --teal: #6fc7bc; --teal-fill: #163530; --teal-line: #6fc7bc;
      --rule: #333944; --panel: #1b1f26; --muted: #8890a0;
    }
  }
  svg { background: var(--paper); }
  text { fill: var(--ink); }
  .mono { font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Menlo, Consolas, monospace; }
  .title { font-size: 25px; font-weight: 600; }
  .subtitle { font-size: 14px; fill: var(--muted); }
  .loop-label { font-size: 12px; fill: var(--graphite); letter-spacing: 0.06em; }
  .lane-badge { font-size: 12px; font-weight: 600; fill: var(--paper); }
  .box-label { font-size: 13px; }
  .stage-label { font-size: 11px; fill: var(--muted); letter-spacing: 0.08em; }
  .caption { font-size: 12.5px; fill: var(--graphite); }
  .node-init rect { fill: var(--panel); stroke: var(--graphite); stroke-width: 1.25; }
  .node-a rect { fill: var(--panel); stroke: var(--rule); stroke-width: 1.25; }
  .node-b rect { fill: var(--panel); stroke: var(--amber-line); stroke-width: 1.5; }
  .node-fma rect { fill: var(--amber-fill); stroke: var(--amber-line); stroke-width: 1.5; }
  .node-reduce rect { fill: var(--teal-fill); stroke: var(--teal-line); stroke-width: 1.5; }
  .node-store rect { fill: var(--panel); stroke: var(--graphite); stroke-width: 1.5; }
  .flow-line { stroke: var(--graphite); stroke-width: 1.25; fill: none; opacity: 0.55; }
  .flow-line-hot { stroke: var(--amber-line); stroke-width: 1.4; fill: none; opacity: 0.75; }
  .flow-line-reduce { stroke: var(--teal-line); stroke-width: 1.4; fill: none; opacity: 0.8; }
  .loop-frame { fill: none; stroke: var(--graphite); stroke-width: 1.25; stroke-dasharray: 5 4; opacity: 0.6; }
</style>
''')

def rect(x, y, w, h, rx=3, cls=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" />'

def node(x, y, w, h, lines, cls, extra=""):
    """A labeled box centered vertically at y, with 1-2 lines of text."""
    top = y - h / 2
    out = [f'<g class="{cls}">', rect(x, top, w, h), '</g>']
    n = len(lines)
    for i, (txt, tcls) in enumerate(lines):
        ty = y - (n - 1) * 8 + i * 16 + 4
        out.append(f'<text x="{x + w/2}" y="{ty}" text-anchor="middle" class="{tcls}">{txt}</text>')
    if extra:
        out.append(extra)
    return "\n".join(out)

def elbow(x1, y1, x2, y2, cls):
    mx = (x1 + x2) / 2
    return f'<path class="{cls}" d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" />'

# ---- header --------------------------------------------------------------
svg.append(f'<text x="48" y="42" class="title">opt_gemm — per-tile dataflow</text>')
svg.append(f'<text x="48" y="64" class="subtitle">'
          f'src/gemm.c · one iteration of row <tspan class="mono">i</tspan>, '
          f'column tile <tspan class="mono">j : j+vl</tspan> · '
          f'8-way unrolled RVV FMA reduction over <tspan class="mono">l</tspan></text>')

# ---- loop frame around a/b/fma columns -----------------------------------
frame_x0, frame_x1 = A_X - 24, FMA_X + FMA_W + 24
frame_y0, frame_y1 = lane_y[0] - LANE_H / 2 - 34, lane_y[-1] + LANE_H / 2 + 46
svg.append(rect(frame_x0, frame_y0, frame_x1 - frame_x0, frame_y1 - frame_y0, rx=10).replace("<rect", '<rect class="loop-frame"'))
svg.append(f'<text x="{(frame_x0 + frame_x1) / 2}" y="{frame_y0 - 10}" text-anchor="middle" class="loop-label mono">'
          f'for (l = 0; l + 8 &#8804; k; l += 8)</text>')
svg.append(f'<text x="{(frame_x0 + frame_x1) / 2}" y="{frame_y1 + 20}" text-anchor="middle" class="caption mono">'
          f'tail: while (l &lt; k) vc0 += &#945;&#183;A[i,l] &#215; B[l, j:j+vl]</text>')

# ---- stage labels ---------------------------------------------------------
svg.append(f'<text x="{INIT_X}" y="{frame_y0 - 34}" class="stage-label">ACCUMULATOR INIT</text>')
svg.append(f'<text x="{A_X}" y="{frame_y0 - 34}" class="stage-label">SCALE A (SCALAR)</text>')
svg.append(f'<text x="{B_X}" y="{frame_y0 - 34}" class="stage-label">LOAD B (VECTOR)</text>')
svg.append(f'<text x="{FMA_X}" y="{frame_y0 - 34}" class="stage-label">FMA ACCUMULATE</text>')
svg.append(f'<text x="{L1_X}" y="42" class="stage-label">REDUCTION TREE</text>')
svg.append(f'<text x="{STORE_X}" y="42" class="stage-label">STORE</text>')

# ---- 8 lanes ---------------------------------------------------------------
for i in range(N_LANES):
    y = lane_y[i]
    # lane badge
    svg.append(f'<circle cx="52" cy="{y}" r="15" fill="var(--graphite)" opacity="0.85" />')
    svg.append(f'<text x="52" y="{y + 4}" text-anchor="middle" class="lane-badge mono">L{i}</text>')

    # init box
    if i == 0:
        init_lines = [("C[i, j:j+vl]", "box-label mono"), ("&#215; &#946;  &#8594; vc0", "box-label mono")]
    else:
        init_lines = [(f"0 &#8594; vc{i}", "box-label mono")]
    svg.append(node(INIT_X, y, INIT_W, LANE_H, init_lines, "node-init"))

    # a_N
    svg.append(node(A_X, y, A_W, LANE_H,
                    [(f"a{i} = &#945;&#183;A[i,l+{i}]", "box-label mono")], "node-a"))
    svg.append(elbow(INIT_X + INIT_W, y, A_X, y, "flow-line"))

    # b_N
    svg.append(node(B_X, y, B_W, LANE_H,
                    [(f"b{i} = B[l+{i}, j:j+vl]", "box-label mono")], "node-b"))
    svg.append(elbow(A_X + A_W, y, B_X, y, "flow-line"))

    # fma (destination vcN is implied by the lane badge -- omitted from the
    # label itself so "vfmacc(vcN, aN, bN)" fits the box at 13px mono
    # without overflowing; spelling out "vcN = vfmacc(vcN, ...)" ran ~13px
    # past the 190px box width)
    svg.append(node(FMA_X, y, FMA_W, LANE_H,
                    [(f"vfmacc(vc{i}, a{i}, b{i})", "box-label mono")], "node-fma"))
    svg.append(elbow(B_X + B_W, y, FMA_X, y, "flow-line-hot"))
    # self-loop arrow (accumulates across l iterations)
    lx = FMA_X + FMA_W / 2
    svg.append(f'<path class="flow-line-hot" marker-end="url(#arrow)" '
              f'd="M{lx - 24},{y - LANE_H/2} C{lx-40},{y-LANE_H/2-16} {lx+40},{y-LANE_H/2-16} {lx+24},{y - LANE_H/2}" />')

# arrowhead marker
svg.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
          '<path d="M0,0 L6,3 L0,6 z" fill="var(--amber-line)" /></marker></defs>')

# ---- reduction tree ---------------------------------------------------------
pair_labels_l1 = [("vc01", 0, 1), ("vc23", 2, 3), ("vc45", 4, 5), ("vc67", 6, 7)]
for idx, (label, a, b) in enumerate(pair_labels_l1):
    y = l1_y[idx]
    svg.append(node(L1_X, y, L1_W, 46, [(f"{label} =", "box-label mono"), (f"vc{a}+vc{b}", "box-label mono")], "node-reduce"))
    svg.append(elbow(FMA_X + FMA_W, lane_y[a], L1_X, y, "flow-line-reduce"))
    svg.append(elbow(FMA_X + FMA_W, lane_y[b], L1_X, y, "flow-line-reduce"))

pair_labels_l2 = [("vc0123", "vc01+vc23"), ("vc4567", "vc45+vc67")]
for idx, (label, expr) in enumerate(pair_labels_l2):
    y = l2_y[idx]
    svg.append(node(L2_X, y, L2_W, 46, [(f"{label} =", "box-label mono"), (expr, "box-label mono")], "node-reduce"))
    svg.append(elbow(L1_X + L1_W, l1_y[2 * idx], L2_X, y, "flow-line-reduce"))
    svg.append(elbow(L1_X + L1_W, l1_y[2 * idx + 1], L2_X, y, "flow-line-reduce"))

svg.append(node(L3_X, l3_y, L3_W, 50, [("vc =", "box-label mono"), ("vc0123+vc4567", "box-label mono")], "node-reduce"))
svg.append(elbow(L2_X + L2_W, l2_y[0], L3_X, l3_y, "flow-line-reduce"))
svg.append(elbow(L2_X + L2_W, l2_y[1], L3_X, l3_y, "flow-line-reduce"))

# ---- store -------------------------------------------------------------------
svg.append(node(STORE_X, l3_y, STORE_W, 54,
                [("vse &#8594;", "box-label mono"), ("C[i, j:j+vl]", "box-label mono")], "node-store"))
svg.append(elbow(L3_X + L3_W, l3_y, STORE_X, l3_y, "flow-line"))

# ---- footer / loop-back annotations -------------------------------------------
svg.append(f'<circle cx="53" cy="{H-44}" r="4" fill="var(--amber)" />')
svg.append(f'<text x="64" y="{H - 40}" class="caption">vector load / FMA compute</text>')
svg.append(f'<circle cx="270" cy="{H-44}" r="4" fill="var(--teal)" />')
svg.append(f'<text x="281" y="{H - 40}" class="caption">reduction-tree merge</text>')
svg.append(f'<text x="48" y="{H - 18}" class="caption">'
          f'8 independent vc0..vc7 chains break the vc&#8594;vc serial FMA dependency, '
          f'so MinorCPU&#8217;s dual-issue pipeline overlaps FMA latency across lanes instead of stalling on one chain.</text>')
svg.append(f'<text x="{W - 48}" y="{H - 18}" text-anchor="end" class="caption mono">j += vl &#8594; next tile &#160;&#183;&#160; i += 1 &#8594; next row</text>')

svg.append('</svg>')

out_path = "doc/opt_gemm_dataflow.svg"
with open(out_path, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {out_path}")
