#!/usr/bin/env python3
"""Generate the opt_gemm_blocked per-tile dataflow diagram for
doc/opt_gemm_blocked_dataflow.svg.

No external deps (pure stdlib string building), same convention as
gen_roofline_svg.py / gen_opt_gemm_dataflow_svg.py.

Diagrams the innermost tile computation of src/gemm.c's opt_gemm_blocked():
one column-tile (j:j+vl) iteration across all M=16 rows of the block --
16 independent per-row accumulators (vc0..vc15), each fed by its own scalar
a_row = alpha*A[row,l] but ALL sharing a single vector load of
B[l, j:j+vl] per l-step. Contrast with opt_gemm's per-tile diagram:
- opt_gemm splits ONE row's l-reduction 8 ways (needs a reduction tree to
  merge the 8 chains back into one output row).
- opt_gemm_blocked splits across 16 ROWS instead (each lane already owns a
  distinct, final output row -- no merge needed, straight to store) and
  loads B exactly once per l instead of once per (row, l), cutting B's
  reload factor 16x -> 1x.
See the docstring on opt_gemm_blocked() in src/gemm.c for the full "why".

Usage:
  python3 script/gen_opt_gemm_blocked_dataflow_svg.py
"""

W, H = 1260, 900

# ---- lane geometry (one lane per row of the M=16 block) -------------------
N_LANES = 16
LANE_Y0 = 170        # center-y of lane 0 (row 0)
LANE_DY = 40          # vertical spacing between lane centers
LANE_H = 26           # box height within a lane row
lane_y = [LANE_Y0 + i * LANE_DY for i in range(N_LANES)]
mid_y = (lane_y[0] + lane_y[-1]) / 2

# ---- column x-ranges --------------------------------------------------------
INIT_X, INIT_W = 70, 190
A_X, A_W = 300, 150
B_X, B_W = 490, 200     # single shared box, not one per lane
FMA_X, FMA_W = 750, 210
STORE_X, STORE_W = 1010, 180

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
  .lane-badge { font-size: 10.5px; font-weight: 600; fill: var(--paper); }
  .box-label { font-size: 11px; }
  .shared-label { font-size: 14px; font-weight: 600; }
  .shared-note { font-size: 10.5px; fill: var(--muted); }
  .fanout-badge { font-size: 12px; font-weight: 700; fill: var(--paper); }
  .stage-label { font-size: 11px; fill: var(--muted); letter-spacing: 0.08em; }
  .caption { font-size: 12.5px; fill: var(--graphite); }
  .node-init rect { fill: var(--panel); stroke: var(--graphite); stroke-width: 1.25; }
  .node-a rect { fill: var(--panel); stroke: var(--rule); stroke-width: 1.25; }
  .node-b rect { fill: var(--amber-fill); stroke: var(--amber-line); stroke-width: 1.75; }
  .node-fma rect { fill: var(--amber-fill); stroke: var(--amber-line); stroke-width: 1.5; }
  .node-store rect { fill: var(--panel); stroke: var(--graphite); stroke-width: 1.5; }
  .flow-line { stroke: var(--graphite); stroke-width: 1.1; fill: none; opacity: 0.5; }
  .flow-line-hot { stroke: var(--amber-line); stroke-width: 1.2; fill: none; opacity: 0.6; }
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
svg.append(f'<text x="48" y="42" class="title">opt_gemm_blocked — per-tile dataflow</text>')
svg.append(f'<text x="48" y="64" class="subtitle">'
          f'src/gemm.c · one column tile <tspan class="mono">j : j+vl</tspan>, '
          f'all 16 rows of the block · single shared B load fans out to '
          f'16 row-parallel RVV FMA lanes</text>')

# ---- loop frame around a/b(shared)/fma columns -----------------------------
frame_x0, frame_x1 = A_X - 24, FMA_X + FMA_W + 24
frame_y0, frame_y1 = lane_y[0] - LANE_H / 2 - 34, lane_y[-1] + LANE_H / 2 + 46
svg.append(rect(frame_x0, frame_y0, frame_x1 - frame_x0, frame_y1 - frame_y0, rx=10).replace("<rect", '<rect class="loop-frame"'))
svg.append(f'<text x="{(frame_x0 + frame_x1) / 2}" y="{frame_y0 - 10}" text-anchor="middle" class="loop-label mono">'
          f'for (l = 0; l &lt; k; l++)  &#8212;  no unroll needed</text>')

# ---- stage labels ---------------------------------------------------------
svg.append(f'<text x="{INIT_X}" y="{frame_y0 - 34}" class="stage-label">ACCUMULATOR INIT (&#215;16)</text>')
svg.append(f'<text x="{A_X}" y="{frame_y0 - 34}" class="stage-label">SCALE A (SCALAR, &#215;16)</text>')
svg.append(f'<text x="{B_X}" y="{frame_y0 - 34}" class="stage-label">LOAD B (VECTOR, SHARED)</text>')
svg.append(f'<text x="{FMA_X}" y="{frame_y0 - 34}" class="stage-label">FMA ACCUMULATE (&#215;16)</text>')
svg.append(f'<text x="{STORE_X}" y="{frame_y0 - 34}" class="stage-label">STORE (&#215;16, DIRECT)</text>')

# ---- 16 row lanes ------------------------------------------------------------
for i in range(N_LANES):
    y = lane_y[i]
    # lane badge -- "R" for row, distinguishing from opt_gemm's k-split "L" lanes
    svg.append(f'<circle cx="46" cy="{y}" r="13" fill="var(--graphite)" opacity="0.85" />')
    svg.append(f'<text x="46" y="{y + 3.5}" text-anchor="middle" class="lane-badge mono">R{i}</text>')

    # init: every lane reads its OWN row of C (unlike opt_gemm, where only
    # lane 0 initializes from C and the rest start at zero -- here all 16
    # lanes are distinct output rows from the start, so all 16 must load).
    svg.append(node(INIT_X, y, INIT_W, LANE_H,
                    [(f"vc{i} = &#946;&#183;C[{i},j:j+vl]", "box-label mono")], "node-init"))

    # a_row (scalar)
    svg.append(node(A_X, y, A_W, LANE_H,
                    [(f"a{i} = &#945;&#183;A[{i},l]", "box-label mono")], "node-a"))
    svg.append(elbow(INIT_X + INIT_W, y, A_X, y, "flow-line"))

    # fma -- consumes a_row (scalar) and vb (shared vector, fanned in from
    # the single B box below), self-accumulates across l iterations
    svg.append(node(FMA_X, y, FMA_W, LANE_H,
                    [(f"vfmacc(vc{i}, a{i}, vb)", "box-label mono")], "node-fma"))
    svg.append(elbow(A_X + A_W, y, FMA_X, y, "flow-line"))
    svg.append(elbow(B_X + B_W, mid_y, FMA_X, y, "flow-line-hot"))
    # self-loop arrow (accumulates across l iterations)
    lx = FMA_X + FMA_W / 2
    svg.append(f'<path class="flow-line-hot" '
              f'd="M{lx - 22},{y - LANE_H/2} C{lx-36},{y-LANE_H/2-13} {lx+36},{y-LANE_H/2-13} {lx+22},{y - LANE_H/2}" />')

    # store -- direct, no reduction: this lane's vc IS the final output row
    svg.append(node(STORE_X, y, STORE_W, LANE_H,
                    [(f"vse &#8594; C[{i},j:j+vl]", "box-label mono")], "node-store"))
    svg.append(elbow(FMA_X + FMA_W, y, STORE_X, y, "flow-line"))

# ---- shared B load: ONE box, fanned out to all 16 FMA lanes -----------------
B_H = 74
svg.append(node(B_X, mid_y, B_W, B_H,
                [("vb = B[l, j:j+vl]", "shared-label mono"),
                 ("loaded once per l", "shared-note"),
                 ("&#8212; reused by all 16 rows", "shared-note")],
                "node-b"))
svg.append(f'<circle cx="{B_X + B_W + 20}" cy="{mid_y - B_H/2 - 4}" r="15" fill="var(--amber-line)" />')
svg.append(f'<text x="{B_X + B_W + 20}" y="{mid_y - B_H/2}" text-anchor="middle" class="fanout-badge mono">&#215;16</text>')

# ---- footer / callouts ------------------------------------------------------
svg.append(f'<circle cx="53" cy="{H-58}" r="4" fill="var(--amber)" />')
svg.append(f'<text x="64" y="{H - 54}" class="caption">shared vector load / FMA compute</text>')
svg.append(f'<circle cx="290" cy="{H-58}" r="4" fill="var(--graphite)" />')
svg.append(f'<text x="301" y="{H - 54}" class="caption">direct store &#8212; no reduction tree needed</text>')
svg.append(f'<text x="48" y="{H - 32}" class="caption">'
          f'16 independent per-row accumulators give the pipeline row-level ILP without unrolling '
          f'<tspan class="mono">l</tspan> at all; B[l, j:j+vl] loads once per <tspan class="mono">l</tspan> '
          f'instead of once per (row, l), cutting B&#8217;s reload factor 16&#215; &#8594; 1&#215;.</text>')
svg.append(f'<text x="48" y="{H - 12}" class="caption mono">'
          f'j += vl &#8594; next tile &#160;&#183;&#160; cf. opt_gemm_dataflow.svg &#8212; k-unrolled single row + reduction tree</text>')

svg.append('</svg>')

out_path = "doc/opt_gemm_blocked_dataflow.svg"
with open(out_path, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {out_path}")
