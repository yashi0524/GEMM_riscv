#!/usr/bin/env python3
"""Generate an array/memory-layout view of opt_gemm_blocked's access pattern
for doc/opt_gemm_blocked_arrays.svg -- companion to
gen_opt_gemm_blocked_dataflow_svg.py's register/instruction-level lane
diagram, in the same "three grids" style as gen_opt_gemm_arrays.svg.

Shows which cells of A, B, C one column-tile iteration (j:j+vl, all M=16
rows of the block) touches, for one l-step: A contributes one scalar per
row (a0..a15 = alpha*A[row,l], the SAME column l down all 16 rows -- a
vertical strip, not opt_gemm's horizontal l..l+7 segment within one row);
B contributes exactly one shared vector load (vb = B[l,j:j+vl], no
"touched but not used" slack the way opt_gemm's l..l+7 rows have, since
here the load is precisely the tile slice); C receives 16 direct vector
stores, one per row, with no reduction step.

Geometrically this is the mirror image of opt_gemm_arrays.svg's funnel:
that diagram fans 8 A-cells into 8 B-rows that reduce into ONE C row;
this one collapses 16 A-cells into ONE B-row that fans back out into 16
C-rows -- an hourglass instead of a funnel, matching
gen_opt_gemm_blocked_dataflow_svg.py's shared-B-box fan-out at the
register level.

No external deps (pure stdlib string building), matching this project's
existing SVG generator convention.

Usage:
  python3 script/gen_opt_gemm_blocked_arrays_svg.py
"""

# ---- illustrative grid sizes ------------------------------------------------
# M=16 is the real hardcoded block size (opt_gemm_blocked errors out at
# compile time if M != 16); K/N are illustrative, matching gen_opt_gemm_arrays.svg's scale.
M, K, N = 16, 12, 16          # A: M x K, B: K x N, C: M x N
CELL = 24                     # px per matrix cell
L = 5                          # highlighted single l (0-indexed)
J0, J_SPAN = 6, 4               # highlighted j:j+vl column tile (in B / C)

def grid_size(rows, cols):
    return cols * CELL, rows * CELL

A_W, A_H = grid_size(M, K)
B_W, B_H = grid_size(K, N)
C_W, C_H = grid_size(M, N)

# ---- canvas layout -----------------------------------------------------------
# A and C share the same top-y so their rows line up 1:1 (row r of A is
# literally the same row as row r of C) -- unlike opt_gemm_arrays.svg,
# where A/C only shared one highlighted row and alignment didn't matter.
A_X, A_Y = 70, 230
OP_GAP = 140
B_X = A_X + A_W + OP_GAP
EQ_GAP = 140
C_X = B_X + B_W + EQ_GAP
C_Y = A_Y

# B is shorter than A/C (K < M rows); center its highlighted row L on
# A/C's vertical midpoint, same idea as the shared-B-box placement in
# gen_opt_gemm_blocked_dataflow_svg.py.
_row_l_target_center = A_Y + A_H / 2
B_Y = _row_l_target_center - (L * CELL + CELL / 2)

OP_SYM_Y = A_Y - 40   # own band above the grids, clear of the fan zone

W = C_X + C_W + 80
H = 850

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
          f'font-family="ui-sans-serif,-apple-system,\'Segoe UI\',Helvetica,Arial,sans-serif">')

svg.append('''
<style>
  :root {
    --ink: #14171c; --paper: #f1eee6; --graphite: #5b6472;
    --amber: #c97a1f; --amber-fill: #f3ddb9; --amber-line: #c97a1f;
    --scalar: #4a63a3; --scalar-fill: #dbe1f3; --scalar-line: #4a63a3;
    --rule: #d8d3c6; --panel: #ffffff; --muted: #7b8291; --cell: #d8d3c6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ink: #f1eee6; --paper: #14171c; --graphite: #a7afbd;
      --amber: #e7a24a; --amber-fill: #4a3818; --amber-line: #e7a24a;
      --scalar: #8fa4e0; --scalar-fill: #232c47; --scalar-line: #8fa4e0;
      --rule: #333944; --panel: #1b1f26; --muted: #8890a0; --cell: #333944;
    }
  }
  svg { background: var(--paper); }
  text { fill: var(--ink); }
  .mono { font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Menlo, Consolas, monospace; }
  .title { font-size: 25px; font-weight: 600; }
  .subtitle { font-size: 14px; fill: var(--muted); }
  .matrix-label { font-size: 16px; font-weight: 600; }
  .dim-label { font-size: 12px; fill: var(--muted); }
  .callout { font-size: 12.5px; fill: var(--ink); }
  .callout-amber { font-size: 12.5px; fill: var(--amber); font-weight: 600; }
  .callout-scalar { font-size: 12.5px; fill: var(--scalar); font-weight: 600; }
  .op-symbol { font-size: 28px; fill: var(--graphite); font-weight: 300; }
  .legend-label { font-size: 12.5px; fill: var(--ink); }
  .legend-sub { font-size: 11px; fill: var(--muted); }
  .cell-plain { fill: var(--panel); stroke: var(--cell); stroke-width: 1; }
  .cell-vector { fill: var(--amber-fill); stroke: var(--amber-line); stroke-width: 1.4; }
  .cell-scalar { fill: var(--scalar-fill); stroke: var(--scalar-line); stroke-width: 1.4; }
  .bracket { stroke: var(--graphite); stroke-width: 1.25; fill: none; }
  .bracket-amber { stroke: var(--amber-line); stroke-width: 1.5; fill: none; }
  .bracket-scalar { stroke: var(--scalar-line); stroke-width: 1.5; fill: none; }
  .frame { fill: none; stroke: var(--rule); stroke-width: 1; }
  .idx-scalar { font-size: 10px; font-weight: 700; fill: var(--scalar); text-anchor: middle; }
  .idx-shared { font-size: 11px; font-weight: 700; fill: var(--amber); text-anchor: middle; }
  .fan-scalar { stroke: var(--scalar-line); stroke-width: 1; fill: none; opacity: 0.5; }
  .fan-vector { stroke: var(--amber-line); stroke-width: 1; fill: none; opacity: 0.5; }
</style>
''')

def matrix_grid(x, y, rows, cols, cell_vector=None, cell_scalar=None):
    """Draw a rows x cols grid at (x,y); cell_vector/cell_scalar are explicit
    {(r,c), ...} sets, checked in that order over a plain default."""
    cell_vector = cell_vector or set()
    cell_scalar = cell_scalar or set()
    out = ['<g>']
    for r in range(rows):
        for c in range(cols):
            cls = "cell-plain"
            if (r, c) in cell_vector:
                cls = "cell-vector"
            elif (r, c) in cell_scalar:
                cls = "cell-scalar"
            out.append(f'<rect class="{cls}" x="{x + c*CELL}" y="{y + r*CELL}" width="{CELL}" height="{CELL}" />')
    out.append(f'<rect class="frame" x="{x}" y="{y}" width="{cols*CELL}" height="{rows*CELL}" />')
    out.append('</g>')
    return "\n".join(out)

def bracket_h(x0, x1, y, cls, label, label_dy=18, above=False):
    d = -1 if above else 1
    y2 = y + d * 10
    ty = y + d * label_dy
    out = [f'<path class="{cls}" d="M{x0},{y} L{x0},{y2} L{x1},{y2} L{x1},{y}" />']
    out.append(f'<text x="{(x0+x1)/2}" y="{ty}" text-anchor="middle" class="callout mono">{label}</text>')
    return "\n".join(out)

def bracket_v(x, y0, y1, cls, label, side="right"):
    d = 10 if side == "right" else -10
    x2 = x + d
    tx = x2 + (14 if side == "right" else -14)
    anchor = "start" if side == "right" else "end"
    out = [f'<path class="{cls}" d="M{x},{y0} L{x2},{y0} L{x2},{y1} L{x},{y1}" />']
    out.append(f'<text x="{tx}" y="{(y0+y1)/2 + 4}" text-anchor="{anchor}" class="callout mono">{label}</text>')
    return "\n".join(out)

def fan_curve(x1, y1, x2, y2, cls, marker=""):
    """Diverges toward y2 within a short, capped x-step -- see
    gen_opt_gemm_arrays.svg's fan_curve for why (avoids skimming through
    labels that share a starting or ending y)."""
    dx = min(40, abs(x2 - x1) * 0.35)
    c1x, c1y = x1 + dx, y1 + (y2 - y1) * 0.75
    c2x, c2y = x2 - dx, y2
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path class="{cls}"{m} d="M{x1},{y1} C{c1x},{c1y} {c2x},{c2y} {x2},{y2}" />'

# ---- header ------------------------------------------------------------------
svg.append(f'<text x="48" y="42" class="title">opt_gemm_blocked — array access pattern</text>')
svg.append(f'<text x="48" y="64" class="subtitle">'
          f'src/gemm.c · which cells of A, B, C one column-tile iteration '
          f'(<tspan class="mono">j:j+vl</tspan>, all 16 rows of the block) touches, for one <tspan class="mono">l</tspan> step</text>')

# ---- legend --------------------------------------------------------------------
LEG_Y = 100
legend_items = [
    ("cell-vector", "vector instruction", "B: 1 shared vle_v load — C: 16 vse_v stores", 48),
    ("cell-scalar", "scalar instruction", "A: 16 separate float loads, one per row", 420),
]
for cls, label, sub, lx in legend_items:
    svg.append(f'<rect class="{cls}" x="{lx}" y="{LEG_Y - 13}" width="20" height="20" rx="2" />')
    svg.append(f'<text x="{lx + 28}" y="{LEG_Y + 1}" class="legend-label">{label}</text>')
    svg.append(f'<text x="{lx + 28}" y="{LEG_Y + 16}" class="legend-sub">{sub}</text>')

# ---- arrowhead markers -----------------------------------------------------
svg.append('<defs>'
          '<marker id="arrow-scalar" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto">'
          '<path d="M0,0 L5,2.5 L0,5 z" fill="var(--scalar-line)" /></marker>'
          '<marker id="arrow-vector" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto">'
          '<path d="M0,0 L5,2.5 L0,5 z" fill="var(--amber-line)" /></marker>'
          '</defs>')

# ---- A: highlight column l down ALL 16 rows (SCALAR reads, one per row) ----
svg.append(f'<text x="{A_X}" y="{A_Y - 52}" class="matrix-label">A <tspan class="dim-label mono">(m &#215; k)</tspan></text>')
svg.append(matrix_grid(A_X, A_Y, M, K, cell_scalar={(r, L) for r in range(M)}))
svg.append(bracket_v(A_X - 4, A_Y, A_Y + M*CELL, "bracket", "0..15", side="left"))
svg.append(bracket_h(A_X + L*CELL, A_X + (L+1)*CELL, A_Y - 6, "bracket-scalar", "l", above=True))
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 46}" class="callout-scalar mono">a0..a15 = &#945;&#183;A[0..15, l]  (scalar)</text>')
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 64}" class="callout mono">16 separate float loads — same column l,</text>')
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 80}" class="callout mono">a different row each (opt_gemm_blocked&#8217;s</text>')
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 96}" class="callout mono">row-parallelism, not opt_gemm&#8217;s k-unroll)</text>')
# per-cell row-index labels, matching gen_opt_gemm_arrays.svg's embedded style
for r in range(M):
    cx = A_X + L * CELL + CELL / 2
    cy = A_Y + r * CELL + CELL / 2
    svg.append(f'<text x="{cx}" y="{cy + 3.5}" class="idx-scalar">{r}</text>')

# ---- B: highlight one row l, sub-tile j:j+vl (VECTOR, shared/reused x16) ----
svg.append(f'<text x="{B_X}" y="{B_Y - 50}" class="matrix-label">B <tspan class="dim-label mono">(k &#215; n)</tspan></text>')
svg.append(matrix_grid(B_X, B_Y, K, N, cell_vector={(L, c) for c in range(J0, J0 + J_SPAN)}))
svg.append(bracket_h(B_X + J0*CELL, B_X + (J0+J_SPAN)*CELL, B_Y - 6, "bracket-amber", "j : j+vl", above=True))
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 46}" class="callout-amber mono">vb = B[l, j:j+vl]  (vector, loaded once)</text>')
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 64}" class="callout mono">1 &#215; vle_v load — no teal slack here: unlike</text>')
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 82}" class="callout mono">opt_gemm, the load IS exactly j:j+vl, reused</text>')
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 100}" class="callout mono">by all 16 rows instead of reloaded per row</text>')
# single "l" label, offset clear of the arrow landing point (B_X-14)
svg.append(f'<text x="{B_X - 26}" y="{B_Y + L*CELL + CELL/2 + 3.5}" class="idx-shared">l</text>')

# ---- op symbols (own band, above the fan zone) ------------------------------
svg.append(f'<text x="{A_X + A_W + OP_GAP/2}" y="{OP_SYM_Y}" text-anchor="middle" class="op-symbol">&#215;</text>')
svg.append(f'<text x="{B_X + B_W + EQ_GAP/2}" y="{OP_SYM_Y}" text-anchor="middle" class="op-symbol">=</text>')

# ---- C: highlight ALL 16 rows' tile j:j+vl (the output, VECTOR store x16) --
svg.append(f'<text x="{C_X}" y="{C_Y - 52}" class="matrix-label">C <tspan class="dim-label mono">(m &#215; n)</tspan></text>')
svg.append(matrix_grid(C_X, C_Y, M, N, cell_vector={(r, c) for r in range(M) for c in range(J0, J0 + J_SPAN)}))
svg.append(bracket_v(C_X - 4, C_Y, C_Y + M*CELL, "bracket", "0..15", side="left"))
svg.append(bracket_h(C_X + J0*CELL, C_X + (J0+J_SPAN)*CELL, C_Y - 6, "bracket-amber", "j : j+vl", above=True))
svg.append(f'<text x="{C_X}" y="{C_Y + M*CELL + 46}" class="callout-amber mono">vse write &#215;16  (vector, direct)</text>')
svg.append(f'<text x="{C_X}" y="{C_Y + M*CELL + 64}" class="callout mono">C[0..15, j:j+vl] &#8592; vc0..vc15 (no reduction)</text>')

# ---- fan arrows: A's 16 scalar cells -> B's 1 shared row -> C's 16 rows ----
# Each row r gets a single dedicated y-offset within B's row-l band (a
# "lane"), used for BOTH legs -- entering from A at that lane height and
# leaving toward C at that same lane height -- so each thread reads
# cleanly left-to-right instead of re-sorting itself at B.
# A -> B: exits from the TOP of each of A's cells (not the side, at that
# row's mid-height) -- same fix as gen_opt_gemm_arrays.svg's arrows,
# applied here from the start since every row's own index label sits
# exactly at that mid-height.
for r in range(M):
    ax = A_X + L * CELL + CELL / 2
    ay = A_Y + r * CELL - 4
    lane_y = B_Y + L * CELL + (r + 0.5) * (CELL / M)
    svg.append(fan_curve(ax, ay, B_X - 14, lane_y, "fan-scalar", marker="arrow-scalar"))

# B -> C: all 16 threads leave the SAME shared row (the hourglass's
# narrow waist) and fan back out to their own C row, at that row's
# vertical center -- no label there, so no exit-point trick is needed.
for r in range(M):
    lane_y = B_Y + L * CELL + (r + 0.5) * (CELL / M)
    cy = C_Y + r * CELL + CELL / 2
    svg.append(fan_curve(B_X + N * CELL, lane_y, C_X - 14, cy, "fan-vector", marker="arrow-vector"))

# ---- footer ------------------------------------------------------------------
svg.append(f'<text x="48" y="{H - 96}" class="callout">'
          f'Amber = RVV vector instructions (B&#8217;s single <tspan class="mono">vle_v</tspan> load, '
          f'C&#8217;s 16 <tspan class="mono">vse_v</tspan> stores) — <tspan class="mono">vl</tspan> elements moved per instruction. '
          f'Blue = plain scalar float loads (A) — one element per instruction.</text>')
svg.append(f'<text x="48" y="{H - 74}" class="callout">'
          f'The mirror image of opt_gemm&#8217;s funnel: there, many B rows reduce into one C row; '
          f'here, one shared B row (the waist) fans back out into all 16 C rows — '
          f'no reduction tree because each row already owns its own accumulator.</text>')
svg.append(f'<text x="48" y="{H - 52}" class="callout">'
          f'Repeating this over all <tspan class="mono">l</tspan> accumulates every row at once; '
          f'repeating over all <tspan class="mono">j</tspan> (rightward through B/C) covers the whole '
          f'output — all 16 rows advance together, unlike opt_gemm&#8217;s one-row-at-a-time <tspan class="mono">i</tspan> loop.</text>')
svg.append(f'<text x="{W - 48}" y="{H - 18}" text-anchor="end" class="callout mono">'
          f'see opt_gemm_blocked_dataflow.svg for the register view · cf. opt_gemm_arrays.svg for the k-unrolled single-row variant</text>')

svg.append('</svg>')

out_path = "doc/opt_gemm_blocked_arrays.svg"
with open(out_path, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {out_path}")
