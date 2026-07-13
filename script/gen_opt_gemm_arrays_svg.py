#!/usr/bin/env python3
"""Generate an array/memory-layout view of opt_gemm's access pattern for
doc/opt_gemm_arrays.svg -- companion to gen_opt_gemm_dataflow_svg.py's
register/instruction-level lane diagram. This one shows the classic
"three grids" GEMM picture: which cells of A, B, and C one output tile
(row i, column tile j:j+vl) actually touches, for one unrolled l-step --
with vector-instruction-accelerated accesses (B's __riscv_vle64_v_f64m1
loads, C's vector load/store) colored distinctly from A's plain scalar
reads (a0..a7 = alpha*A[i,l+N], one float at a time -- opt_gemm never
vectorizes across l, only across the j:j+vl output columns) -- plus
explicit numbered arrows tracing each of the 8 scalar A-cells to its
paired B-row, and each of those 8 rows converging into the single C tile
(the same 8-way-unroll-then-reduce shape gen_opt_gemm_dataflow_svg.py
shows at the register level).

No external deps (pure stdlib string building), matching this project's
existing SVG generator convention.

Usage:
  python3 script/gen_opt_gemm_arrays_svg.py
"""

# ---- illustrative grid sizes (not tied to a specific M/N/K build) --------
M, K, N = 10, 12, 16          # A: M x K, B: K x N, C: M x N
CELL = 24                     # px per matrix cell
ROW_I = 4                      # highlighted row i (0-indexed)
L0, L_SPAN = 2, 8              # highlighted l..l+7 segment (columns in A / rows in B)
J0, J_SPAN = 6, 4               # highlighted j:j+vl column tile (in B / C)

def grid_size(rows, cols):
    return cols * CELL, rows * CELL

A_W, A_H = grid_size(M, K)
B_W, B_H = grid_size(K, N)
C_W, C_H = grid_size(M, N)

# ---- canvas layout ---------------------------------------------------------
# OP_GAP/EQ_GAP widened (vs. the no-arrows version) to give the 8-way fan of
# A->B and B->C arrows room to spread without crowding the x/= symbols,
# which are pulled up into their own band above the fan zone instead of
# being centered in the gap.
A_X, A_Y = 60, 300
OP_GAP = 150
B_X = A_X + A_W + OP_GAP
B_Y = 190
EQ_GAP = 150
C_X = B_X + B_W + EQ_GAP
C_Y = 300
OP_SYM_Y = 178   # fixed height for the x / = symbols, above the fan zone

W = C_X + C_W + 80
H = 800

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
          f'font-family="ui-sans-serif,-apple-system,\'Segoe UI\',Helvetica,Arial,sans-serif">')

svg.append('''
<style>
  :root {
    --ink: #14171c; --paper: #f1eee6; --graphite: #5b6472;
    --amber: #c97a1f; --amber-fill: #f3ddb9; --amber-line: #c97a1f;
    --teal: #2f7d74; --teal-fill: #cfe9e5; --teal-line: #2f7d74;
    --scalar: #4a63a3; --scalar-fill: #dbe1f3; --scalar-line: #4a63a3;
    --rule: #d8d3c6; --panel: #ffffff; --muted: #7b8291; --cell: #d8d3c6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ink: #f1eee6; --paper: #14171c; --graphite: #a7afbd;
      --amber: #e7a24a; --amber-fill: #4a3818; --amber-line: #e7a24a;
      --teal: #6fc7bc; --teal-fill: #163530; --teal-line: #6fc7bc;
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
  .callout-teal { font-size: 12.5px; fill: var(--teal); font-weight: 600; }
  .callout-scalar { font-size: 12.5px; fill: var(--scalar); font-weight: 600; }
  .op-symbol { font-size: 28px; fill: var(--graphite); font-weight: 300; }
  .legend-label { font-size: 12.5px; fill: var(--ink); }
  .legend-sub { font-size: 11px; fill: var(--muted); }
  .cell-plain { fill: var(--panel); stroke: var(--cell); stroke-width: 1; }
  .cell-row-i { fill: var(--panel); stroke: var(--graphite); stroke-width: 1; }
  .cell-l-band { fill: var(--teal-fill); stroke: var(--teal-line); stroke-width: 1; }
  .cell-vector { fill: var(--amber-fill); stroke: var(--amber-line); stroke-width: 1.4; }
  .cell-scalar { fill: var(--scalar-fill); stroke: var(--scalar-line); stroke-width: 1.4; }
  .bracket { stroke: var(--graphite); stroke-width: 1.25; fill: none; }
  .bracket-amber { stroke: var(--amber-line); stroke-width: 1.5; fill: none; }
  .bracket-teal { stroke: var(--teal-line); stroke-width: 1.5; fill: none; }
  .bracket-scalar { stroke: var(--scalar-line); stroke-width: 1.5; fill: none; }
  .frame { fill: none; stroke: var(--rule); stroke-width: 1; }
  .idx-scalar { font-size: 10px; font-weight: 700; fill: var(--scalar); text-anchor: middle; }
  .fan-scalar { stroke: var(--scalar-line); stroke-width: 1.1; fill: none; opacity: 0.55; }
  .fan-vector { stroke: var(--amber-line); stroke-width: 1.1; fill: none; opacity: 0.55; }
</style>
''')

def matrix_grid(x, y, rows, cols, row_hi=None, col_lo_hi=None, row_band=None,
                cell_vector=None, cell_scalar=None):
    """Draw a rows x cols grid of cells at (x,y). Highlighting knobs:
    row_hi: a single row index to tint as 'row i' (cell-row-i class, no accel color)
    col_lo_hi: (lo, span) column range within row_hi to mark -- scalar unless
               overridden by cell_vector/cell_scalar per-cell sets below
    row_band: (lo, span) row range to tint teal (cell-l-band) -- touched-but-
              not-this-tile's-share context (used for B's l..l+7 rows)
    cell_vector / cell_scalar: explicit {(r,c), ...} sets, checked before the
              row_hi/col_lo_hi fallback -- lets a caller mark an exact
              sub-tile (e.g. B's actual 8xvl vector-loaded block) precisely.
    """
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
            elif row_band and row_band[0] <= r < row_band[0] + row_band[1]:
                cls = "cell-l-band"
            elif row_hi is not None and r == row_hi:
                if col_lo_hi and col_lo_hi[0] <= c < col_lo_hi[0] + col_lo_hi[1]:
                    cls = "cell-scalar"
                else:
                    cls = "cell-row-i"
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
    """A curve from (x1,y1) to (x2,y2) that diverges toward y2 quickly, within
    a short distance of x1 -- not the symmetric-S midpoint-control-point
    curve (C(mx,y1)(mx,y2)), which stays flat at y1 for a long initial
    stretch. That flat stretch is a real problem when many curves share the
    same starting y1 (e.g. 8 arrows all leaving row i at the same height,
    each starting where the *next* cell's index label sits): the curve
    visually skims straight through those labels before it ever starts
    turning. Reaching most of the way to y2 within a small, capped x-step
    keeps the path clear of anything still close to (x1, y1)."""
    dx = min(40, abs(x2 - x1) * 0.35)
    c1x, c1y = x1 + dx, y1 + (y2 - y1) * 0.75
    c2x, c2y = x2 - dx, y2
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<path class="{cls}"{m} d="M{x1},{y1} C{c1x},{c1y} {c2x},{c2y} {x2},{y2}" />'

# ---- header ----------------------------------------------------------------
svg.append(f'<text x="48" y="42" class="title">opt_gemm — array access pattern</text>')
svg.append(f'<text x="48" y="64" class="subtitle">'
          f'src/gemm.c · which cells of A, B, C one output tile '
          f'(row <tspan class="mono">i</tspan>, column tile <tspan class="mono">j:j+vl</tspan>) '
          f'touches, for one unrolled <tspan class="mono">l..l+7</tspan> step</text>')

# ---- legend ------------------------------------------------------------------
LEG_Y = 100
legend_items = [
    ("cell-vector", "vector instruction", "vle64/vse64 — RVV-accelerated, vl elements/op", 48),
    ("cell-scalar", "scalar instruction", "one float at a time — not vector-accelerated", 330),
    ("cell-l-band", "touched, not this tile's share", "same l-step, different j (other tiles read it)", 660),
]
for cls, label, sub, lx in legend_items:
    svg.append(f'<rect class="{cls}" x="{lx}" y="{LEG_Y - 13}" width="20" height="20" rx="2" />')
    svg.append(f'<text x="{lx + 28}" y="{LEG_Y + 1}" class="legend-label">{label}</text>')
    svg.append(f'<text x="{lx + 28}" y="{LEG_Y + 16}" class="legend-sub">{sub}</text>')

# ---- arrowhead markers ---------------------------------------------------
svg.append('<defs>'
          '<marker id="arrow-scalar" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto">'
          '<path d="M0,0 L5,2.5 L0,5 z" fill="var(--scalar-line)" /></marker>'
          '<marker id="arrow-vector" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto">'
          '<path d="M0,0 L5,2.5 L0,5 z" fill="var(--amber-line)" /></marker>'
          '</defs>')

# ---- A: highlight row i, and within it the l..l+7 segment (SCALAR reads) ---
svg.append(f'<text x="{A_X}" y="{A_Y - 34}" class="matrix-label">A <tspan class="dim-label mono">(m &#215; k)</tspan></text>')
svg.append(matrix_grid(A_X, A_Y, M, K, row_hi=ROW_I, col_lo_hi=(L0, L_SPAN)))
svg.append(bracket_v(A_X - 4, A_Y + ROW_I*CELL, A_Y + (ROW_I+1)*CELL, "bracket", "row i", side="left"))
svg.append(bracket_h(A_X + L0*CELL, A_X + (L0+L_SPAN)*CELL, A_Y + (ROW_I+1)*CELL + 4, "bracket-scalar", "l .. l+7"))
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 46}" class="callout-scalar mono">a0..a7 = &#945;&#183;A[i, l..l+7]  (scalar)</text>')
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 64}" class="callout mono">8 separate float loads — opt_gemm never</text>')
svg.append(f'<text x="{A_X}" y="{A_Y + M*CELL + 80}" class="callout mono">vectorizes across l, only across j:j+vl</text>')
# per-cell index labels (0..7), matching B's per-row labels below
for n in range(L_SPAN):
    cx = A_X + (L0 + n) * CELL + CELL / 2
    cy = A_Y + ROW_I * CELL + CELL / 2
    svg.append(f'<text x="{cx}" y="{cy + 3.5}" class="idx-scalar">{n}</text>')

# ---- B: highlight column tile j:j+vl, and within it rows l..l+7 (VECTOR) --
svg.append(f'<text x="{B_X}" y="{B_Y - 50}" class="matrix-label">B <tspan class="dim-label mono">(k &#215; n)</tspan></text>')
svg.append(matrix_grid(B_X, B_Y, K, N, row_band=(L0, L_SPAN),
                       cell_vector={(r, c) for r in range(L0, L0 + L_SPAN) for c in range(J0, J0 + J_SPAN)}))
svg.append(bracket_v(B_X + N*CELL + 4, B_Y + L0*CELL, B_Y + (L0+L_SPAN)*CELL, "bracket-teal", "l .. l+7", side="right"))
svg.append(bracket_h(B_X + J0*CELL, B_X + (J0+J_SPAN)*CELL, B_Y - 6, "bracket-amber", "j : j+vl", above=True))
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 46}" class="callout-amber mono">b0..b7 = B[l..l+7, j:j+vl]  (vector)</text>')
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 64}" class="callout mono">8 &#215; vle64_v_f64m1 loads, vl elements each</text>')
svg.append(f'<text x="{B_X}" y="{B_Y + K*CELL + 82}" class="callout-teal mono">teal = full rows l..l+7; only j:j+vl feeds this tile</text>')
# per-row index labels (0..7) just left of the grid, landing points for A's arrows
for n in range(L_SPAN):
    ry = B_Y + (L0 + n) * CELL + CELL / 2
    svg.append(f'<text x="{B_X - 26}" y="{ry + 3.5}" class="idx-scalar">{n}</text>')

# ---- op symbols (own band, above the fan zone) ------------------------------
svg.append(f'<text x="{A_X + A_W + OP_GAP/2}" y="{OP_SYM_Y}" text-anchor="middle" class="op-symbol">&#215;</text>')
svg.append(f'<text x="{B_X + B_W + EQ_GAP/2}" y="{OP_SYM_Y}" text-anchor="middle" class="op-symbol">=</text>')

# ---- C: highlight row i, column tile j:j+vl (the output, VECTOR load+store) -
svg.append(f'<text x="{C_X}" y="{C_Y - 34}" class="matrix-label">C <tspan class="dim-label mono">(m &#215; n)</tspan></text>')
svg.append(matrix_grid(C_X, C_Y, M, N, row_hi=ROW_I,
                       cell_vector={(ROW_I, c) for c in range(J0, J0 + J_SPAN)}))
svg.append(bracket_v(C_X - 4, C_Y + ROW_I*CELL, C_Y + (ROW_I+1)*CELL, "bracket", "row i", side="left"))
svg.append(bracket_h(C_X + J0*CELL, C_X + (J0+J_SPAN)*CELL, C_Y + (ROW_I+1)*CELL + 4, "bracket-amber", "j : j+vl"))
svg.append(f'<text x="{C_X}" y="{C_Y + M*CELL + 46}" class="callout-amber mono">vle64 read (&#215;&#946;), vse64 write  (vector)</text>')
svg.append(f'<text x="{C_X}" y="{C_Y + M*CELL + 64}" class="callout mono">C[i, j:j+vl] &#8592; vc (after reduction)</text>')

# ---- fan arrows: A's 8 scalar cells -> B's 8 rows -> converge into C -------
# A -> B: each a_N exits from the TOP of its own cell (not the side, at row
# i's mid-height) -- every index label sits at that mid-height, shared by
# all 8 cells in the row, so a side-exit's initial (near-)horizontal travel
# skims right past neighboring cells' labels regardless of how quickly the
# curve turns. Exiting from the top puts the whole path a clear 12px above
# the label row from the very first pixel, independent of curve shape.
# n=0 (leftmost in A / topmost in B) to n=7 map monotonically, so the fan
# doesn't cross itself.
for n in range(L_SPAN):
    ax = A_X + (L0 + n) * CELL + CELL / 2
    ay = A_Y + ROW_I * CELL - 4
    bx = B_X - 14
    by = B_Y + (L0 + n) * CELL + CELL / 2
    svg.append(fan_curve(ax, ay, bx, by, "fan-scalar", marker="arrow-scalar"))

# B -> C: each row l+N's vector tile (right edge) converges into
# C[i, j:j+vl] (left edge) -- the reduction. Entry points are spread across
# the row's own height (not one exact pixel) so the 8 arrowheads fan into
# the row instead of stacking into a single overlapping blob -- also more
# accurate, since they really do converge into the same row, not a point.
cx_entry = C_X - 14
row_top = C_Y + ROW_I * CELL
for n in range(L_SPAN):
    bx = B_X + N * CELL
    by = B_Y + (L0 + n) * CELL + CELL / 2
    cy_entry = row_top + (n + 0.5) * (CELL / L_SPAN)
    svg.append(fan_curve(bx, by, cx_entry, cy_entry, "fan-vector", marker="arrow-vector"))

# ---- footer ------------------------------------------------------------------
svg.append(f'<text x="48" y="{H - 96}" class="callout">'
          f'Amber = RVV vector instructions (B&#8217;s <tspan class="mono">vle64_v_f64m1</tspan> loads, '
          f'C&#8217;s vector load/store) — <tspan class="mono">vl</tspan> elements moved per instruction. '
          f'Blue = plain scalar float loads (A) — one element per instruction, never vectorized here.</text>')
svg.append(f'<text x="48" y="{H - 74}" class="callout">'
          f'Numbered arrows: each scalar <tspan class="mono">a</tspan><tspan class="mono">N</tspan> pairs '
          f'with row <tspan class="mono">l+N</tspan> of B (blue, 1-to-1) — those 8 vector products then '
          f'all converge into the same <tspan class="mono">C[i, j:j+vl]</tspan> (amber funnel, the reduction).</text>')
svg.append(f'<text x="48" y="{H - 52}" class="callout">'
          f'Repeating this over all <tspan class="mono">l</tspan> (rightward through B/A) accumulates '
          f'the full row; repeating over all <tspan class="mono">j</tspan> (rightward through C) and '
          f'<tspan class="mono">i</tspan> (downward through A/C) covers the whole output.</text>')
svg.append(f'<text x="{W - 48}" y="{H - 18}" text-anchor="end" class="callout mono">'
          f'see opt_gemm_dataflow.svg for the register/instruction-level view of this same tile</text>')

svg.append('</svg>')

out_path = "doc/opt_gemm_arrays.svg"
with open(out_path, "w") as f:
    f.write("\n".join(svg))
print(f"wrote {out_path}")
