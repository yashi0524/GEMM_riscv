#!/usr/bin/env python3
"""Per-kernel stall breakdown from a gem5 O3PipeView trace, as used for
doc/perf_analysis_O3CPU.md.

Usage:
    script/analyze_o3_pipeview.py TRACE BINARY [KERNEL ...]
    e.g. script/analyze_o3_pipeview.py test/m5out_pipeview/pipeview.trace \
             test/gemm_riscv opt_gemm opt_gemm_blocked

The trace comes from
    gem5.opt --debug-flags=O3PipeView --debug-file=pipeview.trace ...
Kernel PC ranges are taken from the binary's symbol table (llvm-nm-18 -S),
so they track rebuilds. Default kernels: opt_gemm, opt_gemm_blocked.

For each kernel (records filtered by PC range; retire=0 = squashed) prints:
  1. summary: committed/squashed micro-ops, span, IPC, commit-width histogram
  2. mean per-stage latency, overall and per instruction class
  3. top PCs by total dispatch->issue wait
  4. zero-commit cycles attributed to the ROB head's state in that cycle
  5. squash events, keyed by the last committed instruction before each
  6. fetch-gap cycles split into cold I-line first touch / after squash / other

Pure stdlib apart from llvm-nm-18 for symbol lookup.
"""
import bisect
import collections
import statistics
import subprocess
import sys

TICKS_PER_CYCLE = 1000  # 1 GHz core clock, 1 ps ticks
ICACHE_LINE = 64
STAGES = ['fetch', 'decode', 'rename', 'dispatch', 'issue', 'complete', 'retire']


def kernel_ranges(binary, names):
    out = subprocess.run(['llvm-nm-18', '-S', binary], check=True,
                         capture_output=True, text=True).stdout
    syms = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 4:
            syms[f[3]] = (int(f[0], 16), int(f[0], 16) + int(f[1], 16))
    missing = [n for n in names if n not in syms]
    if missing:
        sys.exit(f'symbols not found in {binary}: {missing}')
    return {n: syms[n] for n in names}


def parse_trace(path):
    recs, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line.startswith('O3PipeView:'):
                continue
            stage = line.split(':', 2)[1]
            if stage == 'fetch':
                f = line.split(':', 6)
                cur = {'fetch': int(f[2]), 'pc': int(f[3], 16), 'upc': int(f[4]),
                       'sn': int(f[5]), 'dis': f[6]}
                recs.append(cur)
            else:
                cur[stage] = int(line.split(':')[2])
    return recs


def cyc(ticks):
    return ticks // TICKS_PER_CYCLE


def instr_class(dis):
    m = dis.split()[0]
    if m.startswith('vfmacc'):
        return 'vfmacc'
    if m.startswith('vl') and not m.startswith('vlm'):
        return 'vload'
    if m.startswith('vs') and m[2:3] == 'e':
        return 'vstore'
    if m.startswith('v'):
        return 'v-other'
    if m in ('ld', 'lw', 'lh', 'lb', 'lbu', 'lhu', 'lwu', 'fld', 'flw', 'flh',
             'c_ld', 'c_lw', 'c_fld', 'c_ldsp', 'c_lwsp', 'c_fldsp'):
        return 'sload'
    if m.startswith(('b', 'j', 'c_b', 'c_j')):
        return 'branch'
    return 'int/other'


def mean_seg(rs, a, b):
    v = [(r[b] - r[a]) / TICKS_PER_CYCLE for r in rs if r.get(a) and r.get(b)]
    return statistics.mean(v) if v else 0.0


def analyze(name, rs):
    rs = sorted(rs, key=lambda r: r['sn'])
    com = [r for r in rs if r.get('retire', 0) > 0]
    sq = [r for r in rs if r.get('retire', 0) == 0]
    t0, t1 = cyc(min(r['fetch'] for r in rs)), cyc(max(r['retire'] for r in com))
    span = t1 - t0
    commits = collections.Counter(cyc(r['retire']) for r in com)
    cycles = range(t0, t1 + 1)

    print(f'== {name}: committed={len(com)} squashed={len(sq)} '
          f'span={span} cyc IPC={len(com) / span:.2f}')
    hist = collections.Counter(commits[c] for c in cycles)
    print('   commits/cycle histogram:', sorted(hist.items()))

    # 2. per-stage latency
    print('   mean cyc:', ' '.join(
        f'{a[:3]}->{b[:3]}={mean_seg(com, a, b):.1f}' for a, b in zip(STAGES, STAGES[1:])))
    groups = collections.defaultdict(list)
    for r in com:
        groups[instr_class(r['dis'])].append(r)
    print(f"   {'class':10}{'n':>6}{'dis->iss':>9}{'iss->cpl':>9}{'cpl->ret':>9}")
    for c, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f'   {c:10}{len(g):6}{mean_seg(g, "dispatch", "issue"):9.1f}'
              f'{mean_seg(g, "issue", "complete"):9.1f}{mean_seg(g, "complete", "retire"):9.1f}')

    # 3. top dispatch->issue waits by static instruction
    by_pc = collections.defaultdict(list)
    for r in com:
        by_pc[(r['pc'], r['upc'], r['dis'])].append(r)
    top = sorted(by_pc.items(), key=lambda kv: -sum(x['issue'] - x['dispatch'] for x in kv[1]))
    print('   top PCs by total dispatch->issue wait:')
    for (pc, upc, dis), g in top[:10]:
        print(f'     {pc:#x}.{upc} n={len(g):4} wait={mean_seg(g, "dispatch", "issue"):4.1f} '
              f'exe={mean_seg(g, "issue", "complete"):4.1f} '
              f'c->r={mean_seg(g, "complete", "retire"):4.1f}  {dis}')

    # 4. zero-commit cycles -> ROB head state (retire is monotonic in sn)
    rets = [cyc(r['retire']) for r in com]
    reason, head_pcs = collections.Counter(), collections.Counter()
    for c in cycles:
        if commits[c]:
            continue
        h = com[bisect.bisect_right(rets, c)]
        if cyc(h['fetch']) > c:
            why = 'frontend: head not fetched yet'
        elif cyc(h['dispatch']) > c:
            why = 'frontend: head in decode/rename'
        elif cyc(h['issue']) > c:
            why = 'head waiting to issue'
        elif cyc(h['complete']) > c:
            why = 'head executing'
        else:
            why = 'head complete, awaiting commit'
        reason[why] += 1
        head_pcs[(h['pc'], h['dis'], why)] += 1
    print(f'   zero-commit cycles: {sum(reason.values())}')
    for why, n in reason.most_common():
        print(f'     {n:5}  {why}')
    for (pc, dis, why), n in head_pcs.most_common(8):
        print(f'       {n:4} {pc:#x} {dis:40} [{why}]')

    # 5. squash events: runs of squashed sn, keyed by last committed before
    events, prev, run = [], None, None
    for r in rs:
        if r.get('retire', 0) == 0:
            if run is None:
                run = [prev, 0]
                events.append(run)
            run[1] += 1
        else:
            run, prev = None, r
    print(f'   squash events: {len(events)}, squashed micro-ops: {sum(e[1] for e in events)}')
    culprits = collections.Counter((e[0]['pc'], e[0]['dis']) for e in events if e[0])
    for (pc, dis), n in culprits.most_common(6):
        print(f'     {n:3}x after {pc:#x} {dis}')

    # 6. fetch gaps between consecutive committed fetches
    sq_fetch = sorted(cyc(r['fetch']) for r in sq)
    seen, last = set(), None
    cold_lines = cold = after_sq = other = 0
    for r in com:
        f, line = cyc(r['fetch']), r['pc'] // ICACHE_LINE
        if last is not None and f - last > 1:
            gap = f - last - 1
            i = bisect.bisect_right(sq_fetch, f) - 1
            if line not in seen:
                cold_lines += 1
                cold += gap
            elif i >= 0 and sq_fetch[i] >= last:
                after_sq += gap
            else:
                other += gap
        seen.add(line)
        last = f if last is None else max(last, f)
    print(f'   fetch-gap cycles: cold I-line={cold} ({cold_lines} lines, '
          f'{100 * cold / span:.0f}% of span), after squash={after_sq}, other={other}')


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    trace, binary = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or ['opt_gemm', 'opt_gemm_blocked']
    ranges = kernel_ranges(binary, names)
    recs = parse_trace(trace)
    print(f'{len(recs)} micro-op records in {trace}')
    for name, (lo, hi) in ranges.items():
        rs = [r for r in recs if lo <= r['pc'] < hi]
        if not rs:
            print(f'== {name}: no records in [{lo:#x}, {hi:#x})')
            continue
        analyze(name, rs)


if __name__ == '__main__':
    main()
