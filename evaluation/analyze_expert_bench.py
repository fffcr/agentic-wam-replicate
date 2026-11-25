#!/usr/bin/env python3
"""Analyze detail_calls.csv and optional PyTorch trace.json, without a GPU."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys


def select_round(events):
    """Prefer CPU user annotations; Kineto can also emit same-name GPU annotations."""
    rounds = [e for e in events if e.get('name') == 'BENCH_ROUND'
              and e.get('ph') == 'X' and 'dur' in e]
    cpu = [e for e in rounds if 'user_annotation' in e.get('cat', '').split(',')]
    if len(cpu) == 1:
        selected = cpu[0]
    elif not cpu and len(rounds) == 1 and not rounds[0].get('cat'):
        selected = rounds[0]  # older exporters without categories
    else:
        diagnostics = [{k: e.get(k) for k in ('cat', 'ph', 'pid', 'tid', 'ts', 'dur')}
                       for e in events if e.get('name') == 'BENCH_ROUND']
        raise ValueError('Cannot identify exactly one CPU BENCH_ROUND: '
                         f'cpu_ranges={len(cpu)}, complete_ranges={len(rounds)}, '
                         f'markers={diagnostics[:20]}')
    return selected['ts'], selected['ts'] + selected['dur']


def union_duration(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(right, end) if end is not None else right
    return total


def analyze_detail(path):
    stages = defaultdict(list)
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            stages[row['stage']].append(float(row['ms']))
    totals = {k: sum(v) for k, v in stages.items()}
    action = {k: v for k, v in totals.items() if k.startswith('action.')}
    total = sum(action.values())
    return dict(action_total_ms=total, action_stages={
        k: dict(total_ms=v, calls=len(stages[k]), mean_call_ms=v / len(stages[k]),
                action_share=v / total if total else 0)
        for k, v in sorted(action.items(), key=lambda item: -item[1])},
        warning='Event intervals include launch gaps and instrumentation overhead, not kernel busy time.')


def analyze_trace(path):
    with Path(path).open() as f:
        events = json.load(f)['traceEvents']
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    left, right = select_round(events)
    complete = [e for e in complete if left <= e['ts'] < right]
    # Index expert ranges by CPU thread. Segment queries avoid scanning all ranges per kernel.
    markers = defaultdict(list)
    for i, e in enumerate(complete):
        if (e.get('name', '').startswith('expert.') and
                'gpu_user_annotation' not in e.get('cat', '')):
            key = (e.get('pid'), e.get('tid'))
            markers[key].append((e['ts'], 1, i, e['name']))
            markers[key].append((e['ts'] + e['dur'], 0, i, e['name']))
    segments = {}
    for key, points in markers.items():
        active, starts, labels = {}, [], []
        for ts, start, ident, name in sorted(points):
            if start:
                active[ident] = (ts, name)
            else:
                active.pop(ident, None)
            starts.append(ts)
            labels.append(max(active.values(), default=(0, 'unattributed'))[1])
        segments[key] = (starts, labels)

    def annotation(e):
        segment = segments.get((e.get('pid'), e.get('tid')))
        if segment is None:
            return 'unattributed'
        starts, labels = segment
        i = bisect_right(starts, e['ts']) - 1
        return labels[i] if i >= 0 else 'unattributed'

    external = {}
    correlations = {}
    # CPU operator -> expert range, then CUDA runtime correlation -> expert range.
    for e in complete:
        if e.get('cat') == 'cpu_op':
            ext = e.get('args', {}).get('External id')
            if ext is not None:
                external[str(ext)] = annotation(e)
    for e in complete:
        if 'cuda_runtime' in e.get('cat', '') or 'cuda_driver' in e.get('cat', ''):
            args = e.get('args', {})
            corr = args.get('correlation')
            if corr is not None:
                label = external.get(str(args.get('External id')), 'unattributed')
                correlations[str(corr)] = annotation(e) if label == 'unattributed' else label

    intervals, by_group, by_operator = [], defaultdict(list), defaultdict(list)
    unattributed, kernels = 0, []
    for e in complete:
        if 'kernel' not in e.get('cat', '').lower():
            continue
        args = e.get('args', {})
        interval = (max(left, e['ts']), min(right, e['ts'] + e['dur']))
        label = external.get(str(args.get('External id')), 'unattributed')
        if label == 'unattributed':
            label = correlations.get(str(args.get('correlation')), 'unattributed')
        group = label
        if label.startswith('expert.action.block_'):
            group = 'expert.action.' + label.rsplit('.', 1)[1]
        intervals.append(interval)
        by_group[group].append(interval)
        by_operator[e.get('name', 'unknown')].append(interval)
        kernels.append(e)
        unattributed += label == 'unattributed'
    if not kernels:
        raise ValueError('No GPU kernel events found; check CUDA profiler support')
    busy = union_duration(intervals)
    span = max(e for _, e in intervals) - min(s for s, _ in intervals)
    groups = {name: dict(kernel_count=len(v),
                         summed_kernel_ms=sum(e-s for s, e in v) / 1000,
                         union_kernel_ms=union_duration(v) / 1000)
              for name, v in by_group.items()}
    return dict(profiled_round_ms=(right-left)/1000, kernel_count=len(kernels),
                kernel_union_ms=busy/1000, kernel_span_ms=span/1000,
                kernel_uncovered_within_span_ms=(span-busy)/1000,
                kernel_coverage_of_span=busy/span if span else 0,
                unattributed_kernel_count=unattributed,
                attributed_groups=groups,
                top_kernels=sorted((dict(name=k, calls=len(v),
                    summed_ms=sum(e-s for s, e in v)/1000) for k, v in by_operator.items()),
                    key=lambda x: -x['summed_ms'])[:20],
                warning='Kernel-uncovered gaps can include copies, synchronization and CPU launch '
                        'gaps; this is not a direct measurement of CPU overhead or SM utilization. '
                        'Per-group sums may overlap on multiple streams. Attribution uses trace '
                        'external/correlation IDs; inspect unattributed count. Profiler changes latency.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--details-only', action='store_true', help='Analyze CUDA-event CSV without opening trace')
    p.add_argument('--strict-trace', action='store_true', help='Exit nonzero if trace analysis fails')
    a = p.parse_args()
    result = {}
    if (a.run / 'detail_calls.csv').exists():
        result['detail'] = analyze_detail(a.run / 'detail_calls.csv')
    if not a.details_only and (a.run / 'trace.json').exists():
        try:
            result['trace'] = analyze_trace(a.run / 'trace.json')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result['trace_error'] = f'{type(exc).__name__}: {exc}'
            print('TRACE_ANALYSIS_FAILED (existing capture and timings retained): '
                  + result['trace_error'], file=sys.stderr)
    if not result:
        p.error('No detail_calls.csv or trace.json in run directory')
    result['analysis_status'] = 'partial' if 'trace_error' in result else 'complete'
    result['trace_status'] = ('failed' if 'trace_error' in result else
                              'ok' if 'trace' in result else 'not_analyzed')
    text = json.dumps(result, indent=2)
    (a.run / 'analysis.json').write_text(text)
    print(text)
    if a.strict_trace and 'trace_error' in result:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
