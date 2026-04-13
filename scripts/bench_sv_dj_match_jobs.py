#!/usr/bin/env python3
"""
Compare wall-clock time for SV_DJ matching with --jobs 1 vs --jobs N on the same
enrolled folder and pair list (plan: LFW).

Usage (from repo root):
  python3 scripts/bench_sv_dj_match_jobs.py \\
    --folder results/sv_dj/lfw \\
    --pair_list data/lfw/pair.list \\
    --feat_list data/lfw/lfw_feat.list \\
    --key_size 512 --K 64 --s 1 \\
    --compare_jobs 8

Writes results/sv_dj/bench_match_jobs_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def run_match(repo_root: str, jobs: int, args_ns: argparse.Namespace) -> dict:
    base, ext = os.path.splitext(args_ns.score_list)
    if not ext:
        ext = '.list'
    score_path = '{}.jobs{}{}'.format(base, jobs, ext)

    cmd = [
        sys.executable,
        os.path.join(repo_root, 'libs', 'SV_DJ', 'crypto_system.py'),
        '--folder',
        args_ns.folder,
        '--pair_list',
        args_ns.pair_list,
        '--score_list',
        score_path,
        '--feat_list',
        args_ns.feat_list,
        '--key_size',
        str(args_ns.key_size),
        '--K',
        str(args_ns.K),
        '--s',
        str(args_ns.s),
        '--jobs',
        str(jobs),
    ]
    if args_ns.crt_decrypt:
        cmd.append('--crt_decrypt')
    env = os.environ.copy()
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    out = (proc.stdout or '') + (proc.stderr or '')

    total_dur = None
    m = re.search(r'total duration ([0-9.]+)s', out)
    if m:
        total_dur = float(m.group(1))

    metrics_path = os.path.join(args_ns.folder, 'metrics_dj.json')
    metrics_eff = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                full = json.load(f)
            metrics_eff = full.get('efficiency', {})
        except (json.JSONDecodeError, OSError):
            pass

    return {
        'jobs': jobs,
        'wall_seconds': round(wall, 4),
        'reported_total_duration_s': total_dur,
        'metrics_dj_efficiency': metrics_eff,
        'returncode': proc.returncode,
        'stdout_tail': out[-6000:] if out else '',
    }


def main():
    cpu_n = os.cpu_count() or 8
    p = argparse.ArgumentParser(description='Benchmark SV_DJ crypto_system --jobs 1 vs N')
    p.add_argument('--folder', default='results/sv_dj/lfw')
    p.add_argument('--pair_list', default='data/lfw/pair.list')
    p.add_argument('--feat_list', default='data/lfw/lfw_feat.list')
    p.add_argument('--score_list', default='results/sv_dj/lfw/bench_score.tmp')
    p.add_argument('--key_size', type=int, default=512)
    p.add_argument('--K', type=int, default=64)
    p.add_argument('--s', type=int, default=1)
    p.add_argument(
        '--compare_jobs',
        type=int,
        default=max(2, min(cpu_n, 16)),
        help='Second run uses this many workers (default: min(16, cpu_count))',
    )
    p.add_argument('--crt_decrypt', action='store_true')
    args = p.parse_args()

    repo = _repo_root()

    results = []
    for j in (1, args.compare_jobs):
        print('=== Running --jobs {} ==='.format(j), flush=True)
        row = run_match(repo, j, args)
        results.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        if row['returncode'] != 0:
            print('Run failed; see stdout_tail in JSON', file=sys.stderr)
            break

    speedup = None
    if len(results) == 2 and results[0]['wall_seconds'] and results[0]['wall_seconds'] > 0:
        speedup = round(results[0]['wall_seconds'] / results[1]['wall_seconds'], 3)

    out_path = os.path.join(
        repo,
        'results',
        'sv_dj',
        'bench_match_jobs_{}.json'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'config': vars(args),
        'wall_clock_speedup_jobs1_over_N': speedup,
        'runs': results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
