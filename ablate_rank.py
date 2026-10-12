# -*- coding: utf-8 -*-
"""
ablate_rank.py — one-variable ablation matrix for the rank-aware optimizer
=========================================================================
The paired evaluation compares the *whole* improved optimizer against the
original, so the +Δpt is measured but not attributed per change.  This script
runs the same improved optimizer while varying exactly one knob at a time, which
is what turns the mechanism story into a measured one.

Configurations (each differs from ``full`` in a single field):

    full        -- all changes on (reference)
    snapshot    -- snapshot_by=loss            (undo: rank-best snapshot)
    tau-strict  -- tau=0.95                    (how strict the target constraint is)
    no-escape   -- stall_patience=huge         (undo: stall escape)
    small-cand  -- n_random=16                 (search budget / candidate set)
    long-prefix -- firstk=20                   (attack capacity, not a loss change)
    no-target   -- lam_t=0                     (does the target constraint matter at all)

Usage
-----
    # all configs, 40 queries, one GPU
    python ablate_rank.py --start 208 --end 248

    # a subset
    python ablate_rank.py --start 208 --end 248 --configs full snapshot no-target

    # compare each config against an existing orig arm when done
    python ablate_rank.py --start 208 --end 248 --orig-dir ./results_pair/orig

Output: ``<ablate-dir>/<config>/*.pkl``, plus ``ABLATION_SUMMARY.csv``/``.md``
when ``--orig-dir`` is given.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'SFR_mistral_IMDB_rank.py')

CONFIGS = {
    'full':        {},
    'snapshot':    {'snapshot_by': 'loss'},
    'tau-strict':  {'tau': 0.95},
    'no-escape':   {'stall_patience': 10 ** 9},
    'small-cand':  {'n_random': 16},
    'long-prefix': {'firstk': 20},
    'no-target':   {'lam_t': 0.0},
}

p = argparse.ArgumentParser(description='One-variable ablation for the rank-aware GGPP optimizer')
p.add_argument('--start', type=int, default=208)
p.add_argument('--end', type=int, default=248)
p.add_argument('--iterations', type=int, default=500)
p.add_argument('--seed', type=int, default=20261008)
p.add_argument('--batch', type=int, default=1)
p.add_argument('--baseline-dir', type=str, default='./results_baseline')
p.add_argument('--ablate-dir', type=str, default='./results_ablation')
p.add_argument('--orig-dir', type=str, default=None,
               help='an existing orig arm; if given, each config is compared against it')
p.add_argument('--configs', nargs='+', choices=sorted(CONFIGS), default=sorted(CONFIGS))
p.add_argument('--gpu', type=int, default=None)
p.add_argument('--limit', type=int, default=0)
args = p.parse_args()

os.makedirs(args.ablate_dir, exist_ok=True)

FLAGS = {'snapshot_by': '--snapshot-by', 'tau': '--tau', 'stall_patience': '--stall-patience',
         'n_random': '--n-random', 'firstk': '--firstk', 'lam_t': '--lam-t'}

summary = []
t_all = time.time()

for name in args.configs:
    over = CONFIGS[name]
    outdir = os.path.join(args.ablate_dir, name)
    os.makedirs(outdir, exist_ok=True)

    cmd = [sys.executable, RUNNER, '--arm', 'rank', '--batch', str(args.batch),
           '--start', str(args.start), '--end', str(args.end),
           '--iterations', str(args.iterations), '--seed', str(args.seed),
           '--baseline-dir', args.baseline_dir, '--outdir', outdir,
           '--tag', '_' + name]
    for k, v in over.items():
        cmd += [FLAGS[k], str(v)]
    if args.limit > 0:
        cmd += ['--limit', str(args.limit)]

    env = dict(os.environ)
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"\n{'=' * 70}\n[ablate] config={name}  overrides={over}\n{'=' * 70}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    dt = (time.time() - t0) / 60
    print(f"[ablate] config={name} rc={rc} in {dt:.1f} min", flush=True)
    if rc != 0:
        print(f"[warn] config={name} failed, keeping whatever it wrote", flush=True)

    # ---- optional comparison against the orig arm ----
    if args.orig_dir:
        import pickle
        def load(d):
            out = {}
            if not os.path.isdir(d):
                return out
            for fn in sorted(os.listdir(d)):
                if fn.endswith('.pkl'):
                    with open(os.path.join(d, fn), 'rb') as f:
                        out.update({int(k): v for k, v in pickle.load(f).items()})
            return out
        O, R = load(args.orig_dir), load(outdir)
        ids = sorted(set(O) & set(R))
        if ids:
            o_s = sum(1 for i in ids if O[i].get('succ') == 2)
            r_s = sum(1 for i in ids if R[i].get('succ') == 2)
            f2s = sum(1 for i in ids if O[i].get('succ') != 2 and R[i].get('succ') == 2)
            s2f = sum(1 for i in ids if O[i].get('succ') == 2 and R[i].get('succ') != 2)
            row = {'config': name, 'n': len(ids), 'orig_strict': o_s,
                   'rank_strict': r_s, 'delta_pt': round((r_s - o_s) / len(ids) * 100, 1),
                   'fail_to_success': f2s, 'success_to_fail': s2f, 'minutes': round(dt, 1)}
            summary.append(row)
            print(f"[ablate] {name}: {o_s} -> {r_s} strict  ({row['delta_pt']:+.1f} pt), "
                  f"{f2s} up / {s2f} down", flush=True)

if summary:
    os.makedirs(args.ablate_dir, exist_ok=True)
    csv_path = os.path.join(args.ablate_dir, 'ABLATION_SUMMARY.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    md_path = os.path.join(args.ablate_dir, 'ABLATION_SUMMARY.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# One-variable ablation (rank-aware optimizer)\n\n")
        f.write(f"- queries: {summary[0]['n']} (ids {args.start}~{args.end - 1})\n")
        f.write(f"- reference arm: original optimizer ({args.orig_dir})\n\n")
        f.write("| config | change relative to 'full' | original strict | config strict | delta | fail->success | success->fail |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in summary:
            over = CONFIGS.get(r['config'], {})
            desc = ", ".join(f"{k}={v}" for k, v in over.items()) or "(reference, all changes on)"
            f.write(f"| {r['config']} | {desc} | {r['orig_strict']}/{r['n']} | "
                    f"{r['rank_strict']}/{r['n']} | {r['delta_pt']:+.1f} pt | "
                    f"{r['fail_to_success']} | {r['success_to_fail']} |\n")
    print(f"\nwritten: {csv_path}")
    print(f"written: {md_path}")

print(f"\ntotal {(time.time() - t_all) / 60:.1f} min")
