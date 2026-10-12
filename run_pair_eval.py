# -*- coding: utf-8 -*-
"""
run_pair_eval.py — paired A/B harness: improved optimizer vs. original
====================================================================
Runs the *same* queries, with the *same* target, the *same* initial prefix and
the *same* seed, through both optimizers, so the comparison is per-query paired
instead of two independent samples.

Paired design
-------------
* ``user_prompt`` / ``target_prompt_id`` / ``search_range`` are read from the same
  ``--baseline-dir`` for both arms (or, when absent, derived by the official rule
  with the same per-query seed) -> identical attack targets.
* The initial prefix is recomputed deterministically from ``target_text``
  (``rank_tokens_by_importance`` -> ``get_initial_ids``) -> identical starting point.
* Per query, seed = ``--seed`` + index for both arms.

Usage
-----
    # sequential (one GPU)
    python run_pair_eval.py --start 208 --end 308

    # two GPUs in parallel
    python run_pair_eval.py --start 208 --end 308 --gpu-orig 2 --gpu-rank 0

    # only the improved arm (e.g. to extend an existing run)
    python run_pair_eval.py --start 308 --end 408 --arms rank

Output: ``<pair-dir>/orig/`` and ``<pair-dir>/rank/`` (pkl + csv per arm),
plus ``<pair-dir>/PAIR_SUMMARY.md`` and ``PAIR_SUMMARY.csv``.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'SFR_mistral_IMDB_rank.py')
SUMMARIZER = os.path.join(HERE, 'summarize_pair.py')

p = argparse.ArgumentParser(description='Paired A/B evaluation for the rank-aware optimizer')
p.add_argument('--start', type=int, default=208)
p.add_argument('--end', type=int, default=308)
p.add_argument('--iterations', type=int, default=500)
p.add_argument('--seed', type=int, default=20261008)
p.add_argument('--firstk', type=int, default=10)
p.add_argument('--batch', type=int, default=1, help='output batch number')
p.add_argument('--baseline-dir', type=str, default='./results_baseline',
               help='existing pkl dir; both arms read target/search_range from here')
p.add_argument('--pair-dir', type=str, default='./results_pair')
p.add_argument('--arms', nargs='+', choices=['orig', 'rank'], default=['orig', 'rank'])
p.add_argument('--gpu-orig', type=int, default=None, help='run the orig arm on this GPU index')
p.add_argument('--gpu-rank', type=int, default=None, help='run the rank arm on this GPU index')
p.add_argument('--limit', type=int, default=0)
p.add_argument('--no-summary', action='store_true')
args = p.parse_args()

os.makedirs(args.pair_dir, exist_ok=True)


def build_cmd(arm):
    cmd = [sys.executable, RUNNER,
           '--arm', arm,
           '--batch', str(args.batch),
           '--start', str(args.start),
           '--end', str(args.end),
           '--iterations', str(args.iterations),
           '--seed', str(args.seed),
           '--firstk', str(args.firstk),
           '--baseline-dir', args.baseline_dir,
           '--outdir', os.path.join(args.pair_dir, arm)]
    if args.limit > 0:
        cmd += ['--limit', str(args.limit)]
    return cmd


def launch(arm, gpu):
    env = dict(os.environ)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return subprocess.Popen(build_cmd(arm), env=env)


def run_sequential(arms):
    for arm in arms:
        gpu = args.gpu_orig if arm == 'orig' else args.gpu_rank
        print(f"\n===== arm={arm} (gpu={gpu}) =====", flush=True)
        env = dict(os.environ)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        rc = subprocess.call(build_cmd(arm), env=env)
        if rc != 0:
            sys.exit(f"arm={arm} exited with code {rc}")


def run_parallel(arms):
    procs = []
    for arm in arms:
        gpu = args.gpu_orig if arm == 'orig' else args.gpu_rank
        print(f"launch arm={arm} on gpu={gpu}", flush=True)
        procs.append((arm, launch(arm, gpu)))
    rc = 0
    for arm, pr in procs:
        code = pr.wait()
        print(f"arm={arm} exited with code {code}", flush=True)
        rc = rc or code
    if rc != 0:
        sys.exit(f"at least one arm failed (rc={rc})")


def main():
    t0 = time.time()
    print(f"pair dir: {args.pair_dir}  ids=[{args.start},{args.end})  "
          f"arms={args.arms}", flush=True)

    if len(args.arms) == 2 and (args.gpu_orig is not None or args.gpu_rank is not None):
        run_parallel(args.arms)
    else:
        run_sequential(args.arms)

    print(f"\npair eval finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not args.no_summary:
        rc = subprocess.call([sys.executable, SUMMARIZER, '--pair-dir', args.pair_dir])
        if rc != 0:
            print(f"[warn] summarizer exited with code {rc}", flush=True)


if __name__ == '__main__':
    main()
