# -*- coding: utf-8 -*-
"""
summarize_pair.py — paired statistics for improved vs. original GGPP optimizer
==============================================================================
Reads the two arms produced by ``run_pair_eval.py`` (or two arbitrary pkl files),
pairs them by query id, and writes ``PAIR_SUMMARY.md`` / ``PAIR_SUMMARY.csv``.

Reported per arm:
    * strict success  = target at rank 1 AND correct passage out of top-k
                        (official ``check_success`` == 2)
    * correct-passage eviction rate (exact-cosine rank > top-k)
    * target reaches rank 1
    * median rank of the correct passage (capped at top-k + 1)
    * median cosine of the correct passage

Also reported:
    * per-query flips (failure->success / success->failure)
    * stratification by whether the target already held rank 1 *before* the attack
      (this is the condition that predicts whether the target can be lost)
    * the convention effect (bare query vs. Instruct wrapping) recorded per arm

Usage:
    python summarize_pair.py --pair-dir ./results_pair
    python summarize_pair.py --orig path/orig.pkl --rank path/rank.pkl
    python summarize_pair.py --pair-dir ./results_pair --out PAIR_SUMMARY.md
"""

import argparse
import csv
import os
import pickle
import statistics as st
import sys

p = argparse.ArgumentParser(description='Paired statistics for the GGPP optimizer A/B')
p.add_argument('--pair-dir', type=str, default='./results_pair')
p.add_argument('--orig', type=str, default=None, help='explicit orig pkl (overrides --pair-dir)')
p.add_argument('--rank', type=str, default=None, help='explicit rank pkl (overrides --pair-dir)')
p.add_argument('--out', type=str, default=None, help='markdown output path')
args = p.parse_args()

TOPK = 10
CAP = TOPK + 1


def load_dir(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(d, fn), 'rb') as f:
                rec = pickle.load(f)
            for k, v in rec.items():
                out[int(k)] = v
        except Exception as e:
            print(f"[warn] cannot read {os.path.join(d, fn)}: {type(e).__name__} {e}")
    return out


if args.orig and args.rank:
    with open(args.orig, 'rb') as f:
        ORIG = {int(k): v for k, v in pickle.load(f).items()}
    with open(args.rank, 'rb') as f:
        RANK = {int(k): v for k, v in pickle.load(f).items()}
    out_md = args.out or os.path.join(os.path.dirname(os.path.abspath(args.rank)), 'PAIR_SUMMARY.md')
else:
    ORIG = load_dir(os.path.join(args.pair_dir, 'orig'))
    RANK = load_dir(os.path.join(args.pair_dir, 'rank'))
    out_md = args.out or os.path.join(args.pair_dir, 'PAIR_SUMMARY.md')

ids = sorted(set(ORIG) & set(RANK))
if not ids:
    sys.exit("no overlapping query ids between the two arms — nothing to compare")

L = []
w = L.append


def pct(a, b):
    return f"{a / b:.1%}" if b else "n/a"


def med(xs):
    return st.median(xs) if xs else float('nan')


def cap(x):
    return min(max(int(x), 1), CAP)


# ---------------- header ----------------
w("# GGPP optimizer A/B — paired statistics")
w("")
w(f"- paired queries: **{len(ids)}** (ids {min(ids)} ~ {max(ids)})")
w(f"- strict success = target at rank 1 **and** correct passage pushed out of "
  f"top-{TOPK} (official `check_success == 2`)")
w("- ranks and cosines use exact corpus-wide cosine; strict success comes from the "
  "official judge")
w("")

# ---------------- headline ----------------
w("## 1. Headline")
w("")
w("| metric | original | rank-aware | change |")
w("|---|---|---|---|")
o_s = [r for r in (ORIG[i] for i in ids) if r.get('succ') == 2]
m_s = [r for r in (RANK[i] for i in ids) if r.get('succ') == 2]
w(f"| **strict success** | {len(o_s)}/{len(ids)} = {pct(len(o_s), len(ids))} "
  f"| {len(m_s)}/{len(ids)} = {pct(len(m_s), len(ids))} "
  f"| {(len(m_s) - len(o_s)) / len(ids) * 100:+.1f} pt |")
o_e = [i for i in ids if ORIG[i].get('evict') == 1]
m_e = [i for i in ids if RANK[i].get('evict') == 1]
w(f"| correct passage evicted from top-{TOPK} | {pct(len(o_e), len(ids))} "
  f"| {pct(len(m_e), len(ids))} "
  f"| {(len(m_e) - len(o_e)) / len(ids) * 100:+.1f} pt |")
o_t = [i for i in ids if ORIG[i].get('rank_t') == 1]
m_t = [i for i in ids if RANK[i].get('rank_t') == 1]
w(f"| target reaches rank 1 | {pct(len(o_t), len(ids))} | {pct(len(m_t), len(ids))} "
  f"| {(len(m_t) - len(o_t)) / len(ids) * 100:+.1f} pt |")
w(f"| median rank of correct passage (cap {CAP}) | {med([cap(ORIG[i]['rank_u']) for i in ids]):.1f} "
  f"| {med([cap(RANK[i]['rank_u']) for i in ids]):.1f} "
  f"| {med([cap(RANK[i]['rank_u']) for i in ids]) - med([cap(ORIG[i]['rank_u']) for i in ids]):+.1f} |")
w(f"| median cosine of correct passage | {med([ORIG[i]['cos_u'] for i in ids]):.4f} "
  f"| {med([RANK[i]['cos_u'] for i in ids]):.4f} "
  f"| {med([RANK[i]['cos_u'] for i in ids]) - med([ORIG[i]['cos_u'] for i in ids]):+.4f} |")
w("")

# ---------------- flips ----------------
f2s = [i for i in ids if ORIG[i].get('succ') != 2 and RANK[i].get('succ') == 2]
s2f = [i for i in ids if ORIG[i].get('succ') == 2 and RANK[i].get('succ') != 2]
both = [i for i in ids if ORIG[i].get('succ') == 2 and RANK[i].get('succ') == 2]
neither = [i for i in ids if ORIG[i].get('succ') != 2 and RANK[i].get('succ') != 2]
w("## 2. Per-query flips")
w("")
w("| category | queries |")
w("|---|---|")
w(f"| fail -> success | **{len(f2s)}** |")
w(f"| success -> fail | **{len(s2f)}** |")
w(f"| both succeed | {len(both)} |")
w(f"| both fail | {len(neither)} |")
w("")

# ---------------- stratification ----------------
w("## 3. Stratified by whether the target already held rank 1 before the attack")
w("")
w("| group | n | original strict success | rank-aware strict success | target kept at rank 1 |")
w("|---|---|---|---|---|")
for label, sel in (("target **already** rank 1", [i for i in ids if RANK[i].get('init_rank_t') == 1]),
                   ("target **not** rank 1", [i for i in ids if RANK[i].get('init_rank_t') != 1])):
    if not sel:
        continue
    a = sum(1 for i in sel if ORIG[i].get('succ') == 2)
    b = sum(1 for i in sel if RANK[i].get('succ') == 2)
    kept = sum(1 for i in sel if RANK[i].get('rank_t') == 1)
    w(f"| {label} | {len(sel)} | {a} ({pct(a, len(sel))}) | {b} ({pct(b, len(sel))}) "
      f"| {kept}/{len(sel)} |")
lost = [i for i in ids if RANK[i].get('init_rank_t') != 1 and RANK[i].get('rank_t') != 1]
if lost:
    w("")
    w(f"queries where the rank-aware optimizer loses target rank 1: `{lost}` "
      f"(all in the \"target not rank 1\" group)")
w("")

# ---------------- convention side note ----------------
alt_o = [i for i in ids if ORIG[i].get('succ_alt') == 2]
alt_m = [i for i in ids if RANK[i].get('succ_alt') == 2]
w("## 4. Judging-convention side note (same prefixes, re-judged under the opposite wrapper)")
w("")
w("| judging convention | original strict success | rank-aware strict success |")
w("|---|---|---|")
w(f"| primary ({ORIG[ids[0]].get('judge_mode', 'bare')}) | {pct(len(o_s), len(ids))} "
  f"| {pct(len(m_s), len(ids))} |")
w(f"| opposite wrapper | {pct(len(alt_o), len(ids))} | {pct(len(alt_m), len(ids))} |")
w("")

# ---------------- cost ----------------
eo = [ORIG[i].get('elapsed', 0) for i in ids]
em = [RANK[i].get('elapsed', 0) for i in ids]
ep_o = [ORIG[i].get('epochs', 0) for i in ids]
ep_m = [RANK[i].get('epochs', 0) for i in ids]
w("## 5. Cost")
w("")
w(f"- original: median {med(eo):.0f}s/query, total {sum(eo) / 60:.1f} min")
w(f"- rank-aware: median {med(em):.0f}s/query, total {sum(em) / 60:.1f} min")
if ep_m:
    cap_m = max(ep_m)
    w(f"- rank-aware iterations: median {med(ep_m):.0f}, cap {cap_m}; "
      f"hit the cap (no early stop) {sum(1 for x in ep_m if x >= cap_m)}/{len(ids)}")
w("")

# ---------------- per-query table ----------------
w("## 6. Per-query detail")
w("")
w("| id | orig:strict | rank:strict | orig:rank_correct | rank:rank_correct | "
  "orig:rank_target | rank:rank_target | orig:cos_u | rank:cos_u | flip |")
w("|---|---|---|---|---|---|---|---|---|---|")
flip = {**{i: "fail->success" for i in f2s}, **{i: "success->fail" for i in s2f},
        **{i: "both succeed" for i in both}, **{i: "both fail" for i in neither}}
for i in ids:
    o, m = ORIG[i], RANK[i]
    w(f"| {i} | {'✓' if o.get('succ') == 2 else '✗'} | {'✓' if m.get('succ') == 2 else '✗'} "
      f"| {cap(o['rank_u'])} | {cap(m['rank_u'])} | {o.get('rank_t')} | {m.get('rank_t')} "
      f"| {o['cos_u']:.4f} | {m['cos_u']:.4f} | {flip[i]} |")
w("")

# ---------------- write ----------------
os.makedirs(os.path.dirname(os.path.abspath(out_md)) or '.', exist_ok=True)
with open(out_md, 'w', encoding='utf-8') as f:
    f.write("\n".join(L) + "\n")

csv_path = os.path.splitext(out_md)[0] + '.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    wtr = csv.writer(f)
    wtr.writerow(['id', 'orig_succ', 'rank_succ', 'orig_rank_u', 'rank_rank_u',
                  'orig_rank_t', 'rank_rank_t', 'orig_cos_u', 'rank_cos_u',
                  'orig_evict', 'rank_evict', 'init_rank_t', 'flip',
                  'orig_epochs', 'rank_epochs', 'orig_elapsed', 'rank_elapsed'])
    for i in ids:
        o, m = ORIG[i], RANK[i]
        wtr.writerow([i, o.get('succ'), m.get('succ'), o.get('rank_u'), m.get('rank_u'),
                      o.get('rank_t'), m.get('rank_t'), round(o.get('cos_u', 0), 4),
                      round(m.get('cos_u', 0), 4), o.get('evict'), m.get('evict'),
                      m.get('init_rank_t'), flip[i], o.get('epochs'), m.get('epochs'),
                      o.get('elapsed'), m.get('elapsed')])

print(f"pairs: {len(ids)}")
print(f"strict success  orig {len(o_s)}/{len(ids)} = {pct(len(o_s), len(ids))}"
      f"   rank {len(m_s)}/{len(ids)} = {pct(len(m_s), len(ids))}")
print(f"flips: fail->success {len(f2s)}   success->fail {len(s2f)}")
print(f"eviction rate   orig {pct(len(o_e), len(ids))}   rank {pct(len(m_e), len(ids))}")
print(f"written: {out_md}")
print(f"written: {csv_path}")
