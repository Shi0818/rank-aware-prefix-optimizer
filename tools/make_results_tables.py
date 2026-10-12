# -*- coding: utf-8 -*-
"""
Build the result tables shipped in results/ from the per-query paired detail
============================================================================
Everything in ``results/`` is generated, never hand-typed, so the numbers in the
README, in ``docs/RESULTS.md`` and in the CSVs cannot drift apart.

Input is the per-query paired detail table produced by the experiment
(columns in Chinese, one row per query, 100 rows for the reported run).
Point ``--detail-csv`` at it and the three output files are rewritten:

    results/per_query_paired.csv       100 rows, English column names
    results/pair_summary.csv           headline + stratified metrics
    results/pair_flip_and_groups.csv   flip classification + grouped analysis

Usage:
    python tools/make_results_tables.py --detail-csv <path to detail csv>
"""

import argparse
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT_DIR = os.path.join(PKG, "results")

# Chinese -> English column names
COLS = {
    "样本ID": "query_id",
    "攻击目标ID": "target_id",
    "历史_正确文本名次": "orig_rank_correct",
    "历史_名次_截断11": "orig_rank_correct_cap11",
    "历史_目标名次": "orig_rank_target",
    "历史_严格成功": "orig_strict_success",
    "改进_正确文本名次": "rank_rank_correct",
    "改进_名次_截断11": "rank_rank_correct_cap11",
    "改进_目标名次": "rank_rank_target",
    "改进_严格成功": "rank_strict_success",
    "改进_是否出榜": "rank_evicted",
    "配对翻转": "pair_flip",
    "攻击前_正确文本名次": "pre_rank_correct",
    "攻击前_目标名次": "pre_rank_target",
    "攻击前_正确文本余弦": "pre_cos_correct",
    "攻击后_正确文本余弦": "post_cos_correct",
    "攻击后_目标余弦": "post_cos_target",
    "耗时秒": "elapsed_s",
    "迭代轮数": "iterations",
    "Instruct口径_严格成功": "instruct_strict_success",
}

YESNO = {"是": "yes", "否": "no"}
FLIP = {
    "失败→成功": "fail_to_success",
    "成功→失败": "success_to_fail",
    "都成功": "both_success",
    "都失败": "both_fail",
}


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            name = COLS[k]
            if v in YESNO:
                v = YESNO[v]
            elif v in FLIP:
                v = FLIP[v]
            d[name] = v
        out.append(d)
    return out


def pct(n, d):
    return f"{n}/{d} = {100.0 * n / d:.1f}%"


def med(vals):
    return st.median(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-csv", required=True)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    rows = load(args.detail_csv)
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 1) per-query detail ----------
    fields = list(COLS.values())
    with open(os.path.join(args.outdir, "per_query_paired.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] per_query_paired.csv  ({len(rows)} rows)")

    n = len(rows)
    orig_ss = sum(1 for r in rows if r["orig_strict_success"] == "yes")
    rank_ss = sum(1 for r in rows if r["rank_strict_success"] == "yes")
    rank_ev = sum(1 for r in rows if r["rank_evicted"] == "yes")
    rank_t1 = sum(1 for r in rows if r["rank_rank_target"] == "1")
    orig_t1 = sum(1 for r in rows if r["orig_rank_target"] == "1")
    instr_ss = sum(1 for r in rows if r["instruct_strict_success"] == "yes")

    pre_med = med([float(r["pre_cos_correct"]) for r in rows])
    post_med = med([float(r["post_cos_correct"]) for r in rows])
    orig_cap = med([int(r["orig_rank_correct_cap11"]) for r in rows])
    rank_cap = med([int(r["rank_rank_correct_cap11"]) for r in rows])

    el = sorted(float(r["elapsed_s"]) for r in rows)
    it = sorted(int(r["iterations"]) for r in rows)
    full_iters = sum(1 for x in it if x >= 500)

    # ---------- 2) headline + stratified summary ----------
    with open(os.path.join(args.outdir, "pair_summary.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "original_optimizer", "rank_aware_optimizer", "delta"])
        w.writerow(["queries", n, n, ""])
        w.writerow(["strict_success (target rank 1 AND correct passage out of top-10)",
                    pct(orig_ss, n), pct(rank_ss, n), f"+{100.0*(rank_ss-orig_ss)/n:.1f} pt"])
        w.writerow(["correct_passage_evicted_from_top10", pct(orig_ss, n), pct(rank_ev, n),
                    f"+{100.0*(rank_ev-orig_ss)/n:.1f} pt"])
        w.writerow(["target_reaches_rank_1", pct(orig_t1, n), pct(rank_t1, n),
                    f"{100.0*(rank_t1-orig_t1)/n:+.1f} pt"])
        w.writerow(["median_rank_correct_passage_capped11", orig_cap, rank_cap,
                    f"+{rank_cap - orig_cap}"])
        w.writerow(["median_cosine_correct_passage (before -> after attack)",
                    f"{pre_med:.4f}", f"{post_med:.4f}", f"{post_med - pre_med:+.4f}"])
        w.writerow([])
        w.writerow(["judging_convention", "strict_success", "share", ""])
        w.writerow(["bare query (reported convention)", rank_ss, f"{100.0*rank_ss/n:.1f}%", ""])
        w.writerow(["instruct-wrapped query", instr_ss, f"{100.0*instr_ss/n:.1f}%", ""])
        w.writerow([])
        w.writerow(["wall_clock_seconds", "value", "", ""])
        w.writerow(["min", f"{el[0]:.1f}", "", ""])
        w.writerow(["median", f"{med(el):.1f}", "", ""])
        w.writerow(["mean", f"{st.mean(el):.1f}", "", ""])
        w.writerow(["max", f"{el[-1]:.1f}", "", ""])
        w.writerow([f"queries hitting the {max(it)}-iteration cap", full_iters, "", ""])
    print("[ok] pair_summary.csv")

    # ---------- 3) flip classification + groups ----------
    flips = {
        "fail_to_success": [r["query_id"] for r in rows if r["pair_flip"] == "fail_to_success"],
        "success_to_fail": [r["query_id"] for r in rows if r["pair_flip"] == "success_to_fail"],
        "both_success": [r["query_id"] for r in rows if r["pair_flip"] == "both_success"],
        "both_fail": [r["query_id"] for r in rows if r["pair_flip"] == "both_fail"],
    }
    A = [r for r in rows if r["pre_rank_target"] == "1"]
    B = [r for r in rows if r["pre_rank_target"] != "1"]
    b_lost = [r["query_id"] for r in B if r["rank_rank_target"] != "1"]

    with open(os.path.join(args.outdir, "pair_flip_and_groups.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "count", "query_ids"])
        for k, v in flips.items():
            w.writerow(["pair_flip", k, len(v), " ".join(v)])
        w.writerow([])
        w.writerow(["section", "group", "n", "orig_strict_success",
                    "rank_strict_success", "rank_target_lost_from_rank1"])
        w.writerow(["pre_attack_target_rank", "target already rank 1", len(A),
                    sum(1 for r in A if r["orig_strict_success"] == "yes"),
                    sum(1 for r in A if r["rank_strict_success"] == "yes"),
                    sum(1 for r in A if r["rank_rank_target"] != "1")])
        w.writerow(["pre_attack_target_rank", "target not rank 1", len(B),
                    sum(1 for r in B if r["orig_strict_success"] == "yes"),
                    sum(1 for r in B if r["rank_strict_success"] == "yes"),
                    sum(1 for r in B if r["rank_rank_target"] != "1")])
        w.writerow([])
        w.writerow(["queries_losing_target_rank1", " ".join(b_lost)])
    print("[ok] pair_flip_and_groups.csv")


if __name__ == "__main__":
    main()
