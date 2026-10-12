# -*- coding: utf-8 -*-
"""
SFR_mistral_IMDB_rank.py — script form of SFR_mistral_IMDB_rank.ipynb
=====================================================================
Same pipeline as the official notebook, but as a CLI so a batch of queries can
be run unattended on a server.  ``--arm`` selects which optimizer drives the
search, everything else (model, data, judging, output schema) is shared, so the
two arms are directly comparable.

    python SFR_mistral_IMDB_rank.py --arm rank --batch 1 --start 208 --end 308
    python SFR_mistral_IMDB_rank.py --arm orig --batch 1 --start 208 --end 308

Prerequisites (identical to the official pipeline):
    ./models/data_cache.pkl      produced by SFR_mistral_IMDB_rank.ipynb / .py --accuracy
    ./datasets/title.basics.tsv, ./datasets/name.basics.tsv
    Salesforce/SFR-Embedding-Mistral cached under ./models/SFR_Mistral/

Judging convention
------------------
``GGPP_USE_INSTRUCT=0`` (default) -> bare query, the convention used by the
paper's tables/examples in the appendix.  It relies on the ``use_instruct``
switch that ships with this repository in ``DGD/help_functions.py``
(regenerate it with ``tools/build_help_functions_patch.py``).
Without the switch the judge falls back to exact-cosine over the whole corpus and
prints a warning; the two paths agree on ranking, but the official faiss/HNSW
path is preferred for reporting.

Output
------
``<outdir>/SFR_Mistral_IMDB_<batch><tag>.pkl`` — official key schema *plus*
analysis fields, so both the detection notebook and the paired-summary script
can consume it.  A flat ``.csv`` twin is written as well.
"""

import argparse
import csv
import inspect
import os
import pickle
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

# ===== arguments =====
p = argparse.ArgumentParser(description='GGPP adversarial prefix training (rank-aware optimizer)')
p.add_argument('--arm', choices=['rank', 'orig'], default='rank',
               help='rank = DGD.DGD_rank (improved); orig = DGD.DGD (original)')
p.add_argument('--batch', type=int, default=1, help='batch number, used in the output filename')
p.add_argument('--start', type=int, default=208, help='start prompt index (inclusive)')
p.add_argument('--end', type=int, default=1000, help='end prompt index (exclusive)')
p.add_argument('--iterations', type=int, default=500)
p.add_argument('--seed', type=int, default=20261008, help='per-query seed = seed + i')
p.add_argument('--firstk', type=int, default=10, help='initial prefix length (original: 10)')
p.add_argument('--baseline-dir', type=str, default='./results_baseline',
               help='existing pkl dir; supplies target/search_range so runs are paired')
p.add_argument('--outdir', type=str, default='./results_improved')
p.add_argument('--tag', type=str, default='')
p.add_argument('--limit', type=int, default=0, help='>0 = only the first N queries (smoke test)')
# --- rank-arm hyperparameters (ablation knobs; ignored by --arm orig) ---
p.add_argument('--lam-t', type=float, default=1.0)
p.add_argument('--lam-u', type=float, default=1.0)
p.add_argument('--tau', type=float, default=0.90)
p.add_argument('--n-single', type=int, default=8)
p.add_argument('--n-random', type=int, default=48)
p.add_argument('--max-pos', type=int, default=3)
p.add_argument('--k-tok', type=int, default=64)
p.add_argument('--verify-chunk', type=int, default=8)
p.add_argument('--stall-patience', type=int, default=30)
p.add_argument('--snapshot-by', choices=['rank', 'loss'], default='rank')
args = p.parse_args()

ARM = args.arm
START, END = args.start, args.end
ITER, K0, TOPK = args.iterations, args.firstk, 10
SEED_BASE = args.seed
MODEL_NAME = 'mistralai'
device = torch.device('cuda')

WANT_BARE = os.environ.get('GGPP_USE_INSTRUCT', '0') != '1'

print(f"===== arm={ARM} ids=[{START},{END}) iters={ITER} seed={SEED_BASE} "
      f"firstk={K0} judge={'bare' if WANT_BARE else 'instruct'} =====", flush=True)

# ===== model (identical to the official notebook) =====
from transformers import AutoTokenizer, AutoModel

print("Loading SFR-Embedding-Mistral ...", flush=True)
tok = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral',
                                    cache_dir="./models/SFR_Mistral/",
                                    torch_dtype=torch.float16)
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral',
                                  cache_dir="./models/SFR_Mistral/",
                                  torch_dtype=torch.float16).to(device)
for _p in model.parameters():
    _p.requires_grad = False
model.eval()

# ===== data cache =====
CACHE_PATH = './models/data_cache.pkl'
if not os.path.exists(CACHE_PATH):
    sys.exit(f"missing {CACHE_PATH}; run the official notebook's data cell first")
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)
prompts = cache['prompts']
nconst_embedding = cache['nconst_embedding']
print(f"prompts={len(prompts)} embeddings={len(nconst_embedding)}", flush=True)

# ===== optimizers and judging helpers =====
from DGD.help_functions import (get_relevant_documents, rank_tokens_by_importance,
                                get_initial_ids, get_neighbor_ids, check_success,
                                user_prompt_generation)
from DGD.DGD import get_optimized_prefix_embedding as orig_opt
from DGD.DGD_rank import (get_optimized_prefix_embedding_rank as rank_opt,
                          encode_texts, rank_of)

HAS_SWITCH = 'use_instruct' in inspect.signature(get_relevant_documents).parameters
if WANT_BARE and not HAS_SWITCH:
    print("[warn] DGD/help_functions.py has no use_instruct switch -> falling back to an "
          "exact-cosine judge over the whole corpus (ranking-equivalent, but not the "
          "official faiss/HNSW path). Restore DGD/help_functions.py from this repo "
          "to use the official judge.", flush=True)
USE_INSTRUCT = (not WANT_BARE) if HAS_SWITCH else True

PASSAGES_G = F.normalize(torch.stack(nconst_embedding).float(), p=2, dim=1).to(device)


def judge_ids(query, use_instruct=None):
    """Official judge when the switch is available, exact-cosine fallback otherwise."""
    ui = USE_INSTRUCT if use_instruct is None else use_instruct
    if HAS_SWITCH:
        idx = get_relevant_documents(torch.stack(nconst_embedding), query, TOPK,
                                     model, tok, use_instruct=ui)
        return [int(x) for x in idx]
    q = F.normalize(encode_texts(query, model, tok, use_instruct=ui).float(), p=2, dim=0)
    return [int(x) for x in (q @ PASSAGES_G.T).topk(TOPK).indices.tolist()]


def exact_sims(query, use_instruct=None):
    ui = USE_INSTRUCT if use_instruct is None else use_instruct
    q = F.normalize(encode_texts(query, model, tok, use_instruct=ui).float(), p=2, dim=0)
    return q @ PASSAGES_G.T


# ===== paired parameters from an existing run (optional) =====
base_recs = {}
if os.path.isdir(args.baseline_dir):
    for fn in sorted(os.listdir(args.baseline_dir)):
        if not fn.endswith('.pkl'):
            continue
        try:
            with open(os.path.join(args.baseline_dir, fn), 'rb') as f:
                d = pickle.load(f)
            for k, v in d.items():
                base_recs[int(k)] = v
        except Exception as e:
            print(f"[warn] cannot read {fn}: {type(e).__name__} {e}", flush=True)
print(f"baseline records: {len(base_recs)} (from {args.baseline_dir})", flush=True)

# ===== output =====
os.makedirs(args.outdir, exist_ok=True)
stem = f"SFR_Mistral_IMDB_{args.batch}{args.tag}"
pkl_path = os.path.join(args.outdir, stem + '.pkl')
csv_path = os.path.join(args.outdir, stem + '.csv')

CSV_COLS = ['id', 'arm', 'seed', 'target_id', 'search_range', 'init_succ', 'succ',
            'succ_alt', 'init_rank_u', 'rank_u', 'init_rank_t', 'rank_t',
            'init_cos_u', 'cos_u', 'init_cos_t', 'cos_t', 'base_rank_u', 'base_rank_t',
            'evict', 'hist_succ', 'hist_rank_u', 'epochs', 'elapsed', 'opt_prefix', 'token_ids']

results = {}
if os.path.exists(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        print(f"[resume] loaded {len(results)} records", flush=True)
    except Exception:
        results = {}


def dump():
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
        w.writeheader()
        for k in sorted(results):
            w.writerow(results[k])


ids = [i for i in range(START, min(END, len(prompts)))]
if args.limit > 0:
    ids = ids[:args.limit]
print(f"to run: {len(ids)} queries", flush=True)

RANK_KW = dict(use_instruct=USE_INSTRUCT, lam_t=args.lam_t, lam_u=args.lam_u, tau=args.tau,
               n_single=args.n_single, n_random=args.n_random, max_pos=args.max_pos,
               k_tok=args.k_tok, verify_chunk=args.verify_chunk,
               stall_patience=args.stall_patience, snapshot_by=args.snapshot_by)

t_all = time.time()
for n_done, i in enumerate(ids, 1):
    if i in results:
        print(f"[skip] id={i} already done", flush=True)
        continue
    t0 = time.time()
    try:
        seed_i = SEED_BASE + i
        prompt_text = prompts[i]
        user_prompt = user_prompt_generation(prompt_text, "", "imdb", "for_objects")

        # ---- target: reuse the paired baseline if present, else the official rule ----
        if i in base_recs:
            target = int(base_recs[i]['target_prompt_id'])
            sr = int(base_recs[i].get('search_range', -1))
            user_prompt = base_recs[i].get('user_prompt', user_prompt)
        else:
            random.seed(seed_i)
            sr = random.randint(20, 1000)
            target = get_neighbor_ids(i, user_prompt, sr, model, tok, nconst_embedding)
        target_text = prompts[target]

        # ---- deterministic initial prefix (identical across arms) ----
        important_tokens = rank_tokens_by_importance(target_text, model, tok)
        initial_ids = get_initial_ids(important_tokens, target_text, K0, MODEL_NAME, model, tok)
        initial_prefix = tok.decode(initial_ids.cpu().tolist(), skip_special_tokens=True)

        # ---- initial judgement ----
        topk0 = judge_ids(initial_prefix + user_prompt)
        is0 = check_success(target, i, topk0)
        sims_i = exact_sims(initial_prefix + user_prompt)
        init_rank_u, init_rank_t = rank_of(i, sims_i), rank_of(target, sims_i)
        init_cos_u, init_cos_t = float(sims_i[i]), float(sims_i[target])

        epochs = 0
        loss_list = None
        if is0 > 1:
            token_ids = initial_ids.cpu().tolist()
            opt_prefix = initial_prefix
        elif ARM == 'rank':
            tk, opt_prefix, loss_list = rank_opt(
                MODEL_NAME, PASSAGES_G[i], PASSAGES_G[target], user_prompt, prompt_text,
                target_text, initial_ids, ITER, i, target, TOPK, model, tok,
                nconst_embedding, 1, seed=seed_i, **RANK_KW)
            token_ids = tk[0].tolist()
            epochs = len(loss_list['rank_u'])
        else:
            random.seed(seed_i)
            np.random.seed(seed_i)
            torch.manual_seed(seed_i)
            tk, opt_prefix, loss_list = orig_opt(
                MODEL_NAME, PASSAGES_G[i], PASSAGES_G[target], user_prompt, prompt_text,
                target_text, initial_ids, ITER, i, target, TOPK, model, tok,
                nconst_embedding, 1)
            token_ids = tk[0].tolist()
            epochs = ITER

        # ---- final judgement ----
        q_full = opt_prefix + user_prompt
        topk_f = judge_ids(q_full)
        succ = check_success(target, i, topk_f)
        sims_f = exact_sims(q_full)
        rank_u, rank_t = rank_of(i, sims_f), rank_of(target, sims_f)
        cos_u, cos_t = float(sims_f[i]), float(sims_f[target])
        sims_b = exact_sims(user_prompt)
        base_rank_u, base_rank_t = rank_of(i, sims_b), rank_of(target, sims_b)

        # opposite convention, for the reported convention-effect side note
        topk_alt = judge_ids(q_full, use_instruct=not USE_INSTRUCT)
        succ_alt = check_success(target, i, topk_alt)

        rec = {
            # ---- official schema (detection notebook compatibility) ----
            'user_prompt_id': i, 'user_prompt': user_prompt,
            'target_prompt_id': target, 'prompt_text': prompt_text, 'target_text': target_text,
            'token_ids': token_ids, 'optimized_prefix': opt_prefix,
            'is_success': succ, 'search_range': sr, 'topk_result': TOPK,
            'judge_mode': 'instruct' if USE_INSTRUCT else 'bare',
            'topk_ids_main': topk_f, 'is_success_alt': succ_alt, 'topk_ids_alt': topk_alt,
            'topk_ids_baseline': [int(x) for x in sims_b.topk(TOPK).indices.tolist()],
            'base_tgt_rank': rank_of(target, sims_b), 'base_usr_rank': rank_of(i, sims_b),
            # ---- analysis fields ----
            'id': i, 'arm': ARM, 'seed': seed_i, 'target_id': target,
            'initial_ids': initial_ids.cpu().tolist(), 'initial_prefix': initial_prefix,
            'init_succ': is0, 'succ': succ, 'succ_alt': succ_alt,
            'init_rank_u': init_rank_u, 'rank_u': rank_u,
            'init_rank_t': init_rank_t, 'rank_t': rank_t,
            'init_cos_u': init_cos_u, 'cos_u': cos_u,
            'init_cos_t': init_cos_t, 'cos_t': cos_t,
            'base_rank_u': base_rank_u, 'base_rank_t': base_rank_t,
            'evict': 1 if rank_u > TOPK else 0,
            'hist_succ': base_recs.get(i, {}).get('is_success', None),
            'hist_rank_u': (base_recs.get(i, {}).get('topk_ids_main') or []).index(i) + 1
            if i in (base_recs.get(i, {}).get('topk_ids_main') or []) else -1,
            'epochs': epochs, 'elapsed': round(time.time() - t0, 1),
            'nconst_len': len(nconst_embedding),
        }
        if loss_list is not None:
            rec['loss_tail'] = {
                'rank_u': loss_list['rank_u'][-1],
                'best_rank_u': loss_list.get('best_rank_u', [None])[-1],
                'cos_u': loss_list['cos_u'][-1],
                'n_epochs_logged': len(loss_list['rank_u']),
            }
        results[i] = rec
        dump()
        torch.cuda.empty_cache()

        el = time.time() - t0
        eta = (len(ids) - n_done) * (time.time() - t_all) / n_done / 60
        print(f"[run] {n_done}/{len(ids)} id={i} arm={ARM} init_succ={is0} succ={succ} "
              f"| rank_u {init_rank_u}->{rank_u} rank_t {init_rank_t}->{rank_t} "
              f"cos_u {init_cos_u:.4f}->{cos_u:.4f} | {el:.0f}s ETA {eta:.0f}min", flush=True)
    except Exception:
        print(f"[ERROR] id={i} failed:", flush=True)
        traceback.print_exc()
        torch.cuda.empty_cache()
        continue

dump()
n = len(results)
succ_n = sum(1 for r in results.values() if r['succ'] == 2)
ev_n = sum(1 for r in results.values() if r['evict'] == 1)
print(f"\n===== arm={ARM} done: {n} queries in {(time.time() - t_all) / 60:.1f} min =====")
print(f"strict success {succ_n}/{n} = {succ_n / max(1, n):.1%}   "
      f"correct passage evicted {ev_n}/{n} = {ev_n / max(1, n):.1%}")
print(f"saved -> {pkl_path}")
