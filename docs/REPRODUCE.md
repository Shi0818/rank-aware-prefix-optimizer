# Reproducing the results

Commands assume the repository root as the working directory (the runner uses
relative paths such as `./models/…`).

---

## 0. Environment

```bash
pip install -r requirements.txt
# CPU-only and only running the tests? A CPU build of torch is enough:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

A single 24 GB card is enough for one query at a time (`SFR-Embedding-Mistral` is
about 7 B parameters in fp16, ~15 GB). The optimizer keeps one extra forward per
verification batch, so VRAM headroom matters more than throughput.

---

## 1. Generate the upstream files (required first step) and verify the diff

The upstream release ships no licence file, so this repository does not redistribute
its code. The step below is **required before anything else runs**: it downloads the
upstream release and regenerates the two files the package imports from.

```bash
python tools/fetch_upstream.py          # downloads into third_party/upstream/, then installs
                                        # DGD/DGD.py and DGD/help_functions.py (git-ignored)
python tools/verify_upstream_diff.py    # asserts DGD.py is identical and the notebook differs in 1 cell
```

`DGD/DGD.py` is the byte-identical upstream optimizer used as the **original arm**, so
this repository's claim ("only our new files differ") stays machine-checkable. The
install is byte-reproducible: re-running it regenerates both files with unchanged md5.

If you skip this step, `SFR_mistral_IMDB_rank.py` exits with an explicit message rather
than an ImportError. `verify_upstream_diff.py` exits non-zero if the diff is larger than
claimed, so it can be wired into CI.

---

## 2. Data and model

The pipeline is the upstream one; nothing about data preparation changed.

1. **Corpus + queries.** Download `Data.zip` from the upstream repository and follow
   `Data_Preprocessing.py` to build the IMDB name-profile data (`title.basics.tsv`,
   `name.basics.tsv`).
2. **Embedding cache.** Run the first cells of the upstream
   `SFR_mistral_IMDB.ipynb` (or `SFR_mistral_IMDB_rank.ipynb`) to produce
   `./models/data_cache.pkl`, which must contain `prompts` and `nconst_embedding`.
   Both arms read this same cache — it is what makes the comparison paired.
3. **Model weights.** `Salesforce/SFR-Embedding-Mistral`, cached at
   `./models/SFR_Mistral/`. Offline runs need
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
4. **Baseline results.** The original arm reads target ids and `search_range` from an
   existing result directory (`--baseline-dir`, default `./results_baseline` — note the
   `results/` directory in this repo holds the *published summary tables*, not run
   output). If you already have a run of the released optimizer, point `--baseline-dir`
   at it and skip re-running that arm — which is exactly what the reported numbers do.

---

## 3. CPU self-checks (seconds, no GPU)

```bash
python tests/test_rank_optimizer.py
# or: pytest tests/ -q
```

These verify the optimizer's logic with a fake tokenizer and model: acceptance-key
direction, shapes, snapshot monotonicity, end-to-end rank consistency, and both the
`snapshot_by='loss'` and stall-escape branches. They do **not** measure attack
strength.

---

## 4. Smoke test on the GPU

```bash
export GGPP_USE_INSTRUCT=0        # bare query, paper convention (default of these scripts)
python SFR_mistral_IMDB_rank.py --arm rank --start 208 --end 210 \
       --iterations 20 --limit 2 --outdir ./results_smoke
```

Expect: 2 queries finish in ~25 s each, the correct passage's rank climbing, and the
judge convention printed at startup. Delete `./results_smoke` afterwards.

---

## 5. Full paired evaluation

```bash
# both arms, two GPUs
python run_pair_eval.py --start 208 --end 308 --gpu-orig 2 --gpu-rank 0

# improved arm only, when the original arm already exists
python run_pair_eval.py --start 308 --end 408 --arms rank --gpu-rank 0
```

Outputs `results_pair/{orig,rank}/*.pkl|.csv` plus a paired summary. Roughly 6–7 h for
100 queries on one card (median 206 s/query).

Afterwards, or at any time on a partial run:

```bash
python summarize_pair.py --pair-dir ./results_pair     # -> PAIR_SUMMARY.md / .csv
```

---

## 6. Ablations (turning the mechanism story into measurements)

```bash
python ablate_rank.py --start 208 --end 248 --orig-dir ./results_pair/orig
# or a subset
python ablate_rank.py --start 208 --end 248 --configs full snapshot no-target
```

Configurations: `full`, `snapshot`, `tau-strict`, `no-escape`, `small-cand`,
`long-prefix`, `no-target` — each differing from `full` in exactly one field.

---

## 7. Notebook route

If you prefer the notebook, use `SFR_mistral_IMDB_rank.ipynb`: it is the upstream
notebook with **cell 17 (the attack loop) replaced**. The change is a one-line import
swap plus the convention switch; `tools/SFR_mistral_IMDB_notebook.diff` shows it
verbatim, and `tools/build_rank_notebook.py` regenerates the notebook from upstream.

---

## Pitfalls

1. **Never mix judging conventions.** The optimizer's internal search, the initial
   check and the final judge must use the same convention. The scripts wire this up
   (`_JUDGE_KW`, `judge_ids(use_instruct=...)`) — do not drop the argument when
   editing. If the convention changes between search and judging, the search is
   climbing a gradient computed for the wrong signal.
2. **The switch defaults to upstream behaviour.** `GGPP_USE_INSTRUCT` unset means
   `'1'` (Instruct-wrapped), so existing callers are unaffected. The added scripts set
   `0` by default and pass the convention explicitly.
3. **`DGD/DGD_rank.py` sets `tok.padding_side = "left"`** (last-token pooling needs
   left padding). Single-sequence inference is unaffected; restore it if you batch
   other work in the same process.
4. **Prefix length** is `--firstk` (upstream `initial_firstk`), default 10. Changing it
   is an *attack-capacity* change, not a loss change — report it separately from the
   objective changes.
5. **Strict success uses the official `check_success`** (faiss/HNSW). The diagnostic
   exact-cosine ranks disagree with it on 2 of 100 queries; never mix the two in one
   number (see `docs/RESULTS.md` §6).
6. **Detector-side files.** If you feed these results to
   `Prefix_detection_SFR_Mistral`, note that it expects the upstream `_1…_10.pkl`
   naming. The pkl written here keeps the upstream keys
   (`user_prompt_id`, `target_prompt_id`, `token_ids`, `optimized_prefix`,
   `is_success`, `topk_ids_main`, …) so it can be consumed directly; only the batch
   file naming has to be aligned with the detector.
