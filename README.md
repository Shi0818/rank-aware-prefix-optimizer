# GGPP-RankAware

**Rank-aware prefix optimization for prompt-perturbation attacks on RAG retrieval.**

> **Research use only.** This code implements an adversarial method and is released for
> research on the robustness of retrieval-augmented generation, and for building defenses
> against it. No attack artefacts (optimised prefixes, poisoned passages) are distributed
> here — only the optimiser and the evaluation harness. If you use it, cite it and follow
> your institution's rules on adversarial ML research.

`DGD/DGD_rank.py` is a drop-in replacement for the prefix optimizer in
[GGPP (KDD'24)](https://github.com/Hadise-zb/Prompt-Perturbation-in-Retrieval-Augmented-Generation)
that optimises **the quantity the attack is actually graded on** — the *rank* of the
correct passage — instead of a cosine proxy. On IMDB + `SFR-Embedding-Mistral`, over
100 per-query paired queries:

| | released optimizer | `DGD_rank.py` |
|---|---|---|
| **strict success** (target at rank 1 **and** correct passage out of top-10) | 15 % | **72 %** |
| per-query flips | — | **57 fail → success, 0 success → fail** |

*(+57 pt, same queries / target / initial prefix / seed for both arms — see
[`docs/RESULTS.md`](docs/RESULTS.md).)*

---

## The problem in one paragraph

The released objective is `(1 − cos(q, target)) + cos(q, correct)`: a **linear,
threshold-free proxy** for a criterion that is actually **rank-based**. Reducing the
correct passage's cosine from 0.74 to 0.66 and from 0.66 to 0.60 earn identical
gradient, but only the second one crosses the 10th-place line. Measured over 100
queries, successful and unsuccessful attacks show nearly the *same* cosine reduction
(0.1410 vs 0.1366, heavily overlapping ranges): the absolute value is not the
discriminating quantity, crossing the *k*-th competitor is. `DGD_rank.py` changes the
objective, the acceptance rule and the candidate-verification path so that the search
optimises the rank directly. Full derivation in [`docs/METHOD.md`](docs/METHOD.md).

---

## What is in the box

```
DGD/
  DGD.py                     # byte-identical upstream copy — used as the ORIGINAL arm
  DGD_rank.py                # ★ the rank-aware optimizer (drop-in, same signature)
  help_functions.py          # upstream + a switchable query-wrapper convention
SFR_mistral_IMDB_rank.py     # ★ drop-in runner (script form), --arm rank|orig
SFR_mistral_IMDB_rank.ipynb  # ★ same pipeline as a notebook (1 cell changed vs upstream)
run_pair_eval.py             # paired A/B harness (same query/target/prefix/seed)
summarize_pair.py            # paired statistics
ablate_rank.py               # one-variable ablation matrix
tests/                       # CPU-only logic checks, no GPU or weights needed
tools/                       # fetch + verify upstream, regenerate derived files
docs/                        # METHOD.md, RESULTS.md, REPRODUCE.md
results/                     # the measured numbers, as generated CSV
```

### What differs from upstream — and how to check it

| file | relationship to upstream |
|---|---|
| `DGD/DGD.py` | **byte-identical** (md5 `bdb6624c…`) |
| `SFR_mistral_IMDB.ipynb` | upstream notebook, **exactly one cell (17) changed**, other 17 byte-identical |
| `DGD/help_functions.py` | upstream + a `use_instruct` switch (default preserves upstream behaviour) |
| `DGD/DGD_rank.py` | new |

```bash
python tools/fetch_upstream.py
python tools/verify_upstream_diff.py     # fails loudly if any of the above is untrue
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1) regenerate the upstream files this package imports from (required once)
python tools/fetch_upstream.py

# 2) logic checks — CPU, seconds, no weights needed
python tests/test_rank_optimizer.py

# 3) smoke test on one GPU
export GGPP_USE_INSTRUCT=0
python SFR_mistral_IMDB_rank.py --arm rank --start 208 --end 210 --iterations 20 --limit 2

# 4) paired evaluation (both arms; needs the data cache + baseline results, see docs)
python run_pair_eval.py --start 208 --end 308 --gpu-orig 2 --gpu-rank 0
python summarize_pair.py --pair-dir ./results_pair
```

Full prerequisites (data cache, model weights, baseline results) are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Results at a glance

| metric | released | rank-aware | change |
|---|---|---|---|
| strict success | 15/100 = 15.0 % | 72/100 = 72.0 % | +57.0 pt |
| correct passage evicted from top-10 | 15.0 % | 73.0 % | +58.0 pt |
| target reaches rank 1 | 95.0 % | 90.0 % | −5.0 pt |
| median rank of correct passage (cap 11) | 2 | 11 | +9 |
| median cosine of correct passage | 0.7505 → | 0.6089 | −0.1416 |

Split by whether the target already held rank 1 **before** the attack:

| group | n | released | rank-aware | rank-1 lost |
|---|---|---|---|---|
| target already rank 1 | 62 | 15 (24.2 %) | 53 (85.5 %) | **0** |
| target not rank 1 | 38 | 0 (0.0 %) | 19 (50.0 %) | **10** |

Two honest caveats, both documented in full:

* **+57 pt is measured; per-change attribution is not.** The comparison is the *whole*
  improved optimizer versus the released one. Per-change contributions are mechanical
  inference supported by data; `ablate_rank.py` exists to measure them.
* **Attack strength and stealth trade off.** Longer prefixes and heavier query edits
  are more visible to the detector — precisely the features SATe/ACT look at.

---

## Citation

If you use this code, please cite the upstream GGPP paper first, and link this
repository for the rank-aware optimizer:

```bibtex
@inproceedings{ggpp2024,
  title     = {Prompt Perturbation in Retrieval-Augmented Generation},
  booktitle = {KDD},
  year      = {2024}
}
```

See [`CITATION.cff`](CITATION.cff).

## Acknowledgements and licence

The pipeline, the dataset preparation, the original optimizer and the detection
notebook are the work of the GGPP authors — this repository only adds the rank-aware
optimizer and the surrounding evaluation tooling.

The upstream release ships **no licence file**, so its code is not redistributed here.
`DGD/DGD.py` and `DGD/help_functions.py` are generated locally by
`python tools/fetch_upstream.py` (byte-identical; the second one plus the query-wrapper
switch) and are git-ignored. Cite the original paper — their copyright stays with its
authors.
The additions are released under the MIT licence (see [`LICENSE`](LICENSE)).

## Scope

White-box only (gradients through the embedding model are required) and measured on a
single setting (`SFR-Embedding-Mistral` + IMDB). Other embedders and corpora are
untested. See [`docs/METHOD.md`](docs/METHOD.md) §5.
