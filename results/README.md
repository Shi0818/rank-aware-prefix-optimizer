# `results/` — data dictionary

All three files are **generated** by `tools/make_results_tables.py` from the per-query
paired detail table of the reported run (100 queries, ids 208–307). Do not hand-edit;
re-run the script instead so the numbers stay consistent with the README and
`docs/RESULTS.md`.

---

## `per_query_paired.csv` — 100 rows, one per query

| column | meaning |
|---|---|
| `query_id` | query / user-prompt index (208–307) |
| `target_id` | index of the adversarial target passage (the passage that must reach rank 1) |
| `orig_rank_correct` | rank of the correct passage, **original** optimizer (exact corpus-wide cosine) |
| `orig_rank_correct_cap11` | same, capped at 11 (11 = "out of the top-10") |
| `orig_rank_target` | rank of the target, original optimizer |
| `orig_strict_success` | `yes` iff target rank 1 **and** correct passage out of top-10 (original) |
| `rank_rank_correct` | rank of the correct passage, **rank-aware** optimizer |
| `rank_rank_correct_cap11` | same, capped at 11 |
| `rank_rank_target` | rank of the target, rank-aware optimizer |
| `rank_strict_success` | strict success for the rank-aware optimizer |
| `rank_evicted` | `yes` iff the correct passage left the top-10 (rank-aware) |
| `pair_flip` | one of `fail_to_success`, `success_to_fail`, `both_success`, `both_fail` |
| `pre_rank_correct` / `pre_rank_target` | ranks **before** the attack, i.e. shared by both arms (paired design) |
| `pre_cos_correct` | cosine of the correct passage before the attack |
| `post_cos_correct` / `post_cos_target` | cosines after the rank-aware attack |
| `elapsed_s` | wall-clock seconds for that query |
| `iterations` | search iterations used (500 = the cap was hit) |
| `instruct_strict_success` | strict success of the *same prefix* under the Instruct-wrapped judge |

**Ranks vs. judging.** `*_rank_*` columns are exact corpus-wide cosine ranks and are
used for diagnosis. `*_strict_success` columns come from the official
`check_success` (faiss/HNSW top-K). On 2 of the 100 queries (`232`, `305`) the two
conventions disagree on a boundary case — see `docs/RESULTS.md`.

---

## `pair_summary.csv` — headline and stratified metrics

Three blocks separated by a blank line: (1) headline metrics with per-query pairing,
(2) the judging-convention side note, (3) wall-clock distribution.

## `pair_flip_and_groups.csv` — flips and grouped analysis

Flip classification with the query ids in each bucket, plus the stratified breakdown
by whether the target already held rank 1 *before* the attack, and the list of queries
where the rank-aware optimizer loses rank 1.
