# Method — the rank-aware prefix optimizer

`DGD/DGD_rank.py` is a drop-in replacement for `DGD/DGD.py`. It keeps the call
signature and the `(token_ids, optimized_prefix, loss_list)` return value, so the
only change needed in an existing pipeline is the import line:

```python
# from DGD.DGD import get_optimized_prefix_embedding
from DGD.DGD_rank import get_optimized_prefix_embedding_rank as get_optimized_prefix_embedding
```

---

## 1. The mismatch this work addresses

The released objective (`DGD/DGD.py`, `loss_function`, lines 26–37) is

```python
cos_distance     = 1 - cosine_similarity(new_logits, new_target_tokens)   # (1 - cos(q, target))
cos_sim_negative =     cosine_similarity(new_logits, orginal_emb)         #      cos(q, correct)
cos_sim          = cos_distance + cos_sim_negative
```

so the search minimises a **proxy**: the *absolute* cosine between the perturbed
query and the correct passage, with the target term weighted equally. What the
pipeline is actually graded on is the **rank** of the correct passage — it must be
pushed out of the top-`k` retrieved set. Those are not the same quantity:

* the loss is linear and threshold-free, so `cos` going 0.74 → 0.66 and 0.66 → 0.60
  earn identical gradient, yet only the second one crosses the line;
* the two terms are equal-weighted and unrelated, so gradient spent on lifting the
  target is gradient not spent on evicting the correct passage — even though lifting
  the target is already saturated.

**Measured evidence (100 paired queries, IMDB + `SFR-Embedding-Mistral`).**
Attacks that succeed and attacks that fail show nearly the same cosine reduction:

| group | n | median cos before | median cos after | median reduction | median final rank |
|---|---|---|---|---|---|
| correct passage evicted | 73 | 0.7450 | 0.6033 | **0.1410** | 13 |
| not evicted | 27 | 0.7698 | 0.6294 | **0.1366** | 2 |

The reductions are nearly equal and the ranges overlap heavily; the discriminator
is the **endpoint**, i.e. whether the *k*-th competitor was crossed. Four queries
that were never evicted nevertheless ended below the evicted group's median cosine.

---

## 2. The five changes

| # | location in `DGD.py` | released code | `DGD_rank.py` | why |
|---|---|---|---|---|
| 1 | candidate scoring, L210–237 (core: L218 `torch.matmul(candidate_one_hot, embed_weights)`) | candidates are scored in the **one-hot-relaxed embedding space** with `inputs_embeds`, while the pooling mask is recomputed from a *different* tokenisation | candidates are scored on the **text path**: decode → concat with query → tokenise → forward → last-token pool → exact cosine over the whole corpus | the object finally graded is the decoded text; measured prefix round-trip fidelity is only ~80 % |
| 2 | loss term 1, L26–37 | `(1 - cos_target)`, linear | `relu(tau - cos_target)` — a **constraint**, zero gradient once the target holds rank 1 | reaching rank 1 is already saturated (>95 %); spending gradient there is waste |
| 3 | loss term 2, L26–37 | `cos_correct`, linear, no threshold | `softplus(cos_correct - barrier)`, where `barrier` is the *k*-th best cosine **excluding** the correct passage | it is the line that must be crossed, not an absolute level |
| 4 | acceptance / judging, L279–303 | returns the lowest-**loss** prefix; ranked only every 20 iterations (`epoch % 20`) | lexicographic key `(rank_target, -rank_correct, cos_correct)`, judged **every** iteration, rank-best snapshot kept | lowest loss ≠ best rank; verification already computes corpus-wide cosines, so per-iteration judging is free |
| 5 | stall escape, L243–266 | on stall, zeroes one coordinate's one-hot (a zero embedding, not a token) and re-forwards; no seed | replaces a coordinate with a **real token** from the top-k, plus a fixed seed and `run_with_restarts` | the zero-vector candidate is off-manifold; run-to-run variance is real |

> **Direction caution.** The second element of the acceptance key is `-rank_correct`
> because for the correct passage **larger rank is better** (further away). An earlier
> internal revision had this sign flipped, which silently produced a *negative*
> optimisation (pulling the correct passage closer). Keep the sign —
> `tests/test_rank_optimizer.py` asserts the direction explicitly.

### Change 1 in detail — why the embedding-space candidate score is wrong

Inside `greedy_coordinate_gradient_step` (L167–176) the forward pass is built as

```python
combined = cat([prefix_embeddings, input_embeddings])          # [prefix] + [bare query]
combined = cat([Instructed_embeddings, combined])              # [BOS]    + above
outputs  = model(inputs_embeds=combined, ...)
token_ids      = argmax(one_hot, -1)
temp_prefix    = tok.decode(token_ids)                         # decoded text, ~80% round-trip
full_query     = temp_prefix + user_query
batch_dict     = tok(full_query, ...)                          # a DIFFERENT tokenisation
new_logits     = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
```

The pooled sequence is the manually assembled `inputs_embeds`, but the mask handed to
`last_token_pool` comes from tokenising `temp_prefix + user_query` — a string that has
no `[BOS]` entry and whose prefix may decode to a different number of tokens than the
relaxed prefix it was built from. The mask is therefore **not aligned** with the
sequence it is applied to, and the "last token" that gets pooled is not guaranteed to
be the last token of the assembled input.

`DGD_rank.py` removes the ambiguity entirely: every candidate is scored by decoding it
to text, concatenating with the query, tokenising *that*, and forwarding *that* — the
same path the official judge uses.

---

## 3. What is measured vs. what is inferred

The reported **+57 pt** is the *whole* improved optimizer versus the released one, on
100 per-query paired queries. It is a measurement.

Per-change contributions are **mechanistic inference backed by data**, not isolated
measurements — no single-variable ablation has been run yet. The supporting evidence:

* **Push term (changes 2–3)** — 67 % of the gain (38/57) comes from queries whose
  target *already* held rank 1 before the attack, i.e. queries that need nothing but
  eviction. The evicted/not-evicted cosine table above shows the term is doing the
  right thing but the endpoint is what matters.
* **Target term as a constraint (change 2)** — of the 62 queries whose target already
  held rank 1, **zero** lost it (53 became strict successes, 9 stayed as they were).
  Making the target term a constraint therefore costs nothing on this set.
* **Per-iteration judging + rank snapshot (change 4)** — all 72 successes stopped
  early (median 119 iterations, minimum 3), while **none** of the 26 queries that ran
  to the 500-iteration cap succeeded. Stopping and success are cleanly separated; in
  14/100 queries the rank-best snapshot was better than the last iteration.
* **Text-path verification (change 1)** — the internal exact-cosine judge and the
  official faiss judge disagree on only 2 of 100 queries (see `docs/RESULTS.md`),
  which bounds how often the ~80 % round-trip fidelity actually bites.

`ablate_rank.py` is the instrument for turning these into measurements:
`snapshot`, `tau-strict`, `no-escape`, `small-cand`, `no-target`, `long-prefix`.

---

## 4. Known remaining failure modes

Both are localised and both have a concrete next fix.

**a. Losing target rank 1 (10/100 queries).** All 10 fall in the group whose target
was *not* rank 1 before the attack. Root cause: the target constraint uses a **fixed
absolute threshold** `tau = 0.90`, which asks "is the target cosine high enough", not
"does the target beat the current rank-1 competitor". When both cosines slide down
together (e.g. query 210: `cos_target` −0.146, `cos_correct` −0.153) the relative order
never flips. Fix: replace with a ranking margin,
`softplus(cos_blocker.detach() - cos_target + delta)`.

**b. The runner-up wall (8/100 queries, all in group A).** These 8 all started with the
correct passage at **rank 2** and all ran to the iteration cap. The `barrier` drifts
downward together with the correct passage — pushing the query away from the correct
passage also pushes it away from the corpus, lowering the 10th-place line at the same
time. Fix: freeze `barrier`, or switch to a listwise term that compares the correct
passage against the *k*-th entry explicitly.

---

## 5. Scope

* **White-box.** Gradients through the embedding model are required; closed embedding
  APIs are out of scope.
* **Single setting.** Measured on `SFR-Embedding-Mistral` + IMDB only.
* **Stealth is not modelled.** Longer prefixes and heavier query edits are *more*
  detectable — which is exactly what `Prefix_detection_SFR_Mistral` looks for.
  Capability and stealth trade off; report both.
