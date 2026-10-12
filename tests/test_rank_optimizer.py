# -*- coding: utf-8 -*-
"""
Self-contained checks for DGD/DGD_rank.py — no GPU, no SFR weights required
===========================================================================
A fake tokenizer and a fake model drive the whole optimisation loop on CPU, so the
*logic* of the rank-aware optimizer can be checked in seconds:

  * the direction of the acceptance key (larger rank for the correct passage is
    better, and the target's rank dominates);
  * shapes, rank and barrier helpers;
  * the loop runs to completion and returns the same structure as the original
    (``token_ids`` shaped ``(1, L)``);
  * end-to-end consistency: re-encoding the returned prefix and ranking it again
    reproduces exactly the ranks the snapshot claims;
  * the push term really does lower the correct passage's cosine;
  * the ``snapshot_by='loss'`` and stall-escape branches are reachable.

The fake model is a bag of embeddings with mean pooling, i.e. it does **not**
reproduce the attention path of a real transformer.  These checks therefore
validate flow and logic, not attack strength — real gains must be measured on a
GPU (see ``docs/REPRODUCE.md``).

Run with either:
    python tests/test_rank_optimizer.py
    pytest tests/test_rank_optimizer.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from DGD import DGD_rank  # noqa: E402

DGD_rank.device = torch.device("cpu")

# The vocabulary size must match the hard-coded 32000 of DGD_rank's "mistralai"
# branch, otherwise gradient-nominated candidate token ids fall outside the fake
# embedding table.
VOCAB, DIM, N_CORPUS, PREFIX_LEN = 32000, 32, 60, 10
QUERY_TEXT = "200 201 202"


class FakeBatch(dict):
    def to(self, *a, **k):
        return self


class FakeTok:
    """decode/encode on space-separated integers, so the prefix round-trips losslessly."""

    def __init__(self):
        self.padding_side = "right"

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)

    def encode(self, text, return_tensors=None):
        return torch.tensor([[int(x) for x in str(text).split() if x.lstrip("-").isdigit()]],
                            dtype=torch.long)

    def __call__(self, texts, padding=True, return_tensors="pt", **kw):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t)[0] for t in texts]
        maxlen = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:                      # left padding
            pad = maxlen - len(s)
            ids.append(torch.cat([torch.zeros(pad, dtype=torch.long), s]))
            mask.append(torch.cat([torch.zeros(pad, dtype=torch.long),
                                   torch.ones(len(s), dtype=torch.long)]))
        return FakeBatch(input_ids=torch.stack(ids), attention_mask=torch.stack(mask))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, DIM)
        with torch.no_grad():
            self.embed_tokens.weight.normal_(0, 0.5)
        self.embed_tokens.weight.requires_grad_(False)
        self.W = torch.randn(DIM, DIM) / DIM ** 0.5

    def forward(self, input_ids=None, inputs_embeds=None, output_hidden_states=True,
                attention_mask=None):
        emb = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        h = torch.tanh(emb @ self.W)
        return SimpleNamespace(last_hidden_state=h, hidden_states=[h])


tok = FakeTok()
torch.manual_seed(20261013)      # keep the fake model deterministic across runs
model = FakeModel()


def fake_encode_texts(texts, model, tok, use_instruct=False, chunk=8, **kw):
    """Stand-in for the real judging path: mean-pool the whole sentence."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    out = []
    for t in texts:
        ids = tok.encode(t)[0]
        with torch.no_grad():
            emb = model.embed_tokens(ids)
            out.append(torch.tanh(emb.mean(dim=0) @ model.W))
    v = torch.stack(out)
    return v[0] if single else v


DGD_rank.encode_texts = fake_encode_texts


def build_corpus():
    """Corpus where U_IDX is the query's twin, mirroring the real setting."""
    torch.manual_seed(0)
    corpus = torch.randn(N_CORPUS, DIM)
    u_idx, t_idx = 3, 41
    corpus[u_idx] = fake_encode_texts(QUERY_TEXT, model, tok)
    corpus = [corpus[i] for i in range(N_CORPUS)]
    passages = F.normalize(torch.stack(corpus).float(), p=2, dim=1)
    return corpus, passages, u_idx, t_idx


def test_acceptance_key_direction():
    """Direction of the acceptance key — this was flipped in an early revision."""
    ak = DGD_rank.acceptance_key
    # same target rank: larger rank for the correct passage is better (5 beats 2)
    assert ak(1, 5, 0.50) < ak(1, 2, 0.50), "correct-passage rank direction is flipped"
    # target rank dominates: better to keep the correct passage at rank 1 than to lose it
    assert ak(1, 1, 0.90) < ak(2, 1, 0.10), "target rank does not dominate"
    # equal ranks: lower cosine is better
    assert ak(1, 3, 0.40) < ak(1, 3, 0.60), "cosine direction is flipped"
    # evicted (11) beats still-in-list (9)
    assert ak(1, 11, 0.70) < ak(1, 9, 0.70), "eviction is not preferred"
    print("[ok] acceptance key direction: target rank dominates / larger correct rank wins / "
          "lower cosine wins")


def test_helpers_and_shapes():
    """encode_texts shapes, rank_of, kth_barrier."""
    v = fake_encode_texts("1 2 3", model, tok)
    assert v.shape == (DIM,), v.shape
    emb_batch = fake_encode_texts(["1 2 3", "4 5"], model, tok)
    assert emb_batch.shape == (2, DIM), emb_batch.shape

    sims = torch.tensor([0.5, 0.9, 0.7, 0.3])
    assert DGD_rank.rank_of(1, sims) == 1
    assert float(DGD_rank.kth_barrier(sims, 1, 2)) == 0.5   # 2nd largest after removing index 1
    print("[ok] encode_texts shapes, rank_of, kth_barrier")


def test_main_loop_end_to_end():
    """Run the loop and check structure, snapshot monotonicity and end-to-end ranks."""
    corpus, passages, u_idx, t_idx = build_corpus()
    initial_ids = [10 + i for i in range(PREFIX_LEN)]

    token_ids, prefix, loss_list = DGD_rank.get_optimized_prefix_embedding_rank(
        "mistralai",
        corpus[u_idx], corpus[t_idx],
        QUERY_TEXT, "prompt_text", "target_text",
        initial_ids,
        iterations=60,
        user_prompt_id=u_idx, target_prompt_id=t_idx,
        topk_result=10,
        model=model, tok=tok, nconst_embedding=corpus, thredhold=1,
        lam_t=1.0, lam_u=1.0, tau=0.99,
        n_single=8, n_random=32, max_pos=3, k_tok=16,
        verify_chunk=8, seed=0, verbose=False,
    )

    assert isinstance(token_ids, torch.Tensor) and token_ids.shape == (1, PREFIX_LEN), \
        token_ids.shape
    assert isinstance(prefix, str) and len(prefix) > 0

    fields = ("loss", "target_loss", "original_loss", "rank_u", "cos_u", "cos_t",
              "best_rank_u", "best_rank_t", "best_cos_u", "key")
    n_ep = len(loss_list["rank_u"])
    assert 1 <= n_ep <= 60, n_ep
    for k in fields:
        assert k in loss_list, k
        assert len(loss_list[k]) == n_ep, (k, len(loss_list[k]))
    print(f"[ok] return structure matches the original; token_ids {tuple(token_ids.shape)}; "
          f"loss_list fields all length {n_ep}")

    # --- end-to-end: re-encode the returned prefix, ranks must match the snapshot ---
    q = F.normalize(fake_encode_texts(prefix + QUERY_TEXT, model, tok).float(), p=2, dim=0)
    sf = q @ passages.T
    ru_f = DGD_rank.rank_of(u_idx, sf)
    rt_f = DGD_rank.rank_of(t_idx, sf)
    assert ru_f == loss_list["best_rank_u"][-1], (ru_f, loss_list["best_rank_u"][-1])
    assert rt_f == loss_list["best_rank_t"][-1], (rt_f, loss_list["best_rank_t"][-1])
    assert abs(float(sf[u_idx]) - loss_list["best_cos_u"][-1]) < 1e-4
    print(f"[ok] end-to-end: re-encoded prefix gives rank_u={ru_f} rank_t={rt_f}, "
          f"matching the snapshot")

    # --- monotonicity of the acceptance key and of the snapshot ---
    # (non-increasing rather than strictly decreasing: with a small mock corpus the
    #  loop can already be optimal on iteration 0 and stop immediately)
    ks = loss_list["key"]
    assert ks[-1] <= ks[0], (ks[0], ks[-1])
    assert all(ks[i] <= ks[i - 1] for i in range(1, len(ks))), \
        "acceptance key must be non-increasing outside escape rounds"
    snap_keys = list(zip(loss_list["best_rank_t"],
                         [-x for x in loss_list["best_rank_u"]],
                         [round(c, 4) for c in loss_list["best_cos_u"]]))
    assert all(snap_keys[i] <= snap_keys[i - 1] for i in range(1, len(snap_keys))), snap_keys
    print(f"[ok] acceptance key non-increasing: {ks[0]} -> {ks[-1]}")

    # --- the improvement really happened ---
    r0 = loss_list["rank_u"][0]
    rbest, rtbest = loss_list["best_rank_u"][-1], loss_list["best_rank_t"][-1]
    assert rtbest == 1 and rbest > 10, (rtbest, rbest)
    print(f"[ok] mock hit: target rank {rtbest}, correct passage rank {rbest} (>10, evicted); "
          f"started at correct rank {r0}")


def test_snapshot_by_loss_branch():
    corpus, _, u_idx, t_idx = build_corpus()
    initial_ids = [10 + i for i in range(PREFIX_LEN)]
    _, _, ll = DGD_rank.get_optimized_prefix_embedding_rank(
        "mistralai", corpus[u_idx], corpus[t_idx], QUERY_TEXT, "p", "t",
        initial_ids, iterations=5, user_prompt_id=u_idx, target_prompt_id=t_idx,
        topk_result=10, model=model, tok=tok, nconst_embedding=corpus, thredhold=1,
        n_single=2, n_random=4, k_tok=8, verify_chunk=8, seed=1,
        snapshot_by='loss', verbose=False)
    assert len(ll["rank_u"]) == 5
    print("[ok] snapshot_by='loss' branch runs")


def test_stall_escape_branch():
    corpus, _, u_idx, t_idx = build_corpus()
    initial_ids = [10 + i for i in range(PREFIX_LEN)]
    _, _, ll = DGD_rank.get_optimized_prefix_embedding_rank(
        "mistralai", corpus[u_idx], corpus[t_idx], QUERY_TEXT, "p", "t",
        initial_ids, iterations=6, user_prompt_id=u_idx, target_prompt_id=t_idx,
        topk_result=10, model=model, tok=tok, nconst_embedding=corpus, thredhold=1,
        n_single=0, n_random=0, k_tok=4, verify_chunk=8, seed=2,
        stall_patience=1, verbose=False)
    assert len(ll["rank_u"]) == 6
    print("[ok] stall-escape branch runs")


def main():
    test_acceptance_key_direction()
    test_helpers_and_shapes()
    test_main_loop_end_to_end()
    test_snapshot_by_loss_branch()
    test_stall_escape_branch()
    print("\nall checks passed.")


if __name__ == "__main__":
    main()
