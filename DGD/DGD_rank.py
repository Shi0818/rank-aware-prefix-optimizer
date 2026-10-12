"""
DGD_rank.py — rank-aware prefix optimizer for GGPP  (drop-in for DGD/DGD.py)
============================================================================
Place this file in the repository's ``DGD/`` package, next to ``DGD.py``.
It exposes one function whose call signature is identical to the original
``DGD.get_optimized_prefix_embedding``, so integration is a one-line import
change (see ``tools/SFR_mistral_IMDB_notebook.diff``)::

    # from DGD.DGD import get_optimized_prefix_embedding
    from DGD.DGD_rank import get_optimized_prefix_embedding_rank as \
get_optimized_prefix_embedding

Return value ``(token_ids, optimized_prefix, loss_list)`` is unchanged, so all
downstream code (judging, detection notebook, pkl schemas) keeps working.

Motivation
----------
The original objective (``DGD.py``, ``loss = (1 - cos(q, e_target)) + cos(q, e_user)``)
optimises a *proxy*: the absolute cosine to the correct passage.  What is
actually graded is the *rank* — the correct passage must be pushed out of the
top-k.  Measured on 792 IMDB queries, those two are not the same thing:
successful and unsuccessful attacks show nearly identical cosine reductions
(0.1410 vs 0.1366) with heavily overlapping ranges, i.e. the absolute value is
not the discriminating quantity; crossing the k-th competitor is.

What changes vs. ``DGD.py``
---------------------------
1. **Candidates are verified on the text path.**  The original picks candidates
   from a one-hot-relaxed embedding forward pass, but the object that is finally
   graded is the *decoded* prefix text (measured round-trip fidelity: ~80%).
   Here every candidate goes decode -> concatenate with the query -> tokenize ->
   forward -> last-token pooling -> exact cosine against the whole corpus.

2. **The target term is demoted to a constraint.**  Reaching rank 1 for the
   target is already saturated (>95%), so spending gradient on it is waste.
   ``relu(tau - cos_t)`` gives zero gradient once the target holds rank 1.

3. **The push term becomes a thresholded ranking loss.**
   ``softplus(cos_u - barrier)`` where ``barrier`` is the k-th best cosine
   *excluding* the correct passage — exactly the line that must be crossed.

4. **Acceptance rule = lexicographic rank key, judged every iteration, with a
   rank-best snapshot.**  The original judges once every 20 iterations (25 times
   in 500) and returns the prefix with the lowest *loss*, while lowest loss is
   not the best rank.

5. **Stall escape**: after ``stall_patience`` unimproved iterations, take one
   multi-position random perturbation (the snapshot still protects the best).

Key direction (important!)
--------------------------
``key = (rank_target, -rank_correct, cos_correct)``, minimised.
The correct passage's rank is *larger is better* (pushed further away), hence
the minus sign; its cosine is *smaller is better*, hence the plus sign.

Extra keyword arguments over the original (all optional):
    use_instruct, lam_t, lam_u, tau, n_single, n_random, max_pos, k_tok,
    verify_chunk, stall_patience, snapshot_by, seed, verbose
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda")

INSTRUCT_TASK = 'Given a web search query, retrieve relevant passages that answer the query'


def get_detailed_instruct(task, query):
    return f'Instruct: {task}\nQuery: {query}'


# --------------------------------------------------------------------------
# Unified encoding path: identical to the judging path (last-token, left pad)
# --------------------------------------------------------------------------

def encode_texts(texts, model, tok, use_instruct=False, chunk=8, max_length=4096):
    """texts -> embeddings (unnormalised). Left padding + last token, matching SFR."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    if use_instruct:
        texts = [get_detailed_instruct(INSTRUCT_TASK, t) for t in texts]
    if getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"          # SFR needs left padding: last position is the real token

    outs = []
    for i in range(0, len(texts), chunk):
        batch = tok(texts[i:i + chunk], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**batch)
        outs.append(out.last_hidden_state[:, -1].float())
    emb = torch.cat(outs, dim=0)
    return emb[0] if single else emb


def rank_of(idx, sims):
    """1-based rank (ties counted as better)."""
    return int((sims > sims[idx]).sum().item()) + 1


def kth_barrier(sims, u_idx, k):
    """k-th best cosine excluding the correct passage = the line a success must cross."""
    mask = torch.ones(sims.shape[0], dtype=torch.bool, device=sims.device)
    mask[u_idx] = False
    kk = max(1, min(int(k), int(mask.sum().item())))
    return torch.topk(sims[mask], kk).values[-1]


def acceptance_key(rank_t, rank_u, cos_u):
    """Lexicographic key, smallest wins.

    1. rank_target   -- smaller is better (1 = rank 1, hard requirement)
    2. rank_correct  -- **larger is better** (pushed further away), hence negated
    3. cos_correct   -- smaller is better (tie-break between equal ranks)
    """
    return (int(rank_t), -int(rank_u), round(float(cos_u), 4))


# --------------------------------------------------------------------------
# Gradient nomination (direction only -- precise evaluation is on the text path)
# --------------------------------------------------------------------------

def _nominate(one_hot, embed_weights, input_embeddings, bos_embeddings,
              e_target, e_user, barrier, lam_t, lam_u, tau, model):
    one_hot = one_hot.detach().to(embed_weights.dtype).requires_grad_(True)
    prefix_embeddings = torch.matmul(one_hot, embed_weights)
    seq = torch.cat([bos_embeddings, prefix_embeddings.squeeze(0),
                     input_embeddings.squeeze(0)], dim=0)
    out = model(inputs_embeds=seq.unsqueeze(0), output_hidden_states=True)
    q = out.last_hidden_state[0, -1].float()
    qn = F.normalize(q.unsqueeze(0), p=2, dim=1)[0]

    cos_t = torch.dot(qn, e_target)
    cos_u = torch.dot(qn, e_user)
    loss_t = lam_t * torch.relu(tau - cos_t)               # constraint: only penalised if target drops
    loss_u = lam_u * F.softplus(cos_u - barrier.detach())  # push: bounded by the k-th line
    (loss_t + loss_u).backward()
    return (one_hot.grad.detach(), float(cos_t), float(cos_u),
            float(loss_t), float(loss_u))


def _build_candidates(one_hot, top_indices, n_single, n_random, max_pos):
    """(a) per-position top-1 (standard GCG greedy coordinate) (b) single random (c) multi-position"""
    L = one_hot.shape[1]
    K = top_indices.shape[2]
    cands = []

    for pos in range(L):
        c = one_hot.detach().clone()
        tk = top_indices[0, pos, 0]
        c[0, pos, :] = 0.0
        c[0, pos, tk] = 1.0
        cands.append(c)

    for _ in range(n_single):
        pos = random.randrange(L)
        c = one_hot.detach().clone()
        c[0, pos, :] = 0.0
        c[0, pos, top_indices[0, pos, random.randrange(K)]] = 1.0
        cands.append(c)

    for _ in range(n_random):
        c = one_hot.detach().clone()
        for pos in random.sample(range(L), random.randint(1, min(max_pos, L))):
            c[0, pos, :] = 0.0
            c[0, pos, top_indices[0, pos, random.randrange(K)]] = 1.0
        cands.append(c)

    return cands


def _escape(one_hot, top_indices, max_pos):
    """Stall escape: one random multi-position replacement to break the local optimum."""
    c = one_hot.detach().clone()
    L = c.shape[1]
    K = top_indices.shape[2]
    for pos in random.sample(range(L), random.randint(1, min(max_pos, L))):
        c[0, pos, :] = 0.0
        c[0, pos, top_indices[0, pos, random.randrange(K)]] = 1.0
    return c


# --------------------------------------------------------------------------
# Main entry point: signature-compatible with DGD.get_optimized_prefix_embedding
# --------------------------------------------------------------------------

def get_optimized_prefix_embedding_rank(
        model_name, orginal_emb_logits, new_logits, user_prompt, prompt_text,
        target_text, initial_ids, iterations, user_prompt_id, target_prompt_id,
        topk_result, model, tok, nconst_embedding, thredhold, *,
        use_instruct=False,          # judging convention: False = bare query (paper convention)
        lam_t=1.0, lam_u=1.0,        # constraint / push weights
        tau=0.90,                    # hinge threshold: only penalise cos(q,t) below this
        n_single=8, n_random=48, max_pos=3, k_tok=64,
        verify_chunk=8,              # verification batch; lower it if you run out of memory
        stall_patience=30,           # unimproved iterations before one escape attempt
        snapshot_by='rank',          # 'rank' = snapshot by rank (recommended); 'loss' = original behaviour
        seed=None, verbose=True):
    """
    Returns (token_ids, optimized_prefix, loss_list): identical to the original.
    ``token_ids`` has shape (1, L) and can be passed straight to ``.tolist()``.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    u_idx, t_idx = int(user_prompt_id), int(target_prompt_id)

    # ---- corpus vectors (exact cosine over the whole corpus, no HNSW approximation) ----
    passages = F.normalize(torch.stack(list(nconst_embedding)).float(), p=2, dim=1).to(device)
    e_user = F.normalize(orginal_emb_logits.float().reshape(1, -1), p=2, dim=1)[0]
    e_target = F.normalize(new_logits.float().reshape(1, -1), p=2, dim=1)[0]

    # ---- one-hot prefix init (same as the original: <s> + prefix + query without BOS) ----
    q_ids = tok(user_prompt, return_tensors="pt")["input_ids"][:, 1:].to(device)
    bos_ids = torch.tensor([[1]], dtype=torch.long).to(device)

    if model_name == "mistralai":
        vocab_size = 32000
        embed_weights = model.embed_tokens.weight
        input_embeddings = model.embed_tokens(q_ids)
        bos_embeddings = model.embed_tokens(bos_ids).squeeze(0)      # (1, D)
    elif model_name == "EleutherAI/gpt-j-6b":
        vocab_size = 50400
        embed_weights = model.transformer.wte.weight
        input_embeddings = model.transformer.wte(q_ids)
        bos_embeddings = model.transformer.wte(bos_ids).squeeze(0)
    else:
        vocab_size = 151936
        embed_weights = model.transformer.wte.weight
        input_embeddings = model.transformer.wte(q_ids)
        bos_embeddings = model.transformer.wte(bos_ids).squeeze(0)

    L = len(initial_ids)
    one_hot = torch.zeros(1, L, vocab_size, device=device, dtype=embed_weights.dtype)
    for i, tid in enumerate(initial_ids[:L]):
        one_hot[0, i, int(tid)] = 1.0

    def ids_of(oh):
        return torch.argmax(oh.detach(), dim=-1)[0].tolist()

    def prefix_of(ids):
        return tok.decode(ids, skip_special_tokens=True)

    def verify(ids_list):
        """Text-path batched verification -> (B, N) cosine matrix"""
        texts = [prefix_of(ids) + user_prompt for ids in ids_list]
        qs = F.normalize(encode_texts(texts, model, tok, use_instruct=use_instruct,
                                      chunk=verify_chunk),
                         p=2, dim=1)
        return qs @ passages.T

    # ---- initial judgement ----
    cur_ids = ids_of(one_hot)
    sims = verify([cur_ids])[0]
    cur_ru, cur_rt = rank_of(u_idx, sims), rank_of(t_idx, sims)
    cur_su, cur_st = float(sims[u_idx]), float(sims[t_idx])
    cur_key = acceptance_key(cur_rt, cur_ru, cur_su)
    barrier = kth_barrier(sims, u_idx, topk_result)

    best = {"ids": cur_ids, "prefix": prefix_of(cur_ids), "rank_u": cur_ru,
            "rank_t": cur_rt, "cos_u": cur_su, "cos_t": cur_st, "epoch": -1}
    best_key = cur_key
    best_loss_seen = float("inf")   # only used when snapshot_by='loss'
    stall = 0
    loss_list = {"loss": [], "target_loss": [], "original_loss": [],
                 "rank_u": [], "cos_u": [], "cos_t": [], "best_rank_u": [],
                 "best_rank_t": [], "best_cos_u": [], "key": []}

    if verbose:
        print(f"[rank-opt] init  rank_u={cur_ru}  rank_t={cur_rt}  "
              f"cos_u={cur_su:.4f}  barrier={float(barrier):.4f}  key={cur_key}")

    for epoch in range(iterations):
        grad, cos_t_g, cos_u_g, lt, lu = _nominate(
            one_hot, embed_weights, input_embeddings, bos_embeddings,
            e_target, e_user, barrier, lam_t, lam_u, tau, model)

        top_indices = (-grad).topk(min(k_tok, grad.shape[2]), dim=2).indices
        cands = _build_candidates(one_hot, top_indices, n_single, n_random, max_pos)

        cand_ids = [ids_of(c) for c in cands]
        cand_sims = verify(cand_ids)

        # ---- judge every iteration + lexicographic selection (target rank first) ----
        best_j, cand_key = None, None
        best_ru = best_rt = best_su = best_st = None
        for j in range(cand_sims.shape[0]):
            s = cand_sims[j]
            ru, rt = rank_of(u_idx, s), rank_of(t_idx, s)
            su = float(s[u_idx])
            k = acceptance_key(rt, ru, su)
            if cand_key is None or k < cand_key:
                best_j, cand_key = j, k
                best_ru, best_rt, best_su, best_st = ru, rt, su, float(s[t_idx])

        if cand_key is not None and cand_key < cur_key:
            one_hot = cands[best_j].detach().clone()
            cur_key = cand_key
            cur_ids = cand_ids[best_j]
            sims, cur_ru, cur_rt, cur_su, cur_st = (
                cand_sims[best_j], best_ru, best_rt, best_su, best_st)
            barrier = kth_barrier(sims, u_idx, topk_result)
            stall = 0
        else:
            # ---- stall escape: leave the single-coordinate greedy local optimum ----
            stall += 1
            if stall >= stall_patience:
                one_hot = _escape(one_hot, top_indices, max_pos)
                cur_ids = ids_of(one_hot)
                sims = verify([cur_ids])[0]
                cur_ru, cur_rt = rank_of(u_idx, sims), rank_of(t_idx, sims)
                cur_su, cur_st = float(sims[u_idx]), float(sims[t_idx])
                cur_key = acceptance_key(cur_rt, cur_ru, cur_su)
                barrier = kth_barrier(sims, u_idx, topk_result)
                stall = 0
                if verbose:
                    print(f"[rank-opt] ep {epoch:4d} stall escape -> "
                          f"rank_u={cur_ru} rank_t={cur_rt} cos_u={cur_su:.4f}")

        # ---- snapshot ----
        if snapshot_by == 'loss':
            if (lt + lu) < best_loss_seen:
                best_loss_seen = lt + lu
                best.update({"ids": cur_ids, "prefix": prefix_of(cur_ids),
                             "rank_u": cur_ru, "rank_t": cur_rt,
                             "cos_u": cur_su, "cos_t": cur_st, "epoch": epoch})
        else:
            if cur_key < best_key:
                best_key = cur_key
                best.update({"ids": cur_ids, "prefix": prefix_of(cur_ids),
                             "rank_u": cur_ru, "rank_t": cur_rt,
                             "cos_u": cur_su, "cos_t": cur_st, "epoch": epoch})

        loss_list["loss"].append(lt + lu)
        loss_list["target_loss"].append(lt)
        loss_list["original_loss"].append(lu)
        loss_list["rank_u"].append(cur_ru)
        loss_list["cos_u"].append(cur_su)
        loss_list["cos_t"].append(cur_st)
        loss_list["best_rank_u"].append(best["rank_u"])
        loss_list["best_rank_t"].append(best["rank_t"])
        loss_list["best_cos_u"].append(best["cos_u"])
        loss_list["key"].append(cur_key)

        if verbose and (epoch % 10 == 0):
            print(f"[rank-opt] ep {epoch:4d}  loss_t={lt:.4f} loss_u={lu:.4f}  "
                  f"rank_u={cur_ru} rank_t={cur_rt} cos_u={cur_su:.4f}  |  "
                  f"best_rank_u={best['rank_u']} (ep {best['epoch']})")

        # ---- early stop on success: target rank 1 and correct passage out of top-k ----
        if best["rank_t"] == 1 and best["rank_u"] > topk_result:
            if verbose:
                print(f"[rank-opt] hit (ep {epoch}): target rank 1, "
                      f"correct passage rank {best['rank_u']}")
            break

    token_ids = torch.tensor([best["ids"]], dtype=torch.long)
    if verbose:
        print(f"[rank-opt] done  best: rank_u={best['rank_u']} rank_t={best['rank_t']} "
              f"cos_u={best['cos_u']:.4f} cos_t={best['cos_t']:.4f} ep={best['epoch']}")
    return token_ids, best["prefix"], loss_list


def run_with_restarts(n_restarts, *args, **kwargs):
    """Multi-restart, keep the best. Key: (rank_target, -rank_correct, cos_correct)."""
    results = []
    for r in range(n_restarts):
        kw = dict(kwargs)
        kw["seed"] = None if kwargs.get("seed") is None else kwargs["seed"] + r
        token_ids, prefix, loss_list = get_optimized_prefix_embedding_rank(*args, **kw)
        last_key = loss_list["key"][-1] if loss_list["key"] else None
        results.append((last_key, token_ids, prefix, loss_list))
        print(f"[restart {r}] last key={last_key}")
    results.sort(key=lambda x: (x[0] is None, x[0]))
    return results[0][1], results[0][2], results[0][3]
