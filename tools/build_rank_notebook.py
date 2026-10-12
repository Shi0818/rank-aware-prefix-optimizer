# -*- coding: utf-8 -*-
"""
Build SFR_mistral_IMDB_rank.ipynb from the upstream SFR_mistral_IMDB.ipynb
=========================================================================
Exactly one cell is modified (index 17, the attack loop); the other 17 cells
stay byte-identical to upstream.  The output is a drop-in notebook and ships
with a readable diff for review.

Usage:
  python tools/fetch_upstream.py            # once, to get the upstream notebook
  python tools/build_rank_notebook.py

Input : third_party/upstream/SFR_mistral_IMDB.ipynb
Output: SFR_mistral_IMDB_rank.ipynb
        tools/SFR_mistral_IMDB_notebook.diff      (readable diff of cell 17)
"""

import difflib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.join(PKG, "third_party", "upstream", "SFR_mistral_IMDB.ipynb")
DST = os.path.join(PKG, "SFR_mistral_IMDB_rank.ipynb")
DIFF = os.path.join(HERE, "SFR_mistral_IMDB_notebook.diff")

TARGET_CELL = 17

# ---- line-by-line replacements: (old line stripped, new line text). Multi-line values expand at the same indent. ----
REPLACEMENTS = [
    (
        "from DGD.DGD import get_optimized_prefix_embedding",
        "from DGD.DGD_rank import get_optimized_prefix_embedding_rank as get_optimized_prefix_embedding",
    ),
    (
        "top_k = get_relevant_documents(torch.stack(nconst_embedding), intital_prefix+user_prompt, topk_result, model, tok)",
        "top_k = get_relevant_documents(torch.stack(nconst_embedding), intital_prefix+user_prompt, topk_result, model, tok, **_JUDGE_KW)",
    ),
    (
        "token_ids_tensor, optimized_prefix, loss_list = get_optimized_prefix_embedding(MODEL_NAME, passages_embeddings[user_prompt_id], passages_embeddings[target_prompt_id], user_prompt, prompt_text, target_text, initial_ids, iterations, user_prompt_id, target_prompt_id, topk_result, model, tok, nconst_embedding, thredhold)",
        "token_ids_tensor, optimized_prefix, loss_list = get_optimized_prefix_embedding(MODEL_NAME, passages_embeddings[user_prompt_id], passages_embeddings[target_prompt_id], user_prompt, prompt_text, target_text, initial_ids, iterations, user_prompt_id, target_prompt_id, topk_result, model, tok, nconst_embedding, thredhold, seed=20261008+i, use_instruct=USE_INSTRUCT)",
    ),
    (
        "top_k = get_relevant_documents(torch.stack(nconst_embedding), optimized_prefix+user_prompt, topk_result, model, tok)",
        "top_k = get_relevant_documents(torch.stack(nconst_embedding), optimized_prefix+user_prompt, topk_result, model, tok, **_JUDGE_KW)",
    ),
    (
        'results_store_dct[i]["topk_result"] = topk_result',
        'results_store_dct[i]["topk_result"] = topk_result\n'
        'results_store_dct[i]["judge_mode"] = "instruct" if USE_INSTRUCT else "bare"',
    ),
]

# ---- switch block that replaces the "import random" line (top level, no indent; keeps the import) ----
SWITCH_BLOCK = '''import random

# ---------------------------------------------------------------------------
# [FIX] The judging convention and the optimisation convention must agree --
#       otherwise the search climbs a gradient computed for the wrong signal.
#   GGPP_USE_INSTRUCT=0 -> bare query (paper convention, default)
#   GGPP_USE_INSTRUCT=1 -> upstream Instruct-wrapped query
# The judge side relies on the query-wrapper switch in DGD/help_functions.py;
# when that switch is absent both sides degrade to the instruct convention
# together, so "what is optimised" always equals "what is graded" -- never a
# silent mismatch.
# ---------------------------------------------------------------------------
import os, inspect

_WANT_BARE = os.environ.get("GGPP_USE_INSTRUCT", "0") != "1"
_HAS_SWITCH = "use_instruct" in inspect.signature(get_relevant_documents).parameters
if _WANT_BARE and not _HAS_SWITCH:
    print("[warn] DGD/help_functions.py has no use_instruct switch -> judging and "
          "optimisation both degrade to the instruct convention "
          "(results are not comparable to the paper convention)")
USE_INSTRUCT = (not _WANT_BARE) if _HAS_SWITCH else True
_JUDGE_KW = {"use_instruct": USE_INSTRUCT} if _HAS_SWITCH else {}'''


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing upstream notebook: {SRC}\n  -> run `python tools/fetch_upstream.py` first")

    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    cell = nb["cells"][TARGET_CELL]
    if cell["cell_type"] != "code":
        sys.exit(f"cell {TARGET_CELL} is not a code cell; the upstream structure may have changed")

    old_lines = "".join(cell["source"]).splitlines()
    new_lines, used = [], set()

    for ln in old_lines:
        stripped = ln.strip()
        indent = ln[:len(ln) - len(ln.lstrip())]

        if stripped == "import random":
            new_lines.extend(SWITCH_BLOCK.split("\n"))
            used.add("__switch__")
            continue

        hit = False
        for old, new in REPLACEMENTS:
            if stripped == old:
                new_lines.extend(indent + l for l in new.split("\n"))
                used.add(old)
                hit = True
                break
        if not hit:
            new_lines.append(ln)

    missing = [o[:70] for o, _ in REPLACEMENTS if o not in used]
    if "__switch__" not in used:
        missing.append('"import random" (switch-block insertion point)')
    if missing:
        sys.exit("these anchors were not found in the upstream notebook; "
                 "check the upstream revision:\n  - "
                 + "\n  - ".join(missing))

    nb["cells"][TARGET_CELL]["source"] = [l + "\n" for l in new_lines[:-1]] + [new_lines[-1]]

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    os.makedirs(os.path.dirname(DIFF), exist_ok=True)
    with open(DIFF, "w", encoding="utf-8") as f:
        f.writelines(difflib.unified_diff(
            [l + "\n" for l in old_lines], [l + "\n" for l in new_lines],
            fromfile="a/SFR_mistral_IMDB.ipynb  (cell 17)",
            tofile="b/SFR_mistral_IMDB_rank.ipynb  (cell 17)",
            n=3))

    print(f"[ok] wrote {os.path.relpath(DST, PKG)}")
    print(f"[ok] wrote {os.path.relpath(DIFF, PKG)}")
    print(f"[ok] {len(nb['cells'])} cells, only cell {TARGET_CELL} modified")


if __name__ == "__main__":
    main()
