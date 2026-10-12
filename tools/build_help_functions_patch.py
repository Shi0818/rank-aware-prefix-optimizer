# -*- coding: utf-8 -*-
"""
Generate the query-wrapper switch for DGD/help_functions.py
===========================================================
Turns the unconditional Instruct wrapping inside ``get_relevant_documents``
into a switchable behaviour.  Passing nothing keeps the upstream behaviour
(default ``GGPP_USE_INSTRUCT='1'``), so this is fully backward compatible.
The switch is the prerequisite for aligning the *attack-side* convention with
the *judge-side* convention.

Usage:
  python tools/fetch_upstream.py                 # once, to get the upstream file
  python tools/build_help_functions_patch.py

Input : third_party/upstream/DGD_help_functions.py   (byte-identical upstream copy)
Output: tools/help_functions_use_instruct.patch      (git apply / patch -p1)
        DGD/help_functions.py                        (patched full file, shipped as-is)
"""

import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.join(PKG, "third_party", "upstream", "DGD_help_functions.py")
OUT = os.path.join(HERE, "help_functions_use_instruct.patch")

OLD = """# SFR mistral
def get_relevant_documents(nconst_embeddings, query, topK, model, tok):
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    task_query = get_detailed_instruct(task, query)
    print("task_query: ", task_query)
"""

NEW = """# SFR mistral
# ---------------------------------------------------------------------------
# [FIX] Query-wrapper switch.  ``None`` -> fall back to GGPP_USE_INSTRUCT, whose
#       default keeps the upstream behaviour, so callers that pass nothing are
#       completely unaffected (backward compatible).
#   Upstream unconditionally wraps the query as "Instruct: ...\\nQuery: ..."
#   (~25 extra tokens), whereas the paper's Table 2 caption and the appendix
#   A.3 / A.4 retrieval examples use a *bare* query.  Making this layer explicit
#   is what allows the attack side and the judge side to share one convention:
#     use_instruct=False -> bare query (paper convention)
#     use_instruct=True  -> upstream behaviour
# ---------------------------------------------------------------------------
USE_INSTRUCT = os.environ.get('GGPP_USE_INSTRUCT', '1') == '1'


def get_relevant_documents(nconst_embeddings, query, topK, model, tok, use_instruct=None):
    if use_instruct is None:
        use_instruct = USE_INSTRUCT
    if use_instruct:
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        task_query = get_detailed_instruct(task, query)
    else:
        task_query = query          # paper convention: bare query, no task instruction
    print("use_instruct: ", use_instruct)
    print("task_query: ", task_query)
"""


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing upstream file: {SRC}\n  -> run `python tools/fetch_upstream.py` first")

    with open(SRC, encoding="utf-8") as f:
        orig = f.read()

    if orig.count(OLD) != 1:
        sys.exit(f"anchor found {orig.count(OLD)}x (expected 1); upstream file may have changed")

    mod = orig.replace(OLD, NEW)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(difflib.unified_diff(
            orig.splitlines(keepends=True), mod.splitlines(keepends=True),
            fromfile="a/DGD/help_functions.py", tofile="b/DGD/help_functions.py", n=3))

    # also drop the patched full file, for people who prefer copying over patching
    dst = os.path.join(PKG, "DGD", "help_functions.py")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(mod)

    print(f"[ok] wrote {os.path.relpath(OUT, PKG)}")
    print(f"[ok] wrote {os.path.relpath(dst, PKG)} (patched full file)")


if __name__ == "__main__":
    main()
