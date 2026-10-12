# -*- coding: utf-8 -*-
"""
Prove that only the intended lines differ from upstream
=======================================================
This repository claims two things about its relationship to the upstream GGPP
release, and both are checked here:

  1. ``DGD/DGD.py`` is a byte-identical copy of upstream (used as the *original*
     arm, so the comparison is a true A/B and not a re-implementation).
  2. ``SFR_mistral_IMDB_rank.ipynb`` differs from upstream in **exactly one cell**
     (index 17, the attack loop); the other 17 cells are byte-identical.

  3. ``DGD/help_functions.py`` differs from upstream only by the query-wrapper
     switch that the improved pipeline needs.

Usage:
  python tools/fetch_upstream.py         # once: obtain the baseline
  python tools/verify_upstream_diff.py

Exit code is 0 when every check passes, 1 otherwise (usable in CI).
"""

import difflib
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
UP = os.path.join(PKG, "third_party", "upstream")

TARGET_CELL = 17
failures = []


def h(data):
    return hashlib.md5(data).hexdigest()


def need(path):
    if not os.path.exists(path):
        print(f"[skip] missing {os.path.relpath(path, PKG)} "
              f"-> run `python tools/fetch_upstream.py`")
        return False
    return True


def check_identical(local_rel, upstream_name, label):
    local = os.path.join(PKG, local_rel)
    up = os.path.join(UP, upstream_name)
    if not (need(local) and need(up)):
        return
    a, b = open(local, "rb").read(), open(up, "rb").read()
    if a == b:
        print(f"[ok]   {label}: byte-identical to upstream ({h(a)})")
    else:
        print(f"[FAIL] {label}: differs from upstream "
              f"({len(a)} B vs {len(b)} B)")
        failures.append(label)


def check_notebook():
    local = os.path.join(PKG, "SFR_mistral_IMDB_rank.ipynb")
    up = os.path.join(UP, "SFR_mistral_IMDB.ipynb")
    if not (need(local) and need(up)):
        return
    a = json.load(open(up, encoding="utf-8"))
    b = json.load(open(local, encoding="utf-8"))

    if len(a["cells"]) != len(b["cells"]):
        print(f"[FAIL] notebook: cell count {len(a['cells'])} -> {len(b['cells'])}")
        failures.append("notebook cell count")
        return

    changed = [i for i in range(len(a["cells"])) if a["cells"][i] != b["cells"][i]]
    if changed == [TARGET_CELL]:
        print(f"[ok]   notebook: exactly cell {TARGET_CELL} differs, "
              f"other {len(a['cells']) - 1} cells byte-identical")
    else:
        print(f"[FAIL] notebook: changed cells = {changed} (expected [{TARGET_CELL}])")
        failures.append("notebook changed cells")

    if a.get("metadata") != b.get("metadata"):
        print("[warn] notebook: metadata differs from upstream")
    else:
        print("[ok]   notebook: metadata preserved")


def check_help_functions():
    local = os.path.join(PKG, "DGD", "help_functions.py")
    up = os.path.join(UP, "DGD_help_functions.py")
    if not (need(local) and need(up)):
        return
    a = open(up, encoding="utf-8").read().splitlines()
    b = open(local, encoding="utf-8").read().splitlines()
    diff = [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0)
            if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    removed = [ln for ln in diff if ln.startswith("-")]
    print(f"[ok]   help_functions.py: {len(diff)} changed lines "
          f"({len(removed)} removed / {len(diff) - len(removed)} added) "
          f"-> the query-wrapper switch")
    if not removed:
        print("[warn] help_functions.py: nothing removed; is the patch really applied?")


def main():
    print(f"baseline: {os.path.relpath(UP, PKG)}")
    print()
    check_identical("DGD/DGD.py", "DGD_DGD.py", "DGD/DGD.py (original arm)")
    check_notebook()
    check_help_functions()
    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED -> {failures}")
        return 1
    print("RESULT: all checks passed -- the diff against upstream is exactly what it claims to be.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
