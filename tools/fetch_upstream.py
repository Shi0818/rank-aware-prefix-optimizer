# -*- coding: utf-8 -*-
"""
Fetch the upstream GGPP files and install the ones the package imports
======================================================================
The upstream release ships **no licence file**, so this repository does not
redistribute its code. Instead this script downloads a byte-identical copy into
``third_party/upstream/`` and then *installs* the two files the package imports
from:

    DGD/DGD.py             upstream copy, used as the "original" arm of the A/B
    DGD/help_functions.py  upstream file + the query-wrapper switch (regenerated)

Both are git-ignored. After ``python tools/fetch_upstream.py`` the import paths
used everywhere else in the code work unchanged.

The downloaded copies are also the baseline for

  * ``tools/build_rank_notebook.py``     regenerating the improved notebook,
  * ``tools/build_help_functions_patch.py`` regenerating the query-wrapper switch,
  * ``tools/verify_upstream_diff.py``    proving that only the intended lines differ.

Usage:
  python tools/fetch_upstream.py             # download (if needed) and install
  python tools/fetch_upstream.py --force     # re-download everything
  python tools/fetch_upstream.py --check     # only verify existing files by md5
  python tools/fetch_upstream.py --no-install  # download only, do not touch DGD/
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

UPSTREAM_REPO = "Hadise-zb/Prompt-Perturbation-in-Retrieval-Augmented-Generation"
UPSTREAM_REF = "main"
BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_REF}/"

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DST_DIR = os.path.join(PKG, "third_party", "upstream")

# local filename -> path inside the upstream repository
FILES = {
    "DGD_DGD.py": "DGD/DGD.py",
    "DGD_help_functions.py": "DGD/help_functions.py",
    "SFR_mistral_IMDB.ipynb": "SFR_mistral_IMDB.ipynb",
    "Data_Preprocessing.py": "Data_Preprocessing.py",
    "README.md": "README.md",
}

# md5 of the upstream revision this repository was built and measured against
# (recorded so `--check` can flag silent upstream drift)
KNOWN_MD5 = {
    "DGD_DGD.py": "bdb6624c7c20c7adf85587ce4b00c3dc",
    "DGD_help_functions.py": "b8cd2554504094d261eb50d6dc1b76df",
    "Data_Preprocessing.py": "1b0f2f3a5984ef2b0b19394458609547",
    "SFR_mistral_IMDB.ipynb": "15d784d472897947dc50852ea88ee8e9",
}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "GGPP-RankAware/fetch"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def install():
    """Place the upstream files where the package imports them from.

    Neither file is distributed in this repository: the upstream release carries no
    licence, so redistributing its code would have no legal basis. They are
    regenerated here instead, which keeps every import path in the code unchanged
    while the repository itself contains only this project's own work.
    """
    src_dgd = os.path.join(DST_DIR, "DGD_DGD.py")
    if not os.path.exists(src_dgd):
        print("  [skip] install: third_party/upstream/DGD_DGD.py is missing")
        return False

    dst_dgd = os.path.join(PKG, "DGD", "DGD.py")
    shutil.copyfile(src_dgd, dst_dgd)
    print(f"  install DGD/DGD.py             (byte-identical upstream copy)")

    # help_functions.py = upstream + the switchable query wrapper
    builder = os.path.join(HERE, "build_help_functions_patch.py")
    r = subprocess.run([sys.executable, builder], capture_output=True, text=True)
    dst_hf = os.path.join(PKG, "DGD", "help_functions.py")
    if r.returncode != 0 or not os.path.exists(dst_hf):
        err = (r.stderr or r.stdout or "").strip().splitlines()
        print(f"  [FAIL] install DGD/help_functions.py: {err[-1] if err else 'unknown error'}")
        return False
    print(f"  install DGD/help_functions.py  (upstream + query-wrapper switch)")
    return True


def main():
    ap = argparse.ArgumentParser(description="fetch upstream GGPP baseline files")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--check", action="store_true", help="only report md5 of what exists")
    ap.add_argument("--no-install", action="store_true",
                    help="download into third_party/upstream/ but do not write DGD/")
    args = ap.parse_args()

    os.makedirs(DST_DIR, exist_ok=True)
    ok, skipped, failed = [], [], []

    for local, remote in FILES.items():
        dst = os.path.join(DST_DIR, local)
        if args.check:
            if os.path.exists(dst):
                ok.append((local, open(dst, "rb").read()))
            else:
                skipped.append(local)
            continue
        if os.path.exists(dst) and not args.force:
            skipped.append(local)
            continue
        try:
            data = fetch(BASE + remote)
            with open(dst, "wb") as f:
                f.write(data)
            ok.append((local, data))
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append((local, f"{type(e).__name__}: {e}"))

    print(f"upstream: {UPSTREAM_REPO}@{UPSTREAM_REF}")
    print(f"target  : {os.path.relpath(DST_DIR, PKG)}")
    print()
    for name, data in ok:
        md5 = hashlib.md5(data).hexdigest()
        note = ""
        if name in KNOWN_MD5:
            note = "  [matches recorded revision]" if md5 == KNOWN_MD5[name] \
                   else f"  [WARN] recorded md5 was {KNOWN_MD5[name]}"
        print(f"  ok      {name:26s} {len(data):9d} B  md5={md5}{note}")
    for name in skipped:
        print(f"  skip    {name:26s} (already present)")
    for name, err in failed:
        print(f"  FAILED  {name:26s} {err}")

    if failed:
        print("\nSome files could not be downloaded. If you are offline, download them "
              "manually from the upstream repository and place them in "
              f"{os.path.relpath(DST_DIR, PKG)}/ using the names listed above.")
        return 1

    if not args.check and not args.no_install:
        print("\nInstalling into DGD/ (not tracked by git):")
        if not install():
            print("\nInstall failed; the package cannot import the original arm. "
                  "Re-run with --force, or place the files manually.")
            return 1
        print("\nDone. `python tools/verify_upstream_diff.py` should now pass.")

    if skipped and not args.check:
        print("\nTip: pass --force to refresh, or check md5 drift with --check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
