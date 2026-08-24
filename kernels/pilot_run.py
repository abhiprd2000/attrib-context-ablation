"""Kaggle kernel entry point for the ATTRIB pilot - one episode per condition
(6 total), same task, same seed. Purpose: wall-clock and end-to-end plumbing,
not results. See kernels/README.md and PROGRESS.md.

Requires the `attrib-code` dataset (this repo's src/ + kernels/pilot_core.py)
attached as a kernel input.
"""

from __future__ import annotations

import os
import sys
import time

CANDIDATES = [
    "/kaggle/input/attrib-code",
    "/kaggle/input/datasets/abhiprd2000/attrib-code",
]
CODE_DIR = None
for _ in range(30):
    for c in CANDIDATES:
        if os.path.exists(os.path.join(c, "pilot_core.py")):
            CODE_DIR = c
            break
    if CODE_DIR:
        break
    time.sleep(2)
if CODE_DIR is None:
    raise RuntimeError(f"pilot_core.py not found under any of {CANDIDATES}")
sys.path.insert(0, CODE_DIR)

import pilot_core  # noqa: E402

pilot_core.install_deps()
pilot_core.run_pilot()
