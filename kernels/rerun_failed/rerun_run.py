"""Kaggle kernel entry point for the targeted 3-episode rerun (session 1's crashed
cells, after fixing the ContextWindowExceededError handling). See
src/rerun_failed_core.py's module docstring. Requires the `attrib-code` dataset
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
        if os.path.exists(os.path.join(c, "rerun_failed_core.py")):
            CODE_DIR = c
            break
    if CODE_DIR:
        break
    time.sleep(2)
if CODE_DIR is None:
    raise RuntimeError(f"rerun_failed_core.py not found under any of {CANDIDATES}")
sys.path.insert(0, CODE_DIR)

import rerun_failed_core  # noqa: E402

rerun_failed_core.install_deps()
rerun_failed_core.run_rerun_failed()
