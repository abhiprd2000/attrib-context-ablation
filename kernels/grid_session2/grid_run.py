"""Kaggle kernel entry point for ATTRIB grid SESSION 2 - the final 10 cells (5 new
task_ids x 2 seeds x 6 conditions = 60 episodes) completing the N=12 grid decided
2026-08-18 (see ANALYSIS_PLAN.md's "Grid size: REVISED 2026-08-18" section and
src/session2_core.py's module docstring for the real-number projection). Each
episode's RunRecord is written to /kaggle/working/runs/ immediately after it
completes - pull that directory even if the kernel errors out mid-run.

Requires the `attrib-code` dataset (this repo's src/, including session2_core.py
and pilot_core.py) attached as a kernel input, same as session 1.

NOT PUSHED as of this file's creation - built and syntax-checked only, per the
operator's explicit instruction to wait for Kaggle API access confirmation.
"""

from __future__ import annotations

import os
import sys
import time

CANDIDATES = [
    "/kaggle/input/attrib-code",
    "/kaggle/input/datasets/abhiprd20/attrib-code",
]
CODE_DIR = None
for _ in range(30):
    for c in CANDIDATES:
        if os.path.exists(os.path.join(c, "session2_core.py")):
            CODE_DIR = c
            break
    if CODE_DIR:
        break
    time.sleep(2)
if CODE_DIR is None:
    raise RuntimeError(f"session2_core.py not found under any of {CANDIDATES}")
sys.path.insert(0, CODE_DIR)

import session2_core  # noqa: E402

session2_core.install_deps()
session2_core.run_grid_session2()
