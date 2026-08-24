"""Kaggle kernel entry point for ATTRIB grid SESSION 1 - a real subset of the
240-episode grid (7 tasks x 2 seeds x 6 conditions = 84 episodes), sized to fit
inside one 9h Kaggle GPU session with margin. See src/grid_core.py's module
docstring for the real-number split derivation and kernels/README.md for deploy
notes. Each episode's RunRecord is written to /kaggle/working/runs/ immediately
after it completes - pull that directory even if the kernel errors out mid-run.

Requires the `attrib-code` dataset (this repo's src/, including grid_core.py and
pilot_core.py) attached as a kernel input.
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
        if os.path.exists(os.path.join(c, "grid_core.py")):
            CODE_DIR = c
            break
    if CODE_DIR:
        break
    time.sleep(2)
if CODE_DIR is None:
    raise RuntimeError(f"grid_core.py not found under any of {CANDIDATES}")
sys.path.insert(0, CODE_DIR)

import grid_core  # noqa: E402

grid_core.install_deps()
grid_core.run_grid_session1()
