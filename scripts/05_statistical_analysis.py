#!/usr/bin/env python3
"""
Pipeline Step 05: Statistical Validation & ANOVA Wrapper.
Delegates to `experiments/qwen2_5_7b/reproduce_analysis.py` to verify
cell means, domain breakdowns, Two-Way Factorial ANOVA, and paired t-tests.
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET = ROOT_DIR / "experiments" / "qwen2_5_7b" / "reproduce_analysis.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(TARGET)]))
