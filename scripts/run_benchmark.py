#!/usr/bin/env python
"""Run the full agent-strategy benchmark:

    python scripts/run_benchmark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
