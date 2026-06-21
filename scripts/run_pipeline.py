#!/usr/bin/env python3
"""Thin wrapper so `python scripts/run_pipeline.py ...` works from a source
checkout. The real logic lives in :func:`gturf.cli.pipeline_main`.

Examples
--------
    python scripts/run_pipeline.py --synthetic --output-dir demo_run
    python scripts/run_pipeline.py --oja ojas.xlsx --esco-mapping esco.xlsx \
        --pillar knowledge --crossover-rate 0.7 --generations 60
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gturf.cli import pipeline_main

if __name__ == "__main__":
    pipeline_main()
