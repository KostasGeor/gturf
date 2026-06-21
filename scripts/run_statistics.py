#!/usr/bin/env python3
"""Thin wrapper for the statistical analysis. Logic in gturf.cli.statistics_main.

Example
-------
    python scripts/run_statistics.py --synthetic --tost-margin 0.5 --output-dir stats_run
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gturf.cli import statistics_main

if __name__ == "__main__":
    statistics_main()
