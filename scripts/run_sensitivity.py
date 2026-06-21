#!/usr/bin/env python3
"""Thin wrapper for the sensitivity analysis. Logic in gturf.cli.sensitivity_main.

Example
-------
    python scripts/run_sensitivity.py --synthetic --r 10 --output-dir sens_run
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gturf.cli import sensitivity_main

if __name__ == "__main__":
    sensitivity_main()
