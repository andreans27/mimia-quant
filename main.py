#!/usr/bin/env python3
"""Mimia Quant — Systematic Crypto Trading System"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.trading.cli import main

if __name__ == '__main__':
    main()
