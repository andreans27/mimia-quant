#!/usr/bin/env python3
"""Validate project configuration files."""
import sys
try:
    import yaml
    print(f"yaml OK (v{yaml.__version__})")
except ImportError as e:
    print(f"yaml import FAILED: {e}")
    sys.exit(1)

for path in ("config/config.yaml", "configs/strategies.yaml"):
    with open(path) as f:
        data = yaml.safe_load(f)
        print(f"  ✓ {path} loaded")

config = yaml.safe_load(open("config/config.yaml"))
name = config.get("system", {}).get("name", "UNKNOWN")
print(f"  System: {name}")

strategies = yaml.safe_load(open("configs/strategies.yaml"))
keys = list(strategies.get("strategies", {}).keys())
print(f"  Strategies ({len(keys)}): {keys}")

print("All configs valid!")
