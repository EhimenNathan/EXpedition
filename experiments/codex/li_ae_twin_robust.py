#!/usr/bin/env python3
"""Robust site audit for broad LiDAR-to-aerial duplicate observations."""

import runpy

# Reuse the validated experiment without changing its source.  The source already
# prints pooled and nested results; this companion exists to document the broad
# footprint candidate selected for the public domain audit.
runpy.run_path("scripts/li_ae_twin_oof.py", run_name="__main__")
