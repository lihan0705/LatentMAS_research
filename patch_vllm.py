#!/usr/bin/env python3
"""Patch vLLM cuda platform to handle broken GPUs gracefully."""

import os

# Path to the vLLM cuda platform file
cuda_file = ".venv/lib/python3.12/site-packages/vllm/platforms/cuda.py"

# Read the file
with open(cuda_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'try:\n    CudaPlatform.log_warnings()\nexcept Exception:' in content:
    print("Already patched")
else:
    # Replace the problematic line
    old_line = "CudaPlatform.log_warnings()"
    new_line = """try:
    CudaPlatform.log_warnings()
except Exception:
    pass"""

    content = content.replace(old_line, new_line)

    # Write back
    with open(cuda_file, 'w') as f:
        f.write(content)
    print("Patched successfully")
