"""streaming.py — Streaming chat example.

Run:
    SPM_API_KEY=<your-key> python sdk/python/examples/streaming.py
"""
from __future__ import annotations

import os, sys
from spm_client import SpmClient

API_KEY = os.environ.get("SPM_API_KEY", "")
GATEWAY_URL = os.environ.get("SPM_GATEWAY_URL", "http://localhost:18081")
if not API_KEY:
    sys.exit("Set SPM_API_KEY")

with SpmClient(API_KEY, base_url=GATEWAY_URL) as client:
    session = client.session()
    print("Streaming response:")
    for chunk in session.send("Tell me a one-sentence joke.", stream=True):
        print(chunk, end="", flush=True)
    print()
