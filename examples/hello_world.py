"""hello_world.py — Minimal SPM-Maverick Python SDK example.

Run:
    pip install httpx
    pip install -e sdk/python          # local dev install
    SPM_API_KEY=<your-key> python sdk/python/examples/hello_world.py
"""
from __future__ import annotations

import os
import sys

from spm_client import SpmClient, AuthError, SpmError

API_KEY = os.environ.get("SPM_API_KEY", "")
GATEWAY_URL = os.environ.get("SPM_GATEWAY_URL", "http://localhost:18081")

if not API_KEY:
    print("Set SPM_API_KEY before running this example.", file=sys.stderr)
    sys.exit(1)

with SpmClient(API_KEY, base_url=GATEWAY_URL) as client:
    print("Health:", client.health())

    session = client.session()
    reply = session.send("What is 2 + 2?", model="gpt-4o-mini")
    print("Assistant:", reply)

    reply2 = session.send("And multiply that by 3?")
    print("Assistant:", reply2)

    print(f"\nSession history ({len(session.history)} messages)")
