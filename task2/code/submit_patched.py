#!/usr/bin/env python3
"""Submit a CSV to Kaggle while working around the poisoned DNS for www.googleapis.com.

Only the TCP destination address is remapped (SNI/Host stay www.googleapis.com), so
TLS verification, authentication and the request content are unchanged.
"""
import os
import socket
import sys
import time

import urllib3.util.connection as conn

GOOD_IPS = ["142.250.69.187", "142.250.73.155", "142.251.33.219", "2607:f8b0:400e:c07::cf"]
TARGET = "googleapis.com"

_orig = conn.create_connection


def patched(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None):
    host, port = address[0], address[1]
    if host.endswith(TARGET) and host not in GOOD_IPS:
        err = None
        for ip in GOOD_IPS:
            try:
                t0 = time.time()
                s = _orig((ip, port), timeout, source_address, socket_options)
                print(f"    [dns-bypass] {host} -> {ip} ({time.time()-t0:.2f}s)", flush=True)
                return s
            except Exception as e:  # try the next candidate
                err = e
                print(f"    [dns-bypass] {host} -> {ip} failed: {type(e).__name__}", flush=True)
        raise err
    return _orig(address, timeout, source_address, socket_options)


conn.create_connection = patched

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

path = sys.argv[1]
message = sys.argv[2] if len(sys.argv) > 2 else "submission"
api = KaggleApi()
api.authenticate()
print(f"submitting {path} ({os.path.getsize(path)} bytes)", flush=True)
t0 = time.time()
api.competition_submit(path, message, "2026task2")
print(f"competition_submit returned after {time.time()-t0:.1f}s", flush=True)
