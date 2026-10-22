#!/bin/sh
# Hardware self-test on Linux / macOS (close Amin Studio first).
cd "$(dirname "$0")/.." && exec python3 tests/hw_selftest.py "$@"
