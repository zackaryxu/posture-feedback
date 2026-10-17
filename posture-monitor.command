#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Install first: python3.11 -m venv .venv && .venv/bin/python -m pip install ." >&2
  exit 2
fi
exec .venv/bin/python -m posture_feedback "$@"
