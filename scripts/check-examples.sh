#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m py_compile "$repo_dir"/examples/python/*.py "$repo_dir"/scripts/*.py
"$repo_dir"/scripts/check-doc-drift.sh
