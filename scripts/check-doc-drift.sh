#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Stale production-doc strings that should not appear in user examples.
# Keep this list small and tied to real historical drift we have seen.
stale_pattern='api2|kyle[.]pub|som-llm-api[.]kyle[.]pub|Qwen3[.]5|Qwen3-Coder|Coder-Next|122B|post-import|key parity|phoenix-beta|Phoenix beta'

if grep -RInE \
  --exclude-dir=.git \
  --exclude='check-doc-drift.sh' \
  --exclude='*.pyc' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.gif' \
  --exclude='*.svg' \
  "$stale_pattern" "$repo_dir"; then
  echo "stale SOM API documentation reference found" >&2
  exit 1
fi
