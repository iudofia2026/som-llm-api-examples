#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

for script in \
  "$repo_dir/examples/openai-chat/example.py" \
  "$repo_dir/examples/anthropic-messages/example.py" \
  "$repo_dir/examples/efficient-client/example.py" \
  "$repo_dir/examples/json-object-extraction/example.py"
do
  echo "==> ${script#$repo_dir/} --self-test"
  python3 "$script" --self-test
done
