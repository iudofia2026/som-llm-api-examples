#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$repo_dir/scripts/som-current-model.py" --help >/dev/null
python3 "$repo_dir/scripts/configure-pi.py" --help >/dev/null

for script in \
  "$repo_dir/examples/openai-chat/example.py" \
  "$repo_dir/examples/anthropic-messages/example.py" \
  "$repo_dir/examples/efficient-client/example.py" \
  "$repo_dir/examples/json-object-extraction/example.py"
do
  echo "==> ${script#$repo_dir/} --self-test"
  python3 "$script" --self-test
done
