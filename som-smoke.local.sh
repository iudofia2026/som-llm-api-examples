#!/usr/bin/env bash
# som-smoke.local.sh — one-command SOM router smoke test (local-only, untracked).
#
# Run this the moment you are (a) on Yale network / VPN and (b) have pasted a real
# SOM_LLM_KEY into .env. It answers, in order:
#   1. Can this machine reach api.som.chat?
#   2. Does the key work?
#   3. WHICH MODELS ARE SERVED — and is any of them vision-capable (the gating
#      question for the academic-index campus-photo vetting pipeline)?
#   4. Does a basic text completion work?
#   5. If a vision model exists, can it actually read an image?
#
# Usage:  ./som-smoke.local.sh
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && set -a && source .env && set +a
: "${SOM_LLM_BASE_URL:=https://api.som.chat/v1}"

if [ -z "${SOM_LLM_KEY:-}" ] || [[ "$SOM_LLM_KEY" == *REPLACE_ME* ]]; then
  echo "✗ No real SOM_LLM_KEY in .env — get one from the SOM LLM API dashboard"
  echo "  (try https://api.som.chat/keys or the root page — Yale network required)"
  exit 1
fi

echo "→ 1. reachability"
if ! curl -s -m 8 -o /dev/null "$SOM_LLM_BASE_URL/models"; then
  echo "✗ Cannot reach $SOM_LLM_BASE_URL — api.som.chat resolves to a Yale-internal"
  echo "  address (172.29.x.x). Connect to Yale WiFi or Yale VPN and re-run."
  exit 1
fi
echo "  ✓ reachable"

echo "→ 2/3. auth + model list"
MODELS_JSON=$(curl -s -m 15 -H "Authorization: Bearer $SOM_LLM_KEY" "$SOM_LLM_BASE_URL/models")
echo "$MODELS_JSON" | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("  ✗ Non-JSON response (bad key or gateway error):"); sys.exit(1)
if "data" not in d:
    print("  ✗ Unexpected response:", json.dumps(d)[:300]); sys.exit(1)
ids = [m.get("id","?") for m in d["data"]]
print(f"  ✓ key accepted — {len(ids)} model(s) served:")
for i in ids: print(f"    • {i}")
vision = [i for i in ids if any(k in i.lower() for k in ("vl","vision","kimi","image","gemma-3","pixtral","llava","qwen2.5-omni","internvl"))]
print()
if vision:
    print(f"  ★ VISION CANDIDATES: {vision}")
    print("    → campus-photo vetting can run on SOM directly.")
else:
    print("  ★ NO obvious vision model — pipeline falls back to metadata-only ranking")
    print("    (Commons titles/descriptions/categories), per the Aug 5 notebook plan.")
'

echo "→ 4. text smoke"
FIRST_MODEL=$(echo "$MODELS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s -m 30 -H "Authorization: Bearer $SOM_LLM_KEY" -H "Content-Type: application/json" \
  "$SOM_LLM_BASE_URL/chat/completions" -d "{
    \"model\": \"$FIRST_MODEL\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],
    \"max_tokens\": 8, \"temperature\": 0,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" | python3 -c '
import json,sys
d=json.load(sys.stdin)
msg=d.get("choices",[{}])[0].get("message",{}).get("content","<no content>")
print(f"  model reply: {msg!r}")
'

echo
echo "→ 5. vision smoke: if a vision model was listed above, run:"
echo "   ./som-vision-smoke.local.sh <model-id>   (uses a Yale Old Campus photo from Commons)"
echo "→ done — paste this output into the media-pipeline notebook."
