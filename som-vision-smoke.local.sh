#!/usr/bin/env bash
# som-vision-smoke.local.sh <model-id> — test whether a SOM-served model can
# actually judge a campus photo (local-only, untracked).
#
# Sends one CC-licensed Wikimedia Commons photo of Yale Old Campus and asks the
# exact kind of question the academic-index vetting pipeline needs answered.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: ./som-vision-smoke.local.sh <model-id>}"
[ -f .env ] && set -a && source .env && set +a
: "${SOM_LLM_BASE_URL:=https://api.som.chat/v1}"

IMG=/tmp/som-vision-test.jpg
if [ ! -s "$IMG" ]; then
  echo "→ fetching test image (Yale Old Campus, Wikimedia Commons, 640px thumb)"
  curl -sL -m 20 -A "academic-index-photobot/1.0 (https://academicindex.ai; mudofia1@gmail.com)" \
    "https://commons.wikimedia.org/w/thumb.php?f=Old_Campus_at_Yale_University.jpg&w=640" -o "$IMG"
fi
B64=$(base64 < "$IMG" | tr -d '\n')

python3 - "$MODEL" "$B64" <<'PY'
import json, os, sys, urllib.request
model, b64 = sys.argv[1], sys.argv[2]
base = os.environ.get("SOM_LLM_BASE_URL", "https://api.som.chat/v1")
key = os.environ["SOM_LLM_KEY"]
body = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": (
                "You are vetting candidate photos for a college-guide gallery. "
                "Answer in strict JSON: {\"is_campus_photo\": bool, \"quality\": \"high|medium|low\", "
                "\"subject\": \"<3 words>\", \"reject_reasons\": []}"
            )},
        ],
    }],
    "max_tokens": 96,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    print("  ✓ vision reply:", d["choices"][0]["message"]["content"])
    print("  → VISION VETTING VIABLE on SOM. Record model id in the notebook.")
except Exception as e:
    print(f"  ✗ vision call failed ({e}) — model likely text-only; use metadata-only fallback.")
PY
