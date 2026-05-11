#!/bin/bash

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash .ci/scripts/run_precheck.sh

Create runtime output directories, verify required lightweight Python modules,
and reject secret-like files under .ci/.
EOF
  exit 0
fi

mkdir -p reports/nightly/case_results logs/deploy docs/deploy/generated

python3 - <<'PY'
import importlib.util
import sys

missing = []
for module in ("yaml",):
    if importlib.util.find_spec(module) is None:
        missing.append(module)

if missing:
    print("Missing required Python modules:", ", ".join(missing), file=sys.stderr)
    print("Install CI dependencies with: python3 -m pip install -r .ci/requirements.txt", file=sys.stderr)
    raise SystemExit(1)

if importlib.util.find_spec("jinja2") is None:
    print("WARNING: jinja2 is not installed; render scripts will use their built-in fallback renderer.")
PY

if find .ci -type f \( -name '*.pem' -o -name '*.key' -o -name '*secret*' -o -name '*token*' \) | grep -q .; then
  echo "Potential secret-like file found under .ci; do not store credentials in the repository." >&2
  exit 1
fi

echo "Precheck passed."
