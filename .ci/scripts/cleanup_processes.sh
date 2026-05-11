#!/bin/bash

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash .ci/scripts/cleanup_processes.sh [LOG_ROOT]

Stop DeployCase processes recorded in *.pid files under LOG_ROOT.
Defaults to logs/deploy.
EOF
  exit 0
fi

LOG_ROOT="${1:-logs/deploy}"

if [[ ! -d "${LOG_ROOT}" ]]; then
  echo "No deploy log directory found: ${LOG_ROOT}"
  exit 0
fi

while IFS= read -r pid_file; do
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    continue
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping DeployCase process ${pid} from ${pid_file}"
    kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    sleep 5
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
done < <(find "${LOG_ROOT}" -name '*.pid' -type f)

echo "DeployCase cleanup completed."
