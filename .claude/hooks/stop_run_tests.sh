#!/usr/bin/env bash
# Claude가 작업을 끝내려 할 때 단위 테스트 실행. 실패하면 exit 2 → 멈추지 않고 고치게 함.
# stop_hook_active가 true면(이미 한 번 막았음) 무한 반복 방지를 위해 통과.
set -uo pipefail
INPUT=$(cat)
[ "$(printf '%s' "$INPUT" | jq -r '.stop_hook_active')" = "true" ] && exit 0

cd "${CLAUDE_PROJECT_DIR}"
LOG=".claude/logs/hook_events.jsonl"; mkdir -p "$(dirname "$LOG")"

# 바뀐 .py 파일이 없으면 생략 (문서만 고친 경우 등)
CHANGED=$( { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } 2>/dev/null )
[ -z "$CHANGED" ] && exit 0

OUT=$(uv run pytest tests/unit -q -x 2>&1)
STATUS=$?

jq -nc --arg ts "$(date -Iseconds)" --argjson ok "$([ $STATUS -eq 0 ] && echo true || echo false)" \
  '{ts:$ts, hook:"stop_run_tests", passed:$ok}' >> "$LOG"

if [ $STATUS -ne 0 ]; then
  echo "[stop_run_tests] 단위 테스트 실패. 고친 뒤 끝내세요:" >&2
  echo "$OUT" | tail -40 >&2
  exit 2
fi
exit 0
