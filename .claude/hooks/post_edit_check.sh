#!/usr/bin/env bash
# Claude가 .py 파일을 수정하면 해당 파일에 ruff(자동 수정·포맷)와 mypy 실행.
# 실패하면 exit 2 → 오류 내용이 Claude에게 전달되어 바로 고치게 됨.
set -uo pipefail
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')
[[ "$FILE" == *.py ]] || exit 0

cd "${CLAUDE_PROJECT_DIR}"
LOG=".claude/logs/hook_events.jsonl"; mkdir -p "$(dirname "$LOG")"

OUT=$( { uv run ruff check --fix "$FILE" && uv run ruff format "$FILE" && uv run mypy "$FILE"; } 2>&1 )
STATUS=$?

jq -nc --arg ts "$(date -Iseconds)" --arg file "$FILE" --argjson ok "$([ $STATUS -eq 0 ] && echo true || echo false)" \
  '{ts:$ts, hook:"post_edit_check", file:$file, passed:$ok}' >> "$LOG"

if [ $STATUS -ne 0 ]; then
  echo "[post_edit_check] $FILE 검사 실패. 아래 오류를 고치세요:" >&2
  echo "$OUT" | tail -40 >&2
  exit 2
fi
exit 0
