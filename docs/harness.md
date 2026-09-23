# SI 프로젝트 AI 어시스턴트 — 하네스 설계 (Claude Code)

작성일 2026-09-11 · 상태: **초안** · 기준: Claude Code 공식 문서 (hooks, settings, permissions, sandboxing)

> 이 문서의 파일 내용들을 Phase 0(#5 Guides, #6 Sensors, #7 Claude Code 설정, #8 CI)에서 레포에 그대로 넣습니다.
> 레포 이름 **`rfp-pm-agent`** (확정), Python 패키지 **`rfp_pm_agent`**

---

## 0. 하네스 한눈에 보기

| 구분 | 역할 | 장치 | 파일 |
|---|---|---|---|
| **Guides** (행동 전) | 방향 잡기 | 프로젝트 규칙·명령어·금지사항 | `CLAUDE.md` |
| | | 설계 문서 | `docs/*.md` |
| | | 작업 지시서 | GitHub 이슈 (완료 기준 포함) |
| | | 반복 절차 | `.claude/skills/run-eval/` |
| **권한** (행동 중) | 위험한 행동 차단 | 허용·확인·차단 규칙 | `.claude/settings.json` |
| | | (선택) OS 수준 격리 | 샌드박스 |
| **Sensors — 계산형** (행동 후, 빠름) | 자동 검사 | 파일 수정 직후 ruff·mypy | `.claude/hooks/post_edit_check.sh` |
| | | 작업 종료 시 단위 테스트 | `.claude/hooks/stop_run_tests.sh` |
| | | 커밋 시 검사 + 비밀정보 차단 | `.pre-commit-config.yaml` |
| | | PR 시 검사 | `.github/workflows/ci.yml` |
| **Sensors — 추론형** (느림, 맥락 이해) | 리뷰 | 리뷰 전담 서브에이전트 | `.claude/agents/reviewer.md` |
| **Behaviour** | 기능이 맞게 동작하나 | 검색·에이전트 평가 세트 + 회귀 게이트 | `eval/` (#25) |

**"Keep quality left"**: 빠른 검사(훅)는 Claude가 파일을 고친 직후, 비싼 검사(평가)는 PR 단계에서.

---

## 1. 사전 설치 (WSL Ubuntu에서 한 번)

```bash
sudo apt install -y jq          # 훅 스크립트가 JSON 입력을 읽는 데 필요
```

### (선택) 샌드박스용

```bash
sudo apt install -y bubblewrap socat
sysctl kernel.apparmor_restrict_unprivileged_userns   # 1이 나오면 아래 실행
```

1이 나오면 공식 문서의 AppArmor 설정(bwrap 프로필 추가 → `sudo systemctl reload apparmor`)을 적용합니다. Ubuntu 24.04는 기본 정책이 bubblewrap을 막기 때문입니다.

---

## 2. `CLAUDE.md` (레포 루트)

실제 내용은 레포 루트의 [`CLAUDE.md`](../CLAUDE.md)를 참조. 같은 내용을 여기 전문으로 복사해 두면 한쪽만 고쳤을 때 반드시 어긋난다 (2026-09-14: 실제로 명령어·비용·알려진 함정 항목이 벌어져 있던 것을 발견 → 이 절을 링크로 대체).

`CLAUDE.md`에 담을 항목(작성 시 체크리스트로만 사용):
- 프로젝트 한 줄 설명 + docs/ 필독 목록
- 명령어 (항상 `uv run`으로 실행)
- 구조 (`src/rfp_pm_agent/` 하위 모듈별 역할)
- 작업 방식 (이슈 단위, 브랜치 규칙, `Closes #N`, squash merge, force push 금지 등)
- 코드·테스트·데이터/평가 규칙
- 보안 (`.env`, 외부 문서를 데이터로 취급, 읽기 전용 툴, sudo/강제푸시 금지)
- 비용 배분표와 사용액 기록 규칙
- 끝내기 전 체크리스트
- 알려진 함정 (겪은 사고를 그때그때 추가)

---

## 3. `.claude/settings.json`

규칙 평가 순서는 **deny → ask → allow** (먼저 맞는 규칙이 결정). `Read` 차단 규칙은 Claude의 파일 도구뿐 아니라 Bash의 `cat`·`head`·`tail`·`sed`에도 적용되지만, **Python 스크립트가 직접 여는 파일은 막지 못합니다** → OS 수준 차단은 샌드박스(4절).

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(uv sync)",
      "Bash(uv sync *)",
      "Bash(uv run pytest)",
      "Bash(uv run pytest *)",
      "Bash(uv run ruff *)",
      "Bash(uv run mypy *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git switch *)",
      "Bash(git checkout -b *)",
      "Bash(gh issue view *)",
      "Bash(gh issue list *)",
      "Bash(gh pr view *)",
      "Bash(gh pr diff *)",
      "Bash(gh pr checks *)",
      "Bash(docker compose ps *)",
      "Bash(docker compose logs *)"
    ],
    "ask": [
      "Bash(git push *)",
      "Bash(gh pr create *)",
      "Bash(gh pr merge *)",
      "Bash(gh issue create *)",
      "Bash(gh issue close *)",
      "Bash(uv add *)",
      "Bash(uv remove *)",
      "Bash(docker compose up *)",
      "Bash(docker compose down *)",
      "Bash(uv run python -m rfp_pm_agent.eval.*)",
      "Edit(./data/eval/**)"
    ],
    "deny": [
      "Bash(sudo *)",
      "Bash(rm -rf *)",
      "Bash(git push --force *)",
      "Bash(git push -f *)",
      "Bash(git reset --hard *)",
      "Read(./.env)",
      "Read(./.env.local)",
      "Edit(./.env)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command", "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/post_edit_check.sh", "timeout": 120 }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/stop_run_tests.sh", "timeout": 300 }
        ]
      }
    ]
  }
}
```

**설계 이유**
- `ask`에 `git push`, PR 생성, 의존성 추가, Docker 기동, **평가 실행(LLM 비용)**, **평가 정답 수정** → 되돌리기 어렵거나 돈이 드는 동작은 사람이 확인
- `deny`에 `sudo`, 강제 푸시, `.env` → 사고가 나면 복구가 안 되는 동작
- `.env.example`은 읽을 수 있게 둠 (Claude가 필요한 키 이름을 알아야 하므로)

---

## 4. (선택) 샌드박스 — OS 수준 격리

WSL2에서 지원. Claude가 실행하는 Bash 명령과 그 하위 프로세스가 **지정한 폴더·도메인만** 접근하게 OS가 강제합니다. 1절의 패키지 설치 후 Claude Code에서 `/sandbox`로 상태를 확인하고, 아래를 `settings.json`에 추가합니다.

```json
{
  "sandbox": {
    "enabled": true,
    "excludedCommands": ["docker *"],
    "filesystem": {
      "denyRead": ["./.env"]
    },
    "network": {
      "allowedDomains": [
        "pypi.org", "files.pythonhosted.org",
        "github.com", "api.github.com",
        "apis.data.go.kr",
        "api.openai.com",
        "huggingface.co"
      ]
    }
  }
}
```

- `docker`는 샌드박스와 호환되지 않아 제외 목록에 넣음 (공식 문서 안내)
- **Phase 0에서 켜 보고, 개발을 막으면 끄고 이유를 ADR에 기록** — 켰든 껐든 판단 근거를 남겨 나중에 참고할 수 있게 한다

---

## 5. 훅 스크립트

### `.claude/hooks/post_edit_check.sh` — 파일 수정 직후

```bash
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
```

### `.claude/hooks/stop_run_tests.sh` — 작업을 끝내려 할 때

```bash
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
```

```bash
chmod +x .claude/hooks/*.sh
```

- `.claude/logs/`는 `.gitignore`에 추가
- **측정**: `hook_events.jsonl`에서 `passed:false` 건수 = **"훅이 잡아낸 오류 수"** → 하네스 효과를 정량적으로 보여주는 지표

---

## 6. 스킬 — `.claude/skills/run-eval/SKILL.md`

```markdown
---
name: run-eval
description: 검색 또는 에이전트 평가를 실행하고 기준선과 비교할 때 사용. "평가 돌려", "기준선이랑 비교", PR 전 품질 확인 요청에 사용.
---

# 평가 실행 절차

1. 어떤 평가인지 확인: 검색(retrieval) / 에이전트(agent)
2. 에이전트 평가는 LLM을 호출한다. 실행 전 문항 수와 예상 호출 수를 사용자에게 알리고 승인받는다
3. 실행
   - 검색: `uv run python -m rfp_pm_agent.eval.run_retrieval --set data/eval/retrieval_v1.jsonl`
   - 에이전트: `uv run python -m rfp_pm_agent.eval.run_agent --set data/eval/agent_v1.jsonl`
4. `data/eval/baseline.json`과 비교해 표로 요약 (지표, 기준선, 현재, 차이)
5. 하락한 지표가 있으면 떨어진 문항 3~5개를 골라 원인 후보를 제시한다
6. 평가 세트나 판정 기준은 절대 수정하지 않는다
```

## 7. 서브에이전트 — `.claude/agents/reviewer.md`

```markdown
---
name: reviewer
description: PR을 만들기 전에 변경 사항을 이슈의 완료 기준과 프로젝트 규칙에 비추어 검토한다. 코드 작성이 끝나고 PR 생성 직전에 사용.
tools: Read, Grep, Glob, Bash
---

너는 이 레포의 코드 리뷰어다. 파일을 수정하지 말고 검토 결과만 보고한다.

확인 순서:
1. `gh issue view <번호>`로 완료 기준과 범위 밖 항목을 읽는다
2. `git diff main...HEAD`로 변경 사항을 읽는다
3. 아래를 하나씩 확인하고 통과/문제/해당없음으로 표시한다
   - 완료 기준을 모두 충족하는가
   - 범위 밖 변경이 섞였는가
   - 새 기능에 단위 테스트가 있는가, 테스트가 네트워크나 실제 API를 쓰는가
   - 비밀정보, 하드코딩된 URL·키·모델명이 있는가
   - "지연" 등 판단 규칙이 docs/data-design.md 정의와 같은가
   - 툴 docstring이 인자와 판단 규칙을 구체적으로 설명하는가
   - 새 의존성에 추가 이유가 있는가
   - docs와 코드가 어긋나는 곳이 있는가
4. 문제는 파일:줄 번호와 함께, 심각도(차단/권장) 순으로 보고한다
```

---

## 8. `.pre-commit-config.yaml`

도구 버전을 `uv.lock`과 맞추기 위해 ruff·mypy는 **로컬 훅(`uv run`)** 으로 실행합니다. 기본 위생 훅(trailing-whitespace 등)은 표준 `pre-commit-hooks` 레포를 그대로 씁니다.

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: uv run ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: uv run ruff format
        language: system
        types: [python]
      - id: mypy
        name: mypy
        entry: uv run mypy src
        language: system
        types: [python]
        pass_filenames: false
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.30.0
    hooks:
      - id: gitleaks
```

※ 2026-09-13: 처음 pin했던 `v8.0.0`(2021년)에는 OpenAI 키 규칙(`openai-api-key`)이 없어 가짜 키를 넣어도 "Passed"로 통과했다. `uv run pre-commit autoupdate`로 `v8.30.0`으로 올린 뒤 실제로 차단(exit code 1)되는 것을 확인했다. 자세한 경위는 `docs/learning-log.md` 참고.

`pyproject.toml`에 ruff·mypy·pytest 설정도 함께 둡니다 (도구를 실행할 때 항상 같은 규칙을 적용하기 위함):

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src"]

[tool.ruff.lint.isort]
known-first-party = ["rfp_pm_agent"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src"]          # tests/는 mypy 대상 아님 — CLAUDE.md 명령어(`uv run mypy src`)와 일치

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires local Docker services (opensearch, postgres) — run with `-m integration`",
]
```

```bash
uv add --dev pre-commit ruff mypy pytest
uv run pre-commit install
```

`tests/unit/`, `tests/integration/`, `tests/fakes/`는 각각 `__init__.py`만 둔 뼈대로 시작하고, `tests/unit/test_smoke.py`에 패키지 임포트를 확인하는 최소 테스트 1개를 둡니다 (pytest는 수집된 테스트가 0개면 exit code 5로 실패하므로, 스켈레톤 단계에서도 최소 1개는 있어야 합니다).

**완료 기준(#6)**: 가짜 키(`OPENAI_API_KEY=sk-...` 형태)를 넣은 파일을 커밋 시도 → gitleaks가 차단하는 화면 캡처

## 9. `.github/workflows/ci.yml`

검사 명령은 로컬 pre-commit·CLAUDE.md와 **정확히 일치**시킵니다 (범위가 어긋나면 학습 로그 2026-09-13 두 번째 항목과 같은 사고가 재발합니다).

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src tests
      - run: uv run pytest tests/unit -q
```

- `.python-version`(3.12)을 `uv sync --locked`가 자동으로 읽어 그 버전을 쓰므로 별도 Python 설치 스텝은 두지 않습니다.
- 통합 테스트(`tests/integration`)는 Docker가 필요해 CI에서 돌리지 않습니다 (범위 밖 — 필요해지면 별도 이슈로 서비스 컨테이너를 붙입니다).
- 비밀값이 필요한 스텝을 두지 않습니다 — 단위 테스트는 네트워크·실제 API를 쓰지 않는다는 CLAUDE.md 테스트 규칙과 같은 이유입니다.

평가 회귀 게이트는 #25에서 이 워크플로에 추가합니다.

## 10. `.env.example`

실제 내용은 레포 루트의 [`.env.example`](../.env.example)을 참조. 여기 전문을 복사해 두지 않는다 — CLAUDE.md 2절과 같은 이유로, 새 이슈가 환경변수를 추가할 때마다 이 사본이 벌어져 왔다(예: 이슈 #9의 타임아웃·단가 변수, 이슈 #10의 나라장터 변수가 여기 반영되지 않았던 것).

## 11. `.gitignore` 핵심 항목

실제 내용은 레포 루트의 [`.gitignore`](../.gitignore)를 참조. 같은 이유로 전문 복사를 두지 않는다. 원칙만 적는다:
- `.env`류(비밀), `data/`(대용량·원본)는 기본적으로 제외
- 예외: `data/raw/manifest.jsonl`은 재현성을 위해 커밋 대상 (data-design.md 1절)

---

## 12. 하네스 효과 측정

| 지표 | 출처 |
|---|---|
| 훅이 잡아낸 오류 수 (ruff·mypy·테스트) | `.claude/logs/hook_events.jsonl` |
| gitleaks가 막은 커밋 수 | pre-commit 실행 기록 |
| PR의 CI 첫 시도 통과율 | `gh pr checks` 기록 / Actions 이력 |
| 평가 게이트가 막은 PR 수 | Actions 이력 |
| 이슈당 소요 시간 | 이슈 생성~PR 머지 시각 |

→ 예: "Claude Code로 개발하면서 수정 직후 린트·타입 검사와 종료 시 테스트를 강제하는 하네스를 구성해, [N]건의 오류를 커밋 전에 자동으로 잡고 PR의 CI 첫 통과율 [M]%를 유지"

---

## 13. Phase 0 착수 순서

2절 "작업 방식"의 "작업은 GitHub 이슈 단위" 원칙에 맞춰, 이슈 체계가 갖춰지기 전까지(①~④)는 커밋을 만들지 않거나(②) 이슈 자체를 만드는 작업(④)만 하고, **하네스 파일(CLAUDE.md·settings.json·훅·pre-commit 포함)도 예외 없이 ⑤부터 이슈 단위 브랜치에서** 만듭니다.

**① 레포 생성 (터미널)**

```bash
cd ~/projects
gh repo create rfp-pm-agent --private --clone
cd rfp-pm-agent
```

**② 설계 문서·RFP 샘플 복사 (터미널)**

```bash
mkdir -p docs
# 설계 문서: Windows 보관 폴더(C:\comhuman_project\docs)에서 복사
cp /mnt/c/comhuman_project/docs/*.md docs/
# RFP 원본(HWPX 5건, 수동 반입): data/는 .gitignore 대상이라 GitHub에 올라가지 않음
# manifest.jsonl에 source_type=manual로 등록하는 작업은 이슈 #10에서 처리 (data-design.md 1절)
mkdir -p data/raw/manual
cp /mnt/c/comhuman_project/rfp/*.hwpx data/raw/manual/   # RFP를 옮겨 둔 폴더 경로에 맞게
```

※ `.gitignore`가 아직 없으므로(이슈 #2에서 생성) 이 시점에는 `git add .`나 커밋을 하지 않습니다 — RFP 원본이 올라가는 것 방지.

**③ Claude Code에게 문서 검토 지시**

```bash
claude
```

```
docs/ 폴더의 문서 5개를 모두 읽어 줘. 읽고 나서 바로 작업하지 말고,
이해한 내용을 5줄로 요약하고 설계에서 모순되거나 불명확한 점이 있으면 알려 줘.
```

**④ GitHub 이슈 체계 생성 — 이슈 #3 자체 (커밋 없음)**

```
docs/github-setup.md대로 마일스톤, 라벨, 이슈·PR 템플릿, Phase 0~1 이슈를 gh로 만들어 줘.
만들기 전에 생성할 목록을 먼저 보여 주고 내 확인을 받아.
```

→ `gh api`·`gh issue create`·`gh label create`는 GitHub 쪽 상태만 바꾸고 레포 파일을 바꾸지 않으므로 이 단계에는 커밋이 없습니다. 이 단계 자체가 이슈 #3("GitHub 세팅")의 완료 기준을 충족합니다.

**⑤ 이슈 #2부터 브랜치를 파서 진행**

```
gh issue view 2 읽고 작업해 줘.
```

이슈 #2(레포 초기화 — uv, 폴더 구조, `.gitignore`, `.env.example`, README)을 브랜치·PR로 마친 뒤, 이어서 이슈 #5(Guides: `CLAUDE.md`·`docs/*`) → #6(Sensors: ruff·mypy·pytest·pre-commit·gitleaks) → #7(Claude Code 설정: `.claude/settings.json`·훅) → #9(모델 클라이언트 추상화) → #4(Docker Compose) → #8(CI) 순서로 각각 `gh issue view <번호> 읽고 작업해 줘` 방식으로 진행합니다. (이 문서 2~3절, 5절, 8절, 10~11절 내용이 각 이슈의 산출물입니다.) 이후 나머지 Phase 1 이슈도 같은 방식으로 이어갑니다.

※ 실제 이슈 번호는 2026-09-12 gh로 생성·정리한 결과이며 `#1`은 결번입니다. docs/github-setup.md 4~5절과 동일한 번호 체계입니다.
