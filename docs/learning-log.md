# 학습 로그

작업하며 겪은 실수·삽질과 거기서 배운 것을 짧게 기록한다. 형식: 날짜 / 무슨 일 / 원인 / 배운 것.

---

## 2026-09-13 — gitleaks가 오래된 rev라 가짜 키를 못 잡음

**무슨 일**: 이슈 #6에서 `.pre-commit-config.yaml`에 gitleaks를 추가했는데, 가짜 `OPENAI_API_KEY=sk-...` 키를 넣은 파일을 커밋해도 "Detect hardcoded secrets ... Passed"로 통과했다.

**원인**: pin한 `rev: v8.0.0`(2021년 릴리스)에는 `openai-api-key` 탐지 규칙이 아직 없었다. 훅 자체는 정상 실행됐고 "Passed"까지 찍혔기 때문에, 로그만 보면 정상으로 착각하기 쉬웠다.

**배운 것**: pre-commit 훅이 "Passed"라고 초록불을 켜는 것과 "실제로 탐지 규칙이 최신이라 뭔가를 잡아낼 수 있는 상태"는 별개다. `uv run pre-commit autoupdate`로 `v8.30.0`까지 올린 뒤 같은 가짜 키로 다시 커밋을 시도해 exit code 1로 실제 차단되는 것을 확인했다. 앞으로 보안·품질 관련 훅을 새로 추가하거나 버전을 바꿀 때는 "Passed 뜨는 것"이 아니라 "의도한 위반을 넣었을 때 실제로 막히는 것"을 기준으로 검증한다.

---

## 2026-09-13 — 훅과 pre-commit의 mypy 검사 범위가 달라 같은 오류가 한쪽에서만 잡힘

**무슨 일**: 이슈 #7에서 `tests/unit/test_smoke.py`에 일부러 타입 오류(`def f(x: int) -> str: return x`)를 심었다. Claude Code의 `post_edit_check.sh` 훅은 즉시 잡아 편집을 막았는데, 그 상태로 실제 `git commit`을 해보니 pre-commit의 mypy 훅은 "Passed"로 통과해버렸다.

**원인**: `post_edit_check.sh`는 방금 수정한 파일 경로를 그대로 `mypy`에 넘겨 검사하지만(`tests/`도 포함), `.pre-commit-config.yaml`의 mypy 훅과 CLAUDE.md 명령어는 둘 다 `mypy src`만 실행해 `tests/`는 애초에 검사 대상이 아니었다. 검사 층(Claude Code 훅 / pre-commit / CI)마다 스코프가 달랐던 것.

**배운 것**: 같은 도구(mypy)를 여러 층에 걸어 두면, 층마다 검사 범위가 다를 수 있고 그러면 "어느 한 층만 통과"가 "전체가 안전"을 의미하지 않게 된다. `mypy src` → `mypy src tests`로 pre-commit·CLAUDE.md 명령을 맞추고, `tests/*`는 `disallow_untyped_defs`만 완화하는 override를 추가해 나머지 strict 규칙은 그대로 유지했다. 부수 효과로 `rfp_pm_agent`에 `py.typed`가 없어 mypy가 내던 `import-untyped` 경고도 `py.typed` 추가로 해소됨을 확인했다(`uv build`로 wheel에 실제 포함되는 것까지 확인). 검사 층을 여러 개 둘 때는 항상 "범위가 서로 같은지"를 먼저 맞춘다.
