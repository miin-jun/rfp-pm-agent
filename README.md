# rfp-pm-agent

공공 SI 사업 제안요청서(RFP)와 RFP 기반 합성 PMS 데이터를 근거로, "이번 주 지연된 업무", "미해결 Risk", "관련 산출물" 같은 질문에 출처를 달아 답하는 에이전트. 설계 배경은 `docs/`를 참고.

## 실행 방법

(추후 이슈에서 채움 — Docker Compose, API, MCP 서버 실행 방법)

```bash
uv sync
```

## 문서

- `docs/architecture.md` — 전체 구조, 단계(Phase), 결정 로그
- `docs/tech-stack.md` — 기술 스택과 선정 근거
- `docs/data-design.md` — 스키마, 판단 규칙, 평가 세트 형식
- `docs/github-setup.md` — 마일스톤·라벨·이슈
- `docs/harness.md` — 하네스(규칙·권한·훅) 설계
