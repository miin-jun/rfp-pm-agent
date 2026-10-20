# ADR-0015: 시연 화면 — FastAPI + HTML 한 파일, 디자인 스킬은 Anthropic frontend-design

- 상태: 채택 (2026-10-20)
- 이슈: #40 (범위를 줄여 진행 — 아래 "범위")
- 결정한 사람: 소유자(miin-jun) — 방식(FastAPI + 단일 HTML), 디자인 스킬 선택, 범위 축소. 구현은 Claude(#40은 학습 모드 대상이 아니다)
- 관련: [ADR-0010](0010-langgraph-agent.md)(에이전트), [ADR-0006](0006-grounded-answer-refusal-and-citation.md)(거절 문구·번호 인용),
  [ADR-0013](0013-agent-refusal-not-by-prompt-wording.md)(답 없음 거절), 실행 방법 [development.md](../development.md) "시연 화면"

## 맥락

- 면접에서 "질문 → 근거 답변 → 출처 번호를 누르면 원문 청크 → 근거가 없으면 거절"을 화면으로 보여 줘야 한다. 지금까지는 CLI·MCP로만 확인할 수 있었다
- #40 원래 계획은 Streamlit 세 탭(질의응답·RTM·검색 비교)이었고, docs/tech-stack.md는 "UI 없음, 대안 Streamlit"이었다
- 시연에서 가장 오래 보는 것은 한국어 답변과 출처 원문이다. 읽기 편한 것이 1순위다

## 결정

### 1. FastAPI + HTML 한 파일 (`src/rfp_pm_agent/api/`)

- `POST /api/ask`: 기존 에이전트(`Agent.ask`, 프롬프트 기본값 agent_v1)를 그대로 부른다. 에이전트 코드는 바꾸지 않았다
- 응답(`schemas/api.py` `AskResponse`): 답, 거절 여부(`AgentAnswer.refused`), 출처 목록(번호·출처 문자열·사업명·형식·쪽·요구사항 ID·청크 원문), 실행한 툴 이름, 걸린 시간
- 청크 원문: `AgentAnswer`에는 원문과 번호 → chunk_id 표가 없다. 답의 유효 번호(나온 순서)와 `cited_chunk_ids`를 짝짓고(`cited_chunks` — 청크 번호는 대화 안에서
  청크마다 하나라 1:1), chunk_id로 OpenSearch를 한 번 조회한다. OpenSearch에 연결하지 못하면 답과 출처 문자열은 그대로 주고 원문만 비운다
- `GET /`: `api/static/index.html` 한 파일. 빌드 도구 없음, 외부 자원은 Pretendard 글꼴(CDN) 하나
- 질문마다 새 대화(thread_id)로 묻는다 — 화면은 한 번에 질문 하나다
- 실행: `uv run uvicorn rfp_pm_agent.api.app:create_app --factory`

### 2. 디자인 스킬 — Anthropic `frontend-design` (`.claude/skills/frontend-design`)

- 계획(색·글꼴·배치·기억에 남을 요소 1개)을 먼저 쓰고, "어느 화면에나 나올 기본값"인지 검토한 뒤 만든다는 절차가 작업 도구에도 맞았다
- 적용 결과
  - 색: 회백 바탕 `#F6F7F5`, 본문 `#1C2330`, 출처 번호·버튼 청회색 `#2F5B86`, 고른 출처 형광 `#FFE98A`, 거절만 인주색 `#A8261D`
  - 글꼴: Pretendard 하나, 본문 `word-break: keep-all`. 청크 원문은 표가 `|`로 풀려 있어 줄바꿈을 그대로 둔다(`pre-wrap`)
  - 배치: 위에 예시 질문, 왼쪽 질문·답변, 오른쪽 원문. 좁은 화면(860px 이하)은 위아래로 쌓는다
  - 기억에 남을 요소: 출처 번호를 인쇄 문서의 견출지(색인 탭)처럼 그렸다. 답 안의 번호와 원문 종이 가장자리의 탭이 같은 모양이고, 고른 번호는 형광으로 바뀐다
  - 움직임은 로딩 표시 하나뿐(`prefers-reduced-motion`이면 멈춘다)
- 검토 후 제외: supanova·taste-skill 계열 — 랜딩페이지(히어로·CTA·소개 섹션) 지향이라 작업 도구와 맞지 않는다. 3D·셰이더·움직이는 배경은 긴 한국어 원문 읽기를 방해하고
  시연 노트북의 GPU를 쓴다

### 3. 예시 질문 4개 — 평가에서 agent_v1이 맞게 답한 문항

| 버튼 | 문항 | 근거 |
|---|---|---|
| 문서: 스마트폴 기둥 수 | agent_dev_v1 d6 | 2026-10-16 판정 정답, 숫자가 분명해 원문 대조가 쉽다 |
| 문서: AX 사업 예산 | agent_dev_v1 d1 | 에이전트 정답, 기준선(#19 근거 답변)은 거절 — 에이전트가 검색어를 바꿔 찾은 사례 |
| PMS: 이번 주 지연 업무 | agent_dev_v1 p1 | 정답 ID 재현(2026-10-16·10-19) |
| 근거 없음: 문서에 없는 내용 | agent_refusal_v1 a1 | agent_v1이 거절한 사업 관련 질문(2026-10-18). 답 없음 거절은 3/8로 불안정하다(ADR-0013) — 아래 "결과" |

## 대안

| 대안 | 버린 이유 |
|---|---|
| Streamlit (#40 원래 계획) | 레이아웃이 위에서 아래로 쌓는 구조라 "답 옆에 원문"을 나란히 두기 어렵고, 답 안의 `[n]`을 눌러 다른 영역을 바꾸려면 컴포넌트를 따로 만들어야 한다. 버튼을 누를 때마다 스크립트 전체가 다시 실행되는 모델이라 에이전트 호출을 상태로 막아야 한다. 의존성도 크다 |
| React 등 빌드 도구가 있는 프런트엔드 | 화면 하나에 빌드 체계를 들이는 비용이 크다. 시연이 끝나도 유지할 코드가 늘어난다 |
| #23 `/chat` API를 먼저 만들고 화면이 그것을 부르게 | #23은 대화 이어 가기·인증 범위가 따로 있다. 시연 화면은 `/api/ask` 하나로 충분하고, #23에서 같은 응답 구조를 다시 쓸 수 있다 |

## 범위

- 이번 PR: 질의응답 화면만. #40의 RTM 탭·검색 비교 탭은 하지 않았다 — RTM은 `data/out/rtm_*.md`·`.csv` 산출물로 보여 준다(docs/github-setup.md 컷 라인과 같은 대체)
- 원래 완료 기준의 "실제 API 키 없이 화면 구조 확인"은 단위 테스트(가짜 에이전트)로 대신한다. 화면 자체는 에이전트가 있어야 답을 보여 준다

## 결과

- 단위 테스트 `tests/unit/test_api_app.py`(가짜 에이전트·가짜 OpenSearch): 응답 구조, 거절 플래그, 출처 원문 채우기·순서, 조회 실패 시 원문만 비움, 입력 검증, 에이전트 오류 502
- 통합 테스트 `tests/integration/test_api_app_opensearch.py`(`-m integration`): 실제 OpenSearch에서 chunk_id로 원문·문서 필드를 채운다. 에이전트는 가짜 — 실제 LLM을 부르는 테스트는 만들지 않는다(CLAUDE.md 비용 규칙)
- 실제 에이전트(2026-10-20, `gpt-6-luna`·agent_v1)로 예시 버튼을 눌렀다. 캡처 `docs/images/demo_*.png`

| 버튼 | 거절 | 걸린 시간 | 툴 | 출처 |
|---|---|---|---|---|
| 스마트폴 기둥 수 | 아니오 | 6.6초 | search_documents | [2] SCR-001 — 누르면 원문 열림. 답 "15개, 통합형 5·일반형 10" |
| AX 사업 예산 | 아니오 | 5.6초 | search_documents | [1] PDF 3쪽 — 누르면 원문 열림. 답 "499,620,000원" |
| 이번 주 지연 업무 | 아니오 | 3.3초 | get_delayed_tasks | 없음(PMS 답은 WBS ID로 적는다). 답 "WBS 2.5 한 건" |
| 문서에 없는 내용(a1) | 예, 3/3 | 5.1·5.2·5.3초 | search_documents ×2 | 없음 |

- a1은 답 없음 거절이 불안정한 유형이라(ADR-0013) 3번 눌러 3번 모두 거절되면 쓰기로 미리 정했고, 3/3이라 유지했다
- 비용: LLM 15회 $0.008
