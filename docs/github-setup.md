# SI 프로젝트 AI 어시스턴트 — GitHub 세팅 & Phase 0~1 이슈 초안

작성일 2026-09-11 · 상태: **초안** (데이터 설계·하네스 설계 후 Phase 1 이슈 세부 보완)

> 레포 생성 직후 Claude Code에게 **"docs/github-setup.md대로 gh로 마일스톤·라벨·템플릿·이슈를 만들어 줘"** 라고 지시하면 한 번에 세팅됩니다.
> 사전 준비: WSL에서 `sudo apt install -y gh` → `gh auth login`

---

## 1. 마일스톤

| 마일스톤 | 기한 | 목표 |
|---|---|---|
| **M0 · 하네스** | 2026-09-15 | 레포·환경·하네스(guides·sensors)·CI가 동작 |
| **M1 · MVP** | 2026-09-27 | RFP 파싱 → 하이브리드 검색 → 출처 답변 + PMS 툴 + 에이전트 + MCP + 평가 게이트가 **처음부터 끝까지 동작** |
| **M2 · 확장** | 2026-10-20 | RBAC, Airflow·PySpark, Batch API, Private LLM(vLLM) 비교, Langfuse, 배포 |

## 2. 라벨 (10개)

| 라벨 | 색 | 의미 |
|---|---|---|
| `type:feature` | `#1f6feb` | 기능 구현 |
| `type:experiment` | `#8250df` | 비교 실험 (가설·결정 규칙 필수) |
| `type:docs` | `#0969da` | 문서·ADR·README |
| `type:chore` | `#6e7781` | 환경·설정·CI |
| `area:ingest` | `#bf8700` | 수집·파싱·청킹 |
| `area:search` | `#1a7f37` | 색인·검색·리랭킹·답변 |
| `area:agent` | `#cf222e` | PMS 툴·에이전트·API·MCP |
| `area:eval` | `#9a6700` | 평가 세트·지표·회귀 게이트 |
| `area:harness` | `#57606a` | CLAUDE.md·훅·스킬·CI·비용 통제 |
| `priority:mvp` | `#d1242f` | 9/27 MVP에 반드시 포함 |

## 3. 이슈·PR 템플릿

### `.github/ISSUE_TEMPLATE/task.md`

```markdown
---
name: 작업
about: 기능·설정·문서 작업
title: "[area] "
labels: ""
---

## 목적 (왜)
<!-- 이 작업이 없으면 무엇이 안 되는가 -->

## 할 일
- [ ]

## 완료 기준 (DoD)
<!-- 확인 방법을 명령어로. 예: `uv run pytest tests/ingest -q` 통과 -->
- [ ]

## 범위 밖
<!-- 이번에 하지 않는 것 — Claude Code가 넘어가지 않도록 -->

## 참고
<!-- docs/architecture.md 섹션, 관련 이슈 -->
```

### `.github/ISSUE_TEMPLATE/experiment.md`

```markdown
---
name: 실험
about: 모델·방법 비교 실험
title: "[exp] "
labels: "type:experiment, area:eval"
---

## 가설

## 후보
-

## 평가 세트·지표
<!-- 세트 경로, 문항 수, 지표 (Recall@10, MRR, NDCG@10, 지연, VRAM 등) -->

## 결정 규칙 (측정 전에 고정)
<!-- 예: Recall@10 최고. 1위와 1%p 이내면 더 가벼운 모델 -->

## 산출물
- [ ] `eval/results/...`
- [ ] `docs/adr/NNNN-*.md`
```

### `.github/pull_request_template.md`

```markdown
Closes #

## 변경 요약

## 검증 방법
<!-- 실행한 명령과 결과 -->

## 체크리스트
- [ ] `uv run ruff check . && uv run mypy src && uv run pytest` 통과
- [ ] 평가 지표가 기준선 이하로 떨어지지 않음 (해당 시)
- [ ] 새 환경변수는 `.env.example`에 추가
- [ ] 실제 API 호출하는 테스트 없음 (모킹)
```

---

## 4. Phase 0 이슈 (M0 · 9/12~9/13)

※ 번호는 실제 GitHub 이슈 번호(2026-09-12 gh로 생성·정리한 결과)입니다. `#1`은 생성 과정에서 실수로 만들어졌다가 삭제되어 결번입니다.

| # | 제목 | 라벨 | 완료 기준 |
|---|---|---|---|
| 2 | [harness] 레포 초기화 — uv 프로젝트, 폴더 구조, `.gitignore`, `.env.example`, README 뼈대 | chore, harness, mvp | `uv sync` 성공, `.env`·`data/raw/`가 gitignore됨 |
| 3 | [harness] GitHub 세팅 — 마일스톤·라벨·이슈/PR 템플릿 (이 문서) | chore, harness, mvp | 마일스톤 3개·라벨 10개·템플릿 3개 존재 |
| 4 | [harness] Docker Compose — PostgreSQL + OpenSearch(nori) + 헬스체크 | chore, harness, mvp | `docker compose up -d` 후 두 서비스 healthy, nori 분석기 동작 확인 |
| 5 | [harness] Guides — `CLAUDE.md`, `docs/architecture.md`, `docs/tech-stack.md` | docs, harness, mvp | CLAUDE.md에 명령어·규칙·금지사항·완료 기준 명시 |
| 6 | [harness] Sensors — ruff, mypy, pytest, pre-commit, gitleaks | chore, harness, mvp | pre-commit이 커밋 시 자동 실행, 가짜 API 키 커밋 시도가 차단됨 |
| 7 | [harness] Claude Code 설정 — `.claude/settings.json` 권한(sudo·`.env` 읽기 금지), 훅(수정 후 ruff·mypy, 종료 시 pytest) | chore, harness, mvp | 훅이 실제로 동작하는 로그 확인 |
| 8 | [harness] CI — GitHub Actions (lint·type·test) | chore, harness, mvp | PR에서 체크 통과 표시 |
| 9 | [harness] 모델 클라이언트 추상화 — LLM·임베딩·리랭커를 `base_url`로 전환, 호출별 토큰·비용 로그, 테스트용 가짜 클라이언트 | feature, harness, mvp | 설정만 바꿔 로컬/원격 전환, 단위 테스트는 네트워크 없이 통과 |

## 5. Phase 1 이슈 (M1 · 9/16~9/27)

날짜 대신 **선행 이슈**(먼저 끝나야 하는 이슈)로 순서를 표시합니다. 날짜는 M1 마일스톤 기한(9/27)만 기준으로 삼습니다.

| # | 제목 | 라벨 | 완료 기준 | 선행 이슈 |
|---|---|---|---|---|
| 10 | [ingest] 나라장터 수집 + 수동 반입 5건 등록 — API 클라이언트 구현, 착수 시 수동 반입한 RFP 샘플 5건(HWPX)을 `_manifest.jsonl`에 등록, 이후 API로 추가 수집 (멱등) | feature, ingest, mvp | 수동 반입분이 `source_type=manual`로 `_manifest.jsonl`에 등록됨, API 재실행해도 sha256 기준 중복 없음, `data/raw/`에 원본 저장 | #2 |
| 11 | [ingest] 파서 — PDF·HWPX → 공통 스키마(Document/Section/Block) | feature, ingest, mvp | **확보한 형식 전부 파싱**, 표가 Block으로 보존 | #10 |
| 12 | [exp] HWP 변환 — LibreOffice + H2Orestart (**반나절 타임박스**) | experiment, ingest | PDF·HWP 샘플이 없으면 샘플 확보 후 진행. 성공 시 파서에 연결 / 실패 시 결과 기록 후 M2로 이동 | #10, #11 |
| 13 | [ingest] 구조 기반 청킹 — 요구사항 ID 단위 + 메타데이터 | feature, ingest, mvp | 요구사항 ID 누락률 측정·기록 | #11 |
| 14 | [search] TEI 구동 — 로컬(4050) + RunPod 실행 스크립트, 임베딩·리랭커 클라이언트 연결 | chore, search, mvp | 같은 문장의 벡터가 로컬·RunPod에서 동일 | #9 |
| 15 | [eval] 모델 선정용 평가 세트 — 질문 50~80개, 정답 청크 전수 검수 | feature, eval, mvp | `eval/sets/retrieval_v1.jsonl`, 검수 기록 | #13 |
| 16 | [exp] 임베딩·리랭커 선정 — KURE-v1 / bge-m3 / e5, 리랭커 3조건 | experiment, eval, mvp | ADR-0001·0002 작성, 결정 규칙대로 선택 후 `.env.example`(`EMBED_MODEL_ID`·`RERANK_MODEL_ID`)·OpenSearch 인덱스 별칭(`docs/data-design.md` 5절)·ADR을 함께 갱신 | #14, #15 |
| 17 | [search] OpenSearch 색인 — nori + k-NN 매핑, 증분 색인 | feature, search, mvp | 재실행 시 변경분만 색인 | #4, #13, #16 |
| 18 | [search] 하이브리드 검색 — BM25 + 벡터 + RRF + 리랭킹 (`search_documents`) | feature, search, mvp | 평가 세트에서 Recall@10 기록, 단일 방식 대비 비교 | #17 |
| 19 | [search] Grounded 답변 — 출처(문서·페이지·요구사항 ID) + 근거 없으면 거절 | feature, search, mvp | 모든 답변에 출처, 무관 질문에 거절 | #18 |
| 20 | [agent] 합성 PMS — 스키마(Alembic) + RFP 기반 시드 데이터 소량 생성 | feature, agent, mvp | 날짜 역전 등 논리 검증 통과 | #4, #11 |
| 21 | [agent] PMS 조회 툴 — 지연 업무·미해결 Risk·산출물 조회 | feature, agent, mvp | 읽기 전용 DB 계정으로 동작, 단위 테스트 | #20 |
| 22 | [agent] LangGraph 에이전트 — 툴 선택·멀티스텝·최대 반복·체크포인터 | feature, agent, mvp | 예시 질문 3종(공고 원문)에 정답 | #19, #21 |
| 23 | [agent] FastAPI — `/chat`, `/search`, `/health` | feature, agent, mvp | API 문서 페이지에서 호출 성공 | #22 |
| 24 | [agent] MCP 서버 — stdio, 읽기 전용 툴, Claude Desktop 연동 | feature, agent, mvp | Claude Desktop에서 툴 호출 화면 캡처 | #21 |
| 25 | [eval] 평가 게이트 — 에이전트 골든셋(툴 순서) + 답변 근거성 채점 + CI 연결 | feature, eval, mvp | 기준선보다 떨어지는 PR이 CI에서 실패 | #8, #20, #22 |
| 26 | [docs] MVP 마감 — 데모 녹화, README, 이력서 문장 초안 | docs, mvp | 데모 영상·README·문장 초안 | #22, #23, #24, #25 |

### 밀릴 때 자르는 순서 (컷 라인)

1. **#12 HWP** → M2로 이동 (PDF·HWPX만으로 MVP)
2. **#23 FastAPI** 축소 → `/chat`만
3. **#25 평가 게이트** 축소 → 검색 지표만 CI 연결, 에이전트 채점은 수동 실행
4. **#15 평가 세트** 축소 → 50문항
- **자르지 않는 것**: #18 하이브리드 검색, #19 출처 답변, #22 에이전트, #24 MCP — 세 공고의 공통 핵심

### 일정 메모

- 9/16~9/27은 **주말 포함 12일**. 이슈 1개 = 반나절~하루
- 매일 끝에 해당 이슈의 PR 머지 → 진척도가 마일스톤에 표시됨

## 6. M2 이슈 (제목만, M1 끝나고 상세화)

- [agent] RBAC — 역할·권한 테이블, 검색 사전 필터, 권한 유출 테스트
- [ingest] Airflow DAG — 수집→파싱→청킹→색인 스케줄
- [ingest] PySpark — 나라장터 공고 목록 대량 정제·SW사업 분류
- [agent] Batch API — RFP 기반 합성 PMS 대량 생성
- [exp] Private LLM — RunPod vLLM vs OpenAI, 같은 평가 세트로 품질·지연·비용 비교
- [eval] Langfuse 트레이싱
- [ingest] docx·xlsx 파싱 (+ #12가 넘어온 경우 HWP)
- [harness] AWS EC2 배포
- [agent] Streamlit 채팅 UI

## 7. 생성 방법 (Claude Code가 실행)

```bash
# 마일스톤 (gh에는 milestone 전용 명령이 없어 API 사용)
gh api repos/{owner}/{repo}/milestones -f title="M0 · 하네스" -f due_on="2026-09-15T23:59:59Z"
gh api repos/{owner}/{repo}/milestones -f title="M1 · MVP" -f due_on="2026-09-27T23:59:59Z"
gh api repos/{owner}/{repo}/milestones -f title="M2 · 확장" -f due_on="2026-10-20T23:59:59Z"

# 라벨
gh label create "type:feature" --color 1f6feb --description "기능 구현"
# ... (표의 10개)

# 이슈 (본문은 task.md 템플릿 형식으로 채움)
gh issue create --title "[harness] 레포 초기화 ..." --label "type:chore,area:harness,priority:mvp" --milestone "M0 · 하네스" --body-file /tmp/issue1.md
```

**멱등성 원칙**: 이 스크립트를 다시 돌려도 같은 결과가 나와야 한다. `gh issue create` 전에 반드시 `gh issue list --state all --json title`로 같은 제목이 이미 있는지 확인하고, 있으면 생성을 건너뛴다. (2026-09-12: 파일명 오타로 이슈 16개가 잘못된 번호·내용으로 생성된 사고가 있었음 — 원인은 스크립트 자체의 멱등성 부재가 아니라 매니페스트·파일명 불일치였지만, 재발 방지 차원에서 생성 단계에 제목 중복 확인을 넣는다.)
