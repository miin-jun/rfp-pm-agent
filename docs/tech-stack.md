# SI 프로젝트 AI 어시스턴트 — 기술 스택

작성일 2026-09-11 · 상태: **부분 확정** (2026-09-11 — 검색엔진·개발환경·패키지 관리 등 일부 확정, 나머지는 아래 표의 "확정/기본값" 열 참고)

> 레포의 `docs/tech-stack.md`로 넣어 Claude Code가 참고하게 할 수 있습니다.

---

## 0. 한눈에 보기

| 영역 | MVP 선택 | 확정/기본값(실험 대기) | Phase 2 추가 |
|---|---|---|---|
| 문서 검색 엔진 | **OpenSearch** (BM25-nori + k-NN + 하이브리드) ✅확정 | 확정 | — |
| 정형 DB | **PostgreSQL 16** + SQLAlchemy 2 + Alembic | 확정 | — |
| 임베딩 모델 | **KURE-v1** (RunPod) — #16 선정 실험 후 채택 ([ADR-0001](adr/0001-embedding-model.md)) | 확정 (2026-09-27) | — |
| 리랭커 | **bge-reranker-v2-m3** (RunPod), 후보 N=20 — #16 선정 실험 후 채택 ([ADR-0002](adr/0002-reranker.md)) | 확정 (2026-09-27) | — |
| 임베딩·리랭커 서빙 | **Hugging Face TEI** (RunPod, 로컬 대체 가능) | 확정 | — |
| 답변·에이전트 LLM | **OpenAI mini급** | 기본값(모델명 미확정, 착수 시 가격표로 확정) | **vLLM on RunPod** (Private LLM) |
| 평가 채점 LLM | OpenAI mini급 | 기본값(모델명 미확정) | — |
| 에이전트 | **LangGraph** + langchain-openai | 확정 | HITL(쓰기 동작) |
| API / MCP | **FastAPI** / **MCP Python SDK (FastMCP)** | 확정 | Streamable HTTP 원격 배포 |
| 문서 파싱 | PDF: **PyMuPDF** / HWP: **hwp5html(pyhwp) 변환** | 실측(docs/parsing-exploration.md)으로 확정(이슈 #11) · HWPX는 범위 밖(스텁) | docx·xlsx, HWPX 실제 파싱 |
| 평가 | 자체 평가 스크립트 + JSONL 골든셋 | 확정 | **Langfuse** 트레이싱 |
| 파이프라인 | Python 스크립트 (멱등·증분) | 확정 | **Airflow**, **PySpark**, **OpenAI Batch API** |
| 개발 환경 | **WSL2 Ubuntu** ✅ + **uv** ✅ + Docker Compose | 확정 | AWS EC2 |
| 하네스 | Claude Code(`CLAUDE.md`·hooks·skills·subagents), ruff, mypy, pytest, pre-commit, gitleaks, GitHub Actions | 확정 | 평가 회귀 게이트 강화 |
| UI | 없음 (API·MCP·CLI로 시연) | 확정 | Streamlit |

---

## 1. 문서 검색 엔진 — ✅ OpenSearch 확정

| | **OpenSearch** (추천) | Elasticsearch | PostgreSQL + pgvector |
|---|---|---|---|
| 라이선스 | Apache 2.0 | 무료 티어 있음, 기능 제한 | 오픈소스 |
| 한국어 BM25 | **nori** 플러그인 | **nori** 플러그인 | 한국어 형태소 분석 기본 미지원 → 별도 확장 필요 |
| 벡터 검색 | k-NN (HNSW) | dense_vector (HNSW) | pgvector (HNSW) |
| 하이브리드 + RRF | **엔진 내장** (2.19+ 기본 RRF) | **무료 티어에서 RRF 불가** (Enterprise 전용, 호출 시 403) → 앱 코드로 구현해야 함 | 앱 코드로 구현 |
| 권한 사전 필터 | k-NN 쿼리 안에 필터 조건 | 가능 | SQL WHERE로 쉬움 |
| 운영 부담 | JVM, 메모리 1~2GB | 동일 | 가장 가벼움 |

**추천: OpenSearch.** 이유 2가지
1. 한국어 BM25(nori) + 벡터 + RRF를 **무료로 한 엔진에서** 처리
2. PostgreSQL은 PMS용으로 어차피 쓰므로, 문서 검색은 검색 전문 엔진으로 역할 분리

**핵심 근거**: Elasticsearch 무료 티어는 RRF가 막혀 있어, 같은 계열이면서 Apache 2.0으로 RRF를 내장한 OpenSearch를 선택했다.
※ 주의: Docker로 띄울 때 `vm.max_map_count=262144` 설정 필요 (WSL2에서 흔한 첫 오류)

---

## 2. 임베딩 모델 — KURE-v1

| | **KURE-v1** (추천) | bge-m3 | multilingual-e5-large (기존 사용) |
|---|---|---|---|
| 기반 | bge-m3를 한국어 검색용으로 파인튜닝 | 다국어 | 다국어 |
| 크기 | 0.6B | 0.6B | 0.56B |
| 차원 | 1024 | 1024 | 1024 |
| 최대 입력 | **8192 토큰** | 8192 토큰 | **512 토큰** |
| 라이선스 | MIT | MIT | MIT |
| 한국어 검색 (공개 벤치마크 Recall@10) | 0.797 | 0.792 | 0.759 |

- 공개 벤치마크 차이는 작음 → 우리 RFP 평가 세트로 세 모델을 직접 비교했다(#16). **KURE-v1 채택** — [ADR-0001](adr/0001-embedding-model.md)
  - Recall@10(모든 evidence 적중) KURE-v1 30/51, e5 28/51, bge-m3 25/51. McNemar로는 셋 다 동률이었고, 동률 규칙을 기계적으로 적용하면 e5였지만 잘림 없음(8192토큰)·리랭크 지연을 근거로 소유자가 KURE-v1을 골랐다
- RFP는 긴 요구사항 표가 많아 **512 토큰 제한이 없는 모델**이 유리
- 한국어 검색 순위 1위는 KURE-v2지만 **Late-interaction(ColBERT 계열)** 이라 서빙·인덱싱 구조가 달라짐 → MVP에서는 제외

## 3. 리랭커 — bge-reranker-v2-m3

- 다국어 크로스인코더, XLM-RoBERTa 기반 → TEI로 서빙 가능
- **채택, 후보 N=20** (#16, [ADR-0002](adr/0002-reranker.md)): KURE-v1 기준 MRR 0.317 → 0.474, NDCG@10 0.380 → 0.506, 추가 지연 p95 483ms
- 한국어 파인튜닝 변형은 이번 비교에서 빠졌다 — ADR-0002 후속 후보

## 4. 임베딩·리랭커 서빙 — TEI (Text Embeddings Inference)

- Hugging Face의 임베딩·리랭커 전용 서빙 서버. `/embed`, `/rerank` HTTP API 제공
- 지원 아키텍처에 **XLM-RoBERTa(임베딩·리랭커 모두)** 포함 → KURE-v1, bge-reranker-v2-m3 둘 다 가능
- 구성: RunPod 파드 1개에 TEI 컨테이너 2개(임베딩, 리랭커). 두 모델 합쳐 VRAM 수 GB 수준
- **같은 TEI를 로컬(4050 또는 CPU)에서도 실행 가능** → 파드가 꺼지면 설정(base_url)만 바꿔 대체
- RunPod 주소는 외부 공개 → **인증 토큰 필수**, 주소·토큰은 `.env`에만

### 로컬 구동 (이슈 #14, RTX 4050 6GB, 실측 2026-09-24)

- 이미지: `ghcr.io/huggingface/text-embeddings-inference:89-1.9.4` — `89-`는 Ada Lovelace(RTX 40xx, compute capability 8.9)용 태그 ([README 지원 하드웨어 표](https://github.com/huggingface/text-embeddings-inference/blob/main/README.md))
- `docker-compose.yml`의 `tei-embed`(8080)·`tei-rerank`(8081). profile `tei`라서 `docker compose up -d`로는 뜨지 않고 `docker compose up -d tei-embed tei-rerank`로 띄운다. 모델 캐시는 `tei_models` 볼륨(두 서버 공유)
- 모델 교체: `.env`의 `EMBED_MODEL_ID`·`RERANK_MODEL_ID`를 서버(compose)와 클라이언트(`config.py`)가 함께 읽는다. 값을 바꾸고 `docker compose up -d tei-embed`로 컨테이너를 다시 만든 뒤 `/info`의 `model_id`로 확인. 코드 수정 없음
- 실측: 두 서버 동시 기동 시 VRAM 합계 2511MiB(리랭커 단독 1260MiB), fp16, 요청 1회 후 변화 없음. RAM 서버당 약 1.1GiB(`TOKENIZATION_WORKERS=2`)
- 서버 옵션
  - `AUTO_TRUNCATE=false`: 1.9.4 기본값은 true(긴 입력을 앞부분만 임베딩). 끄면 413 에러로 드러난다 — #16에서 multilingual-e5-large(최대 512 토큰) 비교 시 특히 중요
  - `TOKENIZATION_WORKERS=2`: 미지정(워커 19개, `nproc`=20)으로는 WSL RAM 부족으로 종료됐다 (docs/learning-log.md 2026-09-24)
  - 요청 1건의 입력 수 제한 `max_client_batch_size`=32(기본값). 넘으면 422 — 클라이언트(`TEIEmbeddingClient`·`TEIRerankerClient`)가 `TEI_MAX_CLIENT_BATCH_SIZE`씩 나눠 보낸다
- #16 후보 메모: bge-m3는 `pytorch_model.bin`만 있고 safetensors가 없어 TEI 로드 가능 여부를 착수 시 확인. multilingual-e5-large는 `query: `/`passage: ` 접두어 필요
- RunPod 구동과 로컬·RunPod 벡터 일치 확인은 #56

## 5. LLM

- MVP: **OpenAI mini급** — 정확한 모델명·가격은 착수 시 OpenAI 공식 가격표로 확정
- 개발 중: 가장 싼 모델 / 최종 평가: 한 단계 위 모델
- 호출부는 OpenAI SDK의 `base_url` 하나로 추상화 → Phase 2에 **vLLM(RunPod)** 으로 교체
- **구조화 출력(Structured Outputs)** + Pydantic 검증

## 6. 에이전트 — LangGraph

- 교육과정에서 다룬 도구 → 이번엔 서비스 수준으로 (State, 조건 분기, Checkpointer, 최대 반복 제한)
- 툴 함수는 `tools/` 한 곳에 두고 **에이전트와 MCP 서버가 같은 함수를 import**

## 7. 정형 DB — PostgreSQL 16

- PMS 테이블(프로젝트·WBS·업무·일정·인력·Risk·Issue·산출물), 문서 메타, 사용자·역할·권한, 평가 결과
- SQLAlchemy 2 + **Alembic**(스키마 변경 이력 관리)
- 에이전트용 **읽기 전용 DB 계정** 분리
- 스키마 설계는 다음 단계(데이터 설계)에서

## 8. 문서 파싱

| 형식 | 방법 | 비고 |
|---|---|---|
| PDF | **PyMuPDF**(AGPL) 최종 채택 | 실제 샘플(docs/parsing-exploration.md)로 pdfplumber와 비교한 결과 `find_tables()`가 요구사항 정의표를 표 구조로 정상 인식했다 — 아래 "pyhwp 재평가" 이전에 "표 추출은 pdfplumber가 유리"라고 적었던 건 실측 전 추정이었고 틀렸다. pdfplumber는 프로덕션 의존성에서 제거함(이슈 #11) |
| HWP (구형) | **hwp5html(pyhwp, AGPLv3+)로 변환한 HTML을 파싱** — 채택 | 아래 "pyhwp 재평가" 참고. `<table>`이 표준 HTML 표로 그대로 보존됨(실측 37/37). 별도 프로세스로 CLI만 호출하고 `import hwp5`는 하지 않음 |
| HWPX | (이슈 #11 범위 밖, 스텁만) | 수집된 HWPX 샘플에 요구사항 코드가 0건이라(docs/parsing-exploration.md) 구조 파싱은 미룸. `parse_status=unsupported_format`으로 표시만 하고 건너뜀 |
| docx / xlsx | python-docx / openpyxl | Phase 2 |

- 파싱 결과는 공통 스키마(`Document`, Section 없이 `blocks`+`requirements` 평면 구조 — docs/data-design.md 2절, 이슈 #11에서 개정)로 정규화 → 형식이 달라도 청킹 코드는 하나

### pyhwp 재평가 (이슈 #11, 2026-09-16 — 위 표를 "LibreOffice headless + H2Orestart"에서 바꾼 근거)

기존 표는 "pyhwp는 2020년 이후 업데이트 없음·**표 미지원**·Python 3.9+ 호환 문제 →
비추천"이라 적고 LibreOffice headless + H2Orestart(이슈 #12)를 채택했었다. 이슈
#11 사전 조사(docs/parsing-exploration.md)에서 실제로 `hwp5html --html`로 변환해
보니 **표가 `<table>`로 정상 보존됐다**(37개 요구사항 정의표 전부, 오탈자 라벨
포함해도 구조 자체는 깨지지 않음). "표 미지원"이라는 기각 근거를 다시 보니, 이건
`hwp5txt`(텍스트만 뽑는 별도 서브커맨드)를 가리키는 말이었을 가능성이 높다 —
**`hwp5txt`는 표를 버리지만 `hwp5html`은 표를 `<table>`로 보존한다**(실측). "관리
중단"(pyhwp가 2020년 이후 업데이트 없음) 리스크는 여전히 유효해 남겨 둔다.

- 라이선스: **AGPLv3+** (`pyhwp-0.1b15` 패키지 메타데이터로 확인). 이 코드는
  `import hwp5`를 하지 않고 별도 프로세스로 `hwp5html` CLI만 `subprocess`로
  호출한다 — AGPL의 파생저작물·네트워크 조항이 별개 프로세스로 실행되는 외부
  도구 호출에는 일반적으로 적용되지 않는다는 실무 해석에 따른 것이며, 법률
  자문은 아니다
- 이슈 #12(LibreOffice headless + H2Orestart)는 닫지 않고 **M2 백업 계획**으로
  남겨 둔다 — pyhwp가 실패하는 HWP 파일이 나올 경우의 대안

## 9. API · MCP

- **FastAPI** + Pydantic v2: `/chat`, `/search`, `/health`
- **MCP Python SDK의 FastMCP**: stdio(로컬, Claude Desktop·Claude Code 연결) → Phase 2에 Streamable HTTP
- MCP는 **읽기 전용 툴만** 노출

## 10. 평가·관측

- 골든셋: JSONL (질문, 정답 문서/청크, 기대 툴 순서, 허용 역할)
- 지표: Recall@k·MRR·NDCG(검색) / Faithfulness·Citation 정확도(답변, LLM-as-Judge) / 툴 선택 정확도·단계 수·지연·비용(에이전트)
- 결과는 `data/eval/results/`에 버전별 저장 → 기준선 대비 하락 시 CI 실패
- Phase 2: **Langfuse**로 요청별 트레이스

## 11. 개발 환경 — ✅ WSL2 + uv 확정

| 항목 | 추천 | 대안 | 이유 |
|---|---|---|---|
| OS 환경 | **WSL2 Ubuntu** 안에서 레포·Docker·Claude Code 실행 | Windows 네이티브 | LibreOffice headless, OpenSearch, TEI, 훅 스크립트(bash)가 Linux 기준. 배포 환경(EC2·RunPod)과 동일 |
| 패키지 관리 | **uv** (`pyproject.toml` + `uv.lock`) | conda | 잠금 파일로 재현성 보장, 설치 빠름, Claude Code가 `uv run pytest`처럼 일관된 명령 사용 |
| 컨테이너 | Docker Desktop (WSL2 백엔드) + Docker Compose | — | postgres·opensearch·api·mcp 한 번에 |

## 12. 하네스 도구

| 종류 | 도구 |
|---|---|
| Guides | `CLAUDE.md`, `docs/architecture.md`, `docs/tech-stack.md`, 이슈 템플릿 |
| Computational sensors | ruff(린트·포맷), mypy(타입), pytest, pre-commit, **gitleaks**(시크릿 유출 차단) |
| Claude Code | `.claude/settings.json`(권한·훅), hooks(수정 후 ruff·mypy, 종료 시 테스트), skills(평가 실행·비교), subagents(리뷰 전담) |
| CI | GitHub Actions: 린트·타입·테스트 → 평가 회귀 게이트 |
| 비용 통제 | 테스트에서 실제 API 호출 금지(모킹), 에이전트 최대 반복, 호출별 토큰·비용 로그, OpenAI 사용 한도 |

---

## 12-1. 모델·서빙 선정 근거와 검증 계획

> 현재 선택은 **외부 근거로 좁힌 기본값(가설)** 이다. 최종 근거는 **우리 RFP 데이터로 직접 측정한 결과**로 만든다.

### 근거의 두 종류

| 종류 | 역할 | 한계 |
|---|---|---|
| 외부 근거 (공개 벤치마크·공식 문서) | 후보를 좁힘 | 벤치마크 데이터가 우리 도메인(공공 SI RFP)과 다름, 모델 카드 수치는 개발자 자체 보고 |
| **내부 근거 (우리 데이터로 측정)** | **최종 선택** | 평가 세트를 직접 만들어야 함 |

### 현재 외부 근거

**임베딩 — KURE-v1**
- 모델 카드(개발자 자체 보고) 한국어 검색 Recall@10: KURE-v1 0.797 / bge-m3 0.792 / multilingual-e5-large 0.759
- MTEB-ko-retrieval 정리 자료 nDCG@10: KURE-v2 0.816 / KURE-v1 0.762 / bge-m3-ko 0.755 / bge-m3 0.751 / KoE5 0.734
- 최대 입력 8192 토큰(e5는 512), MIT 라이선스
- **한계**: 1위와의 차이가 1%p 안팎으로 작음 → 우리 데이터에서 순위가 뒤집힐 수 있음

**리랭커 — bge-reranker-v2-m3**
- 다국어 크로스인코더, XLM-RoBERTa 기반
- TEI 지원 아키텍처(XLM-RoBERTa sequence classification)에 해당 → **서빙 제약을 통과하는 후보**
- **한계**: 한국어 RFP 도메인 성능은 공개 근거가 없음 → 직접 측정 필수

**서빙 — TEI**
- 공식 지원 목록에 XLM-RoBERTa 임베딩·리랭커 모두 포함 → 두 모델을 **한 도구로** 서빙
- `/embed`, `/rerank` HTTP API, GPU·CPU 이미지 제공 → RunPod와 로컬 대체가 같은 API
- **성격**: 성능보다 **운영 기준**(지원 범위·API 일관성·대체 가능성)으로 고른 것 → 기준을 문서로 남김

### 내부 검증 실험 (Phase 1, 파싱·청킹 직후, 약 1일)

**왜 이 시점인가**: 임베딩 모델을 바꾸면 전체 재색인이 필요 → **본 색인 전에** 결정해야 함

1. **평가 세트** (`data/eval/qa_v2.jsonl`, 57문항, #15): 실제 RFP 10건 대상. #13 BM25 기준선의 qa_v1 30문항은 그대로 두고, Claude Code가 청크 원문을 읽고 쓴 초안 27문항을 더함 → **추가 문항 전부를 사람이 검수** (f1-ragops 정답 라벨 전수 검증과 같은 방식). LLM API는 쓰지 않음
   - 추가 문항 유형: 말 바꾸기 6 / 요구사항 ID·고유명사 그대로(exact) 2 / 여러 청크 6 / 표 안의 정보 6 / 답 없음 6 / 없는 전제(false_premise) 1 — 정의는 data-design.md 9절
2. **후보**
   - 임베딩: KURE-v1 / bge-m3 / multilingual-e5-large(기존 사용 모델, 기준선)
   - 리랭커: 없음(기준선) / bge-reranker-v2-m3 (한국어 파인튜닝 변형은 이번 범위 밖 — ADR-0002에 후속 후보로 기록)
3. **지표**: Recall@5·@10, MRR, NDCG@10 + **지연(ms/쿼리)**, **색인 시간**, **VRAM**
   - 하이브리드(BM25+벡터 RRF) 조건의 측정은 #18로 넘긴다 (소유자 구현 대상, 이슈 #16 결정). #16은 BM25 단독 한 행만 기준선으로 함께 싣는다
4. **결정 규칙 — 측정 전에 미리 고정** (이슈 #16 본문에 2026-09-26 기록)
   - 주 지표: Recall@10, 답 있는 51문항, "모든 evidence 묶음을 찾아야 적중"(0/1)
   - 1위와 각 후보를 **McNemar 정확검정(양측)** 으로 비교해 p ≥ 0.05면 동률. 처음 계획한 "1위와 1%p 이내" 규칙은 51문항에서 1문항이 약 2%p라 해상도보다 작아서 바꿨다
   - 동률이면 가벼운 모델: VRAM → 색인 시간 → 지연 순으로 비교
   - 리랭커: 없음 vs bge-reranker-v2-m3, 후보 N=20. NDCG@10에는 McNemar를 쓰지 않고 개선폭과 p95 지연 증가를 함께 보고한다
   - e5는 자기 최대 길이(512토큰)에서 자르고 자른 청크 수를 기록한다. 색인 대상은 block_requirement 청크
   - 참고용(결정에 쓰지 않음): 적중 문항 수 차이, 비율 Recall
   - 실행: `uv run python -m rfp_pm_agent.eval.run_retrieval` (측정 절차: docs/model-selection-measurement.md)
5. **산출물**: `docs/adr/0001-embedding-model.md`, `docs/adr/0002-reranker.md`, `data/eval/results/model_selection/`

**서빙 근거 보강(선택)**: 청크 1,000개 색인을 TEI vs sentence-transformers 직접 호출로 비교해 처리량 측정

### 근거 요약

공개 벤치마크로 KURE-v1·bge-m3·e5를 후보로 좁혔고, 공공 RFP 청크로 만든 평가 세트 57문항(답 있음 51)에서 Recall@10을 측정해 결정한다. 1위와의 차이가 McNemar 정확검정으로 유의하지 않으면(p ≥ 0.05) 가벼운 모델을 고른다는 규칙을 측정 전에 정해 둔다.

**결과 (2026-09-27)**: 세 모델 모두 1위(KURE-v1)와 동률이었다. 동률 규칙(VRAM → 색인 시간 → 지연)은 세 모델의 로드 VRAM이 같아(1251 MiB) 구별하지 못했고, 기계적으로 적용하면 e5였다. 소유자가 결과를 본 뒤 KURE-v1 + bge-reranker-v2-m3(N=20)를 채택했다 — 근거와 규칙의 빈틈은 [ADR-0001](adr/0001-embedding-model.md)·[ADR-0002](adr/0002-reranker.md).

## 13. 확정 내역

- [x] 검색 엔진: **OpenSearch** (2026-09-11)
- [x] 개발 환경: **WSL2 Ubuntu** (2026-09-11)
- [x] 패키지 관리: **uv** (2026-09-11)
- [x] 임베딩 모델: **KURE-v1** (2026-09-27, #16, [ADR-0001](adr/0001-embedding-model.md))
- [x] 리랭커: **bge-reranker-v2-m3**, 후보 N=20 (2026-09-27, #16, [ADR-0002](adr/0002-reranker.md))

## 14. 다음 단계

1. 데이터 설계: PMS 스키마, 문서 공통 스키마, 청크 메타데이터(권한 포함)
2. 하네스 설계: `CLAUDE.md` 초안, 훅, CI, 평가 게이트
3. 레포 구조 + GitHub 마일스톤·라벨·이슈 초안
4. 9/27 MVP 기준 일정표

## 참고

- [Reciprocal Rank Fusion on free Elasticsearch: licensing, workarounds, and the OpenSearch alternative | u11d](https://u11d.com/blog/reciprocal-rank-fusion-on-free-elasticsearch-licensing-workarounds-and-the-open-search-alternative/)
- [Introducing reciprocal rank fusion for hybrid search | OpenSearch](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Amazon OpenSearch Service 한국어 하이브리드 검색 | AWS 기술 블로그](https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-hybrid-query-korean)
- [nlpai-lab/KURE-v1 | Hugging Face](https://huggingface.co/nlpai-lab/KURE-v1)
- [임베딩 모델 선택 가이드 | 데이터다이나믹스](https://www.data-dynamics.io/ko/blog/embedding-model-guide)
- [TEI supported models | GitHub](https://github.com/huggingface/text-embeddings-inference/blob/main/docs/source/en/supported_models.md)
- [pyhwp 사용법과 한계 | 한컴 블로그](https://blog.hancom.com/python-hwp-hwpx-text-extraction-pyhwp-guide/)
- [H2Orestart | GitHub](https://github.com/ebandal/H2Orestart)
