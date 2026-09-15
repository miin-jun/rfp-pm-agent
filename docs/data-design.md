# SI 프로젝트 AI 어시스턴트 — 데이터 설계

작성일 2026-09-11 · 상태: **초안** (실제 RFP 샘플을 받은 뒤 요구사항 ID 패턴·표 구조를 확인해 보완)

> 레포의 `docs/data-design.md`로 넣어 Claude Code가 참고합니다. 스키마를 바꿀 때는 이 문서부터 고칩니다.

---

## 0. 설계 원칙

1. **원본은 절대 수정하지 않는다** — Bronze(원본) → Silver(파싱) → Gold(청크·인덱스). 파서를 바꾸면 Silver부터 다시 만든다
2. **ID는 내용에서 만든다** — `doc_id = sha256(파일 바이트)`의 앞 16자리. 같은 파일을 다시 받아도 같은 ID → 중복 없음(멱등)
3. **"오늘"은 설정값이다** — 지연·미해결 판단은 시스템 날짜가 아니라 `AS_OF_DATE`(합성 세계의 기준일)로 한다. 그래야 평가 결과가 매번 같다
4. **문서(FMS)와 프로젝트 데이터(PMS)는 요구사항 ID로 연결한다** — RFP의 `SFR-001`이 WBS 업무와 청크를 잇는 다리
5. **권한 필드는 MVP부터 저장한다** — 적용(필터링)은 Phase 2지만, 나중에 전체 재색인하지 않도록 필드는 처음부터 둔다

---

## 1. 데이터 흐름과 디렉토리

```
나라장터 API ──► data/raw/api/{doc_id}_{파일명}       [Bronze] 원본, git 제외
data/raw/manual/{파일} ──► sha256 계산 ────┘         수동 반입(착수 초기 샘플 등) → 같은 manifest.jsonl에 등록
                 data/raw/manifest.jsonl             수집 기록 (doc_id, 파일명, source_type: api|manual,
                                                      공고번호 nullable, 공고명, URL nullable, 파일 크기,
                                                      sha256, 수집 시각) — git 포함(예외)
                        │ 파싱
                        ▼
                 data/parsed/{doc_id}.json           [Silver] 공통 문서 스키마, git 제외
                        │ 청킹
                        ▼
                 data/chunks/{doc_id}.jsonl          [Gold] 청크, git 제외
                        │ 임베딩(TEI) + 색인
                        ▼
                 OpenSearch  rfp_chunks (별칭)

RFP 요구사항 ──► PostgreSQL requirements  ──► 합성 WBS·Risk·Issue·산출물

eval/sets/*.jsonl        평가 세트 (git 포함, 사람 검수 기록 포함)
eval/results/**          평가 결과 요약 (git 포함)
```

- **중복 방지**: `source_type`(api·manual)과 무관하게 `doc_id`(sha256 앞 16자리)가 같으면 같은 문서로 취급한다. 수동 반입한 파일이 이후 나라장터 API로 다시 수집돼도 `manifest.jsonl`에 중복 등록되지 않는다.
- **파일명 충돌 방지**: `data/raw/api/`에 저장하는 파일명은 `{doc_id}_{원본 파일명}`으로 접두해, 서로 다른 공고가 같은 첨부파일 이름(예: "제안요청서.hwp")을 쓰더라도 덮어쓰지 않는다.

---

## 2. 문서 공통 스키마 (Silver)

형식(PDF·HWPX·HWP)이 달라도 파서의 출력은 모두 이 구조 → 청킹 코드는 하나

```
Document
├── doc_id            str    sha256 앞 16자리
├── source
│   ├── source_type   enum   api | manual        수집 경로 (나라장터 API 자동 수집 | 수동 반입)
│   ├── notice_no     str | null   입찰공고번호 (manual 반입 시 확인 전까지 null 가능)
│   ├── notice_title  str    공고명
│   ├── agency        str    수요기관
│   ├── url           str | null   원본 URL (manual 반입은 null 가능)
│   ├── file_name     str
│   ├── file_type     enum   pdf | hwpx | hwp | docx
│   └── downloaded_at datetime
├── title             str
├── page_count        int | null     HWPX·HWP는 페이지 개념이 없을 수 있음
├── security_level    enum   public | internal | confidential   (RFP 원문은 public)
├── allowed_roles     list[str]      (public이면 전체 역할)
├── parser            {name, version}
└── sections: list[Section]
      ├── section_id  str    "s003"
      ├── heading     str    "3. 기능 요구사항"
      ├── heading_path list[str]  ["Ⅲ. 제안요청 내용", "3. 기능 요구사항"]
      ├── level       int
      ├── page_start / page_end   int | null
      └── blocks: list[Block]
            ├── block_id        str   "s003-b012"
            ├── type            enum  paragraph | table | list
            ├── text            str   (표는 행을 "열이름: 값" 형태로 풀어 쓴 텍스트)
            ├── table           list[list[str]] | null   원래 표 구조 보존
            ├── page            int | null
            └── requirement_id  str | null   블록에서 탐지된 요구사항 ID
```

### 출처 표기 규칙 (Grounded 답변에 사용)

| 파일 형식 | 출처 표기 |
|---|---|
| PDF | `[문서명 p.23 · SFR-001]` |
| HWPX·HWP | `[문서명 · 3. 기능 요구사항 · SFR-001]` (페이지 대신 섹션 경로) |

※ HWPX는 화면 크기에 따라 줄바꿈이 달라지는 형식이라 **고정된 페이지 번호가 없습니다.** 페이지를 억지로 만들지 말고 섹션 경로로 표기합니다.

---

## 3. 요구사항 ID

공공 SW사업 제안요청서는 요구사항에 분류 코드를 붙이는 관행이 있습니다 (예시 — **실제 샘플로 반드시 확인**)

| 코드 | 분류 | 코드 | 분류 |
|---|---|---|---|
| ECR | 시스템 장비 구성 | TER | 테스트 |
| SFR | 기능 | SER | 보안 |
| PER | 성능 | QUR | 품질 |
| INR | 인터페이스 | COR | 제약사항 |
| DAR | 데이터 | PMR / PSR | 프로젝트 관리 / 지원 |

- 탐지 정규식은 **느슨하게 시작하고 테스트로 고정**: `SFR-001`, `SFR - 001`, `SFR_001`, 표 셀 안의 ID
- ⚠️ **f1-ragops 교훈**: 정규식 불일치로 조문 청킹이 0건 동작했던 사고 → 이번엔 **"요구사항 ID 탐지 수 / 요구사항 표의 행 수"를 파싱 품질 지표로 매번 출력**하고, 기준 미만이면 테스트 실패

---

## 4. 청크 스키마 (Gold = OpenSearch 문서)

### 청킹 규칙

| 대상 | 규칙 |
|---|---|
| 요구사항 정의 표 | **요구사항 1개 = 청크 1개** (고유번호·명칭·정의·세부내용·산출정보를 한 청크로) |
| 일반 문단 | 같은 섹션 안에서 약 500토큰 단위, 50토큰 겹침 |
| 긴 표 | 행 단위로 나누고 **각 조각에 표 머리글 반복** |
| 모든 청크 | 임베딩용 텍스트 앞에 **섹션 경로를 붙임** (예: `Ⅲ > 3. 기능 요구사항 > SFR-001 사용자 로그인`) → 짧은 청크도 문맥 유지 |

### 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `chunk_id` | keyword | `{doc_id}:{block_id}:{n}` |
| `doc_id` | keyword | |
| `project_id` | keyword | PMS 프로젝트와 연결 (RFP 1건 = 프로젝트 1개) |
| `text` | text (nori) | 원문 (답변 근거로 보여줄 텍스트) |
| `text_for_embedding` | — (저장만) | 섹션 경로 + 원문 |
| `heading_path` | text (nori) | 섹션 경로 |
| `requirement_id` | keyword | `SFR-001` |
| `requirement_category` | keyword | `SFR` |
| `block_type` | keyword | paragraph / table / list |
| `page` | integer | PDF만 |
| `notice_no`, `agency` | keyword | |
| `security_level` | keyword | 권한 필터용 (Phase 2 적용) |
| `allowed_roles` | keyword[] | 권한 필터용 (Phase 2 적용) |
| `embedding` | knn_vector(1024) | KURE-v1 기준 |
| `embedding_model` | keyword | `nlpai-lab/KURE-v1@{revision}` — **모델 혼용 방지** |
| `content_hash` | keyword | 증분 색인용 (내용이 같으면 재임베딩 생략) |
| `indexed_at` | date | |

---

## 5. OpenSearch 인덱스

```jsonc
// PUT rfp_chunks_v1_kure   (실제 파일은 JSON — 주석은 설명용)
{
  "settings": {
    "index": { "knn": true },
    "analysis": {
      "tokenizer": {
        "ko_tokenizer": { "type": "nori_tokenizer", "decompound_mode": "mixed" }
      },
      "analyzer": {
        "korean": {
          "type": "custom",
          "tokenizer": "ko_tokenizer",
          "filter": ["nori_part_of_speech", "lowercase"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "text":          { "type": "text", "analyzer": "korean" },
      "heading_path":  { "type": "text", "analyzer": "korean" },
      "requirement_id":{ "type": "keyword" },
      "allowed_roles": { "type": "keyword" },
      "security_level":{ "type": "keyword" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": { "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil" }
      }
      // 나머지 keyword·date 필드 생략
    }
  }
}
```

- **별칭(alias) 운영**: 코드는 항상 `rfp_chunks`(별칭)만 부르고, 실제 인덱스는 `rfp_chunks_v1_kure`, `rfp_chunks_v1_bgem3`처럼 **모델별로 분리** → 모델 선정 실험(#16)과 모델 교체를 **검색 중단 없이 별칭 전환만으로** 처리
- `engine: lucene` — 소규모 데이터에 충분하고, 필터를 k-NN 탐색 안에서 적용하는 방식을 지원 (RBAC 사전 필터에 사용). 세부 옵션은 구현 시 공식 문서로 확인
- **요구사항 ID 정확 일치 질문**("SFR-012가 뭐야?")은 `requirement_id` keyword 필드 직접 조회를 우선
- 하이브리드: OpenSearch 검색 파이프라인의 **RRF**(2.19+) 사용 + 비교용으로 **앱 코드 RRF**도 구현 (f1-ragops 경험 재사용)

---

## 6. PMS 스키마 (PostgreSQL)

### 관계도

```
projects ─┬─< project_members >── users
          ├─< requirements (RFP에서 추출한 실제 데이터)
          ├─< wbs_items ──< wbs_requirements >── requirements
          │      └─< deliverables
          ├─< risks  ── (related_wbs_id)
          ├─< issues ── (related_wbs_id, related_risk_id)
          └─< documents (FMS 문서 메타 · 권한)
```

### 테이블

| 테이블 | 주요 컬럼 | 비고 |
|---|---|---|
| `projects` | project_id, name, notice_no, agency(발주기관), contractor(수행사), start_date, end_date, status | RFP 1건 = 프로젝트 1개 |
| `users` | user_id, name, org, org_type(발주/수행/협력) | 이름·이메일은 가짜 |
| `project_members` | project_id, user_id, **role**, allocation_pct(투입률), start_date, end_date | role: `pm`, `developer`, `client`, `partner` |
| `requirements` | project_id, requirement_id, category, title, description, source_doc_id, source_chunk_id | **실제 RFP에서 추출** |
| `wbs_items` | wbs_id("2.3.1"), project_id, parent_wbs_id, name, phase, assignee_user_id, planned_start, planned_end, actual_start, actual_end, progress_pct, status | phase: 분석·설계·구현·시험·이행 / status: `not_started`, `in_progress`, `done`, `on_hold` |
| `wbs_requirements` | wbs_id, requirement_id | **PMS ↔ FMS 연결 고리** |
| `deliverables` | deliverable_id, project_id, name(예: 데이터이관계획서), phase, wbs_id, due_date, submitted_date, status, doc_id(null 가능) | 산출물 |
| `risks` | risk_id, project_id, title, description, probability(1~5), impact(1~5), status, owner_user_id, identified_date, due_date, mitigation, related_wbs_id | status: `open`, `mitigating`, `closed` |
| `issues` | issue_id, project_id, title, description, severity, status, assignee_user_id, raised_date, resolved_date, related_wbs_id, related_risk_id | status: `open`, `in_progress`, `resolved` |
| `documents` | doc_id, project_id, title, file_type, sha256, security_level, allowed_roles[], parsed_at | 청크 권한 필드의 원본 |

### 핵심 판단 규칙 — 코드·툴 설명·평가 세트가 모두 이 정의를 따름

| 개념 | 정의 |
|---|---|
| **기준일** | `AS_OF_DATE` 설정값 (예: 2026-09-21). 시스템 날짜 사용 금지 |
| **이번 주** | 기준일이 속한 주의 월요일~일요일 |
| **지연 업무** | `status != 'done'` **AND** `planned_end < 기준일` |
| **지연 일수** | `기준일 − planned_end` (일) |
| **이번 주 지연된 업무** | 지연 업무 중 `planned_end`가 **이번 주 월요일 ~ 기준일 전날** 사이인 것 (이번 주에 새로 지연된 것) |
| **전체 지연 업무** | 지연 업무 전부 (누적). 툴 인자 `scope: "this_week" \| "all"`로 구분하고 툴 설명에 두 정의를 모두 명시 |
| **미해결 Risk** | `status IN ('open', 'mitigating')`, 정렬 = `probability × impact` 내림차순 |
| **관련 산출물** | 업무(wbs)에 연결된 deliverables + 같은 요구사항에 연결된 업무의 deliverables |

→ 설계 원칙: **"지연"처럼 사람마다 다르게 해석하는 말을 코드 한 곳에 정의**하고, 툴 설명·평가 정답이 같은 정의를 쓰게 함

---

## 7. 합성 PMS 생성 규칙

**원칙: 정답을 알고 만든다.** 지연·Risk는 LLM에게 맡기지 않고 **코드가 의도적으로 심는다** → 평가 정답이 생성 과정에서 자동으로 나옴

| 단계 | 방법 | MVP |
|---|---|---|
| 1. 요구사항 추출 | RFP 청크에서 `requirements` 테이블 채움 (실제 데이터) | ✅ |
| 2. WBS 초안 | LLM에 요구사항 목록 + 표준 단계(분석·설계·구현·시험·이행)를 주고 **구조화 출력**으로 업무 30~60개 생성 | ✅ 실시간 API 소량 |
| 3. 일정 배치 | **코드가** 프로젝트 기간 안에 날짜 배치 (고정 seed) | ✅ |
| 4. 상태 주입 | **코드가** 기준일 기준으로 업무 약 15%를 지연, Risk 5~10개, Issue 5~10개 생성 | ✅ |
| 5. 검증 | 아래 규칙 전부 통과해야 DB 적재 | ✅ |
| 6. 정답 기록 | 주입한 지연 업무·Risk 목록을 `eval/sets/pms_truth_v1.json`에 저장 | ✅ |
| 대량 생성 | 여러 RFP × Batch API | Phase 2 |

### 검증 규칙

- `planned_start ≤ planned_end`, 모든 날짜가 프로젝트 기간 안
- `status = 'done'` ⇔ `progress_pct = 100` 그리고 `actual_end` 존재
- 담당자는 해당 프로젝트 멤버, 하위 업무 기간은 상위 업무 기간 안
- 모든 요구사항이 1개 이상 업무와 연결 (연결 안 된 요구사항 수를 출력)
- 이름·이메일 등은 전부 가짜 (실존 인물·기관 담당자명 사용 금지)

---

## 8. 권한 모델 (필드는 MVP, 적용은 Phase 2)

| 등급 | 대상 예시 | 볼 수 있는 역할 |
|---|---|---|
| `public` | RFP 원문 | pm, developer, client, partner |
| `internal` | WBS, 산출물 문서, 이슈 | pm, developer |
| `confidential` | 원가·계약 관련 | pm |

- RFP 원문은 모두 `public`이라 **Phase 2 RBAC 시연용으로 `internal`·`confidential` 합성 문서**(산출물 문서, 내부 회의록 등)를 추가 생성
- 적용 위치: OpenSearch **사전 필터**(`allowed_roles`) + PMS 툴의 SQL 조건

---

## 9. 평가 세트 스키마

### `eval/sets/retrieval_v1.jsonl` — 검색 평가 (#15)

```json
{"qid": "r001", "question": "사용자 로그인 시 필요한 인증 방식은?", "query_type": "semantic",
 "gold_chunk_ids": ["a1b2...:s003-b012:0"], "gold_requirement_ids": ["SFR-003"],
 "doc_id": "a1b2c3d4e5f6a7b8", "verified_by": "minjurry", "verified_at": "2026-09-19", "note": ""}
```

`query_type`: `exact`(ID·고유명사) / `semantic`(다른 표현) / `table`(표 안의 수치)

### `eval/sets/agent_v1.jsonl` — 에이전트 평가 (#25)

```json
{"qid": "a001", "question": "이번 주 지연된 업무랑 관련 요구사항 알려줘",
 "as_of_date": "2026-09-21", "role": "developer",
 "expected_tools": ["get_delayed_tasks", "search_documents"],
 "expected_facts": {"wbs_ids": ["3.2.1", "3.4.2"], "requirement_ids": ["DAR-002"]},
 "must_cite": true, "should_refuse": false}
```

- `expected_facts`는 **7절 6단계의 정답 기록에서 자동 생성** → 사람은 질문 문장만 검수
- `should_refuse: true` 문항 포함 (문서에 없는 내용 질문 → 거절해야 정답)

- **평가 실행 진입점**: `eval/run_retrieval.py`, `eval/run_agent.py`
- **기준선**: `eval/results/baseline.json` (이슈 #17에서 생성)

---

## 10. 설정값 (`.env.example`에 들어갈 데이터 관련 항목)

| 키 | 예시 | 설명 |
|---|---|---|
| `AS_OF_DATE` | `2026-09-21` | 합성 세계의 "오늘" |
| `OPENSEARCH_URL` | `http://localhost:9200` | |
| `OPENSEARCH_INDEX_ALIAS` | `rfp_chunks` | |
| `POSTGRES_DSN` | `postgresql+psycopg://app:app@localhost:5432/si` | |
| `POSTGRES_READONLY_DSN` | `postgresql+psycopg://agent_ro:...@localhost:5432/si` | 에이전트 툴 전용 읽기 계정 |
| `NARA_API_KEY` | (비밀) | 나라장터 입찰공고정보서비스(공공데이터포털) 인증키 |
| `NARA_BASE_URL` | `https://apis.data.go.kr/1230000/ad/BidPublicInfoService` | |
| `SYNTH_SEED` | `42` | 합성 데이터 재현용 |

---

## 11. 확인이 필요한 것 (RFP 샘플 받은 뒤)

- [ ] 요구사항 ID 실제 표기 형태와 분류 코드
- [ ] 요구사항 정의 표의 열 구성 (고유번호·명칭·정의·세부내용·산출정보 등)
- [ ] HWPX 섹션·표 XML 구조
- [ ] PDF 표 추출 품질 (pdfplumber vs PyMuPDF)
