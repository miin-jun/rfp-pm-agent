# SI 프로젝트 AI 어시스턴트 — 데이터 설계

작성일 2026-09-11 · 상태: **초안** (실제 RFP 샘플을 받은 뒤 요구사항 ID 패턴·표 구조를 확인해 보완)

> 레포의 `docs/data-design.md`로 넣어 Claude Code가 참고합니다. 스키마를 바꿀 때는 이 문서부터 고칩니다.

---

## 0. 설계 원칙

1. **원본은 절대 수정하지 않는다** — Bronze(원본) → Silver(파싱) → Gold(청크·인덱스). 파서를 바꾸면 Silver부터 다시 만든다
2. **ID는 내용에서 만든다** — `doc_id = sha256(파일 바이트)`의 앞 16자리. 같은 파일을 다시 받아도 같은 ID → 중복 없음(멱등). **주의**: 이것만으로는 "실행 횟수와 무관하게 보유 건수가 목표치를 넘지 않는다"는 보장되지 않는다 — 이 둘은 서로 다른 성질이며, 목표 건수가 있는 수집 함수(`collect_from_api`의 `limit` 등)는 둘 다 별도로 만족시켜야 한다. 자세한 내용과 실제 사고 사례는 1절 "중복 방지 vs 목표치 유지" 참고
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
                 data/raw/collect_runs.jsonl         실행 기록 (조회 구간별 호출 여부, total_count, 페이지 수,
                                                      응답·필터 통과·신규 저장 건수) — git 제외
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

data/eval/*.jsonl        평가 세트 (git 포함, 사람 검수 기록 포함)
data/eval/runs.jsonl     평가 실행 기록 (git 포함, 실행마다 한 줄 추가)
data/eval/failures/**    틀린 질문의 상위 검색 결과 (git 포함, run_id로 runs.jsonl과 연결)
```

- **조회 구간 (이슈 #50)**: 기준 날짜는 KST(`+09:00`)다 — 나라장터 조회 시각이 한국 시각 기준이라는 가정이며 참고문서로는 확인하지 못했다. 구간(`--from`/`--to` 또는 `--lookback-days`)은 30일 단위로 나누고 경계는 1일 겹친다(겹친 날의 중복은 `doc_id`가 거른다). 구간마다 `totalCount`까지 모든 페이지를 받는다.
- **실행 기록 (`collect_runs.jsonl`)**: manifest는 저장된 공고의 기록이고, 이 파일은 실행의 기록이다. `--limit` 도달로 조회하지 않은 구간도 `called=false, skip_reason=limit_reached`로 남겨 "공고가 없었다"와 "조회하지 않았다"를 구별한다. `response_count < total_count`면 페이지 누락이다.
- **파일명 충돌 방지**: `data/raw/api/`에 저장하는 파일명은 `{doc_id}_{원본 파일명}`으로 접두해, 서로 다른 공고가 같은 첨부파일 이름(예: "제안요청서.hwp")을 쓰더라도 덮어쓰지 않는다.

### 현재 데이터 현황 (2026-09-22 기준)

| 항목 | 건수 |
|---|---|
| 수집 | 40건 (hwp 22 / hwpx 9 / pdf 9) |
| 파싱 성공 | 28건 |
| hwp 파싱 실패 | 3건 |
| hwpx 미지원 | 9건 |
| `has_requirements=true` | 11건 |

출처: `data/raw/manifest.jsonl`, `data/parsed/<doc_id>.json` (이슈 #52). 이전 문서에 있던 "요구사항 문서 2건"은 파일로 남은 출처가 없던 값이다.

### 중복 방지 vs 목표치 유지 — 서로 다른 두 가지 멱등성 (2026-09-15 실측으로 드러남, learning-log.md 여섯 번째 항목)

수집 로직은 서로 다른 두 성질을 **둘 다** 만족해야 "여러 번 실행해도 안전하다"고 말할 수 있다. 하나만 만족하는 것으로는 부족하다.

1. **같은 문서를 중복 저장하지 않는다** — `source_type`(api·manual)과 무관하게 `doc_id`(sha256 앞 16자리)가 같으면 같은 문서로 취급한다. 수동 반입한 파일이 이후 나라장터 API로 다시 수집돼도, 또는 API 수집을 다시 실행해도 `manifest.jsonl`에 중복 등록되지 않는다. `register_manual_files`·`collect_from_api` 둘 다 만족한다.
2. **실행 횟수와 무관하게 보유 건수가 목표치를 넘지 않는다** — 1번을 만족해도 이 성질이 저절로 따라오지는 않는다. 예: `collect_from_api`의 `limit`을 "이번 실행에서 새로 받을 건수"로 해석하면, 이미 알고 있는 문서는 건너뛰면서도(1번 만족) 매 실행마다 **새로운(중복 아닌) 문서**를 `limit`건씩 더 받아버려 총 보유 건수가 계속 늘어난다 — 실제로 1회차 5건, 2회차 3건 추가로 8건이 됐다. 그래서 `limit`은 "manifest에 보유할 API 수집 문서의 목표 총량"으로 정의하고, 이미 그만큼 있으면 API를 아예 호출하지 않도록 고쳤다. `manual_dir`처럼 내용이 애초에 고정된 소스는 1번만으로 2번도 자동으로 만족된다. `limit`은 상한일 뿐 목표치를 강제하지 않는다 — 이미 보유한 건수보다 낮은 `limit`으로 다시 실행해도 기존 파일이나 manifest 항목을 지우지 않는다(그대로 유지).

---

## 2. 문서 공통 스키마 (Silver)

형식(PDF·HWP)이 달라도 파서의 출력은 모두 이 구조. **이슈 #11 구현(2026-09-16)으로
아래처럼 확정** — 원래 있던 `Document→Section→Block`(heading_path 기반) 계층은
실제 문서 구조와 안 맞아 폐기했다: 요구사항 1개는 "섹션"이 아니라 표(또는 페이지에
걸친 표 조각들) 1개였고, HWP·HWPX는 애초에 페이지 개념이 없어 heading_path보다
`requirement_id`·페이지 기준 출처 표기가 실제 구조에 맞았다
(docs/parsing-exploration.md "요구사항 정의표 구조" 절 실측 근거).

```
Document
├── doc_id               str    sha256 앞 16자리
├── source_file          str
├── bid_title             str    manifest의 notice_title
├── format                enum   pdf | hwp | hwpx
├── parse_status          enum   parsed | unsupported_format | failed
├── has_requirements      bool
├── requirement_count     int
├── declared_total        int | null   문서 자체 "합계" 숫자 (없거나 후보가 여럿이면 null)
├── summary_ids           list[str]    목록(요약)표에서 모은 코드 (검증용)
├── validation_warnings   list[str]
├── blocks: list[Block]                요구사항으로 소비되지 않은 나머지 콘텐츠
│     ├── block_id        str    "b0001"
│     ├── type            enum   paragraph | table
│     ├── text            str
│     ├── table           list[list[str|null]] | null
│     ├── source_order    int
│     └── pdf_page        int | null   0-based, PDF만
└── requirements: list[Requirement]
      ├── requirement_id     str    "SFR-001"
      ├── prefix             str    "SFR" — 코드 자체에서 분리, 접두어 하드코딩 없음
      ├── fields             dict   별칭 매핑된 표준 키만(category/id/name/definition/detail/output/related)
      ├── raw_fields         dict   원문 라벨 그대로, 별칭에 없는 라벨도 보존
      ├── text               str
      ├── source_order       int
      ├── pdf_page_start/end int | null   0-based 장 번호, PDF만
      └── printed_page_start/end int | null   페이지 하단 인쇄 쪽번호, 못 읽으면 null
```

### 출처 표기 규칙 (Grounded 답변에 사용)

| 파일 형식 | 출처 표기 |
|---|---|
| PDF | `[문서명 p.23(인쇄) · SFR-001]` — `printed_page_start`가 있으면 그 값, 없으면 `pdf_page_start`(0-based)에 안내 문구를 붙여 구분 |
| HWP·HWPX | `[문서명 · SFR-001]` (페이지 개념이 없어 요구사항 ID만으로 인용) |

※ 페이지를 억지로 계산(예: "장 번호 - 1" 같은 고정 오프셋)하지 않는다 — 인쇄 쪽번호는 페이지 하단 텍스트를 실제로 읽어서 채우고, 못 읽으면 null로 둔다(docs/parsing-exploration.md, 천안시 PDF 실측: 1쪽이 표지라 번호가 없고, 그 뒤로는 우연히 `printed_page == pdf_page(0-based)`였다 — 이 우연을 코드에 고정 규칙으로 넣지 않는다).

---

## 3. 요구사항 ID

공공 SW사업 제안요청서는 요구사항에 분류 코드를 붙이는 관행이 있습니다. 아래 표는
실제 샘플 2건(연구행정 데이터 기반 AI 플랫폼 구축·2024년 천안시 거점형
스마트도시 조성사업)에서 **관찰된 예시일 뿐, 고정된 접두어 목록이 아닙니다** —
천안시 문서는 인터페이스 요구사항에 `INR`이 아니라 `SIR`을 쓴다는 게 실측으로
드러났습니다(docs/parsing-exploration.md, 이슈 #11 구현 중 재확인).

| 코드 | 분류 | 코드 | 분류 |
|---|---|---|---|
| ECR | 시스템 장비 구성 | TER | 테스트 |
| SFR | 기능 | SER | 보안 |
| PER | 성능 | QUR | 품질 |
| SIR(문서에 따라 INR) | 인터페이스 | COR | 제약사항 |
| DAR | 데이터 | PMR / PSR | 프로젝트 관리 / 지원 |

- **코드 탐지는 라벨 문구가 아니라 셀 텍스트로 판정한다.** 실제 문서(연구행정
  AI 플랫폼)의 `SER-004`·`SER-005` 표는 라벨이 "요구사항 고유번호"가 아니라
  "요구사항 교유번호"(오타)다 — 라벨 문구를 매칭 기준으로 쓰면 이 두 건이
  에러 없이 누락된다. 정규식은 "셀 텍스트 전체(공백 제거)가 `^[A-Z]{3}-\d{3}$`에
  일치"로 고정하고, **접두어는 화이트리스트로 제한하지 않는다**(위 SIR 사례).
- **정의표/요약표 판정은 "셀 하나에 코드가 여러 개 뭉쳐 있는지"로 한다.** 천안시
  PDF의 목록표는 `"SFR-001\nSFR-002\n...\nSFR-007"`처럼 코드를 한 셀에 몰아
  쓰고, 그 목록표가 페이지 경계에서 쪼개지면 코드 1개짜리 조각이 생긴다. "코드
  셀 2개 이상=요약표"나 "코드 1개=정의표"만으로는 이 두 경우를 못 가른다 —
  실측(이슈 #11 구현 초기)으로 정확히 71개(정의표)=71개(요약표)=문서 자체
  "합계 71"이 일치하는 규칙을 확인했다.
- ⚠️ **f1-ragops 교훈**: 정규식 불일치로 조문 청킹이 0건 동작했던 사고 → 요구사항
  탐지 수와 문서 자체 "합계" 숫자를 항상 비교해 불일치를 `validation_warnings`에
  남긴다(이슈 #11에서 구현). "느슨하게 시작하고 테스트로 고정"이라는 원래
  방향은 유효했지만, 구분자(공백·언더스코어)를 느슨하게 하는 대신 접두어를
  느슨하게(비고정) 하는 쪽이 실제 필요였다.

---

## 4. 청크 스키마 (Gold = OpenSearch 문서)

> ⚠️ 이슈 #11에서 2절의 Document 스키마가 Section 없는 구조로 바뀌었다. 아래
> "섹션 경로"·`heading_path`는 그 전 스키마 기준이라 지금 구조와 안 맞는다 —
> #13(청킹, 학습 모드 이슈)에서 `heading_path`를 `requirement_id`나 페이지
> 기준으로 다시 정의해야 한다. 여기서는 손대지 않고 표시만 해 둔다.

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
| 6. 정답 기록 | 주입한 지연 업무·Risk 목록을 `data/eval/pms_truth_v1.json`에 저장 | ✅ |
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

### 검색 평가 세트 — `data/eval/qa_v1.jsonl`(#13, 동결) · `data/eval/qa_v2.jsonl`(#15)

- qa_v1은 #13 BM25 기준선의 정답이다. 수정하지 않는다. qa_v2는 v1 30문항을 같은
  `question_id`로 포함하므로, v1 부분만 떼어 기준선과 비교할 수 있다
- 스키마는 `schemas/eval.py`의 `EvalQuestionV2`다. qa_v1 줄도 그대로 읽힌다
  (문자열 `evidence`는 길이 1 목록으로 바뀐다)
- 원천 정답은 `evidence`(원문 문자열 목록)다. 판정: 청크 `doc_id`가 같고, 공백을 지운
  evidence가 청크 text에 들어 있으면 그 청크가 그 evidence를 담은 것이다. evidence 하나는
  블록 하나 또는 요구사항 하나 안에서만 가져온다
- `gold_chunk_ids`는 청크 파일에서 코드로 뽑은 파생값이다. 청킹·파서가 바뀌면 다시
  뽑는다. 키는 청킹 방식, 값은 evidence 순서에 맞춘 "그 evidence를 담은 청크 ID 목록"의
  목록이다 (evidence 하나를 청크 여럿이 담을 수 있음 — 예: q021)
- `tags`: `paraphrase`(문서와 다른 표현) / `multi_chunk`(서로 다른 블록 2개 이상 필요)
  / `table`(2×2 이상 자료 표의 셀 정보, 요구사항 정의표 제외) / `no_answer`(문서에 답 없음)
  / `exact`(요구사항 ID·고유명사 그대로) / `doc_unspecified`(질문만으로 사업 특정 불가)
- `no_answer` ⇔ `evidence == []` ⇔ `answer`가 null. `note`에 부재 확인 기록을 남긴다.
  검색 지표에서 제외하고 #19 거절 평가에 쓴다
- qa_v1에서 옮긴 30문항은 사람 검수 기록이 없어 `verified_by`·`verified_at`이 null이다
- 만들기·검증: `uv run python -m rfp_pm_agent.eval.qa_v2 check`
  (원문 일치·`gold_chunk_ids` 일치·방식별 최고 점수). `fill-gold`는 `gold_chunk_ids`를 다시 뽑는다

```json
{"question_id": "q031", "question": "...", "answer": "...", "evidence": ["...", "..."],
 "doc_id": "a1b2c3d4e5f6a7b8", "page": null, "type": "일반", "tags": ["multi_chunk"],
 "gold_chunk_ids": {"block_requirement": [["a1b2c3d4e5f6a7b8:block_requirement:b0012"],
                                          ["a1b2c3d4e5f6a7b8:block_requirement:b0140"]]},
 "verified_by": "minjurry", "verified_at": "2026-09-..", "note": ""}
```

### `data/eval/agent_v1.jsonl` — 에이전트 평가 (#25)

```json
{"qid": "a001", "question": "이번 주 지연된 업무랑 관련 요구사항 알려줘",
 "as_of_date": "2026-09-21", "role": "developer",
 "expected_tools": ["get_delayed_tasks", "search_documents"],
 "expected_facts": {"wbs_ids": ["3.2.1", "3.4.2"], "requirement_ids": ["DAR-002"]},
 "must_cite": true, "should_refuse": false}
```

- `expected_facts`는 **7절 6단계의 정답 기록에서 자동 생성** → 사람은 질문 문장만 검수
- `should_refuse: true` 문항 포함 (문서에 없는 내용 질문 → 거절해야 정답)

- **평가 실행 진입점**: `rfp_pm_agent.eval` 패키지 안의 모듈 — 청킹 비교는
  `uv run python -m rfp_pm_agent.eval.run_chunk_eval` (#13). 검색·에이전트 평가는
  `run_retrieval`, `run_agent`로 같은 자리에 추가한다
- **기준선**: `data/eval/baseline.json` (이슈 #17에서 생성)

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
