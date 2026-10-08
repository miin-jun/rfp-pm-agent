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
data/eval/runs.jsonl     #13 청킹 비교 실행 기록 (git 포함, 실행마다 한 줄 추가)
data/eval/failures/**    #13 틀린 질문의 상위 검색 결과 (git 포함, run_id로 runs.jsonl과 연결)
data/eval/results/<실험>/ 검색 평가 기록 (git 포함): runs.jsonl · <run_id>.questions.jsonl · failures/<run_id>.jsonl
                         model_selection/ = #16 모델 선정, hybrid/ = #18 OpenSearch 검색
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
| `chunk_id` | keyword | `{doc_id}:{method}:{source_id}`. method는 block / requirement / block_requirement, source_id는 블록 ID(예: b0001) 또는 요구사항 ID. 블록을 더 나누지 않으므로 순번 없음. 파일 안에서 고유함을 확인(#65: 3,077 / 3,499 / 422 모두 중복 0) |
| `doc_id` | keyword | |
| `method` | keyword | 청킹 방식 (block / requirement / block_requirement) |
| `source_ids` | keyword[] | 원본 조각 ID (블록 ID 또는 요구사항 ID, 지금은 항상 1개) |
| `project_id` | keyword | PMS 프로젝트와 연결 (RFP 1건 = 프로젝트 1개) |
| `text` | text (nori) | 원문 (답변 근거로 보여줄 텍스트) |
| `text_for_embedding` | — (저장만) | 섹션 경로 + 원문 |
| `heading_path` | text (nori) | 섹션 경로 |
| `requirement_id` | keyword | `SFR-001` |
| `requirement_category` | keyword | `SFR` |
| `block_type` | keyword | paragraph / table (Silver `Block.type`). 요구사항 청크는 null |
| `page` | integer | PDF만. Silver와 같은 **0부터 세는** PDF 페이지(블록 `pdf_page`, 요구사항 `pdf_page_start`). 인쇄 쪽번호가 아니다 |
| `bid_title` | keyword | 문서명 — Silver `Document.bid_title`(manifest의 공고명). 출처 표기용 (#81) |
| `format` | keyword | `pdf` / `hwp` / `hwpx` — Silver `Document.format`. 출처 표기에서 쪽번호를 붙일지 정한다 (#81) |
| `printed_page` | integer | 인쇄 쪽번호 — 요구사항 청크는 `Requirement.printed_page_start`, 못 읽었으면 null. 블록 청크는 항상 null(Silver `Block`에 인쇄 쪽번호가 없다). `page`로 채우지 않는다 (#81) |
| `notice_no`, `agency` | keyword | |
| `security_level` | keyword | 권한 필터용 (Phase 2 적용) |
| `allowed_roles` | keyword[] | 권한 필터용 (Phase 2 적용) |
| `embedding` | knn_vector(1024) | KURE-v1 기준 |
| `embedding_model` | keyword | `{모델ID}@{revision}` (예: `nlpai-lab/KURE-v1@{revision}`) — **모델 혼용 방지**. revision은 TEI `/info`의 `model_sha`, null이면(TEI 1.9.4 기본) TEI 모델 볼륨의 `snapshots/` 해시 (#17) |
| `content_hash` | keyword | 증분 색인용. **sha256(TEI `/embed`에 실제로 보내는 문자열)**, 16진수 64자 전체. 그 문자열은 `EMBED_PASSAGE_PREFIX + text`라 접두어가 바뀌면 값이 바뀐다. **`embedding_model`과 함께 비교**해, 둘 다 같을 때만 재임베딩을 생략한다 (#17) |
| `metadata_hash` | keyword | 임베딩하지 않는 필드가 바뀐 것을 잡는다(파서 수정 #58·#59 등). 입력 필드는 `doc_id, method, source_ids, requirement_id, block_type, page, bid_title, format, printed_page`(코드 `METADATA_FIELDS`, 뒤의 셋은 #81). 이 필드만 골라 `json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))`로 직렬화한 UTF-8 바이트의 sha256, 16진수 64자. 없는 필드는 null로 넣는다. chunk_id(문서 _id), text(content_hash가 덮음), 색인 과정이 만드는 필드(embedding·embedding_model·content_hash·indexed_at)는 넣지 않는다 (#17) |
| `indexed_at` | date | 실제 색인 시각(UTC). 기준일 `AS_OF_DATE`와 관계없다 |

첫 인덱스(#17)에 실제로 들어간 필드는 5절 "첫 인덱스 필드"를 본다. 위 표에서 빠진 필드는 그 이유와 함께 5절에 적었다.

---

## 5. OpenSearch 인덱스

매핑 원본은 `src/rfp_pm_agent/ingest/index_mapping.json`이다(#17). 아래는 그 요약이고, 값이 다르면 JSON이 맞다
(`tests/unit/test_ingest_index_chunks.py::test_mapping_matches_decision`이 결정값을 고정한다).

```jsonc
// PUT rfp_chunks_v2_kure   (인덱스 이름은 OPENSEARCH_INDEX_NAME, 별칭은 OPENSEARCH_INDEX_ALIAS)
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
    "dynamic": "strict",          // 매핑에 없는 필드가 오면 색인 오류 (모르는 필드가 에러 없이 추가되지 않게)
    "properties": {
      "chunk_id":        { "type": "keyword" },
      "doc_id":          { "type": "keyword" },
      "method":          { "type": "keyword" },
      "source_ids":      { "type": "keyword" },
      "text":            { "type": "text", "analyzer": "korean" },
      "requirement_id":  { "type": "keyword" },
      "block_type":      { "type": "keyword" },
      "page":            { "type": "integer" },
      "bid_title":       { "type": "keyword" },   // #81 출처 필드
      "format":          { "type": "keyword" },
      "printed_page":    { "type": "integer" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil",
          "parameters": { "m": 16, "ef_construction": 100 }
        }
      },
      "embedding_model": { "type": "keyword" },
      "content_hash":    { "type": "keyword" },
      "metadata_hash":   { "type": "keyword" },
      "indexed_at":      { "type": "date" }
    }
  }
}
```

- **문서 `_id` = `chunk_id`** (#65 확인: 3,077 / 3,499 / 422 모두 중복 0). 색인 모듈은 청크 파일에 chunk_id 중복이 있으면 멈춘다
- **HNSW 파라미터**: `m=16`, `ef_construction=100`을 명시한다. OpenSearch 2.19 lucene 엔진의 기본값과 같지만, 기본값이 버전마다 바뀐 적이 있어(2.11 이하 `ef_construction` 512) 매핑에 고정한다. lucene은 `ef_search`를 쓰지 않고 요청의 k를 쓴다 (공식 문서 2.19 "Methods and engines")
- **첫 인덱스 필드 (#17, 2026-10-01 결정)**: chunk_id, doc_id, method, source_ids, text, requirement_id, block_type, page, embedding, embedding_model, content_hash, metadata_hash, indexed_at. 임베딩 입력은 `text`(#16과 같음). `metadata_hash`는 같은 날 추가 결정(파서가 바뀌어 메타데이터만 달라지는 경우)
  - 제외 `agency`, `project_id` — 출처가 없다 (manifest·Silver 어디에도 없음)
  - 제외 `heading_path`, `text_for_embedding` — 4절 경고대로 #11 이후 구조에 맞게 다시 정의해야 한다
  - 제외 `requirement_category`, `notice_no`, `security_level`, `allowed_roles` — 결정 목록에 없다. 필요해지면 매핑에 필드를 추가하고 다시 색인한다 (`dynamic: strict`라 매핑 추가가 먼저다)
- **출처 필드 추가 (#81, 2026-10-08)**: `bid_title`, `format`, `printed_page`. `search_documents`(#18 PR ③) 결과만으로 출처 표기(2절)를 할 수 있게 한다. 새 인덱스 `rfp_chunks_v2_kure`에 전체 색인한 뒤 별칭을 v1 → v2로 옮겼다. v1(`rfp_chunks_v1_kure`)은 지우지 않고 남겨 두었다 — 되돌릴 때는 별칭만 다시 옮긴다. 재색인 전후 검색 비교는 [hybrid-search-measurement.md](hybrid-search-measurement.md) 6절
- **첫 색인 대상 청크 파일**: `data/chunks/block_requirement.jsonl`, #62 이후 3,499개, sha256 `3864758c77d4c3fe2e83437388e28b146670fdbd598b04003627ca7459906b7f` (앞 12자 `3864758c77d4`, PR #66 확인 실행 기록과 같음). #16 모델 선정은 그 전 파일(`e07f4fbbe181`, 3,514개) 기준이다 — 두 파일 차이는 의미 글자 없는 블록 15개
- **증분 색인 판정** (`uv run python -m rfp_pm_agent.ingest.index_chunks`): 위에서부터 처음 맞는 것 하나로 정한다(content > model > metadata). 인덱스에 없음 → create / `content_hash` 다름 → update(content) / `embedding_model` 다름 → update(model) / `metadata_hash`만 다름 → update(metadata): 벡터·`content_hash`·`indexed_at`은 두고 메타데이터 필드와 `metadata_hash`만 부분 update(TEI 호출 없음) / 셋 다 같음 → skip(TEI 호출 없음) / 인덱스에만 있음 → delete. 삭제 대상이 인덱스 문서 수의 5%를 넘으면 멈추고 `--allow-mass-delete`로만 허용한다 — 이 검사는 인덱스 생성·별칭 연결보다 먼저 한다. create 대상은 bulk `create`로 보내 같은 `_id`가 있으면 409로 거절되게 하고, bulk 응답은 배치마다 확인해 오류가 있으면 다음 배치 전에 멈춘다. `OPENSEARCH_INDEX_NAME`이 별칭과 같거나 이미 별칭으로 쓰이는 이름이면 실행 전에 멈춘다(별칭으로도 쓰기가 되므로 다른 모델 인덱스를 덮어쓸 수 있음). 실행 방법은 docs/development.md

- **별칭(alias) 운영**: 코드는 항상 `rfp_chunks`(별칭)만 부르고, 실제 인덱스는 `rfp_chunks_v1_kure`, `rfp_chunks_v1_bgem3`처럼 **모델별로 분리** → 이후 모델 교체를 **검색 중단 없이 별칭 전환만으로** 처리
- **채택 모델 (#16, 2026-09-27)**: 임베딩 **KURE-v1**([ADR-0001](adr/0001-embedding-model.md)), 리랭커 bge-reranker-v2-m3 N=20([ADR-0002](adr/0002-reranker.md)). 별칭 `rfp_chunks` → `rfp_chunks_v1_kure`로 시작했고, #81에서 `rfp_chunks_v2_kure`로 옮겼다. 인덱스·별칭 생성과 색인은 **#17**에서 한다. 색인 모듈은 인덱스가 없을 때만 만들고, 별칭이 아직 없을 때만 연결한다 — 별칭이 다른 인덱스를 가리키고 있으면 그대로 두고, 전환은 사람이 따로 한다
  - #16 모델 선정은 OpenSearch가 아니라 **메모리 내 코사인 검색**(`search/dense.py`, 임시 구현. #13·#16 재현용으로 남긴다)으로 측정했다. #18부터 서비스 검색은 OpenSearch 별칭을 부른다(`search/opensearch.py`의 `bm25_search`·`knn_search`, 측정은 [hybrid-search-measurement.md](hybrid-search-measurement.md)). 모델별 벡터는 `data/cache/embeddings/`(git 제외)에 따로 두었다. 별칭 전환은 #81(매핑 변경, 같은 모델)에서 처음 썼다
- `engine: lucene` — 소규모 데이터에 충분하고, 필터를 k-NN 탐색 안에서 적용하는 방식을 지원 (RBAC 사전 필터에 사용)
- **요구사항 ID 정확 일치 질문**("SFR-012가 뭐야?")은 BM25 `match`(must)에 `requirement_id` keyword 정확 일치 가산(`constant_score` boost 100, should)을 붙여 처리한다 — 직접 조회 대신 쿼리 쪽 가산으로 구현, 재색인 없음 (#75, [ADR-0003](adr/0003-requirement-id-boost.md))
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
- `tags`: `paraphrase`(문서와 다른 표현) / `multi_chunk`(서로 다른 원문 단위(블록·요구사항) 2개 이상 필요)
  / `table`(2×2 이상 자료 표의 셀 정보, 요구사항 정의표 제외) / `no_answer`(문서에 답 없음)
  / `exact`(요구사항 ID·고유명사 그대로) / `doc_unspecified`(질문만으로 사업 특정 불가)
  / `false_premise`(문서에 없는 전제를 깔고 묻지만, 문서의 제약으로 답할 수 있는 질문 — 예: q038)
- 답 있음/없음 판단 기준: 원문이 질문이 묻는 대상(기능·조건)을 직접 규정하면 답 있음, 주변 정보만 있으면 답 없음
- `no_answer` ⇔ `evidence == []` ⇔ `answer`가 null. `note`에 부재 확인 기록을 남긴다.
  검색 지표에서 제외하고 #19 거절 평가에 쓴다
- qa_v1에서 옮긴 30문항은 사람 검수 기록이 없어 `verified_by`·`verified_at`이 null이다
- `paraphrase` 판정: 질문과 evidence에서 한글·영문·숫자만 남긴 글자 bigram 중 evidence에도
  있는 것의 비율(겹침률)이 **0.10 이하**. 불용어는 빼지 않는다. 추가 문항 초안을 쓰기 전에
  qa_v1 분포(0.00~0.36, 중앙값 0.114)를 보고 고정했다. 비율이 기준 이하여도 핵심 용어
  (요구사항 명칭, 법령·제도 이름 등)가 그대로 겹치면 paraphrase로 보지 않는다(사람 검수).
  qa_v1 30문항의 paraphrase 태그는 이 기준으로 코드가 붙였다 (`note`에 표시). `exact` 문항은
  이 기준을 적용하지 않는다
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
  `uv run python -m rfp_pm_agent.eval.run_chunk_eval` (#13), 검색 평가는
  `uv run python -m rfp_pm_agent.eval.run_retrieval` (#16·#18). 에이전트 평가(`run_agent`)는 #25에서 같은 자리에 추가한다
- **기준선 비교**: 지금은 별도 기준선 파일을 두지 않는다. `run_retrieval --compare <run_id> <run_id> ... --out-dir data/eval/results/<실험>`이
  그 실험의 `runs.jsonl` 하나에서 실행 기록을 찾아, Recall@10(전부 적중)이 가장 많은 실행을 1위로 두고 나머지를 문항별 McNemar로
  비교한다 — 같은 기록 위치의 실행끼리만 비교되고, `--out-dir`를 빼면 `model_selection`을 읽는다.
  평가 게이트용 기준선 파일과 형식은 #25에서 정한다

---

## 10. 설정값 (`.env.example`에 들어갈 데이터 관련 항목)

| 키 | 예시 | 설명 |
|---|---|---|
| `AS_OF_DATE` | `2026-09-21` | 합성 세계의 "오늘" |
| `OPENSEARCH_URL` | `http://localhost:9200` | |
| `OPENSEARCH_INDEX_ALIAS` | `rfp_chunks` | |
| `OPENSEARCH_INDEX_NAME` | `rfp_chunks_v2_kure` | 색인 모듈이 쓰는 실제 인덱스(모델별). 검색 코드는 별칭만 쓴다 (#17) |
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
