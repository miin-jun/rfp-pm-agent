# ADR-0005: `search_documents` 툴 계약 — doc_ids 필터 위치, 오류 계약, 없는 doc_id

- 상태: 채택 (2026-10-08)
- 이슈: #18 (PR ③)
- 결정한 사람: 소유자(miin-jun)
- 관련: [ADR-0004 서비스 검색 기본값](0004-hybrid-rerank-default.md), 형식 [data-design.md](../data-design.md) 5절
  "검색 툴 `search_documents`", 코드 `tools/search_documents.py`·`search/opensearch.py`(`hybrid_body`)

`search_documents(query, top_k=10, doc_ids=None)`는 Agent(#22)·MCP(#24)·`/search`(#23)가 함께 쓰는 툴이다.
인자를 정하는 쪽이 LLM이라, 필터가 결과를 어떻게 바꾸는지와 실패를 어떤 예외로 알리는지를 계약으로 고정한다.

## 결정 1: `doc_ids` 필터는 BM25·k-NN 하위 쿼리마다 넣는다

`doc_ids`가 있으면 `hybrid.queries`의 두 하위 쿼리 각각에 `terms: {doc_id: doc_ids}`를 넣는다 —
BM25는 `bool.filter`, k-NN은 `knn.embedding.filter`(lucene 엔진이 HNSW 탐색 안에서 거른다).
hybrid 최상위 `filter`와 요청 `post_filter`는 쓰지 않는다.

### 근거 (2026-10-08 실측)

조건: OpenSearch 2.19.1, 별칭 `rfp_chunks` → `rfp_chunks_v2_kure`(3,499건), qa_v2 57문항 전부(답 없는 6문항도 `doc_id`가 있다).
문항마다 정답 문서 1개로 필터하고 `size` 10, 후보 50, RRF k=60. 기준은 같은 필터를 건 BM25 50개·k-NN 50개를
앱 RRF(`rrf_fuse`)로 합친 상위 10이다.

| 필터 위치 | 결과 |
|---|---|
| hybrid 최상위 `query.hybrid.filter` | **57/57 문항 HTTP 400** `parsing_exception` "Field is not supported by [hybrid] query" |
| `post_filter` | 57/57 문항 200, 다른 문서 0건. 하지만 기준과 **순서가 같은 문항 20/57**, 상위 10 겹침 473/570(83%). 57문항 모두 10건을 돌려줬다. 필터가 하위 쿼리의 후보 선정 뒤에 적용되기 때문으로 보이나 내부 처리 순서는 확인하지 않았다 |
| BM25 하위 쿼리에만 | **29/57 문항**에서 상위 10에 다른 문서가 섞였다(570건 중 85건). k-NN 쪽이 전체 문서에서 후보를 가져오기 때문이다 |
| **BM25·k-NN 하위 쿼리마다** | 다른 문서 0건. 점수 목록이 기준과 **57/57 일치**, 순서는 52/57 일치. 나머지 5문항(q024·q041·q043·q048·q052)은 같은 10개이고 RRF 점수가 같은 결과끼리만 자리가 바뀌었다 — 동점 순서는 OpenSearch가 정한다(data-design.md 5절 "순서와 동점") |

- 측정 스크립트는 레포에 두지 않았다(일회성). 같은 조건의 통합 테스트:
  `tests/integration/test_opensearch_search.py::test_filtered_hybrid_returns_only_requested_docs_and_matches_app_rrf`
  (질문 1개, 점수 목록 일치를 확인한다)

### 버린 대안

- **hybrid 최상위 `filter`** — 2.19.1이 400으로 거절한다
- **`post_filter`** — 200이 와서 오류가 드러나지 않지만 순위가 기준과 다르다(37/57 문항). 같은 필터의 BM25·k-NN 결과를
  합친 것과 다른 결과라, 측정한 하이브리드 방식으로 그 문서 안을 검색했다고 말할 수 없다
- **한쪽 하위 쿼리에만** — 요청하지 않은 문서가 결과에 섞인다. LLM이 그 결과를 지정한 문서의 내용으로 읽는다

## 결정 2: 오류 계약

| 원인 | 예외 | 비고 |
|---|---|---|
| 입력 오류(빈 질의, 범위 밖 `top_k`, 타입 오류, 빈 `doc_ids`, 없는 doc_id, 질의가 토큰 한도 초과) | `ValueError` | 서버를 부르기 전에 거른다(없는 doc_id는 확인 요청 1회 뒤, 토큰 한도는 TEI 응답 뒤) |
| 서버 장애(OpenSearch·tei-embed·tei-rerank 연결 거절, 응답 시간 초과, 5xx) | `SearchUnavailableError(service)` | 원래 예외는 `__cause__`로 남긴다 |
| **리랭커만 꺼짐** | `SearchUnavailableError("tei-rerank")` | hybrid 검색이 성공했어도 실패로 끝낸다 |
| TEI 422 | 본문에 `must have less than`(토큰 한도)이 있을 때만 `ValueError("질의가 너무 깁니다: …")` | 413(요청 본문 한도 초과)도 `ValueError`. 그 밖의 422는 원래 예외 그대로 |
| 그 밖의 4xx(OpenSearch 별칭 없음 404·잘못된 질의 400, TEI 400 등) | 원래 예외 그대로 | 설정·코드 오류라 감싸지 않는다 |

LLM에게는 두 가지만 구별되면 된다: "인자를 고치면 된다"(`ValueError`)와 "인자를 바꿔도 안 된다, 다시 시도하거나
사용자에게 알린다"(`SearchUnavailableError`). 그 밖의 예외는 버그로 보고 그대로 올린다.

### 근거와 버린 대안

- **타입 오류를 `TypeError`로** — 버렸다. LLM이 만든 인자는 타입 힌트와 다를 수 있고, Agent·MCP가 입력 오류를 두
  예외로 나눠 처리할 이유가 없다. 문자열 `doc_ids="docA"`를 그대로 두면 글자 목록으로 읽혀
  "인덱스에 없는 doc_id: ['d','o','c','A']"가 되어 "문서가 없다"로 잘못 읽힌다(#18 PR③ reviewer 재현)
- **리랭커가 꺼지면 hybrid 결과만 돌려주기** — 버렸다. 리랭크 없는 결과는 서비스 기본값으로 측정한 품질이 아니다.
  ADR-0004 측정에서 하이브리드 단독은 R@10 30/51·MRR 0.348, 리랭크 N=20은 33/51·0.497이었다. 품질이 내려간 결과를
  같은 형식으로 돌려주면 호출한 쪽이 차이를 알 수 없다
- **서버 장애 때 빈 목록** — 버렸다. "찾지 못함"(0건, 정상)과 구별되지 않는다
- **422 전부를 `ValueError`로** — 버렸다. TEI는 배치 크기 초과(`batch size 64 > maximum allowed batch size 32`,
  클라이언트 `TEI_MAX_CLIENT_BATCH_SIZE`가 서버 값보다 클 때)도 422로 준다. 이것은 설정 오류인데 LLM에게
  "질의가 너무 깁니다"로 전달된다. 토큰 한도 초과 문구는 TEI 1.9.4 실측이다(2026-10-08:
  `Input validation error: \`inputs\` must have less than 8192 tokens. Given: 30006`)
- **OpenSearch 4xx도 `SearchUnavailableError`로** — 버렸다. 별칭 없음·질의 오류는 다시 시도해도 같다

## 결정 3: 인덱스에 없는 doc_id는 오류다

`doc_ids`에 인덱스에 없는 ID가 하나라도 있으면 `ValueError`(없는 ID 목록을 메시지에 적는다).
확인은 `doc_ids`가 있을 때만, 검색 전에 별칭에 `size: 0` + `terms` 집계 요청 1회로 한다(중복은 고유 ID 기준).

### 근거

LLM이 지어낸 ID나 잘못 옮긴 ID로 필터하면 결과는 0건이다. 0건은 정상 응답("찾지 못함")이라, LLM이 이를
"그 문서에 그런 내용이 없다"로 읽고 답할 위험이 있다. 오류로 돌려주면 LLM이 ID를 고치거나(이전 검색 결과의
`doc_id`를 다시 쓰거나) 전체 검색(`doc_ids=None`)으로 다시 부를 수 있다.

### 버린 대안

- **없는 ID는 빼고 나머지로 검색** — 일부 ID가 무시된 것이 응답에 드러나지 않는다
- **확인 없이 빈 결과 그대로** — 위의 위험 그대로다

### 비용

`doc_ids`가 있는 호출마다 OpenSearch 요청이 1회 늘어난다(지연 시간은 재지 않았다). `doc_ids=None`이면 추가 요청이 없다.

## 재검토

- OpenSearch를 올리면 결정 1의 네 조건을 다시 잰다(`hybrid.filter`가 지원되면 하위 쿼리마다 넣은 결과와 비교)
- TEI를 올리면 토큰 한도 초과 422의 본문 문구를 다시 확인한다(문구가 바뀌면 `ValueError`로 바뀌지 않고 원래
  예외가 올라간다 — 단위 테스트는 가짜 응답이라 이를 잡지 못한다)
