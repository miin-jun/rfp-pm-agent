# ADR-0011: MCP 서버 — `MCPServer`(stdio), 읽기 전용 툴 4개, `deps`·`as_of` 숨김, `doc_ids` 필수

- 상태: 채택 (2026-10-17)
- 이슈: #24
- 결정한 사람: 소유자(miin-jun)
- **학습 모드**: 서버 코드(`src/rfp_pm_agent/mcp_server/server.py`)는 **소유자가 직접 구현**했다. Claude는 리뷰·테스트·문서만 했다
- 관련: [ADR-0005](0005-search-documents-contract.md)(`search_documents` 계약), [ADR-0006](0006-grounded-answer-refusal-and-citation.md) 결정 4(문서 지정 전제),
  [ADR-0009](0009-pms-query-tools.md)(PMS 툴 되묻기·상한), [ADR-0010](0010-langgraph-agent.md) 결정 3·4(에이전트의 같은 선택),
  architecture.md 4-8절, 연결 방법 [development.md](../development.md) "MCP 서버"

## 맥락

- 에이전트(#22)가 쓰는 툴을 Claude Desktop·Claude Code 같은 다른 AI 도구에서도 부를 수 있게 MCP 표준으로 노출한다(architecture.md 4-8절)
- MCP 툴은 연결한 LLM이 스스로 호출 여부를 정한다. 쓰기 툴을 열면 프롬프트 인젝션이 곧 데이터 조작이 된다(CLAUDE.md 보안) — 읽기 전용만 둔다
- 툴 함수는 `tools/` 한 곳에 있고 에이전트·MCP가 같은 함수를 import한다(tech-stack.md). 서버는 얇은 어댑터다

## 결정

### 1. MCP Python SDK 2.x `MCPServer`, stdio

`from mcp.server.mcpserver import MCPServer`, `mcp.run(transport="stdio")`. 의존성 `mcp[cli]>=2.3.0`(설치 버전 2.3.0).
1.x의 `FastMCP`는 2.x에서 `MCPServer`로 이름이 바뀌었다(아래 "시행착오" 1). 원격(Streamable HTTP)은 Phase 2(#24 범위 밖).

### 2. 툴 4개 — 모두 읽기 전용

| MCP 툴 | 원본 | 노출 인자 |
|---|---|---|
| `get_delayed_tasks` | `tools.pms_tools.get_delayed_tasks` | `project_id`(선택)·`scope`·`limit` |
| `get_open_risks` | `tools.pms_tools.get_open_risks` | `project_id`(선택)·`min_level`·`limit` |
| `get_deliverables` | `tools.pms_tools.get_deliverables` | `project_id`(선택)·`status_filter`·`requirement_id`·`wbs_id`·`limit` |
| `search_documents` | `tools.search_documents.search_documents` | `query`·`doc_ids`(**필수**)·`top_k` |

PMS 툴의 `project_id`는 선택으로 둔다 — 빼고 부르면 오류가 아니라 `status="project_required"`와 사업 후보(`available_projects`, doc_ids 포함)를
돌려받아 사용자에게 되묻는다(ADR-0009). DB 계정은 agent_ro(SELECT만)다.

### 3. `deps`·`as_of`는 LLM에게 숨긴다

- `deps`(DB 리더·검색 클라이언트)는 서버 모듈이 기동 때 `PmsDeps.from_env()`·`SearchDeps.from_env()`로 한 번 만들고 래퍼가 넘긴다. 연결은 첫 호출 때 연다
- `as_of`는 노출하지 않는다. 기준일은 `AS_OF_DATE` 고정 — 연결한 LLM이 "오늘" 날짜를 지어 넣을 길을 없앤다(ADR-0010 결정 4와 같다).
  각 PMS 툴 설명에 "기준일은 서버 설정(AS_OF_DATE)을 쓴다"를 적었다
- 래퍼 함수의 시그니처가 곧 MCP 입력 스키마라, 숨긴 인자는 스키마에 나오지 않는다(단위 테스트로 확인)

### 4. `search_documents`의 `doc_ids`는 필수 (원본은 선택)

원본은 `doc_ids=None`이면 전체 검색이다. MCP에서는 필수(`list[str]`, null 불가)로 바꿨다.

- 근거: #19 결정 4(ADR-0006 결정 4) — 근거 답변의 통과 기준은 문서 지정에서만 만족했다. 전체 검색에서는 사업명 없는 질문을 잘못 거절하거나
  다른 사업의 답을 출처까지 달아 냈다(잘못 거절 11/33 → 문서 지정 1/40). 에이전트도 같은 이유로 필수다(ADR-0010 결정 3)
- MCP에는 에이전트의 system 프롬프트 사업 목록이 없다. 대신 PMS 툴 결과의 `doc_ids`나 `project_required` 응답의 `available_projects[].doc_ids`를
  그대로 넣으라고 툴 설명에 적었다

### 5. 결과 직렬화: `model_dump(mode="json")`

원본은 Pydantic 모델(`DelayedTasksResult` 등, `list[DocumentHit]`)을 돌려준다. 래퍼는 `model_dump(mode="json")`으로 dict / list[dict]로 바꿔 돌려준다 —
날짜가 `"2026-09-24"` 문자열이 된다. SDK는 이것을 구조화 결과(`structured_content`)와 JSON 텍스트로 함께 보낸다. list 반환은 구조화 결과에서
`{"result": [...]}`로 감싸진다(단위 테스트로 확인).

## 검증

### 수동 (소유자, Claude Code에서 `claude mcp add rfp-pm`로 연결)

- PRJ-05로 PMS 툴 3개를 불러 `data/eval/pms_truth_v2.json`과 일치: 지연 업무 6건, high Risk `PRJ-05-R05`, 기한 지난 미제출 산출물 7건(ID 일치)
- 되묻기: 사업을 말하지 않은 질문에 `project_required` → 사용자에게 사업을 물었다
- 4개 툴 연쇄: PMS 툴 결과의 `doc_ids`로 `search_documents`를 불러 RFP 원문과 출처를 붙였다

### 자동

- 단위 `tests/unit/test_mcp_server.py`(13개, 가짜 deps): 4개 툴 등록, 스키마에 `deps`·`as_of` 없음, PMS 툴은 필수 인자 없음, `search_documents`는 `query`·`doc_ids` 필수,
  `doc_ids` 없이 부르면 검색 전에 인자 검증 오류, 각 툴이 원본 함수를 올바른 인자로 부르고 dict / list[dict]를 돌려줌, 원본의 `ValueError` 메시지가 클라이언트 쪽 오류에 없음(한계 1 고정)
- 통합 `tests/integration/test_mcp_server_db.py`(4개, `-m integration`, postgres): 위 수동 검증의 PRJ-05 세 항목을 `mcp.call_tool` 경로로 정답 파일과 대조

## 시행착오

1. **`FastMCP` 이름 변경**: 1.x 예제대로 `from mcp.server.fastmcp import FastMCP`를 쓰면 2.x에서 `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`.
   오류 메시지가 `MCPServer`(`mcp.server.mcpserver`)와 마이그레이션 문서를 알려 준다. 문서(tech-stack.md·architecture.md)의 "FastMCP" 표기도 함께 고쳤다
2. **원본 함수 이름 오타(`get_deliverable`)는 기동 때 잡히지 않는다**: 래퍼 본문의 `pms_tools.get_deliverable(...)`은 호출될 때 속성을 찾으므로
   서버는 정상 기동하고 툴 목록에도 나온다. 그 툴을 부를 때만 `AttributeError`가 나고, MCP 클라이언트에는 원래 메시지 없이
   "Error executing tool get_deliverables"만 간다. 단위 테스트가 4개 툴을 모두 실제로 부르는 이유다(mypy도 잡는다)
3. **필수/선택 인자 순서 `SyntaxError`**: `search_documents`에서 `doc_ids`를 필수로 바꾸면서 기본값 있는 `top_k` 뒤에 두면
   `SyntaxError: parameter without a default follows parameter with a default`. 원본은 `top_k`가 앞이지만 래퍼는 `query, doc_ids, top_k` 순서로 둔다
   (원본 호출은 키워드 인자라 순서와 상관없다)

## 한계

- 툴이 던진 `ValueError`·`PmsUnavailableError`·`SearchUnavailableError`는 SDK가 "예상하지 못한 예외"로 보고 메시지를 서버에만 남긴다.
  클라이언트 LLM에는 "Error executing tool <이름>"만 가서, 원래 메시지("질의가 너무 깁니다", "인덱스에 없는 doc_id: […]", 서버 연결 불가)를 보고
  인자를 고치거나 사용자에게 알릴 수 없다. 에이전트는 이를 툴 결과로 돌려준다(ADR-0010 결정 9). MCP 쪽 처리는 정하지 않았다(인자 스키마 검증 오류는 메시지가 그대로 간다)
- MCP에는 PMS가 없는 RFP 문서의 doc_id를 얻을 길이 없다(사업 목록 툴 없음). PMS 사업 9개의 문서만 `available_projects`로 얻는다
- `limit`·`top_k` 범위(1~50·1~20)는 스키마에 없고 설명과 원본 검증(`ValueError`)에만 있다
- 서버는 import 때 deps를 만든다. `AS_OF_DATE`가 없으면 import가 실패하므로 단위 테스트는 환경변수를 넣고 import한 뒤 모듈 변수 `deps`·`search_deps`를 바꾼다
- Claude Desktop 연결 설정은 development.md에 적었지만, 이 ADR 작성 시점에 Desktop 호출 기록(이슈 완료 기준의 캡처)은 여기 없다

## 버린 대안

- **`search_documents`의 `doc_ids` 선택(원본 그대로)** — 결정 4. 전체 검색 실패가 MCP에서도 그대로 난다
- **`as_of` 노출** — 결정 3. 연결한 LLM이 시스템 날짜를 넣을 수 있다
- **에이전트(`Agent.ask`)를 툴 하나로 노출** — 연결한 쪽 LLM 위에서 우리 LLM을 한 번 더 부른다(비용·지연, 출처가 두 번 가공된다). 원본 툴을 그대로 둔다

## 재검토

- 원격 배포(Streamable HTTP)는 Phase 2. 그때 인증과 도구별 권한을 정한다
- 툴 오류 메시지 전달(한계 1)은 클라이언트가 인자를 고칠 수 있게 하려면 정해야 한다. SDK는 툴이 `ToolError`를 던지면 메시지를 그대로 보내므로,
  래퍼에서 `ValueError`·`PmsUnavailableError`·`SearchUnavailableError`를 `ToolError`로 바꾸는 방법이 있다(후속 이슈 후보)
