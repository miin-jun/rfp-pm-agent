# ADR-0009: PMS 조회 툴 설계 — 되묻기 상태, 결과 상한, 규칙 함수 공유

- 상태: 채택 (2026-10-15, PR #92)
- 이슈: #21 (쓰는 곳: #22 LangGraph 에이전트, #24 MCP 서버)
- 결정한 사람: 소유자(miin-jun). 함수 형태·되묻기·상한·읽기 전용은 소유자가 준 기본값, 판단 위치·Risk 경계·단계 업무 포함은 Claude가 정하고 소유자가 승인, 구현·측정은 Claude
- 관련: [data-design.md](../data-design.md) 6절(판단 규칙, "PMS 조회 툴" 계약), [ADR-0006](0006-grounded-answer-refusal-and-citation.md) 결정 4(문서 지정 전제),
  [ADR-0008](0008-synthetic-pms-and-rtm-truth.md)(합성 PMS·정답), [ADR-0005](0005-search-documents-contract.md)(`search_documents` 계약),
  코드 `tools/pms_tools.py`·`pms/reader.py`·`pms/rules.py`·`schemas/pms_tools.py`

## 맥락

- 에이전트(#22)는 "이번 주 지연 업무", "미해결 Risk", "관련 산출물" 질문에 PMS DB를 써야 한다. architecture.md 4-6절은 Text-to-SQL 대신 파라미터화된
  정해진 툴을 쓰기로 했다
- 툴의 이름·설명·인자 스키마는 LLM이 읽는 API 문서다(architecture.md 4-7절). 결과는 LLM 문맥에 그대로 들어간다
- 합성 PMS(#20)는 9개 사업(프로젝트)이고, 정답 `data/eval/pms_truth_v2.json`은 `pms/rules.py`로 계산했다

## 결정

### 1. 툴 3개 — 순수 파이썬 함수 + Pydantic 결과

```python
get_delayed_tasks(project_id: str | None = None, scope: Literal["all", "this_week"] = "all",
                  as_of: date | str | None = None, limit: int = 20, *, deps: PmsDeps) -> DelayedTasksResult
get_open_risks(project_id: str | None = None, min_level: Literal["low", "medium", "high"] = "low",
               as_of: date | str | None = None, limit: int = 20, *, deps: PmsDeps) -> OpenRisksResult
get_deliverables(project_id: str | None = None,
                 status_filter: Literal["all", "not_submitted", "overdue", "submitted", "approved"] = "all",
                 wbs_id: str | None = None, requirement_id: str | None = None,
                 as_of: date | str | None = None, limit: int = 20, *, deps: PmsDeps) -> DeliverablesResult
```

- LangGraph·LangChain 툴 래핑은 #22에서 한다. docstring을 툴 설명으로 그대로 쓴다 — 인자 의미, 판단 규칙, 기준일, 상한, `search_documents`와 함께 쓰는 예 1개씩
- 선택지 인자는 `Literal`이라 스키마에 허용값이 보인다. `Literal`은 실행 시 막지 않으므로 함수 안에서도 검사해 ValueError를 낸다
- `as_of`는 `date` 또는 "YYYY-MM-DD"(LLM 툴 호출은 문자열로 올 수 있다). 생략하면 `AS_OF_DATE`. 시스템 날짜를 쓰지 않는다

### 2. 프로젝트 미지정은 예외가 아니라 되묻기 상태

`project_id`가 없거나 공백이면 `status="project_required"`, 모르는 ID면 `status="project_not_found"`이고, 둘 다 items 0건과
`available_projects`(project_id·사업명·doc_ids)를 돌려준다. message는 "사용자에게 어느 사업인지 물어본 뒤 다시 부른다"다.
잘못된 인자(`scope` 등 목록 밖, `limit` 범위 밖, 날짜가 아닌 `as_of`)는 project_id 확인보다 먼저 ValueError다 — LLM이 고칠 수 있는 인자 오류와
사용자에게 되물어야 하는 상황을 나눈다. 프로젝트에 없는 `wbs_id`·`requirement_id`는 프로젝트 확인 뒤 ValueError다.

### 3. 결과 머리: total·truncated·doc_ids, limit 1~50(기본 20)

세 툴 공통으로 `status`·`message`·`project_id`·`project_name`·`doc_ids`·`as_of_date`·`total`·`limit`·`truncated`·`available_projects`.

- `total`은 조건에 맞는 전체 건수, items는 앞에서부터 `limit`개, `truncated`는 잘림 여부. LLM 문맥에 수백 건을 넣지 않으면서 "더 있다"는 사실은 알린다
- `doc_ids`는 그 사업의 RFP doc_id다. `search_documents(doc_ids=...)`에 그대로 넣어 같은 사업 문서만 찾는다(ADR-0006 결정 4)

### 4. deps는 키워드 전용 — LLM에 보이지 않는다

`PmsDeps(reader, as_of_date)`는 DB 읽기와 기본 기준일 묶음이다. `search_documents`의 `SearchDeps`(ADR-0005)와 같이 키워드 전용 인자로 두어,
Agent·MCP 어댑터가 미리 묶어 넘기고 LLM 툴 스키마에는 나오지 않는다. 테스트는 가짜 리더를 넣는다.

### 5. agent_ro + 읽기 전용 세션 + 파라미터 바인딩

- 접속은 `POSTGRES_READONLY_DSN`(agent_ro, SELECT만)이고, 세션도 `default_transaction_read_only=on`으로 연다(두 겹)
- 조회 조건은 `project_id` 하나이고 SQLAlchemy `where(Model.project_id == project_id)`로 바인딩 파라미터로 넘긴다. 문자열 조립은 없다
- 연결할 수 없으면 `PmsUnavailableError`(`pms/errors.py`) — `SearchUnavailableError`와 같은 역할

### 6. 판단은 `pms/rules.py` — 정답 생성과 같은 함수 (소유자 승인)

SQL은 프로젝트 행을 가져오기만 하고, 지연·이번 주 지연·지연 일수·미해결 Risk·Risk 점수와 등급·기한 지난 미제출은 `pms/rules.py`가 판단한다.
합성 정답을 만든 `pms/synth.py`도 같은 함수를 쓴다. 프로젝트당 업무는 50개 안팎이라 Python에서 걸러도 성능 문제는 없다.

### 7. Risk 등급 경계 15/8 (소유자 승인)

점수 = probability(1~5) × impact(1~5). 15 이상 `high`, 8~14 `medium`, 7 이하 `low`. data-design.md 6절에 없던 정의라 이 이슈에서 추가했다.
1~5의 곱으로 7과 14는 나오지 않으므로 경계 테스트는 5×3(high)·2×4(medium)·2×3(low)로 충분하다.

### 8. 단계(level 1) 업무도 지연에 포함 (소유자 승인)

정의(`status != 'done'` AND `planned_end < 기준일`)를 단계 업무에도 그대로 적용한다 — 하위 업무가 늦으면 단계도 지연일 수 있고, 정답 파일도 그렇게 계산했다.
결과의 `level`(WBS 깊이)로 구분한다. 에이전트 답변에서 단계 업무를 보일지는 #22에서 정한다.

## ADR-0008 "조회 툴 SQL도 같은 정의" 표현이 바뀐 이유

ADR-0008 결정 2는 "조회 툴(#21)의 SQL도 같은 정의를 따른다"고 적었다 — 툴이 6절 정의를 SQL `WHERE`로 옮겨 구현한다는 뜻이었다.
#21 구현에서는 정의를 SQL로 옮기지 않고 같은 Python 함수를 쓰기로 했다(결정 6). SQL로 옮기면 같은 규칙이 두 벌(Python 정답, SQL 툴)이 되고,
둘이 같은지는 정답 대조 테스트로만 알 수 있다. 같은 함수를 쓰면 규칙 변경이 한 곳에서 끝난다. ADR-0008의 해당 문장에 "ADR-0009에서 변경"을 표시했다.

## 버린 대안

- **SQL에서 판단**(`WHERE status <> 'done' AND planned_end < :as_of` 등): DB가 걸러 전송량이 줄지만, 규칙이 SQL과 `rules.py` 두 곳에 생긴다.
  기준일 경계(`<` 대 `<=`), 이번 주 범위(월요일 ~ 기준일 전날) 같은 세부가 한쪽만 바뀌면 툴과 정답이 에러 없이 어긋난다. 지금 규모(프로젝트당 행 수십 개)에서는 이점이 없다.
  프로젝트가 커지면 SQL로 옮기고 정답 대조 테스트로 같음을 확인한다
- **project_id 미지정 시 예외**: #19(ADR-0006)에서 전체 문서 검색의 잘못 거절 11개 중 8개가 사업명 없는 질문("이 사업의 …")이었고, 같은 유형의 q010은
  다른 사업의 금액을 출처까지 달아 답했다. 대상이 정해지지 않은 채 진행하면 틀린 답이 그럴듯하게 나온다. 예외를 내면 에이전트가 오류 처리 경로로 빠지거나
  임의의 사업을 골라 다시 부를 수 있다. 되묻기 상태와 후보 목록을 돌려주면 에이전트가 사용자에게 묻는 경로가 기본이 된다

## 결과 (2026-10-15)

### 정답 대조 — 실제 DB(`si`, agent_ro), `data/eval/pms_truth_v2.json`

| 항목 | 건수 | 비교 | 9개 사업 |
|---|---|---|---|
| 지연 업무(전체) | 63 | wbs_id·지연 일수·상태 | 모두 일치 |
| 이번 주 지연 | 10 | wbs_id | 모두 일치 |
| 미해결 Risk | 45 | risk_id·점수·상태, **순서까지** | 모두 일치 |
| 기한 지난 미제출 산출물 | 70 | deliverable_id | 모두 일치 |

### 테스트

- 단위 92개 추가(툴 86 `tests/unit/test_tools_pms.py`, 규칙 6 `tests/unit/test_pms_rules.py`) — DB 없이 #20 샘플 데이터셋을 읽는 가짜 리더(`tests/fakes/fake_pms_reader.py`)
- 통합 41개 추가(`tests/integration/test_pms_tools_db.py`, `-m integration`) — 위 정답 대조(9 × 4), 세션·계정 쓰기 거부 각각, DB 연결 불가, 인젝션 형태의 project_id
- 전체: 단위 803 passed, 통합+TEI 71 passed. LLM 비용 없음

## 문제 → 원인 → 해결

1. **"누락 후보" 문구 오표시**(PR 전 reviewer 검토에서 발견)
   - 문제: `get_deliverables`가 `wbs_id`+`requirement_id` 교집합만 빈 경우와, 프로젝트에 없는 `requirement_id`(오타)에도 "요구사항에 연결된 업무가 없다(누락 후보일 수 있다)"를 붙였다.
     이 문구는 LLM이 RTM 누락으로 단정하게 만드는 판단이다
   - 원인: 조건이 "필터 결과가 비었다"(`allowed == set()`)였다. 요구사항 단독의 연결 여부와 요구사항 존재 여부를 따로 보지 않았다
   - 해결: 실패하는 테스트 2개를 먼저 추가하고, 요구사항 단독으로 연결 업무가 없을 때만 문구를 붙이도록 고쳤다. 없는 `requirement_id`는 ValueError(리더에 `requirement_ids` 추가)
2. **판단 중복**
   - 문제: 정답 계산(`pms/synth.py`)이 기한 지난 미제출(`status == "not_submitted" and due_date < as_of`)과 Risk 점수(`probability * impact`)를 `rules.py` 함수 없이 직접 계산했고,
     툴의 지난 일수도 직접 계산했다. 문서는 "같은 함수"라고 적고 있었다
   - 원인: #20 때는 `rules.py`에 그 함수가 없었고, #21에서 함수를 추가하면서 정답 쪽을 바꾸지 않았다
   - 해결: 정답 계산과 툴 모두 `is_overdue_deliverable`·`risk_score`·`delay_days`를 쓰게 바꿨다. 정답 파일을 다시 만들어 sha256이 그대로인 것을 확인했다
3. **읽기 전용 테스트가 계정 권한을 못 봄**
   - 문제: "쓰기가 거부된다" 테스트가 `match="read-only|permission denied"`라 두 겹 방어 중 어느 쪽이 막아도 통과했다
   - 원인: PostgreSQL은 읽기 전용 트랜잭션 검사를 권한 검사보다 먼저 한다. 세션이 읽기 전용이면 항상 `read-only` 오류가 나서 agent_ro 권한은 검사되지 않았다
   - 해결: 세션 테스트는 `read-only`만 기대하고, 계정 테스트를 따로 두었다 — `SET TRANSACTION READ WRITE` 뒤 UPDATE가 `permission denied`로 실패하는지 본다

## 재검토

- 프로젝트가 커져 행이 많아지면 판단을 SQL로 옮긴다(정답 대조 테스트로 같음 확인)
- #22에서 단계(level 1) 지연 업무를 답변에 보일지 정한다
  → 2026-10-16: "단계 지연"으로 따로 보인다([ADR-0010](0010-langgraph-agent.md) 결정 5)
- Risk 경계 15/8은 실제 PMS의 Risk 분포를 구하면 다시 본다
