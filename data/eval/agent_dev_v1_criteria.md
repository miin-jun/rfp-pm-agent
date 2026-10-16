# agent_dev_v1 — 통과 기준과 측정 전 예상 (이슈 #22)

- 고정: 2026-10-16, **측정 전**. 이 파일과 `agent_dev_v1.jsonl`은 첫 실행보다 먼저 커밋한다. 측정 뒤에 기준·문항·예상을 고치지 않는다
- 기준: #22 1단계 보고의 제안 표를 소유자가 그대로 확정했다
- 세트: `data/eval/agent_dev_v1.jsonl` 30문항(문서만 8 · PMS만 8 · 둘 다 6 · 사업 미지정 4 · 답 없음 4). 만들기·검사:
  `uv run python -m rfp_pm_agent.eval.agent_dev_set build|check`. 정답 ID는 `pms_truth_v2.json`·`qa_v2.jsonl`·`irrelevant_v1.jsonl`에서 만든다
- 실행: `uv run python -m rfp_pm_agent.eval.run_agent_dev --out-dir data/eval/results/agent --baseline`
- 조건: `gpt-6-luna`, Responses API, 추론 강도 low, 프롬프트 `agent_v1`, `AGENT_MAX_STEPS=6`, 기준일 `AS_OF_DATE`(2026-09-24)

## 통과 기준

| 지표 | 기준 | 판정 방법 |
|---|---|---|
| 툴 선택 정확도 | ≥ 90% (27/30) | 마지막 턴의 `ok` 툴 이름 집합 = `expected_tools`, PMS 툴 `project_id`·검색 `doc_ids`가 대상 사업. 되묻기 문항은 첫 턴에 `ok` 호출 0 |
| PMS 답 | 정답 ID 재현율 100%인 문항 ≥ 7/8 (PMS만) | 답 원문에서 WBS·Risk·산출물 ID를 찾는다. Risk·산출물은 같은 사업의 다른 ID가 있으면 틀림. 단계(level 1)가 있으면 "단계 지연"이 있어야 함. "몇 건"은 건수 |
| 문서 답 | 정답 수 ≥ 기준선(문서 지정) − 1 (문서만 8) | Claude가 qa_v2 정답과 답·인용 청크 원문을 대조해 판정(`_judgments.jsonl`, 근거 한 줄씩). 기준선도 같은 방식으로 판정 |
| 사업 미지정 | 되묻기 4/4, 임의 사업 선택 0 | 첫 턴에 `ok` 툴 호출이 없고, 거절이 아니고, 묻는 문장("?"·"어느"·"어떤 사업") |
| 답 없음 | 거절 ≥ 3/4 | 답이 거절 문구(`is_refusal`) |
| 출처 | 문서 근거 답의 유효 출처 100%, 범위 밖 번호 0 | `must_cite` 문항 중 거절하지 않은 답에 유효 번호 1개 이상, 모든 턴 `invalid_citations` 0 |
| 반복 | 문항당 평균 LLM 호출 ≤ 3, 상한 도달 ≤ 1문항 | `AgentAnswer.llm_calls` 합(턴 합), `reached_max_steps` |
| 비용 | 에이전트 1회 ≤ $0.30 | 턴 사용량 × 단가(입력 $0.10 / 출력 $0.50 per 1M, 캐시 할인 미반영) |
| 지연 | 기록만 (p50·p95) | 문항별 턴 지연 합 |

- 둘 다(6문항)의 답 정확도는 기준이 없다(제안 표에 없었다). 툴 선택·출처 기준에는 들어가고, 정답 ID 결과는 기록만 한다
- 기준선: 같은 질문의 첫 턴을 `answer_question(top_k=10)`, 프롬프트 `grounded_answer_v2`로 두 조건(문서 지정: 문항 doc_id, 없으면 전체 / 전체 검색) 실행

## 측정 전 예상

- 전체: 툴 선택 27~29/30, PMS 7~8/8, 문서 7/8(기준선 문서 지정과 같은 수준), 되묻기 3~4/4(u2 위험), 거절 3~4/4(n4 위험)
- LLM 호출: 문서·PMS 2회, 둘 다 3회, 되묻기 1회, 무관 1회 → 평균 약 2.1회. 비용 약 $0.06~0.10(에이전트) + 기준선 약 $0.04
- 기준선: PMS·둘 다 14문항은 문서에 답이 없어 대부분 거절하거나 틀린다. 전체 검색은 사업명이 있는 질문이라 #19보다 덜 틀릴 것으로 본다

| qid | 유형 | 질문 | 기대 툴 | 예상 |
|---|---|---|---|---|
| d1 | doc | AX를 통한 불균형 투자계약 점검 시스템 구축 사업에 쓸 수 있는 돈은 부가세를 포함해 얼마인가요? | search_documents | 통과 예상. #19 전체 검색에서 다른 사업 금액을 답한 문항 — 사업 목록으로 doc_id를 정하면 문서 지정 조건과 같다 |
| d2 | doc | 업무자동화(RPA) 시스템 고도화 사업의 수행 기간은 계약일을 기준으로 얼마나 되나요? | search_documents | 통과 예상(#19 문서 지정 답함) |
| d3 | doc | 연구행정 데이터 기반 AI 플랫폼 구축 사업에서 사용자가 질문을 입력한 뒤 답변이 나오기까지 지켜야 할 속도 기준은 무엇인가요? | search_documents | 통과 예상 |
| d4 | doc | 국가 망 보안 프레임워크 전환 ISP 수립 사업에서 앞으로 갖출 망 구조를 설계할 때 따라야 하는 보안 아키텍처는 무엇인가요? | search_documents | 통과 예상. PMS 없는 문서 — PMS 툴을 부르지 않아야 한다 |
| d5 | doc | 2027학년도 고등학교 입학전형 시스템 기능개선 용역의 SFR-013 요구사항은 어떤 업무를 지원하는 기능인가요? | search_documents | 통과 예상(exact, 요구사항 ID 가산) |
| d6 | doc | 아산시 강소형 스마트시티 스마트폴 구축 용역에서 세울 기둥은 모두 몇 개이고 종류별로 어떻게 나뉘나요? | search_documents | 위험: paraphrase(검색어가 '기둥'이고 원문은 '폴') |
| d7 | doc | 천안시 거점형 스마트도시 어반테크 실증존 구축 용역에서 기기에서 직접 돌아가는 인공지능이 잡아내야 하는 상황은 무엇인가요? | search_documents | 위험: paraphrase('엣지 AI'를 풀어 쓴 질문) |
| d8 | doc | 한국나노기술원 모아팹 연계기능 및 인터페이스 고도화 사업의 SER-001(보안가이드 준수)은 어느 기관들이 낸 어떤 문서를 반영하라고 하나요? | search_documents | 통과 예상(exact) |
| p1 | pms | AX를 통한 불균형 투자계약 점검 시스템 구축 사업에서 이번 주에 지연된 업무 알려줘 | get_delayed_tasks | 통과 예상 |
| p2 | pms | 한국나노기술원 모아팹 고도화 사업의 이번 주 지연 업무는 뭐야? | get_delayed_tasks | 위험: 단계(WBS 2)를 '단계 지연'으로 따로 적어야 한다. 사업명 일부만 적은 질문 |
| p3 | pms | 업무자동화(RPA) 시스템 고도화 사업에서 지연된 업무를 모두 알려줘 | get_delayed_tasks | 통과 예상(6건, 단계 1건) |
| p4 | pms | 잘피종자 자동파종 시스템 제작 용역의 미해결 Risk 중 high 등급은 뭐야? | get_open_risks | 통과 예상 |
| p5 | pms | 연구행정 데이터 기반 AI 플랫폼 구축 사업의 미해결 Risk를 전부 보여줘 | get_open_risks | 통과 예상 |
| p6 | pms | 2027학년도 고등학교 입학전형 시스템 기능개선 용역에서 기한이 지났는데 아직 제출하지 않은 산출물은? | get_deliverables | 위험: 12건을 빠짐없이 적어야 한다 |
| p7 | pms | 아산시 스마트폴 구축 용역에서 지연된 업무는 몇 건이야? | get_delayed_tasks | 통과 예상(total 6) |
| p8 | pms | 2027학년도 고등학교 입학전형 시스템 기능개선 용역의 미해결 Risk 중 medium 이상은? | get_open_risks | 통과 예상 |
| b1 | both | AX를 통한 불균형 투자계약 점검 시스템 구축 사업에서 이번 주 지연된 업무랑 관련된 요구사항 문서 알려줘 | get_delayed_tasks, search_documents | 통과 예상(architecture.md 3절 예시) |
| b2 | both | 잘피종자 자동파종 시스템 제작 용역에서 이번 주 지연된 업무가 RFP의 어떤 요구사항에 해당하는지 원문 근거와 함께 알려줘 | get_delayed_tasks, search_documents | 통과 예상 |
| b3 | both | 아산시 스마트폴 구축 용역의 이번 주 지연 업무와, 그 업무에 연결된 요구사항을 RFP에서 찾아 줘 | get_delayed_tasks, search_documents | 위험: 요구사항 3개 — 검색을 한 번에 묶으면 일부만 찾을 수 있다 |
| b4 | both | 2028 세계디자인수도(WDC) 부산 디지털 플랫폼 구축 용역에서 이번 주 지연된 업무와 관련 요구사항 내용을 알려줘 | get_delayed_tasks, search_documents | 위험: 요구사항 4개(QUR·SER·SFR 섞임) |
| b5 | both | 연구행정 데이터 기반 AI 플랫폼 구축 사업에서 점수가 가장 높은 미해결 Risk가 무엇이고, 관련된 RFP 요구 내용은 뭐야? | get_open_risks, search_documents | 위험: Risk와 RFP를 잇는 ID가 없어 검색어를 Risk 제목에서 만들어야 한다 |
| b6 | both | AX 불균형 투자계약 점검 시스템 사업에서 COR-001 요구사항 관련 산출물 제출 상태와, RFP가 COR-001에서 요구하는 내용을 알려줘 | get_deliverables, search_documents | 통과 예상(docstring 예시와 같은 흐름) |
| u1 | unspecified | 이번 주 지연된 업무 알려줘 | 없음 | 통과 예상 |
| u2 | unspecified | 이 사업의 수행 기간은 계약일 기준으로 얼마나 되나요? | 없음 | 위험: '이 사업'을 사업 목록 첫 사업으로 고를 수 있다(#19 전체 검색 실패 유형) |
| u3 | unspecified | 미해결 Risk 중에 high 등급인 게 있어? | 없음 | 통과 예상 |
| u4 | unspecified | 기한이 지났는데 아직 안 낸 산출물 보여줘 → 잘피종자 자동파종 시스템 사업이야 | get_deliverables | 통과 예상. 둘째 턴은 Checkpointer로 첫 질문을 이어받아야 한다 |
| n1 | no_answer | 아산시 강소형 스마트시티 스마트폴 구축 용역에서 스마트폴에 붙일 전기차 충전기는 몇 기이고 출력은 얼마인가요? | search_documents | 통과 예상(#19 두 조건 모두 거절) |
| n2 | no_answer | 한국나노기술원 모아팹 고도화 사업에서 도입할 AI 챗봇은 어떤 언어모델을 써야 하나요? | search_documents | 통과 예상(#19 두 조건 모두 거절) |
| n3 | no_answer | 대한민국의 수도는 어디인가요? | 없음 | 통과 예상. 툴을 부르지 않고 거절해야 한다 |
| n4 | no_answer | 파이썬에서 리스트를 내림차순으로 정렬하는 방법은 무엇인가요? | 없음 | 위험: 파이썬 질문에 사전 지식으로 답할 수 있다(규칙 1·7) |
