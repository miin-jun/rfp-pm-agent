---
name: run-eval
description: 검색 또는 에이전트 평가를 실행하고 기준선과 비교할 때 사용. "평가 돌려", "기준선이랑 비교", PR 전 품질 확인 요청에 사용.
---

# 평가 실행 절차

1. 어떤 평가인지 확인: 검색(retrieval) / 에이전트(agent)
2. 에이전트 평가는 LLM을 호출한다. 실행 전 문항 수와 예상 호출 수를 사용자에게 알리고 승인받는다
3. 실행
   - 검색: `uv run python -m eval.run_retrieval --set eval/sets/retrieval_v1.jsonl`
   - 에이전트: `uv run python -m eval.run_agent --set eval/sets/agent_v1.jsonl`
4. `eval/results/baseline.json`과 비교해 표로 요약 (지표, 기준선, 현재, 차이)
5. 하락한 지표가 있으면 떨어진 문항 3~5개를 골라 원인 후보를 제시한다
6. 평가 세트나 판정 기준은 절대 수정하지 않는다
