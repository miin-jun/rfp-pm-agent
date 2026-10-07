---
name: run-eval
description: 검색 또는 에이전트 평가를 실행하고 기준선과 비교할 때 사용. "평가 돌려", "기준선이랑 비교", PR 전 품질 확인 요청에 사용.
---

# 평가 실행 절차

> 상태: 검색 평가는 `rfp_pm_agent.eval.run_retrieval`로 실행한다(#16·#18).
> 에이전트 평가(`run_agent`)와 평가 게이트는 아직 없다 — #25에서 정한다.
> 에이전트 평가를 요청받으면 실행을 시도하지 말고 소유자에게 알린다.

1. 어떤 평가인지 확인: 검색(retrieval) / 에이전트(agent)
2. 에이전트 평가는 LLM을 호출한다. 실행 전 문항 수와 예상 호출 수를 사용자에게 알리고 승인받는다
3. 실행
   - 검색: `uv run python -m rfp_pm_agent.eval.run_retrieval <방식>` — 방식 인자는 하나만, 반드시 준다
     - `--os-bm25`: OpenSearch 별칭 BM25(nori). OpenSearch 필요
     - `--os-knn`: OpenSearch 별칭 k-NN. OpenSearch·`tei-embed` 필요
     - `--os-hybrid [--rerank --rerank-n 50]`: OpenSearch hybrid(BM25 + k-NN, RRF). OpenSearch·`tei-embed` 필요, `--rerank`면 `tei-rerank`도 필요
       - os 모드 사전 조건(별칭·chunk_id·해시 대조, 어긋나면 ValueError로 멈춤): `docs/hybrid-search-measurement.md` 2절
     - `--bm25`: 청크 파일 메모리 BM25 기준선. 서비스 불필요
     - `--dense [--rerank --rerank-n 20]`: TEI 벡터 검색(+리랭크). `tei-embed` 필요, `--rerank`면 `tei-rerank`도 필요
       - 리랭크는 `--dense`와 `--os-hybrid`에만 붙는다(`--bm25`·`--os-bm25`·`--os-knn`에 붙이면 인자 오류). `--rerank-n`은 10~50(10 미만이면 R@10을 N개로 채점하게 되어 막는다)
       - 사전 조건(`EMBED_MODEL_ID` 일치 등): `docs/model-selection-measurement.md`. TEI 두 개 동시 기동 시 RAM 부족은 CLAUDE.md "알려진 함정"
     - 평가 세트는 `--qa`(기본 `data/eval/qa_v2.jsonl`), 청크는 `--chunks`(기본 `data/chunks/block_requirement.jsonl`)
     - 기록 위치(`--out-dir` 생략 시): os 모드 `data/eval/results/hybrid/`, 나머지 `data/eval/results/model_selection/`
   - 에이전트: #25에서 정한다 (`run_agent`, `data/eval/agent_v1.jsonl`은 아직 없다)
4. 기준선과 비교: `uv run python -m rfp_pm_agent.eval.run_retrieval --compare <run_id> <run_id> ... --out-dir data/eval/results/<실험>`
   - 기준을 따로 두지 않는다. Recall@10(전부 적중) 문항 수가 가장 많은 실행을 1위로 두고 나머지를 1위와 McNemar로 비교한다
     (동점이면 앞에 적은 run_id가 1위). MRR·NDCG 등 다른 지표는 검정하지 않는다
   - `--out-dir`의 `runs.jsonl` 하나만 읽으므로 같은 기록 위치의 실행끼리만 비교된다(`hybrid/`와 `model_selection/` 실행을 섞으면 ValueError)
   - `--out-dir`를 빼면 `model_selection` 기록을 읽는다 — hybrid 실행을 비교할 땐 반드시 준다
   - 출력에 지표 표는 없다. `runs.jsonl`의 각 실행 점수로 표를 만든다 (지표, 기준선, 현재, 차이)
   - `--compare`는 `--rerank`·`--rerank-n`을 쓰지 않는다(에러 없이 무시된다)
5. 하락한 지표가 있으면 떨어진 문항 3~5개를 골라 원인 후보를 제시한다
   - 문항별 결과는 `data/eval/results/<실험>/<run_id>.questions.jsonl`, 틀린 문항의 상위 검색 결과는 `failures/<run_id>.jsonl`
6. 평가 세트나 판정 기준은 절대 수정하지 않는다
