# 하이브리드 검색 측정 (이슈 #18)

#18은 PR 3개로 나눈다(이슈 본문 "결정 (2026-10-03)" 1). 이 문서는 각 PR의 측정 절차와 결과를 모은다.
측정 전 고정값(RRF k=60, 검색기별 후보 50, 리랭커 N=20·N=50, 판정은 R@10 McNemar)은 이슈 본문 결정 4에 있다.
LLM을 부르지 않으므로 비용은 $0이다.

## 1. 공통 조건

- 평가 세트 `data/eval/qa_v2.jsonl`: 57문항을 모두 검색하고, 답 있는 51문항만 채점한다
- 정답 묶음: `data/chunks/block_requirement.jsonl`(3,499개, #62 이후)에서 evidence로 뽑는다
- 검색 대상: 별칭 `rfp_chunks` → `rfp_chunks_v1_kure` (KURE-v1, lucene HNSW m=16·ef_construction=100)
- 기록 위치: `data/eval/results/hybrid/` — os 모드는 `--out-dir`를 주지 않아도 여기에 쓴다
  (#16 모델 선정 기록 `data/eval/results/model_selection/`과 섞지 않는다)
- 지연: 워밍업 5문항 뒤 57문항 × 3회 = 표본 171개. os-knn의 검색 지연에는 질의 임베딩(TEI 호출) 시간이
  들어 있다(#16 dense와 같은 방식). 그래서 os-bm25 지연과 그대로 비교하지 않는다

## 2. 실행

```bash
docker compose ps                                  # opensearch healthy, tei-embed Up(os-knn만 필요)
curl -s 'localhost:9200/_cat/aliases/rfp_chunks?v' # rfp_chunks가 인덱스 하나만 가리키는지
curl -s 'localhost:9200/_cat/count/rfp_chunks?v'   # 문서 수
wc -l data/chunks/block_requirement.jsonl          # 위 문서 수와 같아야 한다

uv run python -m rfp_pm_agent.eval.run_retrieval --os-bm25
uv run python -m rfp_pm_agent.eval.run_retrieval --os-knn
uv run python -m rfp_pm_agent.eval.run_retrieval --compare <run_id> <run_id> --out-dir data/eval/results/hybrid
```

`--compare`는 retriever를 받지 않아서 기본 위치가 `model_selection`이다. #18 기록을 비교할 때는 `--out-dir`를 준다.

`run_retrieval`은 검색 전에 다음을 확인하고, 하나라도 맞지 않으면 ValueError로 멈춘다(점수가 검색이 아니라
데이터 차이를 재지 않게 하기 위해서다).
1. 별칭이 인덱스 하나만 가리킨다
2. 인덱스의 chunk_id 집합이 청크 파일과 같다
3. 인덱스의 `content_hash`(EMBED_PASSAGE_PREFIX + text)·`metadata_hash`가 청크 파일·파싱 결과
   (`--parsed-dir`, 기본 `data/parsed`)로 다시 계산한 값과 같다 — 색인 모듈과 같은 함수(`chunk_hashes`)를 쓴다
4. (os-knn) 인덱스의 `embedding_model`이 질의 TEI 서버의 `{모델ID}@{revision}`과 같다

## 3. PR ① 단일 기준선 (2026-10-05)

| run_id | 검색 | R@10 전부 | R@10 비율 | R@5 전부 | MRR | NDCG@10 | 검색 p50 / p95 |
|---|---|---|---|---|---|---|---|
| `20261005T114207Z-os-bm25` | BM25 (nori `korean`, `match`) | **19/51** | 20.0 | 14 | 0.191 | 0.237 | 7.8 / 13.0 ms |
| `20261005T114923Z-os-knn-KURE-v1` | k-NN (`k=10`, `size=10`) — 이력 | 28/51 | 29.0 | 23 | 0.324 | 0.376 | 18.7 / 30.9 ms (질의 임베딩 포함) |
| `20261005T134118Z-os-knn-KURE-v1` | k-NN (`k=50`, `size=10`) — **기준** | **30/51** | 31.0 | 23 | 0.317 | 0.380 | 19.6 / 24.8 ms (질의 임베딩 포함) |

두 os-knn 기록은 `runs.jsonl`에 k 값이 남지 않는다(코드 상수 `KNN_CANDIDATES`). 위 표의 run_id로 구별한다.

McNemar 정확검정(양측, R@10 전부 적중, 1위 = k=50):
- BM25 대비: k-NN만 맞힘 12, BM25만 맞힘 1, **p=0.0034** — k-NN이 BM25보다 높다
- k=10 대비: k=50만 맞힘 3, k=10만 맞힘 1, p=0.625 — 동률
- 이력(k=10 기준, 2026-10-05 첫 측정): k-NN만 11, BM25만 2, p=0.0225

측정 전 예상과 비교

| | 측정 전 예상 | 실제 | 참고 (같은 조건 아님) |
|---|---|---|---|
| BM25 | 15 근처 | 19/51 | #16 rank_bm25 + bigram 15/51 (`20260926T133524Z-bm25-bigram`, 3,514청크) |
| k-NN (k=10) | 약 30 | 28/51 | #16 KURE-v1 메모리 정확 검색 30/51 (`20260926T143013Z-dense-KURE-v1`, 3,514청크) |
| k-NN (k=50) | 28~30 (재측정 전 고정) | 30/51 | 같음 |

- BM25는 예상보다 4문항 많다. nori 형태소 분석과 rank_bm25 bigram 토큰화의 차이로 보이지만, 문항별 원인은
  확인하지 않았다. 질문 어미("인가") 처리(#70)는 아직 적용하지 않은 값이다
- k=10 k-NN은 #16 정확 검색보다 2문항 적었다. 두 측정은 청크 수(3,514 → 3,499)와 탐색 방식(정확 → HNSW 근사,
  `k=10`)이 함께 달라서 원인을 나눌 수 없었다. 청크 15개를 뺀 것만으로는 gold 순위가 내려가지 않으므로
  (정확 검색이라면 같거나 올라감) HNSW 근사를 후보로 보고 아래처럼 k를 바꿔 다시 쟀다

### k-NN `k` 재측정 (#18 리뷰 권장 6)

lucene HNSW는 `k`가 탐색 후보 수이기도 해서 `k=size=10`은 근사 오차가 가장 큰 설정이다. `k=50`(결정 4의
"검색기별 후보 50")으로 바꾸고 `size=10`으로 잘라 다시 잰다. `k=10` 실행 기록은 지우지 않는다.

- 측정 전 예상(고정): **28~30/51**
- 결과: `20261005T134118Z-os-knn-KURE-v1` R@10 30/51, MRR 0.317, NDCG@10 0.380 (예상 범위 안)
- 해석: k=10 28/51 → k=50 30/51, 차이는 McNemar p=0.625로 유의하지 않음. k=50에서 #16 정확 검색 30/51(3,514청크)과
  같은 값. PR② 하이브리드 비교 기준은 k=50 30/51.
