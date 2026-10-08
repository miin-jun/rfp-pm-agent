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
docker compose ps                                  # opensearch healthy, tei-embed Up(os-knn·os-hybrid), tei-rerank Up(--rerank)
curl -s 'localhost:9200/_cat/aliases/rfp_chunks?v' # rfp_chunks가 인덱스 하나만 가리키는지
curl -s 'localhost:9200/_cat/count/rfp_chunks?v'   # 문서 수
wc -l data/chunks/block_requirement.jsonl          # 위 문서 수와 같아야 한다

uv run python -m rfp_pm_agent.eval.run_retrieval --os-bm25
uv run python -m rfp_pm_agent.eval.run_retrieval --os-knn
uv run python -m rfp_pm_agent.eval.run_retrieval --os-hybrid [--rerank --rerank-n 20]   # N은 10~50
uv run python -m rfp_pm_agent.eval.run_retrieval --compare <run_id> <run_id> --out-dir data/eval/results/hybrid
```

`--compare`는 retriever를 받지 않아서 기본 위치가 `model_selection`이다. #18 기록을 비교할 때는 `--out-dir`를 준다.

`run_retrieval`은 검색 전에 다음을 확인하고, 하나라도 맞지 않으면 ValueError로 멈춘다(점수가 검색이 아니라
데이터 차이를 재지 않게 하기 위해서다).
1. 별칭이 인덱스 하나만 가리킨다
2. 인덱스의 chunk_id 집합이 청크 파일과 같다
3. 인덱스의 `content_hash`(EMBED_PASSAGE_PREFIX + text)·`metadata_hash`가 청크 파일·파싱 결과
   (`--parsed-dir`, 기본 `data/parsed`)로 다시 계산한 값과 같다 — 색인 모듈과 같은 함수(`chunk_hashes`)를 쓴다
4. (os-knn·os-hybrid) 인덱스의 `embedding_model`이 질의 TEI 서버의 `{모델ID}@{revision}`과 같다

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

## 4. BM25 요구사항 ID 가산 (#75, 2026-10-06)

질문에 요구사항 ID가 있으면 `requirement_id` 정확 일치 가산(+100)을 붙인다(`bm25_search`,
[ADR-0003](adr/0003-requirement-id-boost.md)). 이후 서비스 BM25는 이 방식이므로, **PR ② 하이브리드 비교의
BM25 기준은 아래 `20261006T141105Z-os-bm25`(21/51)**다. 3절의 `20261005T114207Z-os-bm25`(19/51)는 가산 전 이력이다.

| run_id | 검색 | R@10 전부 | R@10 비율 | R@5 전부 | MRR | NDCG@10 | 검색 p50 / p95 |
|---|---|---|---|---|---|---|---|
| `20261006T141105Z-os-bm25` | BM25 (nori `korean`, `match`) + ID 가산 — **기준** | **21/51** | 22.0 | 16 | 0.230 | 0.276 | 7.7 / 13.0 ms |

- 가산 전 대비 McNemar: 가산 후만 맞힘 2(q041·q055, 둘 다 ID 질문), 가산 전만 맞힘 0, p=0.5 — 동률
- ID 없는 문항은 상위 10 목록까지 같다. 해석과 한계는 ADR-0003

## 5. PR ② 하이브리드(RRF) + 리랭크

조건: `hybrid_search`(OpenSearch `hybrid` 질의 + 요청 본문의 임시 RRF 파이프라인, `rank_constant=60`,
검색기별 후보 50 = `pagination_depth`·k-NN `k`), 리랭크는 hybrid 상위 N개를 `bge-reranker-v2-m3`로 다시 정렬한다.

### 측정 전 예상 (2026-10-07, 결과를 보기 전에 고정)

| 조건 | 예상 R@10 전부 | 근거 |
|---|---|---|
| os-hybrid | 29/51 | 직감 |
| os-hybrid + rerank N=20 | 31/51 | 직감 |
| os-hybrid + rerank N=50 | 31/51 | 직감 |

비교 기준: k-NN(k=50) `20261005T134118Z-os-knn-KURE-v1` 30/51, BM25(ID 가산) `20261006T141105Z-os-bm25` 21/51.

### 실행 (2026-10-07)

OpenSearch·`tei-embed`·`tei-rerank`를 함께 띄워 쟀다(측정 직후 `docker stats`: 1.77 / 1.32 / 1.16GiB, OOM 없음).
실행 전 가드(별칭이 인덱스 하나, chunk_id 집합, `content_hash`·`metadata_hash`, `embedding_model` =
`nlpai-lab/KURE-v1@8b418a5841…`)는 세 실행 모두 통과했다.

```bash
uv run python -m rfp_pm_agent.eval.run_retrieval --os-hybrid
uv run python -m rfp_pm_agent.eval.run_retrieval --os-hybrid --rerank --rerank-n 20
uv run python -m rfp_pm_agent.eval.run_retrieval --os-hybrid --rerank --rerank-n 50
uv run python -m rfp_pm_agent.eval.run_retrieval --compare 20261005T134118Z-os-knn-KURE-v1 20261006T141105Z-os-bm25 \
  20261007T142117Z-os-hybrid-KURE-v1 20261007T142136Z-os-hybrid-KURE-v1+rerank-bge-reranker-v2-m3-n20 \
  20261007T142235Z-os-hybrid-KURE-v1+rerank-bge-reranker-v2-m3-n50 --out-dir data/eval/results/hybrid
```

### 결과

| run_id | 검색 | R@10 전부 | R@10 비율 | R@5 전부 | MRR | NDCG@10 | 검색 p50 / p95 | 리랭크 p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| `20261005T134118Z-os-knn-KURE-v1` | k-NN (k=50) — 기준 | 30/51 | 31.0 | 23 | 0.317 | 0.380 | 19.6 / 24.8 ms | - |
| `20261006T141105Z-os-bm25` | BM25 + ID 가산 — 기준 | 21/51 | 22.0 | 16 | 0.230 | 0.276 | 7.7 / 13.0 ms | - |
| `20261007T142117Z-os-hybrid-KURE-v1` | hybrid RRF | 30/51 | 30.5 | 26 | 0.348 | 0.404 | 46.2 / 87.6 ms | - |
| `…+rerank-bge-reranker-v2-m3-n20` | hybrid + 리랭크 N=20 | **33/51** | 33.5 | 27 | 0.497 | 0.527 | 35.8 / 47.4 ms | 229.7 / 528.3 ms |
| `…+rerank-bge-reranker-v2-m3-n50` | hybrid + 리랭크 N=50 | **33/51** | 34.5 | 28 | 0.519 | 0.551 | 36.5 / 46.3 ms | 667.5 / 1139.6 ms |

- 지연 표본은 각 171개(워밍업 5문항 뒤 57문항 × 3회). hybrid 검색 지연에는 질의 임베딩 시간이 들어 있다.
  첫 실행(리랭크 없음)의 검색 지연만 p50 46ms로 뒤 두 실행(36ms)보다 높았다 — 같은 검색인데 차이가 나는
  원인은 확인하지 않았다
- 앱 RRF 상위 10 겹침(답 있는 51문항): 평균 9.96, 최소 9(q002·q004). 두 문항 모두 10위와 11위의 RRF 점수가
  같고(q002 1/65, q004 1/63 — 한 검색기에서만 5위·3위), OpenSearch와 `rrf_fuse`가 동점을 다르게 끊어
  10위 청크가 바뀐 것이다. 점수 계산 차이는 아니다

McNemar(1위 = 리랭크 N=20, 같은 R@10이면 앞에 적은 실행):

| 실행 | 1위만 | 이쪽만 | p | 판정 |
|---|---|---|---|---|
| k-NN (k=50) | 4 | 1 | 0.375 | 동률 |
| BM25 + ID 가산 | 12 | 0 | 0.0005 | 1위보다 낮음 |
| hybrid | 3 | 0 | 0.25 | 동률 |
| 리랭크 N=50 | 3 | 3 | 1.0 | 동률 |

### 측정 전 예상과 비교

| 조건 | 예상 | 실제 |
|---|---|---|
| os-hybrid | 29/51 | 30/51 |
| os-hybrid + rerank N=20 | 31/51 | 33/51 |
| os-hybrid + rerank N=50 | 31/51 | 33/51 |

### k-NN(k=50) 대비 문항별 차이

| 조건 | 새로 맞힘 | 새로 틀림 | p |
|---|---|---|---|
| hybrid | q013·q016·q017 | q011·q022·q053 | 1.0 |
| 리랭크 N=20 | q013·q016·q017·q038 | q053 | 0.375 |
| 리랭크 N=50 | q008·q013·q016·q020·q029·q038 | q010·q014·q053 | 0.508 |

새로 틀린 문항의 정답 청크 순위(측정 뒤 같은 인덱스·서버로 다시 검색, 각 방식 후보 50; `>50`은 후보 밖,
리랭크 N=20의 `>20`은 리랭크 대상 밖). 앱 RRF 순위는 모든 문항에서 hybrid와 같았다.

| 문항 | 정답 청크 | BM25 | k-NN | hybrid | 리랭크 N=20 | 리랭크 N=50 |
|---|---|---|---|---|---|---|
| q010 | b0033 | 3 | 6 | 4 | 7 | **11** |
| q011 | b0032 | >50 | 3 | **14** | 1 | 1 |
| q014 | ECR-001 | 5 | 3 | 3 | 10 | **14** |
| q022 | b0065 | >50 | 7 | **19** | 2 | 2 |
| q053 | b0015 | >50 | 10 | **22** | **>20** | **30** |
| q053 | b0016 | >50 | 4 | **21** | **>20** | 10 |

- hybrid가 놓친 q011·q022·q053은 모두 BM25 후보 50 안에 정답이 없다. 두 목록에 다 든 청크가 1/(60+r)를
  두 번 받아 k-NN에만 있는 정답(3·7·4·10위)을 10위 밖으로 밀어냈다
- 리랭크는 q011·q022를 다시 1·2위로 올렸지만, q053은 정답이 hybrid 21·22위라 N=20이면 리랭크 대상에 들지
  못한다. N=50에서는 b0016은 10위로 올라왔고 b0015는 30위다(evidence 2개를 모두 맞혀야 적중)
- q010·q014는 hybrid에서 맞혔는데 리랭크 N=50에서 리랭커가 다른 후보를 위로 올려 10위 밖으로 밀렸다
  (N=20에서는 7·10위로 남았다)
- N=20과 N=50을 직접 대조하면 N=20만 맞힌 문항은 q010·q014·q017, N=50만 맞힌 문항은 q008·q020·q029다(3:3, p=1.0).
  q017도 hybrid·N=20에서 맞혔다가 N=50에서 놓쳤다 — k-NN도 놓친 문항이라 위 k-NN 대비 표에는 나오지 않는다

### 결정 (2026-10-07, 결과를 본 뒤 소유자)

서비스 검색 기본값 = 하이브리드(RRF k=60, 후보 50) + 리랭크 N=20. 근거·한계·재검토 조건은
[ADR-0004](adr/0004-hybrid-rerank-default.md).

## 6. 인덱스 v2 재색인 확인 (#81)

#81에서 출처 필드(`bid_title`·`format`·`printed_page`)를 매핑에 추가하고 새 인덱스 `rfp_chunks_v2_kure`로 전체 색인한 뒤
별칭을 v1 → v2로 옮긴다. 필드만 추가했으므로 BM25는 같아야 하고, k-NN은 벡터·HNSW 그래프를 새로 만들어 근사 결과가 조금
달라질 수 있다.

### 측정 전 예상 (2026-10-08, 측정 전에 고정 — 이슈 #81 본문과 같다)

| 조건 | 비교 대상 (v1) | 예상 | 근거 |
|---|---|---|---|
| os-bm25 (ID 가산) | `20261006T141105Z-os-bm25` 21/51 | 21/51, 문항별 결과(적중·상위 10 목록) 같음 | BM25는 벡터와 무관하고 text·분석기·문서 집합이 같다 |
| os-hybrid + rerank N=20 | `20261007T142136Z-os-hybrid-KURE-v1+rerank-bge-reranker-v2-m3-n20` 33/51 | 33/51, 상위 10이 바뀌는 문항 0~2개, 적중이 바뀌는 문항 0개 | 같은 모델·같은 text라 벡터는 같고, 달라질 수 있는 것은 HNSW 근사 탐색뿐이다. k=50으로 넓게 탐색하고 리랭크가 상위 20을 다시 정렬한다 |

측정 전에 알려진 위험 (os-bm25): v1에는 삭제 표시만 된 문서가 3개 남아 있다(2026-10-08 `_cat/indices`: docs.count 3,499,
docs.deleted 3 — 증분 색인의 update가 남긴 것). Lucene BM25 통계는 병합 전까지 삭제 문서를 포함하므로 새로 만든 v2와 점수가
아주 조금 다를 수 있고, update된 문서는 v1에서 내부 문서 순서가 뒤로 밀려 있어 점수가 같은 결과의 순서가 바뀔 수 있다.
