# 임베딩·리랭커 선정 측정 절차 (이슈 #16)

결정 규칙은 이슈 #16 본문 "결정 규칙" 절과 docs/tech-stack.md 12-1에 측정 전에 고정했다.
이 문서는 그 규칙에 넣을 숫자를 같은 조건으로 재는 순서다. LLM 비용은 없다($0).

## 0. 측정 전 확인
- Docker Desktop의 WSL 연동이 켜져 있어 WSL에서 `docker` 명령이 된다
- 노트북 **전원 연결**, Windows 전원 옵션 **고성능**
- **Pwr Cap 기록**: WSL의 `nvidia-smi`는 전력 한도를 `[N/A]`로 돌려준다. Windows PowerShell에서
  `nvidia-smi.exe -q -d POWER`를 실행해 Power Limit 값을 결과 메모에 적는다
- TEI 서버는 `TEI_TOKENIZATION_WORKERS=2`로 띄운다(기본값). 워커 수를 바꾸면 색인 시간이 달라지므로
  바꿨다면 기록한다 (learning-log 2026-09-24)
- 모델은 측정 전에 한 번 띄워 내려받아 둔다. 색인 시간에 다운로드 시간이 섞이지 않게 하기 위해서다

## 1. 모델별 설정

`.env`를 고치지 않고 셸 환경변수로 덮는다 — compose와 `config.py`(`load_dotenv(override=False)`)
모두 셸 값을 `.env`보다 우선한다. 서버와 `run_retrieval`에 **같은 값**을 줘야 한다
(`run_retrieval`은 `/info`의 model_id가 `EMBED_MODEL_ID`와 다르면 실행하지 않는다).

| 모델 | EMBED_MODEL_ID | EMBED_QUERY_PREFIX | EMBED_PASSAGE_PREFIX | EMBED_TRUNCATE |
|---|---|---|---|---|
| KURE-v1 | `nlpai-lab/KURE-v1` | (빈 값) | (빈 값) | `false` |
| bge-m3 | `BAAI/bge-m3` | (빈 값) | (빈 값) | `false` |
| multilingual-e5-large | `intfloat/multilingual-e5-large` | `query: ` | `passage: ` | `true` |

- 접두어 근거: e5 모델 카드("Each input text should start with 'query: ' or 'passage: '"),
  KURE-v1·bge-m3 모델 카드(접두어 언급 없음 / "no longer requires adding instruction") — 2026-09-26 확인
- e5는 최대 512토큰이라 긴 청크를 잘라 임베딩한다. 클라이언트가 `/tokenize`로 먼저 세어 자른 청크 수를
  `runs.jsonl`의 `truncated_chunks`에 남긴다. compose는 `AUTO_TRUNCATE: "false"`이고 요청마다
  `truncate: true`를 보낸다 — TEI 1.9.4가 요청 단위 `truncate`를 서버 설정보다 우선하는지는 미확인이다.
  e5 실행 뒤 413 오류 없이 끝났는지, `truncated_chunks`가 0보다 큰지를 확인한다

- **빈 청크**: block_requirement 청크 중 text가 빈 것(2026-09-26 기준 4개: 944b의 b0134, d3e2의
  b0005·b0023·b0165 — 모두 내용 없는 1×1 표 블록)은 TEI가 400으로 거절한다. 모든 모델에서 임베딩 요청에서
  빼고 영벡터(코사인 점수 0)로 두며, 개수와 ID를 `runs.jsonl`의 `empty_chunks`·`empty_chunk_ids`에 남긴다
- **모델 revision**: TEI 1.9.4의 `/info`는 revision을 주지 않고 띄우면 `model_sha`가 null이다.
  `run_retrieval`은 TEI 모델 볼륨(`TEI_MODELS_VOLUME`)의 `models--<org>--<name>/snapshots/` 아래 해시를
  읽어 `snapshot_revision`에 남기고, 벡터 캐시 이름에도 쓴다. 스냅샷이 0개거나 2개 이상이면, 또는 못 읽으면
  실행하지 않는다

## 2. 순서 (모델 하나마다)

```bash
export EMBED_MODEL_ID=nlpai-lab/KURE-v1 EMBED_QUERY_PREFIX= EMBED_PASSAGE_PREFIX= EMBED_TRUNCATE=false

# (1) 기동 전 GPU 스냅샷
nvidia-smi --query-gpu=name,driver_version,power.draw,clocks.sm,memory.used,temperature.gpu --format=csv

# (2) 임베딩 서버만 띄우고 로드 확인
docker compose up -d tei-embed
docker compose logs tei-embed | tail -20          # Ready, 로드 오류 여부
curl -s localhost:8080/info                        # model_id, model_sha, max_input_length, version

# (3) VRAM 기록을 켠 채 색인 + 검색 (캐시가 없어야 색인 시간이 잰다)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 > data/tmp/vram_<모델>.csv &
uv run python -m rfp_pm_agent.eval.run_retrieval --dense
kill %1

# (4) 리랭커를 함께 띄워 리랭크 조건 (임베딩 서버는 질의 임베딩에 계속 쓴다)
docker compose up -d tei-rerank
uv run python -m rfp_pm_agent.eval.run_retrieval --dense --rerank

# (5) 측정 후 스냅샷, 서버 내리기
nvidia-smi --query-gpu=name,driver_version,power.draw,clocks.sm,memory.used,temperature.gpu --format=csv
docker compose stop tei-embed tei-rerank
```

- 두 TEI 서버를 함께 띄우는 것은 워커 2개일 때 된다고 확인했다(learning-log 2026-09-24). 워커 수를
  늘리지 않는다
- 벡터 캐시(`data/cache/embeddings/`)가 있으면 색인을 건너뛰고 처음 잰 색인 시간을 기록에 옮긴다.
  색인 시간을 다시 재려면 해당 모델의 캐시 파일을 지운다
- BM25 기준선은 TEI 없이 돈다: `uv run python -m rfp_pm_agent.eval.run_retrieval --bm25`

## 3. 무엇을 어떻게 재나

| 항목 | 방법 | 기록 위치 |
|---|---|---|
| Recall@5·@10(전부 적중·비율), MRR, NDCG@10 | `run_retrieval`이 문항별로 채점 | `runs.jsonl`, `<run_id>.questions.jsonl` |
| 색인 시간 | 청크 3,514개 중 빈 청크 4개를 뺀 3,510개를 배치 32로 임베딩한 전체 시간(1회) | `runs.jsonl`의 `index_seconds` |
| 지연 | 워밍업 5문항 뒤 57문항 × 3회, 질의 1건씩. 검색(질의 임베딩 + 코사인)과 리랭크(상위 20개)를 따로 | `search_latency`, `rerank_latency` (p50·p95) |
| VRAM | 색인 중 `nvidia-smi -lms 500` 기록의 최댓값 − 기동 전 값 | `data/tmp/vram_<모델>.csv`(git 제외) → 요약 `data/eval/results/model_selection/vram_summary.json` |
| 환경 | GPU 이름·드라이버, TEI 버전(`/info`), 이미지 태그(docker-compose.yml), 모델 revision(볼륨 스냅샷 해시) | `runs.jsonl` |

- WSL의 `nvidia-smi`가 `memory.used`를 0으로 보고한 적이 있다(2026-09-26, TEI를 띄우지 않은 상태).
  TEI를 띄운 뒤에도 0이면 Windows PowerShell의 `nvidia-smi.exe`로 같은 값을 읽는다 — WSL 값을 믿을
  수 있는지는 미확인이다
- 지연은 로컬 RTX 4050 Laptop 기준이다. RunPod 등 다른 환경의 값과 섞지 않는다

## 4. 결정 규칙 적용

```bash
uv run python -m rfp_pm_agent.eval.run_retrieval --compare <run_id> <run_id> <run_id>
```

Recall@10(전부 적중) 수가 가장 많은 실행을 1위로 두고, 나머지와 McNemar 정확검정(양측)을 한다.
p ≥ 0.05면 동률이고, 동률인 모델 중 VRAM → 색인 시간 → 지연 순으로 가벼운 모델을 고른다(사람이 기록을
보고 판단). 리랭커는 McNemar 없이 NDCG@10 개선폭과 p95 지연 증가를 함께 적는다.
