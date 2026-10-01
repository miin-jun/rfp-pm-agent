# 개발 환경과 실행 방법

README에서 옮긴 내용이다(2026-09-28). 명령어 전체 목록은 [CLAUDE.md](../CLAUDE.md) "명령어" 절에 있다.

## 실행 방법

(추후 이슈에서 채움 — Docker Compose, API, MCP 서버 실행 방법)

```bash
uv sync
```

임베딩·리랭커 서버(TEI, GPU 필요). `.env`에 `EMBED_MODEL_ID`·`RERANK_MODEL_ID`가 있어야 한다 — 없으면 TEI뿐 아니라 postgres·opensearch만 띄우는 `docker compose up -d`도 `required variable ... is missing a value`로 실패한다(compose가 파일 전체를 먼저 변수 치환하기 때문).

```bash
docker compose up -d tei-embed tei-rerank     # 첫 기동은 모델 다운로드(약 4.5GB)로 수 분
curl -i localhost:8080/health                 # 200이면 준비 완료 (리랭커는 8081)
uv run pytest tests/integration -m tei -q     # 실제 TEI 호출 테스트 (꺼져 있으면 실패)
```

## 알려진 함정

- **WSL2에서 OpenSearch 컨테이너가 계속 재시작되거나 바로 죽는 경우**: 커널의 `vm.max_map_count`가 기본값(65530)이라 OpenSearch(Lucene)가 요구하는 최소값(262144)에 못 미쳐서 발생합니다.
  - 임시 적용 (WSL 재시작 시 초기화됨):
    ```bash
    sudo sysctl -w vm.max_map_count=262144
    ```
  - 영구 적용 (WSL2, `/etc/sysctl.conf`에 추가):
    ```bash
    echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    ```
    Windows 쪽에서 WSL을 재시작(`wsl --shutdown`)해도 유지되지 않는 환경이라면, Windows 사용자 홈의 `.wslconfig`에 아래를 추가하고 `wsl --shutdown` 후 다시 켭니다.
    ```ini
    [wsl2]
    kernelCommandLine = "sysctl.vm.max_map_count=262144"
    ```
  - **확인됨**: `sudo sysctl -w`는 그 부팅 세션에만 적용되는 임시 설정이다. WSL을 재시작하면 값이 초기화되므로, 매번 새로 켤 때마다 다시 설정하지 않으려면 위 영구 적용법(`/etc/sysctl.conf` 또는 `.wslconfig`)을 반드시 함께 적용한다.

- **`agent_ro` 읽기 전용 계정 권한 검증 방법**: `psql`로 `agent_ro`에 접속해 `SELECT`는 성공하고 `CREATE TABLE`은 `permission denied`로 **실패해야 정상**입니다. 실패하지 않고 성공한다면 `docker/postgres/init/01_create_readonly_user.sh`의 권한 부여가 의도보다 넓게 된 것이니 다시 확인합니다.

- **TEI 컨테이너가 137로 종료되는 경우**: `docker inspect -f '{{.State.OOMKilled}}' rfp-pm-agent-tei-embed`가 `true`면 WSL 호스트 RAM 부족입니다. `.env`의 `TEI_TOKENIZATION_WORKERS`(compose 기본 2)를 확인합니다. 미지정(워커 19개)으로 두 서버를 띄웠을 때 이 증상이 났습니다.
