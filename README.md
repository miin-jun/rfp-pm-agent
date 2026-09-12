# rfp-pm-agent

공공 SI 사업 제안요청서(RFP)와 RFP 기반 합성 PMS 데이터를 근거로, "이번 주 지연된 업무", "미해결 Risk", "관련 산출물" 같은 질문에 출처를 달아 답하는 에이전트. 설계 배경은 `docs/`를 참고.

## 실행 방법

(추후 이슈에서 채움 — Docker Compose, API, MCP 서버 실행 방법)

```bash
uv sync
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

## 문서

- `docs/architecture.md` — 전체 구조, 단계(Phase), 결정 로그
- `docs/tech-stack.md` — 기술 스택과 선정 근거
- `docs/data-design.md` — 스키마, 판단 규칙, 평가 세트 형식
- `docs/github-setup.md` — 마일스톤·라벨·이슈
- `docs/harness.md` — 하네스(규칙·권한·훅) 설계
