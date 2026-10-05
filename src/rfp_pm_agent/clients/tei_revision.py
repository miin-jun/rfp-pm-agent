"""TEI 서버가 띄운 모델의 revision(스냅샷 해시) 읽기 — #16 측정과 #17 색인이 함께 쓴다.

TEI 1.9.4의 `/info`는 `model_sha`가 null이라, 모델 파일이 바뀌었는지 알려면 TEI 모델 볼륨의
`models--<org>--<name>/snapshots/` 아래 디렉터리 이름을 직접 읽어야 한다. 볼륨은 root 소유라
WSL에서 바로 읽을 수 없어, TEI 이미지로 `ls`만 하는 일회용 컨테이너를 띄운다.

원래 `eval/run_retrieval.py`(#16)에 있었고 `ingest/index_chunks.py`(#17)가 eval을 import해
쓰던 것을 여기로 옮겼다(#18 댓글, 계층 방향 ingest → eval 제거).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from rfp_pm_agent.config import REPO_ROOT, ClientsConfig

# 레포 루트 기준 절대 경로. 상대 경로였을 때는 레포 루트가 아닌 곳에서 CLI를 실행하면
# compose 파일을 못 찾아 revision 읽기가 실패했다(#18 댓글)
DEFAULT_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


def tei_image_tag(compose_file: Path) -> str | None:
    """docker-compose.yml에 적힌 TEI 이미지 태그. 실제로 떠 있는 컨테이너 이미지는 아니다."""
    if not compose_file.exists():
        return None
    match = re.search(r"image:\s*(\S*text-embeddings-inference\S*)", compose_file.read_text())
    return match.group(1) if match else None


def parse_snapshot_listing(model_id: str, listing: str) -> str:
    """`snapshots/` 목록에서 revision 해시 하나를 고른다. 0개거나 2개 이상이면 ValueError.

    2개 이상이면 어느 스냅샷이 로드됐는지 이 목록만으로는 알 수 없으므로 추측하지 않는다.
    """
    entries = [line.strip() for line in listing.splitlines() if line.strip()]
    if len(entries) != 1:
        raise ValueError(
            f"{model_id}의 TEI 스냅샷이 {len(entries)}개다({entries}) — revision을 정할 수 없다"
        )
    return entries[0]


def read_snapshot_revision(model_id: str, *, volume: str, image: str) -> str:
    """TEI 모델 볼륨의 models--<org>--<name>/snapshots/ 아래 해시를 읽는다.

    볼륨은 root 소유라 WSL에서 바로 읽을 수 없어, TEI 이미지로 `ls`만 하는 일회용 컨테이너를
    띄운다(이미지는 이미 받아 둔 것을 쓴다). 실패하면 RuntimeError — revision 없이 실행하지 않는다.
    """
    folder = "models--" + model_id.replace("/", "--")
    try:
        out = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "ls",
                "-v",
                f"{volume}:/data:ro",
                image,
                f"/data/{folder}/snapshots",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{model_id}의 스냅샷 revision을 읽지 못했다: {exc}") from exc
    return parse_snapshot_listing(model_id, out)


def volume_revision_reader(
    config: ClientsConfig, compose_file: Path = DEFAULT_COMPOSE_FILE
) -> Callable[[str], str]:
    """TEI 모델 볼륨의 snapshots/에서 revision을 읽는 함수(모델 ID → 해시)를 돌려준다.

    `docker run --rm`으로 `ls`만 하는 일회용 컨테이너를 띄운다(볼륨은 읽기 전용으로 붙인다).
    compose 파일에서 TEI 이미지를 못 찾으면 호출할 때 ValueError를 낸다.
    """

    def read(model_id: str) -> str:
        image = tei_image_tag(compose_file)
        if image is None:
            raise ValueError(f"{compose_file}에서 TEI 이미지를 찾지 못해 revision을 읽을 수 없다")
        return read_snapshot_revision(model_id, volume=config.tei_models_volume, image=image)

    return read
