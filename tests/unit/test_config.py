import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

from rfp_pm_agent.config import ClientsConfig, NaraApiConfig, OpenSearchConfig


def test_from_env_reads_base_urls_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://vllm.local:8000/v1")
    monkeypatch.setenv("LLM_MODEL_DEV", "gpt-test-dev")
    monkeypatch.setenv("EMBED_BASE_URL", "http://tei-embed.local:8080")
    monkeypatch.setenv("RERANK_BASE_URL", "http://tei-rerank.local:8081")

    config = ClientsConfig.from_env()

    assert config.llm_base_url == "http://vllm.local:8000/v1"
    assert config.llm_model_dev == "gpt-test-dev"
    assert config.embed_base_url == "http://tei-embed.local:8080"
    assert config.rerank_base_url == "http://tei-rerank.local:8081"


def test_from_env_defaults_prices_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PRICE_INPUT_PER_1M", raising=False)
    monkeypatch.delenv("LLM_PRICE_OUTPUT_PER_1M", raising=False)

    config = ClientsConfig.from_env()

    assert config.llm_price_input_per_1m == 0.0
    assert config.llm_price_output_per_1m == 0.0


def test_from_env_reads_per_1m_price(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenAI 요금 페이지 표기(100만 토큰당)를 그대로 옮겨 적는 값이라는 것을
    # 확인 — 1K로 착각해 1000배 오차가 나지 않는지가 핵심.
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1M", "3.0")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_1M", "15.0")

    config = ClientsConfig.from_env()

    assert config.llm_price_input_per_1m == 3.0
    assert config.llm_price_output_per_1m == 15.0


def test_from_env_missing_api_key_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ClientsConfig.from_env()

    assert config.llm_api_key is None


def test_nara_api_config_from_env_reads_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARA_API_KEY", "test-nara-key")

    config = NaraApiConfig.from_env()

    assert config.api_key == "test-nara-key"


def test_nara_api_config_from_env_default_key_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NARA_API_KEY", raising=False)

    config = NaraApiConfig.from_env()

    assert config.api_key == ""


def test_nara_api_config_warns_when_key_still_encoded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # '%'가 남아 있으면 포털의 "Encoding" 키를 그대로 넣은 것 — 이중 인코딩으로
    # 403이 나는 실제 사고(이슈 #10)의 재발 방지용 경고.
    monkeypatch.setenv("NARA_API_KEY", "abcd%2Bwxyz")

    with caplog.at_level(logging.WARNING):
        config = NaraApiConfig.from_env()

    assert config.api_key == "abcd%2Bwxyz"  # 값 자체는 그대로 통과시킴 (경고만)
    assert "NARA_API_KEY" in caplog.text
    assert "abcd%2Bwxyz" not in caplog.text  # 경고 메시지에 실제 키 값을 그대로 싣지 않음


def test_nara_api_config_no_warning_when_key_decoded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("NARA_API_KEY", "abcd+wxyz")

    with caplog.at_level(logging.WARNING):
        NaraApiConfig.from_env()

    assert caplog.text == ""


# --- 이슈 #10 검증 중 발견한 결함(.env 미로딩) 재발 방지: load_dotenv 계약 자체를 검증 ---
# config.py는 이 계약(override=False, 파일 없으면 예외 없이 통과)에 기대어 동작하므로,
# 여기서 실제 os.environ/임시 .env 파일로 계약을 직접 확인한다. 실제 레포의 .env는
# 절대 참조하지 않고 tmp_path에 만든 임시 파일의 dotenv_path만 명시적으로 사용한다
# — 그래야 이 테스트가 .env 존재 여부와 무관하게 통과한다.


def test_load_dotenv_does_not_override_existing_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NARA_API_KEY", "from-shell")
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("NARA_API_KEY=from-dotenv-should-not-apply\n", encoding="utf-8")

    load_dotenv(dotenv_path=dotenv_path, override=False)

    assert os.environ["NARA_API_KEY"] == "from-shell"


def test_load_dotenv_fills_in_when_not_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NARA_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("NARA_API_KEY=from-dotenv\n", encoding="utf-8")

    load_dotenv(dotenv_path=dotenv_path, override=False)

    assert os.environ.get("NARA_API_KEY") == "from-dotenv"
    monkeypatch.delenv("NARA_API_KEY", raising=False)  # 다음 테스트로 새지 않게 정리


def test_load_dotenv_missing_file_does_not_raise(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.env"

    result = load_dotenv(dotenv_path=missing_path, override=False)

    assert result is False


def test_from_env_reads_tei_max_client_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEI_MAX_CLIENT_BATCH_SIZE", "64")

    assert ClientsConfig.from_env().tei_max_client_batch_size == 64


@pytest.mark.parametrize("value", ["0", "-1"])
def test_from_env_rejects_non_positive_tei_max_client_batch_size(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # 0이면 range()가 에러, 음수면 요청 없이 빈 결과가 나오므로 설정 단계에서 막는다
    monkeypatch.setenv("TEI_MAX_CLIENT_BATCH_SIZE", value)

    with pytest.raises(ValidationError):
        ClientsConfig.from_env()


def test_from_env_reads_embed_prefix_and_truncate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("EMBED_PASSAGE_PREFIX", "passage: ")
    monkeypatch.setenv("EMBED_TRUNCATE", "true")

    config = ClientsConfig.from_env()

    assert config.embed_query_prefix == "query: "
    assert config.embed_passage_prefix == "passage: "
    assert config.embed_truncate is True


def test_from_env_embed_truncate_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_TRUNCATE", "maybe")

    with pytest.raises(ValueError, match="EMBED_TRUNCATE"):
        ClientsConfig.from_env()


def test_opensearch_config_reads_env_names_from_env_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # .env.example의 키 이름 그대로 읽는지 확인 (이슈 #17)
    monkeypatch.setenv("OPENSEARCH_URL", "http://opensearch.test:9200")
    monkeypatch.setenv("OPENSEARCH_INDEX_ALIAS", "alias_test")
    monkeypatch.setenv("OPENSEARCH_INDEX_NAME", "index_test")

    config = OpenSearchConfig.from_env()

    assert config.url == "http://opensearch.test:9200"
    assert config.index_alias == "alias_test"
    assert config.index_name == "index_test"


def test_opensearch_config_index_name_defaults_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # 인덱스 이름에는 모델 이름이 들어가므로 코드에 기본값을 두지 않는다
    monkeypatch.delenv("OPENSEARCH_INDEX_NAME", raising=False)

    assert OpenSearchConfig.from_env().index_name == ""
