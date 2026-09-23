"""토큰화 단위 테스트 (이슈 #13). 네트워크 없음."""

from __future__ import annotations

import pytest

from rfp_pm_agent.search.tokenize import get_tokenizer, tokenize_bigram, tokenize_whitespace


def test_bigram은_공백_없는_문자열도_두_글자씩_자른다():
    # PDF 추출 텍스트에서 실제로 나오는 형태. 띄어쓰기에 기대면 통째로 남는다
    tokens = tokenize_bigram("안정화1개월")
    assert tokens == ["안정", "정화", "화1", "1개", "개월"]


def test_bigram은_기호와_공백을_지운_뒤_자른다():
    assert tokenize_bigram("□사업 기간:") == ["사업", "업기", "기간"]


def test_bigram은_한_글자와_빈_입력에서_예외를_내지_않는다():
    assert tokenize_bigram("가") == ["가"]
    assert tokenize_bigram("") == []
    assert tokenize_bigram("□·|  ") == []


def test_whitespace는_연속_공백과_앞뒤_공백에서_빈_토큰을_만들지_않는다():
    assert tokenize_whitespace("  사업  기간\n3개월\t") == ["사업", "기간", "3개월"]
    assert tokenize_whitespace("   ") == []


def test_whitespace는_요구사항_번호를_통째로_남긴다():
    assert "ECR-003" in tokenize_whitespace("요구사항 고유번호: ECR-003")


def test_get_tokenizer는_이름을_함수로_바꾼다():
    assert get_tokenizer("bigram") is tokenize_bigram
    assert get_tokenizer("whitespace") is tokenize_whitespace


def test_get_tokenizer는_모르는_이름에서_멈춘다():
    # 오타가 기본값으로 넘어가면 runs.jsonl의 tokenizer 값과 실제가 어긋난다
    with pytest.raises(ValueError, match="모르는 토큰화 방식"):
        get_tokenizer("morpheme")
