"""토큰화 (이슈 #13 평가용) — BM25에 넣을 토큰을 만든다.

대상 데이터는 PDF·HWP에서 뽑은 텍스트라 띄어쓰기가 자주 깨져 있다
(예: "□사업기간: 계약일로부터3개월이내(안정화1개월포함)"). 그래서 띄어쓰기에
기대는 토큰화는 "사업 기간" 같은 질문을 맞히지 못한다.

주 방식은 bigram이다. 평가 세트의 정답 판정 규칙 자체가 "공백을 모두 지운 뒤
포함 여부"라서, 공백을 신뢰하지 않는 bigram이 그 전제와 일관된다. whitespace는
bigram이 정말 이득인지 대조하려고 함께 둔다.

형태소 분석기는 여기 없다. 새 의존성이 필요하고, 위처럼 띄어쓰기가 깨진
문자열에서는 분석 품질이 떨어진다. 실제 서비스는 OpenSearch nori(형태소)로
가므로 언젠가 측정해야 하지만, 이번 비교의 대상은 청킹 방식이지 토큰화가 아니다.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Tokenizer = Callable[[str], list[str]]

# 한글(음절·자모)·영문·숫자만 남긴다. 나머지(공백, "□", "·", "|", 문장부호)는
# 문서마다 들쭉날쭉해서 토큰 경계로 삼으면 같은 표현이 다르게 잘린다.
_KEEP = re.compile(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+")


def tokenize_whitespace(text: str) -> list[str]:
    """공백으로 자른다. 빈 토큰은 남기지 않는다.

    "ECR-003" 같은 요구사항 번호가 통째로 보존되는 것이 이 방식의 장점이고,
    "계약일로부터3개월이내"가 한 덩어리로 남는 것이 단점이다.
    """
    return text.split()


def tokenize_bigram(text: str) -> list[str]:
    """공백과 기호를 지운 뒤 인접한 두 글자씩 겹쳐 자른다.

    "사업기간" → ["사업", "업기", "기간"]. 한 글자만 남으면 그 글자 하나를
    토큰으로 돌려준다. 남는 글자가 없으면 빈 목록이다.
    """
    cleaned = _KEEP.sub("", text)
    if len(cleaned) <= 1:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


TOKENIZERS: dict[str, Tokenizer] = {
    "bigram": tokenize_bigram,
    "whitespace": tokenize_whitespace,
}


def get_tokenizer(name: str) -> Tokenizer:
    """이름을 토큰화 함수로 바꾼다. 모르는 이름이면 ValueError를 낸다.

    실행 인자와 runs.jsonl의 tokenizer 값을 잇는 지점이라, 오타가 들어오면
    기본값으로 넘어가지 않고 그 자리에서 멈춘다.
    """
    try:
        return TOKENIZERS[name]
    except KeyError:
        known = ", ".join(sorted(TOKENIZERS))
        raise ValueError(f"모르는 토큰화 방식: {name!r} (가능한 값: {known})") from None
