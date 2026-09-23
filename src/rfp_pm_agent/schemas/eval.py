"""평가 스키마 (이슈 #13) — 청킹 방식 비교 실행에 쓰는 레코드.

data/eval/qa_v1.jsonl 한 줄이 EvalQuestion 하나, data/eval/runs.jsonl 한 줄이
RunRecord 하나다. runs.jsonl은 방식 × 토큰화 조합마다 한 줄씩 쌓인다.

점수를 "전체 / 요구사항 / 일반"으로 나눠 내는 이유는 방식마다 다룰 수 있는
질문 유형이 다르기 때문이다. requirement 방식은 요구사항 정의표만 청크로
만들어서 일반 유형 질문은 원리상 맞힐 수 없다. 전체 점수만 보면 이 방식이
실제 성능보다 낮게 보인다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

QuestionType = Literal["요구사항", "일반"]
TokenizerName = Literal["bigram", "whitespace"]


class EvalQuestion(BaseModel):
    """평가 질문 하나. evidence는 사람이 원문에서 그대로 따온 문자열이다."""

    question_id: str
    question: str
    answer: str
    evidence: str
    doc_id: str
    page: int | None = None
    type: QuestionType


class SearchHit(BaseModel):
    """검색 결과 한 건. 점수 내림차순의 몇 번째인지는 목록 안 위치로 나타낸다."""

    chunk_id: str
    doc_id: str
    score: float
    text: str


class TypeScore(BaseModel):
    """한 유형(또는 전체)의 집계. 비율이 아니라 개수로 담아 합산이 검산된다.

    ceiling(최고 점수)은 검색과 무관하게, 정답 조건을 만족하는 청크가 그 방식의
    청크 전체에 하나라도 있는 질문 수다. hit@k는 ceiling을 넘을 수 없다.
    """

    total: int
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    ceiling: int


class RunScore(BaseModel):
    """한 번의 방식 × 토큰화 실행 결과."""

    overall: TypeScore
    requirement: TypeScore
    general: TypeScore
    # hit@5를 못 맞힌 질문. 상위 5개 청크는 failures 파일에 따로 남긴다
    missed_question_ids: list[str]


class RunRecord(BaseModel):
    """runs.jsonl 한 줄. 점수가 달라졌을 때 원인을 찾을 수 있도록 조건을 모두 적는다."""

    run_id: str  # "{실행시각}-{방식}-{토큰화}" — failures 파일 이름과 같다
    ran_at: str  # ISO 8601 (UTC)
    qa_file: str
    chunks_file: str
    method: str
    tokenizer: TokenizerName
    k1: float
    b: float
    top_k: int
    chunk_count: int
    scores: RunScore
    failures_file: str | None = None
