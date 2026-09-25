"""평가 스키마 (이슈 #13) — 청킹 방식 비교 실행에 쓰는 레코드.

data/eval/qa_v1.jsonl 한 줄이 EvalQuestion 하나, data/eval/runs.jsonl 한 줄이
RunRecord 하나다. runs.jsonl은 방식 × 토큰화 조합마다 한 줄씩 쌓인다.

점수를 "전체 / 요구사항 / 일반"으로 나눠 내는 이유는 방식마다 다룰 수 있는
질문 유형이 다르기 때문이다. requirement 방식은 요구사항 정의표만 청크로
만들어서 일반 유형 질문은 원리상 맞힐 수 없다. 전체 점수만 보면 이 방식이
실제 성능보다 낮게 보인다.

EvalQuestionV2는 data/eval/qa_v2.jsonl 한 줄이다 (이슈 #15). qa_v1도 읽을 수 있다.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, field_validator, model_validator

QuestionType = Literal["요구사항", "일반"]
TokenizerName = Literal["bigram", "whitespace"]

# v2 문항 유형 태그 (docs/data-design.md 9절). 한 문항에 여러 개가 붙을 수 있다
QuestionTag = Literal[
    "paraphrase",
    "multi_chunk",
    "table",
    "no_answer",
    "exact",
    "doc_unspecified",
    "false_premise",  # 문서에 없는 전제를 깔고 묻지만, 문서의 제약으로 답할 수 있는 질문
]


class EvalQuestion(BaseModel):
    """평가 질문 하나. evidence는 사람이 원문에서 그대로 따온 문자열이다."""

    question_id: str
    question: str
    answer: str
    evidence: str
    doc_id: str
    page: int | None = None
    type: QuestionType


class EvalQuestionV2(BaseModel):
    """v2 평가 질문 하나 (이슈 #15, data/eval/qa_v2.jsonl).

    원천 정답은 evidence(원문 문자열 목록)다. 여러 청크 문항은 evidence가 2개
    이상이고, 답 없음 문항은 빈 목록이다. qa_v1 한 줄도 그대로 읽힌다 —
    문자열 evidence는 길이 1 목록으로 바꾸고, 나머지 새 필드는 기본값을 쓴다.

    gold_chunk_ids는 청크 파일에서 코드로 뽑은 파생값이다. 키는 청킹 방식이고,
    값은 evidence 순서에 맞춘 "그 evidence를 담은 청크 ID 목록"의 목록이다.
    evidence 하나를 여러 청크가 담을 수 있어서(예: q021) 평평한 목록으로 두면
    어느 evidence를 맞혔는지 구별할 수 없다.
    """

    question_id: str
    question: str
    answer: str | None = None
    evidence: list[str]
    doc_id: str  # 답 없음 문항도 질문이 겨냥한 문서
    page: int | None = None
    type: QuestionType | None = None  # v1 필드. 답 없음 문항은 None
    tags: list[QuestionTag] = []
    gold_chunk_ids: dict[str, list[list[str]]] = {}
    verified_by: str | None = None
    verified_at: str | None = None
    note: str = ""

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence_str_to_list(cls, value: object) -> object:
        # qa_v1의 evidence는 문자열 하나다
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _no_answer_consistent(self) -> Self:
        """no_answer 태그 ⇔ evidence가 빈 목록 ⇔ answer가 None. 셋 중 하나만 어긋나도 거부한다."""
        flags = {
            "no_answer 태그": "no_answer" in self.tags,
            "evidence == []": not self.evidence,
            "answer is None": self.answer is None,
        }
        if len(set(flags.values())) != 1:
            detail = ", ".join(f"{k}={v}" for k, v in flags.items())
            raise ValueError(f"{self.question_id}: 답 없음 표시가 서로 맞지 않는다 ({detail})")
        if any(len(per) != len(self.evidence) for per in self.gold_chunk_ids.values()):
            raise ValueError(f"{self.question_id}: gold_chunk_ids의 길이가 evidence 개수와 다르다")
        return self


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
