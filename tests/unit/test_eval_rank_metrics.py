"""reciprocal_rank·ndcg_at_k 테스트 (이슈 #16).

함수 본문은 소유자가 직접 구현했다(학습 대상, 2026-09-26). 기대값은 구현 전에
docstring의 정의로 손으로 계산해 둔 값이다. 구현 결과에 맞춰 바꾸지 않는다.

정의 요약 (retrieval.py docstring):
- gold_groups: evidence별 정답 청크 ID 집합의 목록
- RR: 처음으로 어느 묶음이든 적중한 순위의 역수, 상위 k 안에 없으면 0
- NDCG gain: 청크가 새로 덮은 묶음 수. 이미 덮은 묶음의 중복 청크는 0
- DCG = Σ gain_r / log2(r + 1), IDCG는 같은 정의의 이상적 순서
"""

from __future__ import annotations

import math

import pytest

from rfp_pm_agent.eval.retrieval import ndcg_at_k, reciprocal_rank

# 묶음 2개일 때의 IDCG = 1/log2(2) + 1/log2(3)
IDCG_TWO = 1.0 + 1.0 / math.log2(3)  # 1.6309297535714575


def test_1_묶음_1개_1위_적중() -> None:
    groups = [{"a"}]
    ranked = ["a", "x", "y"]
    assert reciprocal_rank(ranked, groups) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(1.0)


def test_2_묶음_1개_3위_적중() -> None:
    groups = [{"a"}]
    ranked = ["x", "y", "a"]
    assert reciprocal_rank(ranked, groups) == pytest.approx(1 / 3)
    # DCG = 1/log2(4) = 0.5, IDCG = 1
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(0.5)


def test_3_k_안에_적중_없음() -> None:
    groups = [{"a"}]
    ranked = ["x", "y", "z"]
    assert reciprocal_rank(ranked, groups, 10) == pytest.approx(0.0)
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(0.0)


def test_4_q021형_같은_묶음의_청크가_1_2위여도_ndcg는_1을_넘지_않는다() -> None:
    groups = [{"a", "b", "c", "d", "e"}]
    ranked = ["a", "b", "x"]
    assert reciprocal_rank(ranked, groups) == pytest.approx(1.0)
    # 2위 b는 이미 덮은 묶음이라 gain 0 → DCG = 1, IDCG = 1
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(1.0)


def test_5_묶음_2개가_1위와_3위() -> None:
    groups = [{"a"}, {"b"}]
    ranked = ["a", "x", "b"]
    assert reciprocal_rank(ranked, groups) == pytest.approx(1.0)
    # DCG = 1/log2(2) + 1/log2(4) = 1.5
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(1.5 / IDCG_TWO)  # ≈ 0.919721


def test_6_묶음_2개_중_1개만_1위() -> None:
    groups = [{"a"}, {"b"}]
    ranked = ["a", "x", "y"]
    assert reciprocal_rank(ranked, groups) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(1.0 / IDCG_TWO)  # ≈ 0.613147


def test_7_q038형_청크_하나가_두_묶음을_함께_덮는다() -> None:
    # evidence 2개가 같은 청크 s에 있다
    groups = [{"s"}, {"s"}]
    ranked = ["x", "s"]
    assert reciprocal_rank(ranked, groups) == pytest.approx(0.5)
    # 2위 s가 두 묶음을 새로 덮어 gain 2 → DCG = 2/log2(3)
    # 이상적 순서는 s가 1위 → IDCG = 2/log2(2) = 2
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx((2 / math.log2(3)) / 2)  # ≈ 0.630930


def test_8_결과가_k개보다_적어도_계산된다() -> None:
    groups = [{"a"}]
    ranked = ["x", "a"]
    assert reciprocal_rank(ranked, groups, 10) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(1 / math.log2(3))  # ≈ 0.630930


def test_9_gold_groups가_비면_ValueError() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], [])
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], [], 10)


def test_10_k가_10이면_11위_적중은_무시한다() -> None:
    groups = [{"a"}]
    ranked = [f"x{i}" for i in range(10)] + ["a"]  # a는 11위
    assert reciprocal_rank(ranked, groups, 10) == pytest.approx(0.0)
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(0.0)
    # k=None이면 전체를 본다
    assert reciprocal_rank(ranked, groups) == pytest.approx(1 / 11)


def test_11_묶음이_겹쳐도_ndcg는_1을_넘지_않는다() -> None:
    # p가 0·2번, q가 1·3번 묶음을 덮는다 → [p, q]가 최적 순서
    groups = [{"x", "p"}, {"x", "q"}, {"p"}, {"q"}]
    ranked = ["p", "q"]
    assert ndcg_at_k(ranked, groups, 10) == pytest.approx(1.0)


def test_12_모든_묶음이_비면_0() -> None:
    # 어떤 evidence도 청크에 담기지 않은 문항 (도달 불가)
    assert ndcg_at_k(["a"], [set()], 10) == pytest.approx(0.0)
