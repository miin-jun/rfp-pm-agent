from pydantic import BaseModel


class RtmResult(BaseModel):
    """요구사항-업무 연결 결과."""

    links: dict[str, list[tuple[str, float]]]
    missing: list[str]
    out_of_scope: list[str]


def build_rtm(
    requirement_ids: list[str],
    wbs_ids: list[str],
    scores: dict[tuple[str, str], float],
    threshold: float = 0.5,
) -> RtmResult:
    """요구사항별로 점수가 threshold 이상인 업무를 연결하고, 누락·범위 외 후보를 찾는다."""
    links: dict[str, list[tuple[str, float]]] = {}
    for req in requirement_ids:
        matched = []
        for wbs in wbs_ids:
            score = scores[(req, wbs)]
            if score >= threshold:
                matched.append((wbs, score))
        matched.sort(key=lambda x: x[1], reverse=True)
        links[req] = matched

    missing = [req for req in requirement_ids if not links[req]]

    linked_wbs = set()
    for matched in links.values():
        for wbs, _score in matched:
            linked_wbs.add(wbs)
    out_of_scope = [wbs for wbs in wbs_ids if wbs not in linked_wbs]

    return RtmResult(links=links, missing=missing, out_of_scope=out_of_scope)
