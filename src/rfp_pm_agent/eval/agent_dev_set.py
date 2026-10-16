"""에이전트 개발 평가 세트 `data/eval/agent_dev_v1.jsonl` 만들기·검사 (이슈 #22, ADR-0010 "평가").

30문항: 문서만 8 · PMS만 8 · 둘 다 6 · 사업 미지정 4 · 답 없음 4. 질문 문장과 측정 전 예상은 이 파일에 적고,
정답 ID는 사람이 옮겨 적지 않고 원천에서 만든다.

- 문서 문항: qa_v2(사람 검수)의 문항을 사업명이 드러나게 바꿔 쓴다. 판정 기준은 qa_v2 `answer` 그대로(`reference_answer`)
- PMS·둘 다 문항: `data/eval/pms_truth_v2.json`(pms/rules.py로 계산한 정답)의 지연 업무·미해결 Risk·기한 지난 미제출·RTM
- 답 없음: qa_v2의 답 없음(C) 2문항 + `irrelevant_v1.jsonl`(사람 검수) 2문항

data/eval/의 다른 파일은 고치지 않는다. 이 세트는 측정 전에 커밋하고, 측정 뒤에 고치지 않는다(CLAUDE.md 평가 규칙).

CLI: `uv run python -m rfp_pm_agent.eval.agent_dev_set build` / `... check`(파일이 원천과 같은지)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rfp_pm_agent.eval.qa_v2 import load_questions_v2
from rfp_pm_agent.schemas.eval import AgentDevQuestion, AgentExpectedFacts, IrrelevantQuestion

DEFAULT_OUT = Path("data/eval/agent_dev_v1.jsonl")
QA_FILE = Path("data/eval/qa_v2.jsonl")
TRUTH_FILE = Path("data/eval/pms_truth_v2.json")
IRRELEVANT_FILE = Path("data/eval/irrelevant_v1.jsonl")

# 문서만: (qid, qa_v2 문항, 바꿔 쓴 질문, 예상)
DOC_QUESTIONS = [
    (
        "d1",
        "q010",
        "AX를 통한 불균형 투자계약 점검 시스템 구축 사업에 쓸 수 있는 돈은 부가세를 포함해 얼마인가요?",
        "통과 예상. #19 전체 검색에서 다른 사업 금액을 답한 문항 — 사업 목록으로 doc_id를 정하면 문서 지정 조건과 같다",
    ),
    (
        "d2",
        "q001",
        "업무자동화(RPA) 시스템 고도화 사업의 수행 기간은 계약일을 기준으로 얼마나 되나요?",
        "통과 예상(#19 문서 지정 답함)",
    ),
    (
        "d3",
        "q005",
        "연구행정 데이터 기반 AI 플랫폼 구축 사업에서 사용자가 질문을 입력한 뒤 답변이 나오기까지 지켜야 할 속도 기준은 무엇인가요?",
        "통과 예상",
    ),
    (
        "d4",
        "q009",
        "국가 망 보안 프레임워크 전환 ISP 수립 사업에서 앞으로 갖출 망 구조를 설계할 때 따라야 하는 보안 아키텍처는 무엇인가요?",
        "통과 예상. PMS 없는 문서 — PMS 툴을 부르지 않아야 한다",
    ),
    ("d5", "q041", None, "통과 예상(exact, 요구사항 ID 가산)"),
    (
        "d6",
        "q017",
        "아산시 강소형 스마트시티 스마트폴 구축 용역에서 세울 기둥은 모두 몇 개이고 종류별로 어떻게 나뉘나요?",
        "위험: paraphrase(검색어가 '기둥'이고 원문은 '폴')",
    ),
    (
        "d7",
        "q021",
        "천안시 거점형 스마트도시 어반테크 실증존 구축 용역에서 기기에서 직접 돌아가는 인공지능이 잡아내야 하는 상황은 무엇인가요?",
        "위험: paraphrase('엣지 AI'를 풀어 쓴 질문)",
    ),
    ("d8", "q055", None, "통과 예상(exact)"),
]

# 답 없음(문서): qa_v2 C 문항. 질문에 이미 사업명이 있다
NO_ANSWER_DOC = [
    ("n1", "q045", "통과 예상(#19 두 조건 모두 거절)"),
    ("n2", "q056", "통과 예상(#19 두 조건 모두 거절)"),
]
NO_ANSWER_IRRELEVANT = [
    ("n3", "u001", "통과 예상. 툴을 부르지 않고 거절해야 한다"),
    ("n4", "u005", "위험: 파이썬 질문에 사전 지식으로 답할 수 있다(규칙 1·7)"),
]


def _project(truth: dict[str, Any], project_id: str) -> dict[str, Any]:
    project: dict[str, Any] = next(p for p in truth["projects"] if p["project_id"] == project_id)
    return project


def delayed(truth: dict[str, Any], project_id: str, *, this_week: bool) -> AgentExpectedFacts:
    tasks = [
        t for t in _project(truth, project_id)["delayed_tasks"] if t["this_week"] or not this_week
    ]
    return AgentExpectedFacts(
        wbs_ids=[t["wbs_id"] for t in tasks],
        stage_wbs_ids=[t["wbs_id"] for t in tasks if t["level"] == 1],
    )


def risks(truth: dict[str, Any], project_id: str, min_score: int) -> list[str]:
    return [
        r["risk_id"] for r in _project(truth, project_id)["open_risks"] if r["score"] >= min_score
    ]


def linked_requirements(truth: dict[str, Any], project_id: str, wbs_id: str) -> list[str]:
    return [
        r["requirement_id"] for r in _project(truth, project_id)["rtm"] if wbs_id in r["wbs_ids"]
    ]


def build_questions(
    qa_file: Path = QA_FILE, truth_file: Path = TRUTH_FILE, irrelevant_file: Path = IRRELEVANT_FILE
) -> list[AgentDevQuestion]:
    qa = {q.question_id: q for q in load_questions_v2(qa_file)}
    truth: dict[str, Any] = json.loads(truth_file.read_text("utf-8"))
    irrelevant = {
        q.question_id: q
        for q in (
            IrrelevantQuestion.model_validate_json(line)
            for line in irrelevant_file.read_text("utf-8").splitlines()
            if line.strip()
        )
    }
    project_by_doc = {p["doc_id"]: p["project_id"] for p in truth["projects"]}
    doc_by_project = {p["project_id"]: p["doc_id"] for p in truth["projects"]}
    questions: list[AgentDevQuestion] = []

    for qid, source_id, text, prediction in DOC_QUESTIONS:
        q = qa[source_id]
        questions.append(
            AgentDevQuestion(
                qid=qid,
                type="doc",
                turns=[text or q.question],
                doc_id=q.doc_id,
                project_id=project_by_doc.get(q.doc_id),
                expected_tools=["search_documents"],
                expected_facts=AgentExpectedFacts(),
                should_clarify=False,
                should_refuse=False,
                must_cite=True,
                reference_answer=q.answer,
                source=f"qa_v2:{source_id}",
                dod=qid == "d1",
                prediction=prediction,
            )
        )

    def pms(
        qid: str,
        project_id: str,
        text: str,
        tool: str,
        facts: AgentExpectedFacts,
        prediction: str,
        *,
        dod: bool = False,
    ) -> AgentDevQuestion:
        return AgentDevQuestion(
            qid=qid,
            type="pms",
            turns=[text],
            doc_id=doc_by_project[project_id],
            project_id=project_id,
            expected_tools=[tool],  # type: ignore[list-item]
            expected_facts=facts,
            should_clarify=False,
            should_refuse=False,
            must_cite=False,
            reference_answer=None,
            source=f"pms_truth_v2:{project_id}",
            dod=dod,
            prediction=prediction,
        )

    questions += [
        pms(
            "p1",
            "PRJ-03",
            "AX를 통한 불균형 투자계약 점검 시스템 구축 사업에서 이번 주에 지연된 업무 알려줘",
            "get_delayed_tasks",
            delayed(truth, "PRJ-03", this_week=True),
            "통과 예상",
            dod=True,
        ),
        pms(
            "p2",
            "PRJ-09",
            "한국나노기술원 모아팹 고도화 사업의 이번 주 지연 업무는 뭐야?",
            "get_delayed_tasks",
            delayed(truth, "PRJ-09", this_week=True),
            "위험: 단계(WBS 2)를 '단계 지연'으로 따로 적어야 한다. 사업명 일부만 적은 질문",
        ),
        pms(
            "p3",
            "PRJ-01",
            "업무자동화(RPA) 시스템 고도화 사업에서 지연된 업무를 모두 알려줘",
            "get_delayed_tasks",
            delayed(truth, "PRJ-01", this_week=False),
            "통과 예상(6건, 단계 1건)",
        ),
        pms(
            "p4",
            "PRJ-08",
            "잘피종자 자동파종 시스템 제작 용역의 미해결 Risk 중 high 등급은 뭐야?",
            "get_open_risks",
            AgentExpectedFacts(risk_ids=risks(truth, "PRJ-08", 15)),
            "통과 예상",
        ),
        pms(
            "p5",
            "PRJ-02",
            "연구행정 데이터 기반 AI 플랫폼 구축 사업의 미해결 Risk를 전부 보여줘",
            "get_open_risks",
            AgentExpectedFacts(risk_ids=risks(truth, "PRJ-02", 0)),
            "통과 예상",
        ),
        pms(
            "p6",
            "PRJ-04",
            "2027학년도 고등학교 입학전형 시스템 기능개선 용역에서 기한이 지났는데 아직 제출하지 않은 산출물은?",
            "get_deliverables",
            AgentExpectedFacts(
                deliverable_ids=_project(truth, "PRJ-04")["overdue_unsubmitted_deliverables"]
            ),
            "위험: 12건을 빠짐없이 적어야 한다",
        ),
        pms(
            "p7",
            "PRJ-05",
            "아산시 스마트폴 구축 용역에서 지연된 업무는 몇 건이야?",
            "get_delayed_tasks",
            AgentExpectedFacts(count=len(_project(truth, "PRJ-05")["delayed_tasks"])),
            "통과 예상(total 6)",
        ),
        pms(
            "p8",
            "PRJ-04",
            "2027학년도 고등학교 입학전형 시스템 기능개선 용역의 미해결 Risk 중 medium 이상은?",
            "get_open_risks",
            AgentExpectedFacts(risk_ids=risks(truth, "PRJ-04", 8)),
            "통과 예상",
        ),
    ]

    def both(
        qid: str,
        project_id: str,
        text: str,
        tool: str,
        facts: AgentExpectedFacts,
        prediction: str,
        *,
        dod: bool = False,
    ) -> AgentDevQuestion:
        return AgentDevQuestion(
            qid=qid,
            type="both",
            turns=[text],
            doc_id=doc_by_project[project_id],
            project_id=project_id,
            expected_tools=[tool, "search_documents"],  # type: ignore[list-item]
            expected_facts=facts,
            should_clarify=False,
            should_refuse=False,
            must_cite=True,
            reference_answer=None,
            source=f"pms_truth_v2:{project_id}",
            dod=dod,
            prediction=prediction,
        )

    def week_both(
        qid: str, project_id: str, text: str, prediction: str, *, dod: bool = False
    ) -> AgentDevQuestion:
        facts = delayed(truth, project_id, this_week=True)
        reqs = [r for w in facts.wbs_ids for r in linked_requirements(truth, project_id, w)]
        facts.requirement_ids = list(dict.fromkeys(reqs))
        return both(qid, project_id, text, "get_delayed_tasks", facts, prediction, dod=dod)

    cor_001 = next(r for r in _project(truth, "PRJ-03")["rtm"] if r["requirement_id"] == "COR-001")
    questions += [
        week_both(
            "b1",
            "PRJ-03",
            "AX를 통한 불균형 투자계약 점검 시스템 구축 사업에서 이번 주 지연된 업무랑 관련된 요구사항 문서 알려줘",
            "통과 예상(architecture.md 3절 예시)",
            dod=True,
        ),
        week_both(
            "b2",
            "PRJ-08",
            "잘피종자 자동파종 시스템 제작 용역에서 이번 주 지연된 업무가 RFP의 어떤 요구사항에 해당하는지 원문 근거와 함께 알려줘",
            "통과 예상",
        ),
        week_both(
            "b3",
            "PRJ-05",
            "아산시 스마트폴 구축 용역의 이번 주 지연 업무와, 그 업무에 연결된 요구사항을 RFP에서 찾아 줘",
            "위험: 요구사항 3개 — 검색을 한 번에 묶으면 일부만 찾을 수 있다",
        ),
        week_both(
            "b4",
            "PRJ-07",
            "2028 세계디자인수도(WDC) 부산 디지털 플랫폼 구축 용역에서 이번 주 지연된 업무와 관련 요구사항 내용을 알려줘",
            "위험: 요구사항 4개(QUR·SER·SFR 섞임)",
        ),
        both(
            "b5",
            "PRJ-02",
            "연구행정 데이터 기반 AI 플랫폼 구축 사업에서 점수가 가장 높은 미해결 Risk가 무엇이고, 관련된 RFP 요구 내용은 뭐야?",
            "get_open_risks",
            AgentExpectedFacts(risk_ids=risks(truth, "PRJ-02", 0)[:1]),
            "위험: Risk와 RFP를 잇는 ID가 없어 검색어를 Risk 제목에서 만들어야 한다",
        ),
        both(
            "b6",
            "PRJ-03",
            "AX 불균형 투자계약 점검 시스템 사업에서 COR-001 요구사항 관련 산출물 제출 상태와, RFP가 COR-001에서 요구하는 내용을 알려줘",
            "get_deliverables",
            AgentExpectedFacts(
                deliverable_ids=cor_001["deliverable_ids"], requirement_ids=["COR-001"]
            ),
            "통과 예상(docstring 예시와 같은 흐름)",
        ),
    ]

    def unspecified(qid: str, turns: list[str], prediction: str, **final: Any) -> AgentDevQuestion:
        return AgentDevQuestion(
            qid=qid,
            type="unspecified",
            turns=turns,
            doc_id=final.get("doc_id"),
            project_id=final.get("project_id"),
            expected_tools=final.get("expected_tools", []),
            expected_facts=final.get("facts", AgentExpectedFacts()),
            should_clarify=True,
            should_refuse=False,
            must_cite=False,
            reference_answer=None,
            source=final.get("source", "agent_dev_v1"),
            dod=False,
            prediction=prediction,
        )

    questions += [
        unspecified("u1", ["이번 주 지연된 업무 알려줘"], "통과 예상"),
        unspecified(
            "u2",
            ["이 사업의 수행 기간은 계약일 기준으로 얼마나 되나요?"],
            "위험: '이 사업'을 사업 목록 첫 사업으로 고를 수 있다(#19 전체 검색 실패 유형)",
        ),
        unspecified("u3", ["미해결 Risk 중에 high 등급인 게 있어?"], "통과 예상"),
        unspecified(
            "u4",
            ["기한이 지났는데 아직 안 낸 산출물 보여줘", "잘피종자 자동파종 시스템 사업이야"],
            "통과 예상. 둘째 턴은 Checkpointer로 첫 질문을 이어받아야 한다",
            doc_id=doc_by_project["PRJ-08"],
            project_id="PRJ-08",
            expected_tools=["get_deliverables"],
            facts=AgentExpectedFacts(
                deliverable_ids=_project(truth, "PRJ-08")["overdue_unsubmitted_deliverables"]
            ),
            source="pms_truth_v2:PRJ-08",
        ),
    ]

    for qid, source_id, prediction in NO_ANSWER_DOC:
        q = qa[source_id]
        questions.append(
            AgentDevQuestion(
                qid=qid,
                type="no_answer",
                turns=[q.question],
                doc_id=q.doc_id,
                project_id=project_by_doc.get(q.doc_id),
                expected_tools=["search_documents"],
                expected_facts=AgentExpectedFacts(),
                should_clarify=False,
                should_refuse=True,
                must_cite=False,
                reference_answer=None,
                source=f"qa_v2:{source_id}",
                dod=False,
                prediction=prediction,
            )
        )
    for qid, source_id, prediction in NO_ANSWER_IRRELEVANT:
        questions.append(
            AgentDevQuestion(
                qid=qid,
                type="no_answer",
                turns=[irrelevant[source_id].question],
                doc_id=None,
                project_id=None,
                expected_tools=[],
                expected_facts=AgentExpectedFacts(),
                should_clarify=False,
                should_refuse=True,
                must_cite=False,
                reference_answer=None,
                source=f"irrelevant_v1:{source_id}",
                dod=False,
                prediction=prediction,
            )
        )
    return questions


def to_jsonl(questions: Sequence[AgentDevQuestion]) -> str:
    return "".join(q.model_dump_json() + "\n" for q in questions)


def load_agent_questions(path: Path = DEFAULT_OUT) -> list[AgentDevQuestion]:
    return [
        AgentDevQuestion.model_validate_json(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    text = to_jsonl(build_questions())
    if args.command == "build":
        args.out.write_text(text, "utf-8")
        print(f"{args.out}: {text.count(chr(10))}문항")
        return 0
    same = args.out.read_text("utf-8") == text
    print("원천과 같다" if same else "원천과 다르다 — build로 다시 만들기 전에 이유를 확인한다")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
