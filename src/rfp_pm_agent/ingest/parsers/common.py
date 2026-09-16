"""요구사항 코드 탐지·표 분류·행 파싱 공통 유틸 (HWP·PDF 파서가 함께 쓴다).

전부 docs/parsing-exploration.md "요구사항 정의표 구조" 절의 실측 근거를 따른다:

1. **코드는 라벨 문구가 아니라 셀 텍스트로 판정한다.** HWP의 SER-004/SER-005는
   라벨이 "요구사항 고유번호"가 아니라 "요구사항 교유번호"(오타)라, 라벨 문구
   매칭으로는 조용히 누락된다.
2. **접두어를 하드코딩하지 않는다.** 천안시 PDF는 인터페이스 요구사항에 흔히 쓰는
   `INR`이 아니라 `SIR`을 쓴다. `^[A-Z]{3}-\\d{3}$` 형태이면 접두어를 가리지 않고
   코드로 인정한다.
3. **정의표/요약표 판정은 "셀 하나에 코드가 여러 개 뭉쳐 있는지"로 한다.** 이슈 #11
   구현 초기 실측(2026-09-16)으로 드러난 세 가지 함정 때문에 라벨·열 수·표 전체
   코드 개수로는 판정할 수 없었다:
   - 천안시 목록표는 `"SFR-001\\nSFR-002\\n...\\nSFR-007"`처럼 코드를 한 셀에
     몰아 쓴다 → "코드 셀이 2개 이상이면 요약표"가 아예 발동하지 않았다
   - 그 목록표가 페이지 경계(23→24쪽)에서 쪼개지며 코드 1개짜리 조각(`QUR-001`
     단독, 4열)이 생겨, "코드 1개 = 정의표"로만 판정하면 가짜 정의표가 된다
   - 반대로 `QUR-003`은 페이지 끝에서 잘려 2열짜리 조각만 남는데, 열 수로
     판정하면 이것도 놓친다. `ECR-007`은 세부내용에 재질 규격 "SUS-316"이
     우연히 코드 패턴과 겹치는데, 표 전체 distinct 코드 개수로 판정하면 이것도
     정의표에서 제외돼 버린다
   → "셀 하나에 느슨한 코드가 2개 이상"이라는 신호 하나로 위 세 가지를 전부
   올바르게 가른다(정의표 71개, summary_ids 71개, 문서 자체 "합계 71"과 일치,
   중복 0건 — 실측 확인됨).
"""

from __future__ import annotations

import re

CODE_EXACT = re.compile(r"^[A-Z]{3}-\d{3}$")
CODE_LOOSE = re.compile(r"[A-Z]{3}-\d{3}")

# 실제 문서의 라벨 문구에서 도출 (docs/parsing-exploration.md). 매핑에 없는 라벨은
# raw_fields에는 그대로 남고 fields(표준 키)에는 들어가지 않는다 — 버리지 않는다.
LABEL_ALIASES: dict[str, str] = {
    "요구사항분류": "category",
    "요구사항고유번호": "id",
    "요구사항교유번호": "id",  # 실측 오타(SER-004, SER-005)
    "요구사항명칭": "name",
    "정의": "definition",
    "세부내용": "detail",
    "산출정보": "output",
    "관련요구사항": "related",
}


def normalize_label(text: str) -> str:
    """공백·개행 제거 (라벨·코드 셀 비교용)."""
    return re.sub(r"\s+", "", text)


def is_code(cell: str | None) -> bool:
    """셀 텍스트 전체(공백 제거)가 요구사항 코드 형태와 정확히 일치하는가."""
    if not cell:
        return False
    return bool(CODE_EXACT.match(normalize_label(cell)))


def cell_has_bundled_codes(cell: str | None) -> bool:
    """셀 하나에 느슨한 코드 패턴이 2개 이상 들어 있는가 (목록표 특징)."""
    if not cell:
        return False
    return len(CODE_LOOSE.findall(cell)) >= 2


def table_role(rows: list[list[str | None]]) -> tuple[str, str | None]:
    """표의 역할을 판정한다.

    반환값: ("definition", code) | ("summary", None) | ("other", None)
    """
    if any(cell_has_bundled_codes(c) for row in rows for c in row):
        return "summary", None
    exact_cells = [c for row in rows for c in row if c and is_code(c)]
    if len(exact_cells) == 1:
        return "definition", normalize_label(exact_cells[0])
    if len(exact_cells) >= 2:
        return "summary", None
    return "other", None


def collect_loose_codes(rows: list[list[str | None]]) -> set[str]:
    """표의 모든 셀에서 느슨하게 코드 문자열을 모은다 (요약표 코드 집합 수집용)."""
    codes: set[str] = set()
    for row in rows:
        for c in row:
            if c:
                codes.update(CODE_LOOSE.findall(c))
    return codes


def parse_row(row: list[str | None]) -> tuple[str, str] | None:
    """행 하나를 (원문 라벨, 값)으로 변환한다. 실측(HWP 220행·PDF 443행 전수
    확인)상 non-null 셀은 항상 2개 아니면 3개다. 3개면 마지막 두 개가
    (라벨, 값)이고 그 앞은 rowspan으로 묶인 그룹 라벨이라 무시한다. non-null이
    1개면 라벨 없이 이어지는 순수 텍스트 조각(페이지 경계로 잘린 부분)이라
    None을 반환 — 호출부가 직전 필드에 이어붙인다.

    non-null 2개는 보통 (라벨, 값)이지만, **값 자체가 비어 있는 행**(실측:
    PSR-003 전체 공란·QUR-003 페이지 조각)에서는 (그룹라벨, 서브라벨)만 남아
    "서브라벨을 값으로" 잘못 읽는다 — 두 번째 셀 자체가 이미 알려진 라벨(정의/
    세부내용 등)이면 그걸 라벨로, 값은 빈 문자열로 본다."""
    non_null = [c for c in row if c not in (None, "")]
    if len(non_null) >= 3:
        return non_null[-2], non_null[-1]
    if len(non_null) == 2:
        if normalize_label(non_null[1]) in LABEL_ALIASES:
            return non_null[1], ""
        return non_null[0], non_null[1]
    return None


class RequirementBuilder:
    """정의표 행들을 순서대로 넣으면 fields/raw_fields/text를 누적한다.

    페이지 조각으로 나뉜 정의표도 이 빌더에 행을 계속 넣는 것만으로 이어붙는다 —
    non-null 1개짜리 이어짐 행은 직전 라벨의 값 뒤에 텍스트를 붙이기만 하면 되기
    때문이다(parse_row 참고).
    """

    def __init__(self) -> None:
        self.fields: dict[str, str] = {}
        self.raw_fields: dict[str, str] = {}
        self._label_order: list[str] = []
        self._current_label: str | None = None

    def add_row(self, row: list[str | None]) -> None:
        parsed = parse_row(row)
        if parsed is None:
            non_null = [c for c in row if c not in (None, "")]
            if not non_null:
                return
            lone = non_null[0]
            # 실측(QUR-003, 문서상 값이 전부 빈 채로 페이지가 잘린 극단 사례):
            # 이 유일한 셀이 알려진 라벨이면 "값이 빈 새 필드"로 본다. 그래야
            # 라벨("요구사항 명칭")이 직전 필드(예: 고유번호)의 값 뒤에 잘못
            # 이어붙지 않는다. 라벨이 아니면 원래 규칙대로 순수 이어짐 텍스트다.
            if normalize_label(lone) in LABEL_ALIASES:
                self._set_field(lone, "")
                return
            if self._current_label is not None:
                self.raw_fields[self._current_label] = (
                    self.raw_fields.get(self._current_label, "") + "\n" + lone
                )
                key = LABEL_ALIASES.get(normalize_label(self._current_label))
                if key and key in self.fields:
                    self.fields[key] = self.fields[key] + "\n" + lone
            return

        label, value = parsed
        self._set_field(label, value)

    def _set_field(self, label: str, value: str) -> None:
        self.raw_fields[label] = value
        self._current_label = label
        if label not in self._label_order:
            self._label_order.append(label)
        key = LABEL_ALIASES.get(normalize_label(label))
        if key:
            self.fields[key] = value

    def build_text(self) -> str:
        return "\n".join(f"{label}: {self.raw_fields[label]}" for label in self._label_order)


DECLARED_TOTAL_LABEL = "합계"


def find_declared_total(tables: list[list[list[str | None]]]) -> tuple[int | None, list[str]]:
    """ "합계" 행(라벨, 값이 순수 정수)을 표 목록 전체에서 찾는다. 문서 전체
    자유 텍스트가 아니라 표 구조 안에서만 찾아, 예산 "합 계 …원" 같은 서식
    문자열은 애초에 순수 정수가 아니라서 걸러진다. 후보가 2개 이상이면 null +
    경고(어느 쪽이 맞는지 애매하기 때문)."""
    candidates: list[int] = []
    for matrix in tables:
        for row in matrix:
            parsed = parse_row(row)
            if parsed is None:
                continue
            label, value = parsed
            if normalize_label(label) != DECLARED_TOTAL_LABEL:
                continue
            stripped = value.strip()
            if stripped.isdigit():
                candidates.append(int(stripped))

    if len(candidates) == 0:
        return None, []
    if len(candidates) >= 2:
        return None, [
            f"declared_total 후보 {len(candidates)}개 발견 — 애매하여 null로 둠: {candidates}"
        ]
    return candidates[0], []


def validate_requirement_ids(
    requirement_ids: list[str],
    summary_ids: set[str],
    declared_total: int | None,
) -> list[str]:
    """정의표에서 뽑은 코드들이 요약표·문서 자체 합계와 맞는지 확인한다.
    불일치는 예외가 아니라 경고 문자열 목록으로 반환한다(호출부가
    Document.validation_warnings에 넣는다)."""
    warnings: list[str] = []

    dup_ids = {rid for rid in requirement_ids if requirement_ids.count(rid) > 1}
    if dup_ids:
        warnings.append(f"requirement_id 중복: {sorted(dup_ids)}")

    req_id_set = set(requirement_ids)
    if req_id_set != summary_ids:
        only_summary = sorted(summary_ids - req_id_set)
        only_requirements = sorted(req_id_set - summary_ids)
        warnings.append(
            f"요약표 코드({len(summary_ids)}) vs 정의표 코드({len(req_id_set)}) 불일치 — "
            f"요약표만: {only_summary}, 정의표만: {only_requirements}"
        )

    if declared_total is not None and declared_total != len(requirement_ids):
        warnings.append(
            f"declared_total({declared_total}) != 파싱된 요구사항 수({len(requirement_ids)})"
        )

    return warnings
