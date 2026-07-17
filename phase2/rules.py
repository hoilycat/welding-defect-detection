from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleResult:
    defect_type: str
    display_name: str
    risk_score: int
    action: str
    likely_cause: str
    reason: str


DEFECT_RULES: dict[str, RuleResult] = {
    "crack": RuleResult(
        defect_type="crack",
        display_name="균열 (Crack)",
        risk_score=100,
        action="즉시 재작업",
        likely_cause="급냉, 잔류 응력, 재료 취성 가능성",
        reason="길고 얇은 선형 결함은 구조적 신뢰도를 크게 낮출 수 있다.",
    ),
    "lack of fusion": RuleResult(
        defect_type="lack of fusion",
        display_name="용입불량 (Lack of Fusion)",
        risk_score=80,
        action="재검사 또는 보수 용접",
        likely_cause="전류 부족, 용접 속도 과다, 개선각 부족 가능성",
        reason="길게 이어지는 어두운 띠는 용접부 결합 강도 저하를 의미할 수 있다.",
    ),
    "fusion": RuleResult(
        defect_type="lack of fusion",
        display_name="용입불량 (Lack of Fusion)",
        risk_score=80,
        action="재검사 또는 보수 용접",
        likely_cause="전류 부족, 용접 속도 과다, 개선각 부족 가능성",
        reason="길게 이어지는 어두운 띠는 용접부 결합 강도 저하를 의미할 수 있다.",
    ),
    "undercut": RuleResult(
        defect_type="undercut",
        display_name="언더컷 (Undercut)",
        risk_score=60,
        action="보수 용접",
        likely_cause="전류 과다, 아크 길이 과다, 운봉 불안정 가능성",
        reason="가장자리 결함은 용접 토우부를 약화시킬 수 있다.",
    ),
    "porosity": RuleResult(
        defect_type="porosity",
        display_name="기공 (Porosity)",
        risk_score=50,
        action="주의 관찰, 밀집 시 재검사",
        likely_cause="가스 배출 불량, 표면 오염, 습기 가능성",
        reason="작고 둥근 어두운 결함은 단독으로는 낮은 위험일 수 있으나, 밀집하면 위험도가 증가한다.",
    ),
    "slag inclusion": RuleResult(
        defect_type="slag inclusion",
        display_name="슬래그 혼입 (Slag Inclusion)",
        risk_score=40,
        action="크기와 위치에 따라 추가 확인 또는 경미 보수",
        likely_cause="슬래그 제거 부족, 용접 각도 문제, 용접 속도 문제 가능성",
        reason="불규칙한 어두운 혼입물은 용접 금속 내부에 이물질이 남은 상태일 수 있다.",
    ),
    "candidate": RuleResult(
        defect_type="candidate",
        display_name="결함 후보 (Candidate)",
        risk_score=30,
        action="전처리 화면 확인 후 학습 모델로 재확인",
        likely_cause="고전 영상처리에서 어두운 후보 영역으로 검출됨",
        reason="YOLO 모델이 제공되지 않은 상태이므로 최종 분류가 아닌 시각적 후보로 판단한다.",
    ),
}


def normalize_defect_name(name: str | None) -> str:
    if not name:
        return "candidate"
    lowered = name.strip().lower().replace("_", " ")
    aliases = {
        "lp": "lack of fusion",
        "lack of penetration": "lack of fusion",
        "porosity": "porosity",
        "pore": "porosity",
        "slag": "slag inclusion",
        "slag inclusion": "slag inclusion",
        "crack": "crack",
        "undercut": "undercut",
    }
    return aliases.get(lowered, lowered)


def explain_detection(defect_name: str | None) -> RuleResult:
    normalized = normalize_defect_name(defect_name)
    return DEFECT_RULES.get(
        normalized,
        RuleResult(
            defect_type=normalized,
            display_name=f"{normalized} (Unknown)",
            risk_score=35,
            action="검사자 확인 필요",
            likely_cause="현재 룰 테이블에 등록되지 않은 결함명",
            reason="추가 라벨 데이터와 룰 정의가 필요하다.",
        ),
    )
