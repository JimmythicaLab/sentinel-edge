"""Unit tests for the rule engine.

Run:
    pytest tests/
"""
from __future__ import annotations

from dataclasses import dataclass

from src.rule_engine import RuleEngine, Zone


@dataclass
class _Det:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


# ---------------------------------------------------------------------------
#  Geometry primitives
# ---------------------------------------------------------------------------
def test_point_in_polygon_square():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert RuleEngine._point_in_polygon((0.5, 0.5), square)
    assert not RuleEngine._point_in_polygon((1.5, 0.5), square)
    assert not RuleEngine._point_in_polygon((-0.1, 0.5), square)


def test_point_in_polygon_triangle():
    tri = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    assert RuleEngine._point_in_polygon((0.5, 0.4), tri)
    assert not RuleEngine._point_in_polygon((0.5, 1.5), tri)


# ---------------------------------------------------------------------------
#  Helmet-mandatory rule
# ---------------------------------------------------------------------------
def _zones():
    return [{
        "name": "crane_zone",
        "camera": "CAM-42",
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "rules": ["helmet_mandatory"],
    }]


def _policies(min_dwell: int = 2):
    return {"min_dwell_frames": min_dwell}


def test_helmet_missing_fires_after_dwell():
    engine = RuleEngine(_zones(), _policies(min_dwell=2))
    person = _Det("person", 0.9, (0.4, 0.1, 0.6, 0.9), track_id=1)

    assert list(engine.evaluate(0, [person])) == []
    events = list(engine.evaluate(1, [person]))
    assert len(events) == 1
    assert events[0].kind == "HELMET_MANDATORY"
    assert events[0].camera_id == "CAM-42"
    assert events[0].zone == "crane_zone"


def test_helmet_present_suppresses_event():
    engine = RuleEngine(_zones(), _policies(min_dwell=2))
    person = _Det("person", 0.9, (0.4, 0.1, 0.6, 0.9), track_id=1)
    # Helmet lands in the top 25% head region
    helmet = _Det("helmet", 0.8, (0.45, 0.12, 0.55, 0.20))

    assert list(engine.evaluate(0, [person, helmet])) == []
    assert list(engine.evaluate(1, [person, helmet])) == []


def test_person_outside_zone_does_not_trigger():
    engine = RuleEngine(_zones(), _policies(min_dwell=1))
    # person centre at (2.0, 0.5) — well outside the (0..1) zone
    person = _Det("person", 0.9, (1.9, 0.1, 2.1, 0.9), track_id=1)
    assert list(engine.evaluate(0, [person])) == []


# ---------------------------------------------------------------------------
#  Zone dataclass
# ---------------------------------------------------------------------------
def test_zone_from_dict_coerces_tuples():
    z = Zone.from_dict({
        "name": "z1",
        "camera": "CAM-1",
        "polygon": [[0, 0], [1, 0], [1, 1]],
        "rules": ["restricted_access"],
    })
    assert isinstance(z.polygon[0], tuple)
    assert z.rules == ["restricted_access"]
