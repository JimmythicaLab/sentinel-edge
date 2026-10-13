"""Zone / PPE / dwell rule engine.

Policy lives in YAML so EHS staff can tune without touching code or
waiting on a vendor rebuild. Geometric zones are stored as polygons
in normalised (0..1) image coordinates so a config survives a camera
resolution change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Event:
    kind: str
    camera_id: str
    zone: str
    detections: list
    dwell_frames: int
    severity: str = "MED"

    def summary(self) -> str:
        return (
            f"camera={self.camera_id} zone={self.zone} kind={self.kind} "
            f"dwell_frames={self.dwell_frames} dets={len(self.detections)} "
            f"severity={self.severity}"
        )


@dataclass
class Zone:
    name: str
    camera: str
    polygon: list[tuple[float, float]]
    rules: list[str]
    severity: str = "MED"

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        return cls(
            name=d["name"],
            camera=d["camera"],
            polygon=[tuple(p) for p in d["polygon"]],
            rules=list(d.get("rules", [])),
            severity=d.get("severity", "MED"),
        )


class RuleEngine:
    """Zone + PPE + dwell rule evaluator.

    Supports multiple rules per zone:
      - helmet_mandatory         : person inside zone must have a helmet overlap
      - reflective_vest_mandatory: person must have a vest overlap
      - restricted_access        : no person allowed in zone at all
      - lone_worker_forbidden    : >=1 person triggers (pair-check deferred)
    """

    _SEVERITY_RANK = {"LOW": 1, "MED": 2, "HIGH": 3}

    def __init__(self, zones: list[dict], policies: dict) -> None:
        self.zones = [Zone.from_dict(z) for z in zones]
        self.policies = policies
        self._dwell: dict[tuple[str, int, str], int] = {}

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def evaluate(self, frame_idx: int, detections: list) -> Iterable[Event]:
        people = [d for d in detections if d.cls == "person"]
        helmets = [d for d in detections if d.cls in ("helmet", "hard_hat")]
        vests = [d for d in detections if d.cls in ("vest", "reflective_vest")]

        events: list[Event] = []
        min_dwell = int(self.policies.get("min_dwell_frames", 15))

        for zone in self.zones:
            persons_in_zone = [p for p in people if self._point_in_polygon(
                self._bbox_center(p.xyxy), zone.polygon
            )]

            for rule in zone.rules:
                violators = self._violators(rule, persons_in_zone, helmets, vests)
                for p in violators:
                    key = (zone.name, p.track_id or -1, rule)
                    self._dwell[key] = self._dwell.get(key, 0) + 1
                    if self._dwell[key] >= min_dwell:
                        events.append(
                            Event(
                                kind=rule.upper(),
                                camera_id=zone.camera,
                                zone=zone.name,
                                detections=[p],
                                dwell_frames=self._dwell[key],
                                severity=zone.severity,
                            )
                        )
                        # Reset dwell to debounce — one event per violation window
                        self._dwell[key] = 0

                # Decay counters for persons that are no longer in the zone
                stale = [
                    k for k in self._dwell
                    if k[0] == zone.name and k[2] == rule and
                    k[1] not in {p.track_id or -1 for p in violators}
                ]
                for k in stale:
                    self._dwell[k] = max(0, self._dwell[k] - 2)
                    if self._dwell[k] == 0:
                        del self._dwell[k]

        return events

    # ------------------------------------------------------------------
    #  Rule dispatch
    # ------------------------------------------------------------------
    def _violators(self, rule, persons_in_zone, helmets, vests) -> list:
        if rule == "helmet_mandatory":
            return [p for p in persons_in_zone if not self._has_overlap_on_head(p, helmets)]
        if rule == "reflective_vest_mandatory":
            return [p for p in persons_in_zone if not self._has_overlap_on_torso(p, vests)]
        if rule == "restricted_access":
            return list(persons_in_zone)
        if rule == "lone_worker_forbidden":
            return list(persons_in_zone) if len(persons_in_zone) == 1 else []
        # Unknown rule — soft-fail
        return []

    # ------------------------------------------------------------------
    #  Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _point_in_polygon(
        pt: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        """Ray-casting point-in-polygon. Polygon in normalised (0..1) coords.
        Detection bbox assumed in normalised coords after upstream rescale.
        """
        x, y = pt
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _has_overlap_on_head(person, helmets) -> bool:
        px1, py1, px2, py2 = person.xyxy
        head_y_max = py1 + (py2 - py1) * 0.25
        for h in helmets:
            hx1, hy1, hx2, hy2 = h.xyxy
            cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            if px1 <= cx <= px2 and py1 <= cy <= head_y_max:
                return True
        return False

    @staticmethod
    def _has_overlap_on_torso(person, vests) -> bool:
        px1, py1, px2, py2 = person.xyxy
        torso_y_min = py1 + (py2 - py1) * 0.20
        torso_y_max = py1 + (py2 - py1) * 0.70
        for v in vests:
            vx1, vy1, vx2, vy2 = v.xyxy
            cx, cy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
            if px1 <= cx <= px2 and torso_y_min <= cy <= torso_y_max:
                return True
        return False
