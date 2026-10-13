"""Reporter smoke tests — ensures the template path works without an LLM."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.reporter import BilingualReporter


@dataclass
class _Event:
    kind: str = "HELMET_MANDATORY"
    camera_id: str = "CAM-42"
    zone: str = "crane_zone"
    dwell_frames: int = 20


def test_template_fallback_without_llm():
    reporter = BilingualReporter({"model_path": None})
    event = _Event()
    report = asyncio.run(reporter.describe(event, frame=None))
    assert "CAM-42" in report.zh
    assert "CAM-42" in report.en
    assert report.severity == "HIGH"


def test_report_pretty_contains_both_languages():
    reporter = BilingualReporter({"model_path": None})
    report = asyncio.run(reporter.describe(_Event(), frame=None))
    pretty = report.pretty()
    assert "[zh]" in pretty
    assert "[en]" in pretty
