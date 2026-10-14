"""Local LLM bilingual incident reporter.

Stage I: template-based + optional Qwen2.5-0.5B via llama.cpp if weights present.
Stage II: deterministic Qwen2.5-0.5B INT8 running on host CPU alongside the NPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Report:
    zh: str
    en: str
    severity: str

    def pretty(self) -> str:
        return (
            f"[severity: {self.severity}]\n"
            f"[zh] {self.zh}\n[en] {self.en}"
        )


class BilingualReporter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._llm = None
        model_path = cfg.get("model_path")
        if model_path and Path(model_path).exists():
            try:
                from llama_cpp import Llama  # type: ignore

                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=cfg.get("n_ctx", 1024),
                    n_threads=cfg.get("n_threads", 4),
                    verbose=False,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[reporter] LLM load failed — falling back to template: {exc}")

    async def describe(self, event, frame) -> Report:
        if self._llm is None:
            return self._template_report(event)
        return self._llm_report(event)

    @staticmethod
    def _template_report(event) -> Report:
        """Deterministic fallback used when no LLM weights are loaded.

        Wording is intentionally rule-agnostic — the rule-engine supplies the
        structured fields (kind, zone, camera, dwell_frames), the template
        only phrases them. Downstream consumers that need PPE-specific or
        helmet-specific wording should either load the LLM path or extend
        this template with per-kind branches.
        """
        kind_zh = {
            "HELMET_MANDATORY": "未配戴安全帽",
            "REFLECTIVE_VEST_MANDATORY": "未穿反光背心",
            "RESTRICTED_ACCESS": "進入限制區域",
            "LONE_WORKER_FORBIDDEN": "單人作業",
        }.get(event.kind, "違規事件")
        kind_en = event.kind.replace("_", " ").lower()
        zh = (
            f"[{event.camera_id} {event.zone}] 偵測到人員{kind_zh}, "
            f"持續 {event.dwell_frames} 幀。建議通知現場主管複查。"
        )
        en = (
            f"{kind_en.capitalize()} detected in {event.zone} on "
            f"{event.camera_id} after {event.dwell_frames} frames of dwell. "
            f"Supervisor review recommended."
        )
        return Report(zh=zh, en=en, severity="HIGH")

    def _llm_report(self, event) -> Report:
        prompt = (
            "You are an industrial safety reporter. Write a 1-sentence incident "
            "summary in Traditional Chinese, then a 1-sentence English summary, "
            "then output severity (LOW/MED/HIGH).\n\n"
            f"Event JSON: kind={event.kind} camera={event.camera_id} "
            f"zone={event.zone} dwell_frames={event.dwell_frames}\n"
            "Format strictly as:\nZH: ...\nEN: ...\nSEV: ..."
        )
        resp = self._llm(prompt, max_tokens=256, temperature=0.3)
        text = resp["choices"][0]["text"]
        zh = _extract(text, "ZH:") or ""
        en = _extract(text, "EN:") or ""
        sev = _extract(text, "SEV:") or "MED"
        return Report(zh=zh.strip(), en=en.strip(), severity=sev.strip())


def _extract(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith(key):
            return line.split(key, 1)[1].strip()
    return None
