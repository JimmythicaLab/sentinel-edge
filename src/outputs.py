"""Output sinks — MQTT, LINE Notify, Teams/Slack webhooks, siren GPIO."""
from __future__ import annotations

import asyncio
import json

import httpx


class OutputDispatcher:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._mqtt = self._init_mqtt() if cfg.get("mqtt", {}).get("enabled") else None

    def _init_mqtt(self):
        import paho.mqtt.client as mqtt  # type: ignore

        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        host = self.cfg["mqtt"].get("host", "localhost")
        port = self.cfg["mqtt"].get("port", 1883)
        c.connect(host, port, keepalive=30)
        c.loop_start()
        return c

    async def push(self, event, report, snapshot) -> None:
        payload = {
            "camera": event.camera_id,
            "zone": event.zone,
            "kind": event.kind,
            "severity": report.severity,
            "zh": report.zh,
            "en": report.en,
        }
        tasks: list = []
        if self._mqtt is not None:
            topic = self.cfg["mqtt"].get("topic", "sentinel/events")
            self._mqtt.publish(topic, json.dumps(payload, ensure_ascii=False))
        line_token = self.cfg.get("line", {}).get("token")
        if line_token:
            tasks.append(self._line_notify(line_token, report))
        teams_url = self.cfg.get("teams", {}).get("webhook")
        if teams_url:
            tasks.append(self._teams_webhook(teams_url, payload))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _line_notify(token: str, report) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {token}"},
                data={"message": f"\n{report.zh}\n{report.en}"},
            )

    @staticmethod
    async def _teams_webhook(url: str, payload: dict) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(url, json=payload)
