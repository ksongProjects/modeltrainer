from __future__ import annotations

from quant_platform.config import server_runtime_config


def test_server_runtime_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_PLATFORM_HOST", "0.0.0.0")
    monkeypatch.setenv("QUANT_PLATFORM_PORT", "9001")
    monkeypatch.setenv("QUANT_PLATFORM_RELOAD", "true")

    payload = server_runtime_config()

    assert payload["host"] == "0.0.0.0"
    assert payload["port"] == 9001
    assert payload["reload"] is True


def test_server_runtime_config_falls_back_on_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_PLATFORM_PORT", "not-a-port")
    monkeypatch.setenv("QUANT_PLATFORM_RELOAD", "no")

    payload = server_runtime_config()

    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 8000
    assert payload["reload"] is False
