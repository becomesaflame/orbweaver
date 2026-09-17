"""Root logging for `orbweaver serve` (issue #69): journald needs named INFO lines.

Uvicorn's default dictConfig configures only uvicorn.* loggers. orbweaver.* INFO
is dropped; ERROR hits logging.lastResort as a bare message with no logger name.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn

from orbweaver.auth import INSECURE_DEFAULT_SECRET
from orbweaver.cli import LOG_FORMAT, gateway_log_config, main, parse_log_level
from orbweaver.config import settings

STRONG_SECRET = "s" * 32
_BACKEND = Path(__file__).resolve().parents[1] / "backend"


def _run_log_script(body: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(_BACKEND)}
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=str(_BACKEND),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


_EMIT = """
log = logging.getLogger("orbweaver.channels.cron")
log.info("cron sweeper started")
log.error("cron: job failed")
try:
    raise RuntimeError("boom")
except RuntimeError:
    logging.getLogger("orbweaver.app").exception("websocket turn failed")
"""


def test_parse_log_level_rejects_unknown():
    with pytest.raises(ValueError, match="ORBWEAVER_LOG_LEVEL"):
        parse_log_level("verbose")


def test_uvicorn_default_config_drops_orbweaver_info():
    """Same path as unfixed serve: uvicorn.run's default log_config."""
    script = f"""
import logging
import logging.config
from uvicorn.config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
{_EMIT}
"""
    proc = _run_log_script(script)
    assert proc.returncode == 0, proc.stderr
    err = proc.stderr
    assert "cron sweeper started" not in err
    assert "orbweaver.channels.cron" not in err
    assert "cron: job failed" in err
    assert "websocket turn failed" in err


def test_gateway_log_config_emits_named_info_and_traceback():
    script = f"""
import logging
import logging.config
from orbweaver.cli import gateway_log_config
logging.config.dictConfig(gateway_log_config("info"))
{_EMIT}
"""
    proc = _run_log_script(script)
    assert proc.returncode == 0, proc.stderr
    err = proc.stderr
    assert "INFO orbweaver.channels.cron cron sweeper started" in err
    assert "ERROR orbweaver.channels.cron cron: job failed" in err
    assert "ERROR orbweaver.app websocket turn failed" in err
    assert "RuntimeError: boom" in err
    assert "Traceback" in err


def test_gateway_log_config_survives_uvicorn_defaults():
    """configure_logging then uvicorn's stock dictConfig must keep named INFO."""
    script = f"""
import logging
import logging.config
from uvicorn.config import LOGGING_CONFIG
from orbweaver.cli import configure_logging
configure_logging("info")
logging.config.dictConfig(LOGGING_CONFIG)
{_EMIT}
"""
    proc = _run_log_script(script)
    assert proc.returncode == 0, proc.stderr
    assert "INFO orbweaver.channels.cron cron sweeper started" in proc.stderr


def test_gateway_log_config_dict_has_named_root_format():
    cfg = gateway_log_config("warning")
    assert cfg["root"]["level"] == "WARNING"
    assert "%(name)s" in cfg["formatters"]["orbweaver"]["format"]
    assert cfg["formatters"]["orbweaver"]["format"] == LOG_FORMAT
    assert "orbweaver" in cfg["root"]["handlers"]


def test_serve_cli_passes_root_log_config(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", STRONG_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", False)
    monkeypatch.setattr(settings, "orbweaver_log_level", "debug")
    monkeypatch.setattr(sys, "argv", ["orbweaver", "serve"])
    main()
    assert len(calls) == 1
    cfg = calls[0]["log_config"]
    assert "%(name)s" in cfg["formatters"]["orbweaver"]["format"]
    assert cfg["root"]["level"] == "DEBUG"
    assert calls[0]["log_level"] == "debug"


def test_serve_cli_rejects_invalid_log_level(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", STRONG_SECRET)
    monkeypatch.setattr(settings, "orbweaver_log_level", "verbose")
    monkeypatch.setattr(sys, "argv", ["orbweaver", "serve"])
    with pytest.raises(SystemExit, match="ORBWEAVER_LOG_LEVEL"):
        main()
    assert calls == []


def test_serve_refusal_includes_logger_name(monkeypatch, capsys):
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.setattr(settings, "orbweaver_jwt_secret", INSECURE_DEFAULT_SECRET)
    monkeypatch.setattr(settings, "orbweaver_dev_insecure", False)
    monkeypatch.setattr(sys, "argv", ["orbweaver", "serve"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "ERROR orbweaver refusing to start" in err
    assert "ORBWEAVER_JWT_SECRET" in err
