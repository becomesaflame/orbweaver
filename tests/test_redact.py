from orbweaver.compact.persist import persist_tool_result
from orbweaver.config import settings
from orbweaver.redact import PLACEHOLDER, is_secret_name, redact_secrets
from orbweaver.workspace import LocalWorkspace

FAKE_BOT = "123456789:AAHfakeTelegramTokenValueForTests"
FAKE_ANTHROPIC = "sk-ant-api03-fake-key-value-for-tests-only"


def test_secret_name_heuristic():
    assert is_secret_name("TELEGRAM_BOT_TOKEN")
    assert is_secret_name("GH_TOKEN")
    assert is_secret_name("GITHUB_TOKEN")
    assert is_secret_name("ANTHROPIC_API_KEY")
    assert is_secret_name("DATABASE_URL")
    assert is_secret_name("MY_SERVICE_PASSWORD")
    assert not is_secret_name("ORBWEAVER_PINNED_TOKEN_CAP")
    assert not is_secret_name("ORBWEAVER_MODEL")
    assert not is_secret_name("MAX_TOKENS")


def test_redacts_env_assignment_keeps_token_cap():
    raw = (
        f"TELEGRAM_BOT_TOKEN={FAKE_BOT}\n"
        "ORBWEAVER_PINNED_TOKEN_CAP=4000\n"
        f"ANTHROPIC_API_KEY={FAKE_ANTHROPIC}\n"
    )
    out = redact_secrets(raw)
    assert FAKE_BOT not in out
    assert FAKE_ANTHROPIC not in out
    assert f"TELEGRAM_BOT_TOKEN={PLACEHOLDER}" in out
    assert f"ANTHROPIC_API_KEY={PLACEHOLDER}" in out
    assert "ORBWEAVER_PINNED_TOKEN_CAP=4000" in out


def test_redacts_export_and_json():
    raw = (
        f'export GH_TOKEN=gho_fakeGitHubTokenValue12345\n'
        f'{{"openai_api_key": "{FAKE_ANTHROPIC}", "model": "gpt"}}\n'
    )
    out = redact_secrets(raw)
    assert "gho_fakeGitHubTokenValue12345" not in out
    assert FAKE_ANTHROPIC not in out
    assert f"GH_TOKEN={PLACEHOLDER}" in out
    assert f'"openai_api_key": "{PLACEHOLDER}"' in out
    assert '"model": "gpt"' in out


def test_redacts_inline_assignment():
    out = redact_secrets(f"curl -H token GH_TOKEN={FAKE_BOT} https://example.test")
    assert FAKE_BOT not in out
    assert f"GH_TOKEN={PLACEHOLDER}" in out
    assert "https://example.test" in out


def test_redacts_live_settings_value(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", FAKE_BOT)
    out = redact_secrets(f"printenv said {FAKE_BOT} at the end")
    assert FAKE_BOT not in out
    assert PLACEHOLDER in out


def test_redact_is_idempotent():
    raw = f"TELEGRAM_BOT_TOKEN={FAKE_BOT}\nORBWEAVER_PINNED_TOKEN_CAP=4000\n"
    once = redact_secrets(raw)
    assert redact_secrets(once) == once


def test_persist_redacts_before_store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "compact_tool_result_chars", 100)
    ws = LocalWorkspace("workspace:default", str(tmp_path))
    dump = f"TELEGRAM_BOT_TOKEN={FAKE_BOT}\n" + ("x" * 200)
    stored, rel = persist_tool_result(ws, "toolu_secret", "Bash", dump)
    assert rel == ".orbweaver/tool-results/toolu_secret.txt"
    assert FAKE_BOT not in stored
    assert FAKE_BOT not in ws.read(rel)
    assert f"TELEGRAM_BOT_TOKEN={PLACEHOLDER}" in ws.read(rel)
