from orbweaver import __version__
from orbweaver.agent import build_agent_system, static_system


def test_version_is_semver_zero():
    assert __version__ == "0.26.0"
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_agent_system_names_running_version():
    text = static_system()
    assert __version__ in text
    assert "semantic version" in text
    blocks = build_agent_system("")
    assert __version__ in blocks[0]["text"]
