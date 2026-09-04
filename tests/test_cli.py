import sys

import pytest

from orbweaver.cli import main


def test_snapshot_cli_is_phase_six(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["orbweaver", "snapshot", "export"])
    with pytest.raises(SystemExit, match="phase 6"):
        main()
