"""Issue #114: bwrap hardening argv, rlimit prologue, and the hand-assembled seccomp filter."""

from __future__ import annotations

import re
import resource
import struct
from pathlib import Path

import pytest
from sandbox_mounts import assert_ro_bind_dests_creatable

from orbweaver.config import settings
from orbweaver.sandbox import bwrap as bwrap_mod
from orbweaver.sandbox import seccomp
from orbweaver.sandbox.bwrap import (
    build_bwrap_argv,
    hardening_args,
    resource_limits,
    rlimit_prologue,
    seccomp_enabled,
)
from orbweaver.sandbox.errors import label_sandbox_output


@pytest.fixture
def all_flags(monkeypatch):
    monkeypatch.setattr(bwrap_mod, "_bwrap_help_flags", lambda exe: None)


def test_argv_has_hardening_flags(tmp_path: Path, all_flags):
    argv = build_bwrap_argv("echo hi", tmp_path, tmp_path / "tmp")
    for flag in ("--unshare-user", "--unshare-pid", "--unshare-net", "--die-with-parent"):
        assert flag in argv
    for flag in ("--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try", "--new-session"):
        assert flag in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert_ro_bind_dests_creatable(argv)


def test_argv_tmpfs_over_sys_after_host_bind(tmp_path: Path, all_flags, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_hide_sys", True)
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp")
    root_i = next(i for i, a in enumerate(argv) if a == "--ro-bind" and argv[i + 1 : i + 3] == ["/", "/"])
    sys_i = next(i for i, a in enumerate(argv) if a == "--tmpfs" and argv[i + 1] == "/sys")
    assert root_i < sys_i
    monkeypatch.setattr(settings, "orbweaver_sandbox_hide_sys", False)
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp")
    assert "/sys" not in argv


def test_argv_fallback_root_does_not_mount_sys(tmp_path: Path, all_flags):
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", host_root=False)
    assert "/sys" not in argv


def test_hardening_flags_are_feature_detected(monkeypatch):
    old = frozenset({"--unshare-user", "--unshare-pid", "--unshare-net", "--die-with-parent", "--ro-bind"})
    monkeypatch.setattr(bwrap_mod, "_bwrap_help_flags", lambda exe: old)
    assert hardening_args() == []
    assert seccomp_enabled() is False
    modern = old | {"--unshare-ipc", "--cap-drop", "--seccomp"}
    monkeypatch.setattr(bwrap_mod, "_bwrap_help_flags", lambda exe: modern)
    assert hardening_args() == ["--unshare-ipc", "--cap-drop", "ALL"]


def test_real_bwrap_help_advertises_hardening_flags():
    """This host's bwrap: the help parser must find the flags production relies on."""
    if not bwrap_mod.bwrap_path():
        pytest.skip("bwrap not installed")
    flags = bwrap_mod.bwrap_supported_flags()
    assert flags is not None
    for flag in ("--cap-drop", "--new-session", "--unshare-ipc", "--seccomp", "--tmpfs"):
        assert flag in flags


def test_argv_passes_seccomp_fd_before_mounts(tmp_path: Path, all_flags):
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", seccomp_fd=7)
    i = argv.index("--seccomp")
    assert argv[i + 1] == "7"
    assert i < argv.index("--ro-bind")
    assert "--seccomp" not in build_bwrap_argv("true", tmp_path, tmp_path / "tmp")


def test_rlimit_prologue_wraps_command(tmp_path: Path, all_flags, monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_procs", 512)
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_mem_mb", 2048)
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_open_files", 4096)
    monkeypatch.setattr(bwrap_mod, "_clamp_to_hard_limit", lambda requested, which: requested)
    argv = build_bwrap_argv("echo hi", tmp_path, tmp_path / "tmp")
    assert argv[-3:-1] == ["bash", "-lc"]
    script = argv[-1]
    assert script.endswith("\necho hi")
    assert "ulimit -u 512 2>/dev/null" in script
    assert "ulimit -v 2097152 2>/dev/null" in script
    assert "ulimit -n 4096 2>/dev/null" in script


def test_rlimit_zero_disables_each_limit(monkeypatch):
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_procs", 0)
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_mem_mb", 0)
    monkeypatch.setattr(settings, "orbweaver_sandbox_max_open_files", 0)
    assert resource_limits() == {}
    assert rlimit_prologue() == ""
    assert rlimit_prologue({"nproc": 64}) == "ulimit -u 64 2>/dev/null\n"


def test_rlimit_clamps_to_host_hard_limit(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda which: (100, 200))
    assert bwrap_mod._clamp_to_hard_limit(4096, resource.RLIMIT_NOFILE) == 200
    assert bwrap_mod._clamp_to_hard_limit(150, resource.RLIMIT_NOFILE) == 150
    monkeypatch.setattr(resource, "getrlimit", lambda which: (100, resource.RLIM_INFINITY))
    assert bwrap_mod._clamp_to_hard_limit(4096, resource.RLIMIT_NOFILE) == 4096


def test_explicit_limits_override_settings(tmp_path: Path, all_flags):
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", limits={})
    assert argv[-1] == "true"
    argv = build_bwrap_argv("true", tmp_path, tmp_path / "tmp", limits={"nofile": 256})
    assert argv[-1] == "ulimit -n 256 2>/dev/null\ntrue"


@pytest.mark.parametrize(
    ("setting", "has_flag", "machine", "expected"),
    [
        ("auto", True, "x86_64", True),
        ("auto", True, "aarch64", True),
        ("auto", False, "x86_64", False),
        ("auto", True, "riscv64", False),
        ("off", True, "x86_64", False),
        ("on", False, "riscv64", True),
        ("bogus", True, "x86_64", True),
    ],
)
def test_resolve_seccomp_mode(setting, has_flag, machine, expected):
    assert seccomp.resolve_seccomp_mode(setting, bwrap_has_flag=has_flag, machine=machine) is expected


def _decode(program: bytes) -> list[tuple[int, int, int, int]]:
    assert len(program) % 8 == 0
    return [struct.unpack_from("<HBBI", program, off) for off in range(0, len(program), 8)]


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_seccomp_program_shape(machine):
    audit_arch, table = seccomp.SYSCALL_TABLES[machine]
    insns = _decode(seccomp.assemble_filter(machine))
    assert insns[0] == (0x20, 0, 0, 4), "load seccomp_data.arch"
    assert insns[1] == (0x15, 1, 0, audit_arch)
    assert insns[2] == (0x06, 0, 0, seccomp.SECCOMP_RET_KILL_PROCESS)
    assert insns[3] == (0x20, 0, 0, 0), "load seccomp_data.nr"
    assert insns[-2] == (0x06, 0, 0, seccomp.SECCOMP_RET_ALLOW)
    assert insns[-1] == (0x06, 0, 0, seccomp.SECCOMP_RET_ERRNO | 1)
    deny_idx = len(insns) - 1
    denied = {table[name] for name in seccomp.DENIED_SYSCALLS}
    seen: set[int] = set()
    for idx in range(4, len(insns) - 2):
        code, jt, jf, k = insns[idx]
        assert jf == 0
        assert idx + 1 + jt == deny_idx, "every check jumps to ret ERRNO"
        if code == 0x35:
            assert machine == "x86_64" and k == 0x40000000
        else:
            assert code == 0x15
            seen.add(k)
    assert seen == denied
    assert len(insns) < 64


def test_seccomp_tables_cover_every_denied_syscall():
    for machine, (_arch, table) in seccomp.SYSCALL_TABLES.items():
        missing = [name for name in seccomp.DENIED_SYSCALLS if name not in table]
        assert not missing, f"{machine} table lacks {missing}"


_HEADERS = {
    "x86_64": Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h"),
    "aarch64": Path("/usr/include/asm-generic/unistd.h"),
}


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_seccomp_syscall_numbers_match_kernel_headers(machine):
    header = _HEADERS[machine]
    if not header.is_file():
        pytest.skip(f"{header} not installed")
    text = header.read_text(encoding="utf-8", errors="replace")
    found = {m.group(1): int(m.group(2)) for m in re.finditer(r"#define __NR_(\w+)\s+(\d+)", text)}
    _arch, table = seccomp.SYSCALL_TABLES[machine]
    for name, nr in table.items():
        if name in found:
            assert found[name] == nr, f"{machine} {name}: table {nr} != header {found[name]}"


def test_seccomp_filter_for_host_matches_machine():
    program = seccomp.seccomp_filter_for_host()
    if not seccomp.seccomp_supported_machine():
        assert program is None
        return
    assert program == seccomp.assemble_filter(seccomp.current_machine())


def test_open_seccomp_fd_is_readable_from_start():
    import os

    program = seccomp.assemble_filter("x86_64")
    fd = seccomp.open_seccomp_fd(program)
    try:
        assert os.read(fd, len(program) + 16) == program
    finally:
        os.close(fd)


def test_unknown_machine_raises():
    with pytest.raises(ValueError):
        seccomp.assemble_filter("riscv64")


def test_label_seccomp_eperm_as_syscall_denial():
    out = label_sandbox_output("strace: attach: ptrace(PTRACE_SEIZE, 11): Operation not permitted")
    assert out.startswith("sandbox_denied: syscall")
    fs = label_sandbox_output("touch: cannot touch '/etc/x': Operation not permitted")
    assert fs.startswith("sandbox_denied: filesystem")
