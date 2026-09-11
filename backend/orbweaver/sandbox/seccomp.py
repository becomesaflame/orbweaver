"""seccomp-BPF deny-list for sandboxed bash, assembled in pure Python.

bubblewrap accepts ``--seccomp FD`` where FD holds a raw ``struct sock_filter[]``
program; it installs it with ``prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER)``
right before exec (and sets ``PR_SET_NO_NEW_PRIVS``). No libseccomp needed: the
program is ~40 instructions, so it is hand-assembled here from the classic BPF
opcodes and the per-architecture syscall numbers below.

Program shape::

    ld  seccomp_data.arch
    jeq AUDIT_ARCH_<this arch>, ok, kill    ; foreign ABI -> SECCOMP_RET_KILL_PROCESS
    ld  seccomp_data.nr
    jge 0x40000000, deny                    ; x86_64 only: refuse the x32 ABI
    jeq <denied nr>, deny                   ; one per denied syscall
    ...
    ret SECCOMP_RET_ALLOW
    ret SECCOMP_RET_ERRNO | EPERM

Denied syscalls fail with ``EPERM`` instead of killing the process so tools
that probe optional features (io_uring, perf) degrade instead of dying.
"""

from __future__ import annotations

import errno
import os
import platform
import struct
from functools import lru_cache

# classic BPF opcodes (linux/bpf_common.h)
_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_JMP_JGE_K = 0x35
_BPF_RET_K = 0x06

# linux/seccomp.h return values
SECCOMP_RET_KILL_PROCESS = 0x8000_0000
SECCOMP_RET_ERRNO = 0x0005_0000
SECCOMP_RET_ALLOW = 0x7FFF_0000

# linux/audit.h
AUDIT_ARCH_X86_64 = 0xC000_003E
AUDIT_ARCH_AARCH64 = 0xC000_00B7

# struct seccomp_data offsets
_OFF_NR = 0
_OFF_ARCH = 4

# x86_64 syscall numbers with this bit set are the x32 ABI; refuse them wholesale.
_X32_SYSCALL_BIT = 0x4000_0000

# Syscalls the sandboxed command never legitimately needs. bwrap has already
# set up namespaces and mounts before the filter is installed, so denying
# unshare/setns/mount here only affects the command it runs.
DENIED_SYSCALLS: tuple[str, ...] = (
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "mount",
    "umount2",
    "pivot_root",
    "open_tree",
    "move_mount",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "mount_setattr",
    "keyctl",
    "add_key",
    "request_key",
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "kexec_load",
    "kexec_file_load",
    "open_by_handle_at",
    "setns",
    "unshare",
    "init_module",
    "finit_module",
    "delete_module",
    "reboot",
    "swapon",
    "swapoff",
    "acct",
    "settimeofday",
    "clock_settime",
)

# arch/x86/entry/syscalls/syscall_64.tbl
_X86_64_NR: dict[str, int] = {
    "ptrace": 101,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "mount": 165,
    "umount2": 166,
    "pivot_root": 155,
    "open_tree": 428,
    "move_mount": 429,
    "fsopen": 430,
    "fsconfig": 431,
    "fsmount": 432,
    "fspick": 433,
    "mount_setattr": 442,
    "keyctl": 250,
    "add_key": 248,
    "request_key": 249,
    "bpf": 321,
    "perf_event_open": 298,
    "userfaultfd": 323,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "kexec_load": 246,
    "kexec_file_load": 320,
    "open_by_handle_at": 304,
    "setns": 308,
    "unshare": 272,
    "init_module": 175,
    "finit_module": 313,
    "delete_module": 176,
    "reboot": 169,
    "swapon": 167,
    "swapoff": 168,
    "acct": 163,
    "settimeofday": 164,
    "clock_settime": 227,
}

# include/uapi/asm-generic/unistd.h (aarch64 uses the generic table)
_AARCH64_NR: dict[str, int] = {
    "ptrace": 117,
    "process_vm_readv": 270,
    "process_vm_writev": 271,
    "mount": 40,
    "umount2": 39,
    "pivot_root": 41,
    "open_tree": 428,
    "move_mount": 429,
    "fsopen": 430,
    "fsconfig": 431,
    "fsmount": 432,
    "fspick": 433,
    "mount_setattr": 442,
    "keyctl": 219,
    "add_key": 217,
    "request_key": 218,
    "bpf": 280,
    "perf_event_open": 241,
    "userfaultfd": 282,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "kexec_load": 104,
    "kexec_file_load": 294,
    "open_by_handle_at": 265,
    "setns": 268,
    "unshare": 97,
    "init_module": 105,
    "finit_module": 273,
    "delete_module": 106,
    "reboot": 142,
    "swapon": 224,
    "swapoff": 225,
    "acct": 89,
    "settimeofday": 170,
    "clock_settime": 112,
}

SYSCALL_TABLES: dict[str, tuple[int, dict[str, int]]] = {
    "x86_64": (AUDIT_ARCH_X86_64, _X86_64_NR),
    "aarch64": (AUDIT_ARCH_AARCH64, _AARCH64_NR),
}
_MACHINE_ALIASES = {"amd64": "x86_64", "arm64": "aarch64"}

SECCOMP_MODES = ("auto", "on", "off")


def _insn(code: int, jt: int, jf: int, k: int) -> bytes:
    if not 0 <= jt <= 255 or not 0 <= jf <= 255:
        raise ValueError("BPF jump offset out of range")
    return struct.pack("<HBBI", code, jt, jf, k & 0xFFFF_FFFF)


def current_machine() -> str:
    m = platform.machine().lower()
    return _MACHINE_ALIASES.get(m, m)


def seccomp_supported_machine(machine: str | None = None) -> bool:
    return (machine or current_machine()) in SYSCALL_TABLES


def assemble_filter(machine: str | None = None, denied: tuple[str, ...] = DENIED_SYSCALLS) -> bytes:
    """Return the raw ``sock_filter[]`` program for ``machine`` (default: this host)."""
    arch_key = machine or current_machine()
    try:
        audit_arch, table = SYSCALL_TABLES[arch_key]
    except KeyError as e:
        raise ValueError(f"no seccomp syscall table for {arch_key}") from e
    nrs = sorted({table[name] for name in denied if name in table})
    prog: list[bytes] = []
    prog.append(_insn(_BPF_LD_W_ABS, 0, 0, _OFF_ARCH))
    prog.append(_insn(_BPF_JMP_JEQ_K, 1, 0, audit_arch))
    prog.append(_insn(_BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS))
    prog.append(_insn(_BPF_LD_W_ABS, 0, 0, _OFF_NR))
    # Every conditional below jumps forward to the final `ret ERRNO`.
    checks = len(nrs) + (1 if arch_key == "x86_64" else 0)
    deny_idx = len(prog) + checks + 1
    if arch_key == "x86_64":
        idx = len(prog)
        prog.append(_insn(_BPF_JMP_JGE_K, deny_idx - (idx + 1), 0, _X32_SYSCALL_BIT))
    for nr in nrs:
        idx = len(prog)
        prog.append(_insn(_BPF_JMP_JEQ_K, deny_idx - (idx + 1), 0, nr))
    prog.append(_insn(_BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    assert len(prog) == deny_idx
    prog.append(_insn(_BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM))
    return b"".join(prog)


@lru_cache(maxsize=4)
def seccomp_filter_for_host() -> bytes | None:
    """Program for this machine, or None when the architecture has no table."""
    if not seccomp_supported_machine():
        return None
    return assemble_filter()


def resolve_seccomp_mode(setting: str, *, bwrap_has_flag: bool, machine: str | None = None) -> bool:
    """`on` / `off` / `auto` -> whether to pass ``--seccomp`` to bwrap.

    `auto` enables the filter when bwrap advertises ``--seccomp`` and a syscall
    table exists for this architecture. `on` forces it (and fails loudly later if
    the platform cannot honour it); `off` disables it.
    """
    mode = (setting or "auto").strip().lower()
    if mode not in SECCOMP_MODES:
        mode = "auto"
    if mode == "off":
        return False
    if mode == "on":
        return True
    return bwrap_has_flag and seccomp_supported_machine(machine)


def open_seccomp_fd(program: bytes) -> int:
    """Write ``program`` to a sealed memfd positioned at 0, ready for ``--seccomp FD``."""
    fd = os.memfd_create("orbweaver-seccomp", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        written = 0
        while written < len(program):
            written += os.write(fd, program[written:])
        os.lseek(fd, 0, os.SEEK_SET)
        import fcntl

        add_seals = getattr(fcntl, "F_ADD_SEALS", None)
        seals = 0
        for name in ("F_SEAL_WRITE", "F_SEAL_SHRINK", "F_SEAL_GROW"):
            val = getattr(fcntl, name, None)
            if isinstance(val, int):
                seals |= val
        if isinstance(add_seals, int) and seals:
            try:
                fcntl.fcntl(fd, add_seals, seals)
            except OSError:
                pass
    except Exception:
        os.close(fd)
        raise
    return fd
