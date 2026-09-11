"""Labeled sandbox denials so the agent can retry with the right permission."""

from __future__ import annotations

import re

_UNIX_HINT = re.compile(
    r"(?:/run/|/var/run/)[^\s:]+(?:\.sock|journal/socket|\.s\.PGSQL\.\d+)",
    re.IGNORECASE,
)
_NET_HINT = re.compile(
    r"network is unreachable|name or service not known|could not resolve|"
    r"temporary failure in name resolution|connection refused|"
    r"proxy.*403|sandbox_denied: network",
    re.IGNORECASE,
)
_FS_HINT = re.compile(
    r"read-only file system|operation not permitted|permission denied",
    re.IGNORECASE,
)
# EPERM from the seccomp deny-list (orbweaver.sandbox.seccomp), not a mount.
_SECCOMP_HINT = re.compile(
    r"\b(?:ptrace|PTRACE_\w+|unshare|setns|namespace|perf_event_open|io_uring\w*|"
    r"keyctl|add_key|request_key|userfaultfd|bpf)\b|"
    r"must be superuser to use mount|no new privileges.*sudo",
    re.IGNORECASE,
)


def sandbox_denied(kind: str, detail: str, retry: str) -> str:
    return f"sandbox_denied: {kind} ({detail}). {retry}"


def network_denied(host: str) -> str:
    return sandbox_denied(
        "network",
        f"{host} not allowed",
        'Ask the user to approve permissions ["full_network"] before retrying',
    )


def unix_socket_denied(path: str) -> str:
    return sandbox_denied(
        "unix_socket",
        path,
        'Ask the user to approve permissions ["all"] or an allowUnixSockets grant',
    )


def filesystem_denied(detail: str) -> str:
    return sandbox_denied(
        "filesystem",
        detail,
        'Ask the user to approve permissions ["all"] before host writes',
    )


def syscall_denied() -> str:
    return sandbox_denied(
        "syscall",
        "ptrace, mount, namespaces, keyrings, bpf, perf, io_uring and setuid are blocked by seccomp/no_new_privs",
        'Ask the user to approve permissions ["all"] if the tool truly needs it',
    )


def label_sandbox_output(out: str) -> str:
    """Prefix unlabeled failures with a constraint the model can act on."""
    text = out or ""
    if "sandbox_denied:" in text or "sandbox_unavailable:" in text:
        return text
    sock = _UNIX_HINT.search(text)
    if sock and ("no such file" in text.lower() or "connection refused" in text.lower()):
        return unix_socket_denied(sock.group(0)) + "\n" + text
    if _NET_HINT.search(text):
        return (
            sandbox_denied(
                "network",
                "host not reachable from the sandbox allowlist",
                'Ask the user to approve permissions ["full_network"] before retrying',
            )
            + "\n"
            + text
        )
    if "operation not permitted" in text.lower() and _SECCOMP_HINT.search(text):
        return syscall_denied() + "\n" + text
    if _FS_HINT.search(text):
        return filesystem_denied("sandbox blocked a filesystem operation") + "\n" + text
    return text
