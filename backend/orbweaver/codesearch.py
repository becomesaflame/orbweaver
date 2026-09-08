"""On-demand hybrid search over workspace source files (not memory chunks)."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbweaver.embeddings import embed_text
from orbweaver.permissions.rules import path_is_always_denied

MAX_FILE_BYTES = 256_000
MAX_FILES = 800
MAX_CHUNKS = 2_000
CHUNK_CHARS = 900
CHUNK_OVERLAP = 80
DEFAULT_RESULTS = 8
MAX_RESULTS = 20
SNIPPET_CHARS = 280

DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
    }
)
BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".tar",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".pyc",
        ".pyo",
        ".class",
        ".o",
        ".a",
        ".wasm",
        ".mp3",
        ".mp4",
        ".wav",
        ".ogg",
        ".webm",
        ".sqlite",
        ".db",
    }
)
_TERM_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


@dataclass(frozen=True)
class FileChunk:
    path: str
    start_line: int
    text: str
    embedding: list[float]


def clamp_results(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = DEFAULT_RESULTS
    return max(1, min(n, MAX_RESULTS))


def _rel_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parse_gitignore(text: str) -> list[tuple[bool, bool, str]]:
    rules: list[tuple[bool, bool, str]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        negated = line.startswith("!")
        line = line.removeprefix("!")
        dir_only = line.endswith("/")
        line = line.removesuffix("/")
        line = line.removeprefix("/")
        if line:
            rules.append((negated, dir_only, line))
    return rules


def _load_root_gitignore(root: Path) -> list[tuple[bool, bool, str]]:
    path = root / ".gitignore"
    try:
        return _parse_gitignore(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return []


def _gitignore_match(rel: str, is_dir: bool, rules: list[tuple[bool, bool, str]]) -> bool:
    ignored = False
    name = rel.rsplit("/", 1)[-1]
    for negated, dir_only, pat in rules:
        if dir_only and not is_dir:
            continue
        candidates = {rel, name}
        if not pat.startswith("**/"):
            candidates.add(rel.split("/", 1)[-1] if "/" in rel else rel)
        hit = any(
            fnmatch.fnmatch(c, pat)
            or fnmatch.fnmatch(c, pat.removeprefix("**/"))
            or fnmatch.fnmatch(Path(c).name, pat)
            for c in candidates
        )
        if not hit and "**" in pat:
            hit = fnmatch.fnmatch(rel, pat.replace("**/", "*").replace("**", "*"))
        if hit:
            ignored = not negated
    return ignored


def _git_ls_files(root: Path) -> list[str] | None:
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.decode("utf-8", errors="replace").split("\0") if p]


def _walk_files(root: Path) -> list[Path]:
    git_files = _git_ls_files(root)
    if git_files is not None:
        out: list[Path] = []
        for rel in git_files:
            if path_is_always_denied(rel):
                continue
            parts = Path(rel).parts
            if any(p in DEFAULT_SKIP_DIRS for p in parts):
                continue
            path = (root / rel).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                continue
            if path.is_file():
                out.append(path)
            if len(out) >= MAX_FILES:
                break
        return out

    rules = _load_root_gitignore(root)
    out = []
    root_res = root.resolve()
    for dirpath, dirnames, filenames in root.walk():
        try:
            rel_dir = dirpath.resolve().relative_to(root_res).as_posix()
        except ValueError:
            dirnames[:] = []
            continue
        keep: list[str] = []
        for name in dirnames:
            if name in DEFAULT_SKIP_DIRS:
                continue
            child_rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if _gitignore_match(child_rel, True, rules):
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if path_is_always_denied(rel) or _gitignore_match(rel, False, rules):
                continue
            path = dirpath / name
            if path.is_file():
                out.append(path)
            if len(out) >= MAX_FILES:
                return out
    return out


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0 or size > MAX_FILE_BYTES:
        return False
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in raw[:8192]:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _chunk_text(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        start = i
        buf: list[str] = []
        size = 0
        while i < len(lines) and size < CHUNK_CHARS:
            buf.append(lines[i])
            size += len(lines[i]) + 1
            i += 1
        blob = "\n".join(buf).strip()
        if blob:
            chunks.append((start + 1, blob))
        if i >= len(lines):
            break
        # Overlap a few lines so a match on a boundary still lands in a chunk.
        back = 0
        used = 0
        while back < len(buf) and used < CHUNK_OVERLAP:
            used += len(buf[-1 - back]) + 1
            back += 1
        i = max(start + 1, i - max(1, back))
        if i <= start:
            i = start + 1
    return chunks


def index_workspace(root: Path) -> list[FileChunk]:
    chunks: list[FileChunk] = []
    for path in _walk_files(root):
        if not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            rel = _rel_posix(root, path)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        for start_line, blob in _chunk_text(text):
            chunks.append(
                FileChunk(
                    path=rel,
                    start_line=start_line,
                    text=blob,
                    embedding=embed_text(blob),
                )
            )
            if len(chunks) >= MAX_CHUNKS:
                return chunks
    return chunks


def _lexical_score(query: str, text: str) -> float:
    terms = _TERM_RE.findall(query.lower())
    if not terms:
        return 0.0
    blob = text.lower()
    present = sum(1 for t in terms if t in blob)
    counts = sum(blob.count(t) for t in terms)
    return (present / len(terms)) + 0.08 * min(counts, 12)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _hybrid_score(query_vec: list[float], query: str, chunk: FileChunk) -> float:
    lexical = _lexical_score(query, f"{chunk.path}\n{chunk.text}")
    cosine = _cosine(query_vec, chunk.embedding)
    # Hash embeddings are orthogonal across distinct strings; lexical carries tests
    # and exact-token lookups. Real models still contribute via cosine.
    return 0.40 * cosine + 0.60 * lexical


def search_workspace(root: Path | str, query: str, *, k: int = DEFAULT_RESULTS) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    base = Path(root)
    indexed = index_workspace(base)
    q_vec = embed_text(q)
    limit = clamp_results(k)
    scored = [(_hybrid_score(q_vec, q, c), c) for c in indexed]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for score, chunk in scored:
        key = (chunk.path, chunk.start_line)
        if key in seen:
            continue
        seen.add(key)
        snippet = chunk.text.replace("\n", " ")
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[: SNIPPET_CHARS - 1] + "…"
        hits.append(
            {
                "path": chunk.path,
                "line": chunk.start_line,
                "score": round(float(score), 4),
                "snippet": snippet,
            }
        )
        if len(hits) >= limit:
            break
    return hits


def format_hits(query: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return (
            f"WorkspaceSearch: {query}\n"
            "No matching project files. Try a shorter identifier, or Grep for an exact token."
        )
    lines = [
        f"WorkspaceSearch: {query} ({len(hits)} hits)",
        "These are workspace files, not MemorySearch facts. Read a hit to see the full file.",
        "",
    ]
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit['path']}:{hit['line']}  score={hit['score']:.3f}")
        if hit.get("snippet"):
            lines.append(f"   {hit['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_workspace_search(workspace: Any, inp: dict[str, Any]) -> str:
    query = str(inp.get("query") or "").strip()
    if not query:
        return "WorkspaceSearch requires a query."
    root = getattr(workspace, "root", None)
    if root is None:
        return "WorkspaceSearch: no workspace root."
    hits = search_workspace(root, query, k=inp.get("max_results"))
    return format_hits(query, hits)
