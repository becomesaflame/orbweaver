"""Snapshot pack format lives here. Export/import CLI is phase 6.

Packs must omit host and workspace MCP config (``mcp.json`` and its ``env``
blocks) and ``hooks.json``. Those secrets stay on the destination host, like API keys.
"""
