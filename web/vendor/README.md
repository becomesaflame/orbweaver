# Vendored browser libraries

There is no build step for `web/`; third-party scripts are committed here and
served by the `/ui` static mount (`app.py`). Update by re-fetching the pinned
URL, replacing the file, and updating this table.

| File | Library | Version | Source | SHA-256 |
| --- | --- | --- | --- | --- |
| `marked.min.js` | marked (Markdown parser, MIT) | 15.0.12 | https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js | `3e7e7d7feb3e5d58cb6c804f68ab5c24cc7e5eb6270fd6e5cbb9124739217d0c` |
| `purify.min.js` | DOMPurify (HTML sanitizer, Apache-2.0 / MPL-2.0) | 3.2.6 | https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js | `89e1fa7647cb495370d3a997ace4387f5d15d9f4c5af12352c53daa400956287` |

`index.html` loads them as `/ui/vendor/<file>`; if either fails to load the
chat falls back to plain-text rendering.
