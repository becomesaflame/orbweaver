from __future__ import annotations

import tempfile
from pathlib import Path


def transcribe_bytes(data: bytes, filename: str = "audio.wav") -> str:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError("faster-whisper is not installed (pip install orbweaver[stt])") from e
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        model = WhisperModel("base", device="cpu")
        segments, _info = model.transcribe(tmp.name)
        return "".join(s.text for s in segments).strip()
