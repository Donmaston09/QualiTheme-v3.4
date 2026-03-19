"""
utils/transcription.py
Audio & video -> text transcription -- 100% open-source, zero API keys -- v3.3

Primary backend  : faster-whisper  (CTranslate2, MIT licence)
                   3-4x faster than openai-whisper on CPU; identical accuracy;
                   produces word-level timestamps; lower memory footprint.
                   https://github.com/SYSTRAN/faster-whisper

Fallback backend : openai-whisper  (original OpenAI implementation, MIT licence)
                   Used automatically if faster-whisper is not installed.
                   https://github.com/openai/whisper

Both run entirely locally -- no internet connection, no API key, no data leaves
the researcher's machine. This matters for sensitive interview data (GDPR, NHS, etc).

Model sizes (shared between both backends)
  tiny   -- ~39 MB   : fast, good for clear English audio
  base   -- ~74 MB   : recommended default; great speed/accuracy balance
  small  -- ~244 MB  : better for accented speech, noisy audio
  medium -- ~769 MB  : best CPU accuracy; allow ~3-5 min for a 30-min recording

Supported file types
  Audio : mp3, wav, m4a, ogg, flac, webm, opus, aac
  Video : mp4, mov, avi, mkv, webm  (ffmpeg extracts the audio track)

Requirements (add to requirements.txt)
  faster-whisper>=1.0.0    <- recommended
  # openai-whisper>=20231117  <- fallback (uncomment if needed)

System packages (packages.txt on Streamlit Cloud)
  ffmpeg   <- required for video and most audio formats
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# Supported extensions
AUDIO_EXTENSIONS: frozenset = frozenset({
    "mp3", "wav", "m4a", "ogg", "flac", "webm", "opus", "aac", "mpeg", "mpga",
})
VIDEO_EXTENSIONS: frozenset = frozenset({
    "mp4", "mov", "avi", "mkv", "webm",
})
SUPPORTED_MEDIA_EXTENSIONS: frozenset = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# 200 MB hard cap -- practical limit for CPU inference on Streamlit Cloud
MAX_FILE_BYTES: int = 200 * 1024 * 1024

# Model info for the UI
MODEL_INFO: dict = {
    "tiny":   {"size": "~39 MB",  "note": "Fastest - good for clear English"},
    "base":   {"size": "~74 MB",  "note": "Recommended - great speed/accuracy balance"},
    "small":  {"size": "~244 MB", "note": "Better for accents and noisy audio"},
    "medium": {"size": "~769 MB", "note": "Best CPU quality - slower (~5 min/30 min audio)"},
}


def transcribe_media(
    file_bytes: bytes,
    filename: str,
    language: str = "",
    model_size: str = "base",
    task: str = "transcribe",
) -> dict:
    """
    Transcribe audio or video bytes to text using local Whisper models.
    No API key required. All processing happens on this machine.

    Parameters
    ----------
    file_bytes  : raw bytes of the uploaded file
    filename    : original filename -- used for extension detection
    language    : ISO 639-1 code hint e.g. "en", "fr". Empty = auto-detect.
    model_size  : "tiny" | "base" | "small" | "medium"
    task        : "transcribe" (default) or "translate" (translate to English)

    Returns
    -------
    dict with keys:
        "text"      str   -- full transcript text
        "segments"  list  -- [{start, end, text}, ...]
        "language"  str   -- detected / used language
        "duration"  float -- recording length in seconds
        "backend"   str   -- "faster-whisper" | "openai-whisper"
        "model"     str   -- model size used
    """
    ext = _safe_ext(filename)
    if ext not in SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MEDIA_EXTENSIONS))}."
        )

    if len(file_bytes) > MAX_FILE_BYTES:
        mb_actual = len(file_bytes) / (1024 * 1024)
        raise ValueError(
            f"File is {mb_actual:.1f} MB -- limit is "
            f"{MAX_FILE_BYTES // (1024 * 1024)} MB. "
            f"Trim the recording to shorter sections for best results."
        )

    if model_size not in MODEL_INFO:
        model_size = "base"

    suffix = Path(filename).suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if _faster_whisper_available():
            return _transcribe_faster_whisper(tmp_path, language, model_size, task)
        elif _openai_whisper_available():
            return _transcribe_openai_whisper(tmp_path, language, model_size, task)
        else:
            raise ImportError(
                "No transcription backend found.\n"
                "Install faster-whisper:  pip install faster-whisper>=1.0.0\n"
                "Or openai-whisper:       pip install openai-whisper>=20231117\n"
                "Also add 'ffmpeg' to packages.txt (Streamlit Cloud)."
            )

    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Transcription failed for '{filename}': {exc}\n"
            "Common causes: corrupt file, missing ffmpeg, "
            "or unsupported codec. Try converting to .mp3 or .wav first."
        ) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---- faster-whisper (primary) ------------------------------------------------

def _transcribe_faster_whisper(audio_path, language, model_size, task):
    model = _load_faster_whisper_model(model_size)

    kwargs = {
        "task":       task,
        "beam_size":  5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        kwargs["language"] = language

    segments_iter, info = model.transcribe(audio_path, **kwargs)

    segments = []
    parts = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end,   2),
                "text":  text,
            })
            parts.append(text)

    duration = 0.0
    if hasattr(info, "duration") and info.duration:
        duration = round(info.duration, 2)
    elif segments:
        duration = segments[-1]["end"]

    return {
        "text":     " ".join(parts),
        "segments": segments,
        "language": info.language if not language else language,
        "duration": duration,
        "backend":  "faster-whisper",
        "model":    model_size,
    }


def _load_faster_whisper_model(model_size: str):
    try:
        import streamlit as st

        @st.cache_resource(
            show_spinner=f"Downloading Whisper '{model_size}' model -- first run only..."
        )
        def _load(size: str):
            from faster_whisper import WhisperModel
            return WhisperModel(size, device="cpu", compute_type="int8")

        return _load(model_size)
    except Exception:
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device="cpu", compute_type="int8")


# ---- openai-whisper (fallback) -----------------------------------------------

def _transcribe_openai_whisper(audio_path, language, model_size, task):
    model = _load_openai_whisper_model(model_size)

    kwargs = {
        "task":    task,
        "verbose": False,
        "fp16":    False,
    }
    if language:
        kwargs["language"] = language

    result = model.transcribe(audio_path, **kwargs)

    segments = [
        {
            "start": round(float(s["start"]), 2),
            "end":   round(float(s["end"]),   2),
            "text":  s["text"].strip(),
        }
        for s in result.get("segments", [])
        if s.get("text", "").strip()
    ]

    return {
        "text":     result["text"].strip(),
        "segments": segments,
        "language": result.get("language", language or "auto"),
        "duration": segments[-1]["end"] if segments else 0.0,
        "backend":  "openai-whisper",
        "model":    model_size,
    }


def _load_openai_whisper_model(model_size: str):
    try:
        import streamlit as st

        @st.cache_resource(
            show_spinner=f"Downloading Whisper '{model_size}' model -- first run only..."
        )
        def _load(size: str):
            import whisper
            return whisper.load_model(size)

        return _load(model_size)
    except Exception:
        import whisper
        return whisper.load_model(model_size)


# ---- Availability checks -----------------------------------------------------

def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _openai_whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_backend() -> str:
    if _faster_whisper_available():
        return "faster-whisper"
    if _openai_whisper_available():
        return "openai-whisper"
    return "none"


# ---- Utility helpers ---------------------------------------------------------

def _safe_ext(filename: str) -> str:
    name  = re.sub(r"[/\\]", "", filename)
    parts = name.rsplit(".", 1)
    return parts[-1].lower() if len(parts) == 2 else ""


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds  % 60)
    return f"{m:02d}:{s:02d}"


def format_timestamp_long(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def segments_to_transcript_text(segments: list) -> str:
    """
    Convert timed segment dicts to timestamped plain text for the coding pipeline.
    Example:  [00:04]  The onboarding process was really confusing at first.
    """
    if not segments:
        return ""
    lines = []
    for seg in segments:
        ts   = format_timestamp(seg.get("start", 0))
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{ts}]  {text}")
    return "\n".join(lines)


def get_model_info_html() -> str:
    """HTML table of model sizes for display in the UI."""
    rows = "".join(
        f"<tr><td><b>{name}</b></td>"
        f"<td>{info['size']}</td>"
        f"<td style='color:#555'>{info['note']}</td></tr>"
        for name, info in MODEL_INFO.items()
    )
    return (
        "<table style='font-size:0.85em;border-collapse:collapse;width:100%'>"
        "<tr style='background:#1F77B4;color:white'>"
        "<th style='padding:4px 8px;text-align:left'>Model</th>"
        "<th style='padding:4px 8px;text-align:left'>Download size</th>"
        "<th style='padding:4px 8px;text-align:left'>Best for</th></tr>"
        + rows + "</table>"
    )
