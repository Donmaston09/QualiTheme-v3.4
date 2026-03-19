"""
main_app.py – QualiTheme v3.1: Universal Thematic Analysis App

Fixes vs v3.0:
  - st.stop() inside tab blocks replaced with early-return guards
    (st.stop() inside a tab halts the entire Streamlit script, breaking
     other tabs that have already rendered)
  - html.escape() applied to all user-supplied strings interpolated into
    unsafe_allow_html markup (XSS hardening)
  - Filenames sanitised before display
  - st.query_params.pop() used instead of deprecated del st.query_params[key]
  - participant_id editing now triggers st.rerun() so metric row updates
  - Added @st.cache_data on codebook generation per transcript to avoid
    redundant computation on every re-render
  - Sidebar domain select stored to session state correctly
  - Upload limit check fixed: allows exact limit (>= instead of >)
  - Minor UX: "Jump to next uncoded" button shows participant count badge

Run with:  streamlit run main_app.py
"""

from __future__ import annotations

import html as _html
import re
import streamlit as st
import pandas as pd

from utils.ingestion import parse_transcript
from utils.subscription import (
    init_subscription, is_pro, tier_config,
    render_subscription_widget, pro_gate, max_transcripts,
)
from utils.analysis import (
    auto_code_with_embeddings,
    enrich_with_sentiment,
    generate_codebook,
    compute_cooccurrence_matrix,
    visualize_code_frequencies,
    visualize_code_cooccurrence,
    plot_cooccurrence_heatmap,
    plot_sentiment_distribution,
    plot_wordcloud,
    plot_theme_distribution,
    plot_code_timeline,
    export_plot,
    merge_all_transcripts,
    plot_participant_code_heatmap,
    plot_participant_theme_bars,
    plot_participant_sentiment_summary,
    plot_cross_participant_prevalence,
    # ── Group / pooled analysis ──
    group_auto_code,
    group_enrich_sentiment,
    generate_group_codebook,
    plot_group_code_frequencies,
    plot_group_theme_prevalence,
    plot_group_sentiment_overview,
    plot_group_cooccurrence_network,
    plot_group_theme_timeline,
    # ── Corpus synthesis ──
    synthesize_corpus,
    plot_synthesis_frequency,
    plot_synthesis_saturation,
    plot_synthesis_heatmap,
    plot_synthesis_sentiment_breakdown,
    plot_synthesis_cooccurrence,
    plot_synthesis_theme_journey,
    plot_synthesis_quotes_table,
)
from utils.export import (
    export_coded_segments_to_csv,
    export_coded_segments_to_word,
    export_codebook_to_word,
    export_to_excel,
    generate_ai_report,
    export_all_transcripts_csv,
    export_multi_transcript_excel,
    generate_synthesis_report,
)
from utils.transcription import (
    transcribe_media,
    SUPPORTED_MEDIA_EXTENSIONS,
    MODEL_INFO,
    segments_to_transcript_text,
    format_timestamp,
    format_timestamp_long,
)


def _segments_to_srt(segments: list[dict]) -> str:
    """Convert Whisper segment dicts to standard SRT subtitle format."""
    def _srt_ts(s: float) -> str:
        s = max(0.0, float(s))
        h  = int(s // 3600)
        m  = int((s % 3600) // 60)
        sc = int(s % 60)
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QualiTheme – Thematic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Layout ───────────────────────────────────────────────── */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1  { color: #1F77B4; font-size: 2rem; }
h2  { color: #2c3e50; border-bottom: 2px solid #e8ecf0;
      padding-bottom: 5px; margin-top: 1.5rem; }
h3  { color: #34495e; }

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]  { gap: 6px; }
.stTabs [data-baseweb="tab"]       { border-radius: 6px 6px 0 0; padding: 6px 16px; }
.stTabs [aria-selected="true"]     { background: #1F77B4 !important; color: white !important; }
.stDownloadButton > button         { width: 100%; border-radius: 6px; }

/* ── Segment cards (coding tab) ───────────────────────────── */
.seg-card {
    background: #f8f9fa; border-left: 4px solid #1F77B4;
    padding: 8px 12px; border-radius: 0 6px 6px 0;
    margin-bottom: 4px; font-size: 0.91em; line-height: 1.5;
}
.seg-card-neg { border-left-color: #D62728; }
.seg-card-pos { border-left-color: #2CA02C; }

/* ── Metric boxes ─────────────────────────────────────────── */
.metric-box {
    background: white; border: 1px solid #dee2e6; border-radius: 10px;
    padding: 14px 18px; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

/* ── Participant pills ────────────────────────────────────── */
.participant-pill {
    display: inline-block; background: #e8f4fd; color: #1F77B4;
    border: 1px solid #b8d9f3; border-radius: 20px;
    padding: 3px 12px; font-size: 0.82em; margin: 2px; font-weight: 600;
}
.participant-pill-done {
    background: #e6f9ef; color: #1a7f5a; border-color: #a3dfc0;
}

/* ── Theme-highlighted transcript viewer ──────────────────── */
.hl-transcript {
    font-size: 0.93em; line-height: 1.85;
    font-family: Georgia, "Times New Roman", serif;
    background: #fff; border: 1px solid #dde3ea;
    border-radius: 10px; padding: 20px 24px;
    max-height: 640px; overflow-y: auto;
}
.hl-seg {
    display: inline;
    border-radius: 3px;
    padding: 1px 3px;
    margin: 0 1px;
    cursor: default;
    position: relative;
}
.hl-seg:hover::after {
    content: attr(data-theme);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0,0,0,0.82);
    color: #fff;
    font-size: 0.75em;
    white-space: nowrap;
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
    z-index: 999;
    font-family: sans-serif;
}
.hl-uncoded {
    color: #555;
}
/* Theme legend */
.theme-legend {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin: 10px 0 14px 0;
}
.theme-chip {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 20px; padding: 3px 11px;
    font-size: 0.8em; font-weight: 600;
    border: 1px solid rgba(0,0,0,0.1);
    cursor: default;
}
.theme-chip-dot {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
}

/* ── Media upload panel ───────────────────────────────────── */
.media-drop-zone {
    border: 2px dashed #b0c4de; border-radius: 12px;
    padding: 28px 20px; text-align: center;
    background: #f7fafd; color: #555;
    font-size: 0.92em;
}
.transcription-segment {
    background: #f8f9fa; border-left: 3px solid #7F7F7F;
    padding: 6px 12px; border-radius: 0 6px 6px 0;
    margin-bottom: 3px; font-size: 0.88em; line-height: 1.5;
}
.ts-badge {
    background: #34495e; color: #fff; border-radius: 4px;
    padding: 1px 6px; font-size: 0.75em; margin-right: 6px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "transcripts":      [],
    "active_idx":       0,
    "domain":           "General / Mixed",
    "research_context": "",
    "memos":            {},
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

init_subscription()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active() -> dict | None:
    txs = st.session_state.transcripts
    idx = st.session_state.active_idx
    return txs[idx] if txs and 0 <= idx < len(txs) else None


def _set_active(idx: int) -> None:
    st.session_state.active_idx = idx


def _new_transcript(filename: str, text: str, pid: str) -> dict:
    return {
        "filename":         filename,
        "participant_id":   pid,
        "transcript_text":  text,
        "coded_segments":   [],
        "cluster_keywords": {},
        "cluster_labels":   {},
        "coding_done":      False,
    }


def _build_segments(t: dict) -> None:
    if not t["coded_segments"]:
        raw = [s.strip() for s in t["transcript_text"].splitlines() if s.strip()]
        t["coded_segments"] = [{"Segment": s, "Code": ""} for s in raw]


def _safe_display(value: str) -> str:
    """Escape a user-supplied string for safe interpolation into HTML."""
    return _html.escape(str(value))


def _sanitise_filename(name: str) -> str:
    """Strip path separators and control chars from a filename for display."""
    return re.sub(r"[/\\<>&\"']", "_", name)


# ── Theme colour palette (20 distinct, accessible background colours) ────────
_THEME_BG_COLOURS: list[str] = [
    "#D0E8FF", "#FFE0B2", "#C8E6C9", "#FFCDD2", "#E1BEE7",
    "#FFECB3", "#B2EBF2", "#F8BBD0", "#DCEDC8", "#B3E5FC",
    "#FFD180", "#CCFF90", "#84FFFF", "#FF9E80", "#B388FF",
    "#CFD8DC", "#F0F4C3", "#FFCCBC", "#C5CAE9", "#D7CCC8",
]
_THEME_BORDER_COLOURS: list[str] = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#BCBD22", "#17BECF", "#E377C2", "#7F7F7F", "#8C564B",
    "#FF9900", "#00CC44", "#00AACC", "#FF4400", "#7733FF",
    "#607D8B", "#A0A000", "#FF5722", "#3F51B5", "#795548",
]


def _build_theme_colour_map(coded_segments: list[dict]) -> dict[str, dict]:
    """
    Return {code: {"bg": "#xxx", "border": "#xxx", "text": "#000"}} for all
    distinct non-empty codes in *coded_segments*.
    """
    codes = list(dict.fromkeys(           # preserve first-seen order, dedupe
        s["Code"] for s in coded_segments if s.get("Code", "").strip()
    ))
    colour_map: dict[str, dict] = {}
    for i, code in enumerate(codes):
        colour_map[code] = {
            "bg":     _THEME_BG_COLOURS[i % len(_THEME_BG_COLOURS)],
            "border": _THEME_BORDER_COLOURS[i % len(_THEME_BORDER_COLOURS)],
            "text":   "#1a1a1a",
        }
    return colour_map


def _render_highlighted_transcript(
    coded_segments: list[dict],
    colour_map: dict[str, dict],
    show_uncoded: bool = True,
    filter_codes: list[str] | None = None,
) -> str:
    """
    Build an HTML string of the full transcript with colour-coded theme spans.
    Each span has a data-theme tooltip on hover.
    """
    lines: list[str] = []
    for seg in coded_segments:
        code = seg.get("Code", "").strip()
        text = _safe_display(seg.get("Segment", ""))

        if not code:
            if show_uncoded:
                lines.append(f"<span class='hl-uncoded'>{text}</span>")
            continue

        if filter_codes and code not in filter_codes:
            # show as uncoded-grey when filtered out
            lines.append(f"<span class='hl-uncoded'>{text}</span>")
            continue

        c = colour_map.get(code, {"bg": "#eee", "border": "#aaa", "text": "#111"})
        safe_code = _safe_display(code)
        lines.append(
            f"<span class='hl-seg' "
            f"style='background:{c['bg']};border-bottom:2px solid {c['border']};color:{c['text']};' "
            f"data-theme='{safe_code}'>{text}</span>"
        )

    # Join with paragraph spacing between each segment
    return " ".join(lines)


def _render_theme_legend(colour_map: dict[str, dict]) -> str:
    """Return HTML for the colour-coded theme legend chips."""
    chips = []
    for code, c in colour_map.items():
        safe_code  = _safe_display(code)
        bg_col     = c["bg"]
        border_col = c["border"]
        chips.append(
            f"<span class='theme-chip' "
            f"style='background:{bg_col};border-color:{border_col};'>"
            f"<span class='theme-chip-dot' style='background:{border_col};'></span>"
            f"{safe_code}</span>"
        )
    return f"<div class='theme-legend'>{''.join(chips)}</div>"




# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#1F77B4;margin-bottom:0'>🔍 QualiTheme</h2>"
        "<p style='color:#666;font-size:0.85em;margin-top:2px'>"
        "Universal Thematic Analysis</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    render_subscription_widget()
    st.markdown("---")

    st.header("📂 Upload Transcripts")
    domain_choice = st.selectbox(
        "Research domain",
        ["General / Mixed", "Healthcare & Medicine", "Education & Learning",
         "Business & Organisational", "Social Sciences", "UX / Product Research",
         "Policy & Government", "Psychology & Wellbeing", "Other"],
        key="domain_select",
    )
    st.session_state["domain"] = domain_choice

    limit  = max_transcripts()
    loaded = len(st.session_state.transcripts)

    if is_pro():
        st.caption(f"Pro plan · up to **{limit}** transcripts per session.")
        uploaded_files = st.file_uploader(
            "Select files (.txt · .docx · .pdf · .csv)",
            type=["txt", "docx", "pdf", "csv"],
            accept_multiple_files=True,
        )
    else:
        st.caption("Free plan · **1 transcript** at a time.")
        single = st.file_uploader(
            "Upload a transcript",
            type=["txt", "docx", "pdf", "csv"],
            accept_multiple_files=False,
        )
        uploaded_files = [single] if single else []

    if uploaded_files:
        existing_names = {t["filename"] for t in st.session_state.transcripts}
        new_files      = [f for f in uploaded_files if f.name not in existing_names]

        if not is_pro() and loaded >= 1:
            st.warning("Clear the current transcript first, or upgrade to Pro for batch upload.")
        elif len(new_files) + loaded > limit:
            st.error(f"Plan allows {limit} transcript(s). Remove some or upgrade to Pro.")
        else:
            for f in new_files:
                safe_name = _sanitise_filename(f.name)
                try:
                    text = parse_transcript(f.read(), f.name)
                    n    = len([s for s in text.splitlines() if s.strip()])
                    if n > tier_config()["max_segments"] and not is_pro():
                        st.error(
                            f"**{safe_name}** has {n} segments — Free plan allows "
                            f"{tier_config()['max_segments']}. Upgrade to Pro."
                        )
                        continue
                    pid = f.name.rsplit(".", 1)[0]
                    st.session_state.transcripts.append(
                        _new_transcript(f.name, text, pid)
                    )
                    st.success(f"✅ {safe_name}")
                except Exception as e:
                    st.error(f"Could not read {safe_name}: {e}")

            if new_files:
                st.session_state.active_idx = max(
                    0, len(st.session_state.transcripts) - len(new_files)
                )

    # Participant manager
    if st.session_state.transcripts:
        st.markdown("---")
        st.markdown("**Loaded Transcripts**")
        for i, t in enumerate(st.session_state.transcripts):
            is_active = (i == st.session_state.active_idx)
            is_coded  = any(s["Code"].strip() for s in t["coded_segments"])
            label     = f"{'✅' if is_coded else '⬜'} {t['participant_id']}"
            ca, cb    = st.columns([4, 1])
            with ca:
                if st.button(
                    label, key=f"sel_{i}", use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    _set_active(i)
                    st.rerun()
            with cb:
                if st.button("✕", key=f"del_{i}", use_container_width=True):
                    st.session_state.transcripts.pop(i)
                    st.session_state.active_idx = max(
                        0, st.session_state.active_idx - 1
                    )
                    st.rerun()

        if st.button("🗑️ Clear all", use_container_width=True):
            st.session_state.transcripts = []
            st.session_state.active_idx  = 0
            st.rerun()

    st.markdown("---")

    # LLM settings
    st.header("🤖 LLM Settings")
    st.caption("Optional — AI theme labels & report generation.")
    llm_provider = st.selectbox(
        "Provider",
        ["— None —", "Claude (Anthropic)", "OpenAI (GPT-4)", "Google Gemini"],
    )
    llm_api_key = ""
    if llm_provider != "— None —":
        llm_api_key = st.text_input(
            "API Key", type="password", placeholder="Paste your key here…"
        )
        if llm_api_key and len(llm_api_key.strip()) > 10:
            st.success("Key set ✓")
        elif llm_api_key:
            st.warning("Key looks too short.")
    use_llm = llm_provider != "— None —" and len(llm_api_key.strip()) > 10

    st.markdown("---")
    st.session_state.research_context = st.text_area(
        "Research context / questions",
        value=st.session_state.research_context,
        height=70,
        placeholder="e.g. 'What barriers do staff face?'",
    )
    st.markdown("---")

    # About section
    about_col, author_col = st.columns([3, 1])
    with about_col:
        st.markdown("### About QualiTheme")
        st.markdown(
            """
            **QualiTheme** is a browser-based thematic analysis tool for researchers,
            students, and practitioners across any discipline. Supports the complete
            analysis workflow — transcript upload, manual or AI-assisted coding,
            visualisation, and export — with no installation required.

            Built on Reflexive Thematic Analysis principles. Accepts **.txt**,
            **.docx**, **.pdf**, and **.csv** formats.

            **Free** — 1 transcript, frequency chart, CSV export.  
            **Pro** ($10 one-time) — batch up to 50 participants, all
            visualisations, Word/Excel exports, AI report generation.
            """
        )
    with author_col:
        st.markdown(
            """
            <div style='background:linear-gradient(135deg,#1F77B4 0%,#2d9cdb 100%);
                        color:white;border-radius:14px;padding:22px 18px;
                        text-align:center;box-shadow:0 4px 14px rgba(31,119,180,.25);'>
                <div style='font-size:2.2em;margin-bottom:6px;'>👤</div>
                <div style='font-size:1em;font-weight:700;line-height:1.3;'>
                    Anthony Onoja
                </div>
                <div style='font-size:0.82em;font-weight:600;opacity:0.9;
                            margin-bottom:2px;'>PhD</div>
                <hr style='border-color:rgba(255,255,255,0.3);margin:10px 0;'>
                <div style='font-size:0.78em;opacity:0.92;line-height:1.6;'>
                    School of Health Sciences<br>
                    University of Surrey<br>
                    United Kingdom
                </div>
                <div style='margin-top:12px;'>
                    <a href='mailto:a.onoja@surrey.ac.uk'
                       style='background:rgba(255,255,255,0.18);color:white;
                              text-decoration:none;padding:6px 12px;
                              border-radius:20px;font-size:0.76em;
                              border:1px solid rgba(255,255,255,0.35);
                              display:inline-block;margin-top:2px;'>
                        ✉ a.onoja@surrey.ac.uk
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("QualiTheme v3.1 · Built with Streamlit")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>🔍 QualiTheme</h1>"
    "<p style='color:#666;margin-top:-8px'>"
    "Universal Qualitative Thematic Analysis · Single or Batch Participant Transcripts</p>",
    unsafe_allow_html=True,
)

if not st.session_state.transcripts:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    for col, em, ttl, body in [
        (c1, "📂", "Batch Upload",
         "Upload up to 50 participant transcripts at once (Pro). Free: 1 at a time."),
        (c2, "🤖", "AI Auto-Code",
         "Sentence-transformer embeddings cluster all participants in one click."),
        (c3, "👥", "Cross-Participant",
         "Heatmaps, bubble charts and grouped bars compare themes across all participants."),
        (c4, "📤", "Smart Exports",
         "One Excel sheet per participant + a cross-participant summary sheet."),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-box'><div style='font-size:2em'>{em}</div>"
                f"<b>{_safe_display(ttl)}</b><br>"
                f"<small style='color:#666'>{_safe_display(body)}</small></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.info("👈 Upload one or more transcripts from the sidebar to begin.")
    st.stop()  # Safe: only called at top-level, not inside a tab

# ── Active transcript guard ───────────────────────────────────────────────────
active = _active()
if active is None:
    st.warning("No transcript selected. Use the sidebar.")
    st.stop()  # Safe: top-level

_build_segments(active)
segs         = active["coded_segments"]
n_loaded     = len(st.session_state.transcripts)
coded_count  = sum(1 for s in segs if s["Code"].strip())
unique_codes = len({s["Code"] for s in segs if s["Code"].strip()})

# Participant pill bar
if n_loaded > 1:
    pills = ""
    for i, t in enumerate(st.session_state.transcripts):
        done = any(s["Code"].strip() for s in t["coded_segments"])
        cls  = "participant-pill-done" if done else "participant-pill"
        safe_pid = _safe_display(t["participant_id"])
        pills += (
            f"<span class='{cls}'>{'✅ ' if done else ''}{safe_pid}</span> "
        )
    st.markdown(pills, unsafe_allow_html=True)
    coded_transcripts = sum(
        1 for t in st.session_state.transcripts
        if any(s["Code"].strip() for s in t["coded_segments"])
    )
    st.progress(
        coded_transcripts / n_loaded,
        text=f"{coded_transcripts} / {n_loaded} transcripts coded",
    )
    st.markdown("")

# Metrics row
total_across = sum(
    sum(1 for s in t["coded_segments"] if s["Code"].strip())
    for t in st.session_state.transcripts
)
m1, m2, m3, m4, m5 = st.columns(5)
for col, val, label in [
    (m1, len(segs),     f"Segments ({_safe_display(active['participant_id'])})"),
    (m2, coded_count,   "Coded (this file)"),
    (m3, unique_codes,  "Unique Codes"),
    (m4, n_loaded,      "Transcripts Loaded"),
    (m5, total_across,  "Total Coded Segs"),
]:
    with col:
        st.markdown(
            f"<div class='metric-box'>"
            f"<div style='font-size:1.7em;font-weight:bold;color:#1F77B4'>{val}</div>"
            f"<div style='font-size:0.8em;color:#666'>{label}</div></div>",
            unsafe_allow_html=True,
        )
st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab_transcript, tab_media, tab_coding, tab_analysis,
 tab_group, tab_cross, tab_synthesis, tab_report, tab_export) = st.tabs([
    "📄 Transcript",
    "🎙️ Media",
    "🏷️ Coding",
    "📊 Single Analysis",
    "🧩 Group Analysis",
    "👥 Cross-Participant",
    "🔬 Synthesis",
    "📝 Report",
    "⬇️ Export",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — TRANSCRIPT  (colour-highlighted theme viewer)
# ═══════════════════════════════════════════════════════════════════
with tab_transcript:
    safe_pid = _safe_display(active["participant_id"])
    st.subheader(f"Viewing: {safe_pid}")

    # Participant label edit
    new_pid = st.text_input(
        "Edit participant label",
        value=active["participant_id"],
        key="pid_editor",
        max_chars=100,
    )
    if new_pid.strip() and new_pid != active["participant_id"]:
        active["participant_id"] = new_pid.strip()
        st.rerun()

    # ── Toolbar row ──────────────────────────────────────────────────────
    tb1, tb2, tb3 = st.columns([3, 2, 1])
    with tb1:
        view_mode = st.radio(
            "View mode",
            ["🎨 Theme Highlighted", "📄 Plain Text"],
            horizontal=True,
            key="tx_view_mode",
        )
    with tb2:
        show_uncoded_segs = st.checkbox(
            "Show uncoded segments", value=True, key="tx_show_uncoded"
        )
    with tb3:
        wc = len(active["transcript_text"].split())
        st.markdown(
            f"<div class='metric-box' style='padding:8px 10px;'>"
            f"<b>{wc:,}</b> words<br><small>{len(segs)} segments</small></div>",
            unsafe_allow_html=True,
        )

    has_codes = any(s.get("Code", "").strip() for s in segs)

    # ── Highlighted view ─────────────────────────────────────────────────
    if view_mode == "🎨 Theme Highlighted" and has_codes:
        colour_map = _build_theme_colour_map(segs)

        # Optional per-theme filter
        all_theme_names = list(colour_map.keys())
        with st.expander("🔍 Filter by theme (show/hide specific themes)", expanded=False):
            filter_cols = st.columns(min(4, len(all_theme_names)))
            active_themes: list[str] = []
            for fi, tname in enumerate(all_theme_names):
                c = colour_map[tname]
                with filter_cols[fi % len(filter_cols)]:
                    checked = st.checkbox(
                        tname, value=True, key=f"tf_{fi}",
                        help=f"Toggle visibility of theme: {tname}",
                    )
                    if checked:
                        active_themes.append(tname)

        filter_set = active_themes if len(active_themes) < len(all_theme_names) else None

        # Legend
        st.markdown(_render_theme_legend(colour_map), unsafe_allow_html=True)

        # Highlighted transcript body
        html_body = _render_highlighted_transcript(
            segs, colour_map,
            show_uncoded=show_uncoded_segs,
            filter_codes=filter_set,
        )
        st.markdown(
            f"<div class='hl-transcript'>{html_body}</div>",
            unsafe_allow_html=True,
        )

        # Segment-by-segment detail panel
        st.markdown("---")
        st.markdown("#### 📋 Segment Detail")
        st.caption(
            "Click any theme chip above to filter. "
            "Hover over highlighted text to see its theme name."
        )
        detail_filter = st.selectbox(
            "Jump to theme",
            ["All themes"] + all_theme_names,
            key="tx_detail_filter",
        )

        show_segs = segs if detail_filter == "All themes" else [
            s for s in segs if s.get("Code") == detail_filter
        ]
        for i, seg in enumerate(show_segs[:60]):
            code = seg.get("Code", "").strip()
            sent = seg.get("Sentiment", "")
            c    = colour_map.get(code, {"bg": "#f0f0f0", "border": "#aaa"})

            # Sentiment badge
            sent_html = ""
            if sent:
                sent_col = {"Positive": "#2CA02C", "Negative": "#D62728"}.get(sent, "#888")
                sent_html = (
                    f"<span style='background:{sent_col};color:white;border-radius:4px;"
                    f"padding:1px 5px;font-size:0.72em;margin-left:5px;'>{sent}</span>"
                )

            code_html = ""
            if code:
                safe_code = _safe_display(code)
                code_html = (
                    f"<span style='background:{c['border']};color:white;"
                    f"border-radius:4px;padding:1px 7px;font-size:0.76em;"
                    f"margin-left:4px;'>{safe_code}</span>"
                )

            safe_seg_text = _safe_display(seg["Segment"])
            seg_border = c["border"]
            seg_bg     = c["bg"]
            st.markdown(
                f"<div style='border-left:4px solid {seg_border};"
                f"background:{seg_bg};padding:7px 12px;border-radius:0 6px 6px 0;"
                f"margin-bottom:5px;font-size:0.9em;line-height:1.55;'>"
                f"<span style='color:#999;font-size:0.78em;'>#{i+1}</span>"
                f"{code_html}{sent_html}"
                f"<br>{safe_seg_text}</div>",
                unsafe_allow_html=True,
            )
        if len(show_segs) > 60:
            st.caption(f"… {len(show_segs)-60} more segments. Use theme filter to focus.")

    elif view_mode == "🎨 Theme Highlighted" and not has_codes:
        st.info("🏷️ Code this transcript first (Coding tab) to see theme highlights.")
        st.text_area(
            "Raw transcript", value=active["transcript_text"],
            height=320, disabled=True, label_visibility="collapsed",
        )

    else:
        # Plain text fallback
        st.text_area(
            "Raw transcript", value=active["transcript_text"],
            height=380, disabled=True, label_visibility="collapsed",
        )
        st.markdown("---")
        st.subheader("Segments")
        for i, seg in enumerate(segs[:30]):
            sent  = seg.get("Sentiment", "")
            cls   = {"Positive": "seg-card-pos", "Negative": "seg-card-neg"}.get(sent, "")
            badge = ""
            if seg["Code"]:
                safe_code = _safe_display(seg["Code"])
                badge = (
                    f" <span style='background:#1F77B4;color:white;border-radius:4px;"
                    f"padding:1px 6px;font-size:0.78em'>{safe_code}</span>"
                )
            safe_seg = _safe_display(seg["Segment"])
            st.markdown(
                f"<div class='seg-card {cls}'>"
                f"<span style='color:#999;font-size:0.8em'>#{i+1}</span>{badge}"
                f"<br>{safe_seg}</div>",
                unsafe_allow_html=True,
            )
        if len(segs) > 30:
            st.caption(f"… {len(segs)-30} more segments.")

    if n_loaded > 1:
        st.markdown("---")
        st.subheader("Navigate Participants")
        nav_cols = st.columns(min(6, n_loaded))
        for i, t in enumerate(st.session_state.transcripts):
            done = any(s["Code"].strip() for s in t["coded_segments"])
            with nav_cols[i % len(nav_cols)]:
                if st.button(
                    f"{'✅' if done else '⬜'} {t['participant_id']}",
                    key=f"nav_{i}", use_container_width=True,
                    type="primary" if i == st.session_state.active_idx else "secondary",
                ):
                    _set_active(i)
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — MEDIA  (audio/video → local Whisper transcription, no API key)
# ═══════════════════════════════════════════════════════════════════
with tab_media:
    st.subheader("🎙️ Audio & Video Transcription")

    # Import helper for backend detection (lazy — only when this tab is open)
    from utils.transcription import get_available_backend, get_model_info_html

    _backend = get_available_backend()
    _backend_colours = {
        "faster-whisper":  ("✅ faster-whisper active",  "#e6f9ef", "#1a7f5a"),
        "openai-whisper":  ("✅ openai-whisper active",  "#fff8e1", "#7d6000"),
        "none":            ("⚠️ No Whisper backend installed", "#fdecea", "#c0392b"),
    }
    _blabel, _bbg, _bfg = _backend_colours[_backend]
    st.markdown(
        f"<div style='background:{_bbg};color:{_bfg};border-radius:8px;"
        f"padding:8px 14px;font-size:0.88em;font-weight:600;margin-bottom:8px;'>"
        f"🔒 100% local processing — no API key, no data sent anywhere &nbsp;·&nbsp; {_blabel}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if _backend == "none":
        st.error(
            "No transcription backend is installed.  \n"
            "Add one of these to **requirements.txt** and redeploy:\n\n"
            "```\nfaster-whisper>=1.0.0        # recommended — 3-4x faster on CPU\n"
            "# openai-whisper>=20231117   # fallback alternative\n```\n\n"
            "Also add **`ffmpeg`** to **packages.txt** for video and most audio formats."
        )
    else:
        st.markdown(
            "Upload any audio or video recording. Whisper transcribes it locally — "
            "everything stays on this server, nothing is sent to any third-party service. "
            "The transcript is added straight to your participant list, ready for coding."
        )

        # ── Settings row ─────────────────────────────────────────────────────
        ms1, ms2, ms3 = st.columns([2, 2, 2])

        with ms1:
            local_model = st.selectbox(
                "Model size",
                list(MODEL_INFO.keys()),
                index=1,          # default: "base"
                key="whisper_local_model",
                help=(
                    "Larger models are more accurate but slower to load and run.\n"
                    "The model is downloaded once and cached for the session."
                ),
            )
            st.markdown(
                f"<small style='color:#555'>"
                f"{MODEL_INFO[local_model]['size']}  ·  {MODEL_INFO[local_model]['note']}"
                f"</small>",
                unsafe_allow_html=True,
            )

        with ms2:
            lang_options = {
                "Auto-detect": "",
                "English": "en",    "French": "fr",    "Spanish": "es",
                "German": "de",     "Dutch": "nl",     "Italian": "it",
                "Portuguese": "pt", "Arabic": "ar",    "Chinese": "zh",
                "Japanese": "ja",   "Korean": "ko",    "Hindi": "hi",
                "Russian": "ru",    "Turkish": "tr",   "Polish": "pl",
                "Swedish": "sv",    "Norwegian": "no", "Danish": "da",
            }
            lang_choice  = st.selectbox(
                "Language",
                list(lang_options.keys()),
                key="whisper_lang",
                help=(
                    "Auto-detect works well for most recordings. "
                    "Specify a language if audio is short or has heavy background noise."
                ),
            )
            language_iso = lang_options[lang_choice]

        with ms3:
            task_choice = st.radio(
                "Task",
                ["Transcribe (keep original language)",
                 "Translate to English"],
                key="whisper_task",
                help=(
                    "Transcribe: output in the spoken language.\n"
                    "Translate: Whisper translates directly to English — "
                    "useful for non-English interviews."
                ),
            )
            whisper_task = "translate" if task_choice.startswith("Translate") else "transcribe"

        # Model info table in expander
        with st.expander("ℹ️ Model size guide", expanded=False):
            st.markdown(get_model_info_html(), unsafe_allow_html=True)
            st.markdown(
                "<small>Models are downloaded from Hugging Face on first use "
                "and cached for the session. No data leaves your server.</small>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Upload zone ───────────────────────────────────────────────────────
        supported_exts = sorted(SUPPORTED_MEDIA_EXTENSIONS)
        st.markdown(
            f"<div class='media-drop-zone'>"
            f"<b>🎵 Audio:</b> mp3 · wav · m4a · ogg · flac · webm · opus · aac<br>"
            f"<b>🎬 Video:</b> mp4 · mov · avi · mkv · webm"
            f"<br><small style='color:#888'>Max file size: 200 MB  ·  "
            f"Tip: use .mp3 or .wav for fastest processing</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        limit        = max_transcripts()
        loaded_now   = len(st.session_state.transcripts)
        can_add_more = loaded_now < limit or is_pro()

        if not can_add_more and not is_pro():
            st.warning(
                "Free plan allows 1 transcript at a time. "
                "Clear the current transcript or upgrade to Pro for batch uploads."
            )
        else:
            media_file = st.file_uploader(
                "Upload audio or video file",
                type=supported_exts,
                accept_multiple_files=False,
                key="media_uploader",
            )

            if media_file is not None:
                safe_mname  = _sanitise_filename(media_file.name)
                file_bytes_val = media_file.getvalue()
                file_mb     = len(file_bytes_val) / (1024 * 1024)

                if any(t["filename"] == media_file.name
                       for t in st.session_state.transcripts):
                    st.info(f"**{safe_mname}** is already loaded as a transcript.")
                else:
                    # File summary card
                    fc1, fc2, fc3 = st.columns(3)
                    with fc1:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<b>{safe_mname}</b><br>"
                            f"<small style='color:#666'>filename</small></div>",
                            unsafe_allow_html=True,
                        )
                    with fc2:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<b>{file_mb:.1f} MB</b><br>"
                            f"<small style='color:#666'>file size</small></div>",
                            unsafe_allow_html=True,
                        )
                    with fc3:
                        ext_upper = _sanitise_filename(
                            media_file.name.rsplit(".", 1)[-1].upper()
                        )
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<b>.{ext_upper}</b><br>"
                            f"<small style='color:#666'>format</small></div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("")
                    transcribe_btn = st.button(
                        f"▶ Transcribe with Whisper ({local_model})",
                        type="primary",
                        key="transcribe_btn",
                    )

                    if transcribe_btn:
                        spinner_msg = (
                            f"Transcribing `{safe_mname}` using "
                            f"**{_backend}** · model: **{local_model}**  \n"
                            f"*This runs locally — larger files take a few minutes. "
                            f"The model loads once and is cached.*"
                        )
                        with st.spinner(spinner_msg):
                            try:
                                result = transcribe_media(
                                    file_bytes = file_bytes_val,
                                    filename   = media_file.name,
                                    language   = language_iso,
                                    model_size = local_model,
                                    task       = whisper_task,
                                )

                                # Build transcript with timestamps if available
                                if result["segments"]:
                                    tx_text = segments_to_transcript_text(
                                        result["segments"]
                                    )
                                else:
                                    tx_text = result["text"]

                                pid = media_file.name.rsplit(".", 1)[0]
                                st.session_state.transcripts.append(
                                    _new_transcript(media_file.name, tx_text, pid)
                                )
                                st.session_state.active_idx = (
                                    len(st.session_state.transcripts) - 1
                                )
                                st.session_state[f"transcription_result_{pid}"] = result

                                duration_str = format_timestamp(result["duration"])
                                n_segs = len(result["segments"])
                                st.success(
                                    f"✅ **Transcription complete!**  \n"
                                    f"Language detected: **{result['language']}**  ·  "
                                    f"Duration: **{duration_str}**  ·  "
                                    f"Segments: **{n_segs}**  ·  "
                                    f"Backend: **{result['backend']}**  \n"
                                    f"Participant **{pid}** added — "
                                    f"go to the **Coding tab** to start analysis."
                                )
                                st.rerun()

                            except ValueError as exc:
                                st.error(f"**File error:** {exc}")
                            except ImportError as exc:
                                st.error(
                                    f"**Missing package:** {exc}  \n\n"
                                    "Add `faster-whisper>=1.0.0` to **requirements.txt** "
                                    "and `ffmpeg` to **packages.txt**, then redeploy."
                                )
                            except RuntimeError as exc:
                                st.error(f"**Transcription error:** {exc}")
                            except Exception as exc:
                                st.error(
                                    f"**Unexpected error:** {exc}  \n"
                                    "Check that ffmpeg is installed (packages.txt)."
                                )

    # ── Transcription previews ───────────────────────────────────────────────
    media_participants = [
        (i, t) for i, t in enumerate(st.session_state.transcripts)
        if f"transcription_result_{t['participant_id']}" in st.session_state
    ]
    if media_participants:
        st.markdown("---")
        st.subheader("Transcription Previews")
        for idx, t in media_participants:
            pid    = t["participant_id"]
            result = st.session_state.get(f"transcription_result_{pid}", {})
            dur    = format_timestamp(result.get("duration", 0))
            lang   = result.get("language", "?")
            bk     = result.get("backend",  "?")
            mdl    = result.get("model",    "?")

            with st.expander(
                f"🎙️ {pid}  ·  {lang}  ·  {dur}  ·  {bk} ({mdl})",
                expanded=(idx == st.session_state.active_idx),
            ):
                segs_ts = result.get("segments", [])
                if segs_ts:
                    st.caption(f"{len(segs_ts)} timed segments")
                    for seg in segs_ts[:50]:
                        ts_str   = format_timestamp(seg.get("start", 0))
                        safe_seg = _safe_display(seg.get("text", ""))
                        st.markdown(
                            f"<div class='transcription-segment'>"
                            f"<span class='ts-badge'>{ts_str}</span>"
                            f"{safe_seg}</div>",
                            unsafe_allow_html=True,
                        )
                    if len(segs_ts) > 50:
                        st.caption(f"… {len(segs_ts)-50} more segments")
                else:
                    st.text_area(
                        "Full transcript",
                        value=result.get("text", ""),
                        height=200, disabled=True,
                        key=f"ts_preview_{pid}",
                    )

                dl1, dl2 = st.columns(2)
                with dl1:
                    if result.get("text"):
                        st.download_button(
                            "⬇ Download plain text (.txt)",
                            data=result["text"].encode("utf-8"),
                            file_name=f"{pid}_transcript.txt",
                            mime="text/plain",
                            key=f"dl_ts_plain_{pid}",
                        )
                with dl2:
                    if segs_ts:
                        # SRT subtitle format download
                        srt = _segments_to_srt(segs_ts)
                        st.download_button(
                            "⬇ Download subtitles (.srt)",
                            data=srt.encode("utf-8"),
                            file_name=f"{pid}_subtitles.srt",
                            mime="text/plain",
                            key=f"dl_ts_srt_{pid}",
                        )


with tab_coding:
    st.subheader(f"Coding: {_safe_display(active['participant_id'])}")

    apply_to_all = False
    if n_loaded > 1:
        apply_to_all = st.checkbox(
            "🔁 Apply auto-coding to **all** loaded transcripts at once",
            value=True,
            help="Processes every participant's transcript in a single run.",
        )

    coding_mode = st.radio(
        "",
        ["Manual Coding",
         "Auto-Code (Embeddings + Keywords)",
         "Auto-Code + LLM Theme Labels (Pro)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if coding_mode in (
        "Auto-Code (Embeddings + Keywords)",
        "Auto-Code + LLM Theme Labels (Pro)",
    ):
        use_llm_labels = coding_mode == "Auto-Code + LLM Theme Labels (Pro)"
        if use_llm_labels and not is_pro():
            pro_gate("LLM Theme Labels")
            use_llm_labels = False
        if use_llm_labels and not use_llm:
            st.warning("No valid API key — using keyword labels.")
            use_llm_labels = False

        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            n_clusters = st.slider("Number of themes", 2, 20, 5)
        with col_b:
            run_sentiment = st.checkbox(
                "Sentiment analysis after coding",
                value=True,
                disabled=not is_pro(),
            )
            if not is_pro():
                run_sentiment = False
        with col_c:
            run_btn = st.button("▶ Run Auto-Code", type="primary",
                                use_container_width=True)

        if run_btn:
            targets = st.session_state.transcripts if apply_to_all else [active]
            prog    = st.progress(0, text="Starting…")

            for idx, t in enumerate(targets):
                _build_segments(t)
                prog.progress(
                    idx / max(len(targets), 1),
                    text=f"Coding {t['participant_id']} ({idx+1}/{len(targets)})…",
                )
                codes, ckws, clabels, _, _ = auto_code_with_embeddings(
                    [s["Segment"] for s in t["coded_segments"]],
                    n_clusters=n_clusters,
                    use_llm=use_llm_labels,
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                )
                for i, seg in enumerate(t["coded_segments"]):
                    seg["Code"] = codes[i]
                t["cluster_keywords"] = ckws
                t["cluster_labels"]   = clabels
                t["coding_done"]      = True

                if run_sentiment and is_pro():
                    t["coded_segments"] = enrich_with_sentiment(t["coded_segments"])

            prog.progress(1.0, text="Done!")
            st.success(
                f"✅ Auto-coded **{len(targets)} transcript(s)** "
                f"into {n_clusters} themes."
            )

        if active.get("cluster_labels"):
            st.markdown(f"### Themes — {_safe_display(active['participant_id'])}")
            cls_cols = st.columns(min(4, len(active["cluster_labels"])))
            for ci, (k, label) in enumerate(active["cluster_labels"].items()):
                kws   = active["cluster_keywords"].get(k, [])
                count = sum(1 for s in segs if s["Code"] == label)
                with cls_cols[ci % len(cls_cols)]:
                    safe_label = _safe_display(label)
                    safe_kws   = _safe_display(", ".join(kws[:5]))
                    st.markdown(
                        f"<div class='metric-box' style='text-align:left;padding:10px 14px'>"
                        f"<b style='color:#1F77B4'>{safe_label}</b><br>"
                        f"<small style='color:#888'>{safe_kws}</small><br>"
                        f"<span style='font-size:0.82em'>{count} segs</span></div>",
                        unsafe_allow_html=True,
                    )

    else:  # Manual coding
        if is_pro() and unique_codes > 1:
            with st.expander("🔀 Merge Codes (Pro)"):
                all_codes = sorted({s["Code"] for s in segs if s["Code"].strip()})
                mfrom = st.multiselect("Merge these…", all_codes, key="merge_from")
                mto   = st.text_input("…into", key="merge_to", max_chars=100)
                if st.button("Merge") and mfrom and mto:
                    for s in segs:
                        if s["Code"] in mfrom:
                            s["Code"] = mto.strip()
                    st.success(f"Merged → '{_safe_display(mto)}'")
                    st.rerun()

        for i, seg in enumerate(segs):
            c1, c2 = st.columns([4, 1])
            with c1:
                sent = seg.get("Sentiment", "")
                cls  = {"Positive": "seg-card-pos",
                        "Negative": "seg-card-neg"}.get(sent, "")
                safe_seg = _safe_display(seg["Segment"])
                st.markdown(
                    f"<div class='seg-card {cls}'><b>#{i+1}</b> {safe_seg}</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                seg["Code"] = st.text_input(
                    "Code", value=seg["Code"], key=f"code_{i}",
                    label_visibility="collapsed",
                    max_chars=100,
                )

    if is_pro() and unique_codes > 0:
        st.markdown("---")
        st.subheader("📌 Code Memos (Pro)")
        current_codes = sorted({s["Code"] for s in segs if s["Code"].strip()})
        sel_code = st.selectbox("Select code", current_codes, key="memo_code")
        if sel_code:
            mv = st.text_area(
                "Note", value=st.session_state.memos.get(sel_code, ""),
                height=90, key=f"memo_{sel_code}",
            )
            if st.button("Save memo"):
                st.session_state.memos[sel_code] = mv
                st.success("Saved.")

    if n_loaded > 1:
        st.markdown("---")
        uncoded = [
            i for i, t in enumerate(st.session_state.transcripts)
            if not any(s["Code"].strip() for s in t["coded_segments"])
            and i != st.session_state.active_idx
        ]
        if uncoded:
            st.caption(f"**{len(uncoded)} transcript(s) still uncoded.**")
            next_i = uncoded[0]
            next_pid = st.session_state.transcripts[next_i]["participant_id"]
            if st.button(f"→ Jump to next uncoded: {next_pid}"):
                _set_active(next_i)
                st.rerun()
        else:
            st.success("🎉 All transcripts are coded!")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SINGLE-TRANSCRIPT ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_analysis:
    coded = [s for s in segs if s["Code"].strip()]
    if not coded:
        st.info(f"🏷️ Code '{_safe_display(active['participant_id'])}' first.")
    else:
        codebook_df = generate_codebook(segs)
        st.subheader(f"Analysis: {_safe_display(active['participant_id'])}")

        (vtab_freq, vtab_dist, vtab_sent, vtab_net,
         vtab_heat, vtab_wc, vtab_time) = st.tabs([
            "📊 Frequencies", "🥧 Distribution", "😊 Sentiment",
            "🕸️ Network", "🔥 Heatmap", "☁️ Word Cloud", "📈 Timeline",
        ])

        pid_sfx = re.sub(r"[^A-Za-z0-9_\-]", "_", active["participant_id"])

        def _dl(fig, name: str) -> None:
            files = export_plot(fig, name)
            dc1, dc2, dc3 = st.columns(3)
            for col, (ext, buf) in zip([dc1, dc2, dc3], files.items()):
                with col:
                    mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                    st.download_button(
                        f"⬇ {ext.upper()}", data=buf,
                        file_name=f"{name}_{pid_sfx}.{ext}", mime=mime,
                        use_container_width=True,
                        key=f"dl_{name}_{ext}_{pid_sfx}",
                    )

        with vtab_freq:
            st.dataframe(codebook_df, use_container_width=True, hide_index=True)
            fig = visualize_code_frequencies(codebook_df)
            st.pyplot(fig, use_container_width=True)
            _dl(fig, "code_frequencies")

        with vtab_dist:
            if not is_pro():
                pro_gate("Theme Distribution Chart")
            else:
                fig = plot_theme_distribution(segs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _dl(fig, "theme_distribution")

        with vtab_sent:
            if not is_pro():
                pro_gate("Sentiment Analysis")
            elif "Sentiment" not in (segs[0] if segs else {}):
                st.info("Run Auto-Code with sentiment enabled, or click below.")
                if st.button("Run Sentiment Now", key="run_sent_now"):
                    active["coded_segments"] = enrich_with_sentiment(segs)
                    st.rerun()
            else:
                fig = plot_sentiment_distribution(segs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _dl(fig, "sentiment")
                df_sent  = pd.DataFrame(segs)
                cols_show = [c for c in
                             ["Code", "Segment", "Sentiment", "SentimentScore"]
                             if c in df_sent.columns]
                df_sent = df_sent[df_sent["Code"].astype(str).str.strip() != ""]
                st.dataframe(df_sent[cols_show], use_container_width=True,
                             hide_index=True)

        with vtab_net:
            if not is_pro():
                pro_gate("Co-occurrence Network")
            else:
                fig = visualize_code_cooccurrence(segs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _dl(fig, "cooccurrence_network")
                else:
                    st.info("Need ≥2 different codes.")

        with vtab_heat:
            if not is_pro():
                pro_gate("Co-occurrence Heatmap")
            else:
                matrix = compute_cooccurrence_matrix(segs)
                if not matrix.empty and matrix.values.sum() > 0:
                    fig = plot_cooccurrence_heatmap(matrix)
                    st.pyplot(fig, use_container_width=True)
                    _dl(fig, "cooccurrence_heatmap")
                else:
                    st.info("Not enough co-occurrence data.")

        with vtab_wc:
            if not is_pro():
                pro_gate("Keyword Word Cloud")
            elif active.get("cluster_keywords"):
                fig = plot_wordcloud(active["cluster_keywords"])
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _dl(fig, "wordcloud")
            else:
                st.info("Run Auto-Coding first.")

        with vtab_time:
            if not is_pro():
                pro_gate("Code Prevalence Timeline")
            else:
                fig = plot_code_timeline(segs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _dl(fig, "code_timeline")
                else:
                    st.info("Need more coded segments (at least 20 recommended).")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — GROUP / POOLED ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_group:
    if not is_pro():
        pro_gate("Group Analysis")
    elif n_loaded < 2:
        st.info("Load **2 or more** transcripts to run a group analysis.")
    else:
        st.subheader("🧩 Group Analysis — All Transcripts Pooled")
        st.markdown(
            "Group analysis treats every transcript as part of **one shared corpus**. "
            "A single K-Means pass assigns the same code vocabulary to all participants, "
            "making frequency comparisons and co-occurrence patterns directly comparable. "
            "You can still inspect and edit individual transcripts in the other tabs."
        )

        # ── Group Auto-Code panel ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Step 1 — Run Group Auto-Coding")
        st.caption(
            "This pools all segments from every loaded transcript and fits "
            "one shared theme model. Each transcript's segments receive codes "
            "from the same vocabulary."
        )

        ga1, ga2, ga3 = st.columns([2, 2, 1])
        with ga1:
            g_n_clusters = st.slider("Number of shared themes", 2, 20, 5,
                                     key="g_n_clusters")
        with ga2:
            g_run_sentiment = st.checkbox(
                "Include sentiment analysis", value=True,
                key="g_sentiment",
            )
        with ga3:
            g_run_btn = st.button("▶ Run Group Code",
                                  type="primary", use_container_width=True,
                                  key="g_run_btn")

        if g_run_btn:
            # Ensure segments are built for all transcripts
            for t in st.session_state.transcripts:
                _build_segments(t)

            with st.spinner(
                f"Pooling segments from {n_loaded} transcripts and clustering…"
            ):
                try:
                    g_ckws, g_clabels = group_auto_code(
                        st.session_state.transcripts,
                        n_clusters=g_n_clusters,
                        use_llm=use_llm,
                        llm_provider=llm_provider,
                        llm_api_key=llm_api_key,
                    )
                    if g_run_sentiment:
                        group_enrich_sentiment(st.session_state.transcripts)

                    st.session_state["group_codebook"] = generate_group_codebook(
                        st.session_state.transcripts
                    )
                    st.success(
                        f"✅ Group coding complete — {n_loaded} transcripts, "
                        f"{g_n_clusters} shared themes."
                    )
                except Exception as exc:
                    st.error(f"Group coding failed: {exc}")

        # ── Check if group codebook exists ──────────────────────────────
        g_cb = st.session_state.get("group_codebook")

        if g_cb is None or g_cb.empty:
            st.info("Run Group Auto-Code above to see group-level results.")
        else:
            n_p = n_loaded

            # ── Codebook summary ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Group Codebook")
            st.caption(
                "Shows total frequency across all participants and how many "
                "distinct participants used each code."
            )
            st.dataframe(g_cb, use_container_width=True, hide_index=True)

            def _gdl(fig, name: str) -> None:
                files = export_plot(fig, name)
                gc1, gc2, gc3 = st.columns(3)
                for col, (ext, buf) in zip([gc1, gc2, gc3], files.items()):
                    with col:
                        mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                        st.download_button(
                            f"⬇ {ext.upper()}", data=buf,
                            file_name=f"group_{name}.{ext}", mime=mime,
                            use_container_width=True,
                            key=f"gdl_{name}_{ext}",
                        )

            # ── Visualisation tabs ───────────────────────────────────────
            (gt_freq, gt_prev, gt_sent,
             gt_net, gt_time) = st.tabs([
                "📊 Frequencies",
                "📈 Prevalence",
                "😊 Sentiment",
                "🕸️ Network",
                "⏱️ Timeline",
            ])

            with gt_freq:
                st.markdown(
                    "**Total segment count per code** (pooled across all "
                    f"{n_p} participants). Numbers in brackets = distinct "
                    "participants who used that code."
                )
                fig = plot_group_code_frequencies(g_cb)
                st.pyplot(fig, use_container_width=True)
                _gdl(fig, "group_code_frequencies")

            with gt_prev:
                st.markdown(
                    "**Left axis** = total frequency.  "
                    "**Red dots** = % of participants who used this code.  \n"
                    "A code with high frequency but low participant % indicates "
                    "it is concentrated in a few voices — a potential outlier theme."
                )
                fig = plot_group_theme_prevalence(g_cb, n_p)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _gdl(fig, "group_theme_prevalence")
                else:
                    st.info("Run group coding first.")

            with gt_sent:
                st.markdown(
                    "Overall sentiment distribution across every coded "
                    "segment from all participants."
                )
                fig = plot_group_sentiment_overview(st.session_state.transcripts)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _gdl(fig, "group_sentiment_overview")
                else:
                    st.info(
                        "Enable sentiment analysis in the Group Auto-Code "
                        "settings above and re-run."
                    )

            with gt_net:
                st.markdown(
                    "**Co-occurrence network** built from all participants' "
                    "pooled segments. Edge weight = how often two codes appear "
                    "in the same 4-segment window across the entire corpus."
                )
                fig = plot_group_cooccurrence_network(st.session_state.transcripts)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _gdl(fig, "group_cooccurrence_network")
                else:
                    st.info("Need ≥2 different codes.")

            with gt_time:
                st.markdown(
                    "Code prevalence rolling over the pooled segment sequence. "
                    "**Dashed vertical lines** mark participant boundaries."
                )
                fig = plot_group_theme_timeline(st.session_state.transcripts)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _gdl(fig, "group_theme_timeline")
                else:
                    st.info("Need more coded segments across participants.")

            # ── Download group codebook CSV ──────────────────────────────
            st.markdown("---")
            from utils.export import export_coded_segments_to_csv
            all_segs_flat = merge_all_transcripts(st.session_state.transcripts)
            st.download_button(
                "📄 Download Group Codebook (CSV)",
                data=g_cb.to_csv(index=False).encode("utf-8"),
                file_name="group_codebook.csv",
                mime="text/csv",
                use_container_width=True,
            )



with tab_cross:
    if not is_pro():
        pro_gate("Cross-Participant Analysis")
    elif n_loaded < 2:
        st.info("Load 2 or more transcripts to see cross-participant charts.")
    else:
        any_coded = any(
            any(s["Code"].strip() for s in t["coded_segments"])
            for t in st.session_state.transcripts
        )
        if not any_coded:
            st.info("Code at least one transcript first.")
        else:
            st.subheader("👥 Cross-Participant Analysis")
            st.caption(
                "All charts draw on every loaded transcript that has been coded. "
                "Code more participants to enrich the comparison."
            )

            txs = st.session_state.transcripts

            (xtab_heat, xtab_bars, xtab_bubble,
             xtab_sent, xtab_table) = st.tabs([
                "🔥 Participant × Code Heatmap",
                "📊 Grouped Bars",
                "🫧 Prevalence Bubbles",
                "😊 Sentiment Comparison",
                "📋 Master Table",
            ])

            def _xdl(fig, name: str) -> None:
                files = export_plot(fig, name)
                xc1, xc2, xc3 = st.columns(3)
                for col, (ext, buf) in zip([xc1, xc2, xc3], files.items()):
                    with col:
                        mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                        st.download_button(
                            f"⬇ {ext.upper()}", data=buf,
                            file_name=f"{name}.{ext}", mime=mime,
                            use_container_width=True,
                            key=f"xdl_{name}_{ext}",
                        )

            with xtab_heat:
                st.markdown(
                    "**Rows = participants · Columns = codes · Cell = segment count**  \n"
                    "Reveals which themes dominate per participant and which are shared."
                )
                fig = plot_participant_code_heatmap(txs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _xdl(fig, "participant_code_heatmap")
                else:
                    st.info("No coded data yet.")

            with xtab_bars:
                st.markdown(
                    "**Grouped bars** — for each code (top 10), compare counts per participant."
                )
                fig = plot_participant_theme_bars(txs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _xdl(fig, "participant_theme_bars")
                else:
                    st.info("No coded data yet.")

            with xtab_bubble:
                st.markdown(
                    "**Bubble chart** — x = participants, y = codes (top 12).  \n"
                    "Bubble size = segment count."
                )
                fig = plot_cross_participant_prevalence(txs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _xdl(fig, "prevalence_bubble")
                else:
                    st.info("No coded data yet.")

            with xtab_sent:
                st.markdown(
                    "**Stacked bars** — Positive / Neutral / Negative per participant."
                )
                fig = plot_participant_sentiment_summary(txs)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    _xdl(fig, "participant_sentiment")
                else:
                    st.info(
                        "Run sentiment analysis first "
                        "(Auto-Code tab → sentiment checkbox)."
                    )

            with xtab_table:
                st.markdown("**All coded segments** across every loaded participant.")
                all_segs = merge_all_transcripts(txs)
                if all_segs:
                    df_all    = pd.DataFrame(all_segs)
                    show_cols = [c for c in
                                 ["Participant", "Segment", "Code",
                                  "Sentiment", "SentimentScore"]
                                 if c in df_all.columns]
                    st.dataframe(df_all[show_cols], use_container_width=True,
                                 hide_index=True)
                else:
                    st.info("No coded segments yet.")



# ═══════════════════════════════════════════════════════════════════
# TAB 7 — CORPUS SYNTHESIS
# ═══════════════════════════════════════════════════════════════════
with tab_synthesis:
    if not is_pro():
        pro_gate("Corpus Synthesis")
    elif n_loaded < 2:
        st.info("Load and code **2 or more** transcripts to run a corpus synthesis.")
    else:
        any_coded_syn = any(
            any(s["Code"].strip() for s in t["coded_segments"])
            for t in st.session_state.transcripts
        )
        if not any_coded_syn:
            st.info("Code at least one transcript first, then return here.")
        else:
            st.subheader("🔬 Corpus Synthesis")
            st.markdown(
                "Synthesis combines **all participants** into a single analytical "
                "picture — not side-by-side comparisons, but one unified view of "
                "what themes emerged, how strongly, across how many voices, and "
                "what the data says overall. Use this to write up your findings."
            )

            # ── Run synthesis ─────────────────────────────────────────────
            st.info(
                "💡 **Tip:** For best results, use **5–10 themes** when coding "
                f"{n_loaded} participants. Too many clusters (e.g. 50) produces "
                "thin, single-participant themes that aren't meaningful across the group. "
                "Re-run Auto-Code in the Coding tab with fewer clusters if needed.",
                icon="🎯",
            )
            syn_run = st.button(
                "▶ Run Corpus Synthesis",
                type="primary",
                key="syn_run_btn",
            )
            if syn_run:
                with st.spinner("Synthesising corpus across all participants…"):
                    st.session_state["synthesis"] = synthesize_corpus(
                        st.session_state.transcripts
                    )
                st.success(
                    f"✅ Synthesis complete — "
                    f"{st.session_state['synthesis']['n_participants']} participants · "
                    f"{st.session_state['synthesis']['n_coded_segments']} coded segments · "
                    f"{st.session_state['synthesis']['n_unique_codes']} unique themes."
                )

            syn = st.session_state.get("synthesis")

            if not syn:
                st.info("Click **Run Corpus Synthesis** above to generate results.")
            else:
                n_p_syn   = syn["n_participants"]
                n_seg_syn = syn["n_coded_segments"]
                n_cod_syn = syn["n_unique_codes"]
                sat_df    = syn["saturation"]
                cb_df     = syn["codebook"]

                # ── Metrics strip ─────────────────────────────────────────
                sm1, sm2, sm3, sm4 = st.columns(4)
                for col, val, label in [
                    (sm1, n_p_syn,   "Participants"),
                    (sm2, n_seg_syn, "Coded Segments"),
                    (sm3, n_cod_syn, "Unique Themes"),
                    (sm4,
                     sum(1 for _, r in sat_df.iterrows()
                         if r.get("% Participants", 0) >= 50)
                         if not sat_df.empty else 0,
                     "Major Themes (≥50% participants)"),
                ]:
                    with col:
                        st.markdown(
                            f"<div class='metric-box'>"
                            f"<div style='font-size:1.7em;font-weight:bold;"
                            f"color:#1F77B4'>{val}</div>"
                            f"<div style='font-size:0.8em;color:#666'>{label}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown("")

                # ── Theme saturation table ────────────────────────────────
                with st.expander("📋 Theme Saturation Table", expanded=True):
                    st.caption(
                        "Saturation = % of participants who used each theme. "
                        "Universal >80% · Major 50–80% · Moderate 25–49% · Minority <25%"
                    )
                    if not sat_df.empty:
                        # Colour-code the Saturation column
                        def _colour_sat(val: str) -> str:
                            return {
                                "Universal (>80%)":  "background-color:#c8e6c9",
                                "Major (50–80%)":    "background-color:#fff9c4",
                                "Moderate (25–49%)": "background-color:#ffe0b2",
                                "Minority (<25%)":   "background-color:#ffcdd2",
                            }.get(str(val), "")

                        st.dataframe(
                            sat_df.style.applymap(
                                _colour_sat, subset=["Saturation"]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No coded data available.")

                # ── Representative quotes ─────────────────────────────────
                with st.expander("💬 Representative Quotes by Theme"):
                    quotes = syn.get("top_quotes", [])
                    seen_q: set = set()
                    for q in quotes:
                        if q["Code"] in seen_q:
                            continue
                        seen_q.add(q["Code"])
                        code_col  = _THEME_BORDER_COLOURS[
                            list(seen_q).index(q["Code"]) % len(_THEME_BORDER_COLOURS)
                        ]
                        safe_code  = _safe_display(q["Code"])
                        safe_quote = _safe_display(q["Quote"])
                        safe_part  = _safe_display(str(q["Participant"]))
                        st.markdown(
                            f"<div style='border-left:4px solid {code_col};"
                            f"background:#f8f9fa;padding:8px 14px;"
                            f"border-radius:0 8px 8px 0;margin-bottom:8px;'>"
                            f"<span style='background:{code_col};color:white;"
                            f"border-radius:4px;padding:1px 8px;font-size:0.78em;"
                            f"font-weight:600'>{safe_code}</span>"
                            f"<br><em style='font-size:0.92em'>\"{safe_quote}\"</em>"
                            f"<br><small style='color:#888'>— {safe_part}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                st.subheader("Synthesis Visualisations")

                # ── Build all figures (cached in session_state) ───────────
                if "synthesis_figures" not in st.session_state or syn_run:
                    figs: dict = {}
                    with st.spinner("Generating synthesis charts…"):
                        figs["synthesis_frequency"]  = plot_synthesis_frequency(cb_df)
                        figs["synthesis_saturation"] = plot_synthesis_saturation(
                            sat_df, n_p_syn)
                        figs["synthesis_heatmap"]    = plot_synthesis_heatmap(
                            st.session_state.transcripts)
                        figs["synthesis_sentiment"]  = plot_synthesis_sentiment_breakdown(syn)
                        figs["synthesis_network"]    = plot_synthesis_cooccurrence(
                            syn.get("all_segments", []))
                        figs["synthesis_journey"]    = plot_synthesis_theme_journey(
                            st.session_state.transcripts)
                        figs["synthesis_quotes"]     = plot_synthesis_quotes_table(
                            syn.get("top_quotes", []))
                    st.session_state["synthesis_figures"] = figs
                else:
                    figs = st.session_state["synthesis_figures"]

                # ── Chart tabs ────────────────────────────────────────────
                (stab_freq, stab_sat, stab_heat, stab_sent,
                 stab_net, stab_journey, stab_quotes) = st.tabs([
                    "📊 Frequencies",
                    "🎯 Saturation Map",
                    "🔥 Heatmap",
                    "😊 Sentiment",
                    "🕸️ Network",
                    "🌊 Theme Journey",
                    "💬 Quotes Table",
                ])

                def _sdl(fig, name: str) -> None:
                    """Download buttons for a synthesis chart."""
                    if fig is None:
                        return
                    files = export_plot(fig, name)
                    sc1, sc2, sc3 = st.columns(3)
                    for col, (ext, buf) in zip([sc1, sc2, sc3], files.items()):
                        with col:
                            mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                            st.download_button(
                                f"⬇ {ext.upper()}", data=buf,
                                file_name=f"{name}.{ext}", mime=mime,
                                use_container_width=True,
                                key=f"sdl_{name}_{ext}",
                            )

                with stab_freq:
                    st.markdown(
                        "Total coded segments per theme across all participants. "
                        "Labels show *(number of participants)*."
                    )
                    fig = figs.get("synthesis_frequency")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_frequency")
                    else:
                        st.info("Not enough data.")

                with stab_sat:
                    st.markdown(
                        "**x-axis** = % of participants who used this theme.  "
                        "**y-axis** = total frequency.  \n"
                        "**Top-right quadrant** = themes that are both frequent "
                        "and broadly shared — your core findings.  \n"
                        "**Top-left** = deep but narrow — important to a few.  \n"
                        "**Bottom-right** = mentioned widely but briefly — background themes."
                    )
                    fig = figs.get("synthesis_saturation")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_saturation")
                    else:
                        st.info("Not enough data.")

                with stab_heat:
                    st.markdown(
                        "Each row is a participant, each column is a theme. "
                        "**Colour intensity** shows relative emphasis (row-normalised). "
                        "**Numbers** show raw segment counts. "
                        "Reveals participant subgroups and themes that cluster together."
                    )
                    fig = figs.get("synthesis_heatmap")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_heatmap")
                    else:
                        st.info("Not enough data.")

                with stab_sent:
                    st.markdown(
                        "**Left** — overall corpus sentiment across all coded segments.  \n"
                        "**Right** — per-theme sentiment breakdown (top 12 themes).  \n"
                        "Requires sentiment analysis to have been run (Auto-Code tab)."
                    )
                    fig = figs.get("synthesis_sentiment")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_sentiment")
                    else:
                        st.info(
                            "Run sentiment analysis first — Auto-Code tab → "
                            "enable 'Sentiment analysis after coding'."
                        )

                with stab_net:
                    st.markdown(
                        "Co-occurrence network built from all participants' pooled segments. "
                        "Thicker edges = themes that appear together more frequently. "
                        "Larger nodes = higher degree centrality (connected to many others)."
                    )
                    fig = figs.get("synthesis_network")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_network")
                    else:
                        st.info("Need at least 2 different themes.")

                with stab_journey:
                    st.markdown(
                        "How the **mix of themes shifts** from participant to participant. "
                        "Each band is a theme — wider = more of that participant's data. "
                        "Reveals whether your study reached saturation "
                        "(stable theme mix) or if late participants introduced new themes."
                    )
                    fig = figs.get("synthesis_journey")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_journey")
                    else:
                        st.info("Need at least 2 participants with coded data.")

                with stab_quotes:
                    st.markdown(
                        "Representative quotes — one per major theme — "
                        "automatically selected by segment length and coded theme."
                    )
                    fig = figs.get("synthesis_quotes")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        _sdl(fig, "synthesis_quotes")
                    else:
                        st.info("No quotes available.")

                # ── Full synthesis report download ────────────────────────
                st.markdown("---")
                st.subheader("📄 Download Full Synthesis Report")
                st.markdown(
                    "Generates a comprehensive Word document containing: "
                    "AI-written narrative synthesis, all 7 charts embedded, "
                    "theme saturation appendix, and representative quotes appendix."
                )

                if not use_llm:
                    st.warning(
                        "No LLM key set in the sidebar — report will use a "
                        "structured template narrative instead of AI-written text."
                    )

                gen_report_btn = st.button(
                    "🚀 Generate Synthesis Report (.docx)",
                    type="primary",
                    key="syn_report_btn",
                )
                if gen_report_btn:
                    with st.spinner(
                        "Writing synthesis report — this may take 30–60 seconds…"
                    ):
                        try:
                            report_bytes = generate_synthesis_report(
                                synthesis        = syn,
                                figures          = figs,
                                llm_provider     = llm_provider,
                                llm_api_key      = llm_api_key,
                                research_context = st.session_state.research_context,
                                n_participants   = n_p_syn,
                                transcripts      = st.session_state.transcripts,
                            )
                            st.session_state["synthesis_report"] = report_bytes
                            st.success("✅ Report ready — click below to download.")
                        except Exception as exc:
                            st.error(f"Report generation failed: {exc}")

                if "synthesis_report" in st.session_state:
                    st.download_button(
                        "⬇ Download Synthesis Report (.docx)",
                        data=st.session_state["synthesis_report"],
                        file_name="corpus_synthesis_report.docx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document"
                        ),
                        type="primary",
                        key="syn_report_dl",
                    )

                # ── Saturation CSV export ─────────────────────────────────
                if not sat_df.empty:
                    st.download_button(
                        "📄 Download Saturation Table (CSV)",
                        data=sat_df.to_csv(index=False).encode("utf-8"),
                        file_name="theme_saturation.csv",
                        mime="text/csv",
                        key="syn_sat_csv",
                    )



with tab_report:
    _segs_coded = any(s["Code"].strip() for s in segs)
    if not _segs_coded:
        st.info("Complete coding first.")
    elif not is_pro():
        pro_gate("AI Research Report")
    else:
        st.subheader("📝 AI Research Report Generator")

        report_scope = st.radio(
            "Generate report for:",
            ["Active transcript only", "All transcripts (merged)"],
            horizontal=True,
            disabled=(n_loaded < 2),
        )

        if not use_llm:
            st.warning("No LLM key set — a template report will be generated.")

        if st.button("🚀 Generate Report", type="primary"):
            if report_scope == "All transcripts (merged)" and n_loaded >= 2:
                report_segs = merge_all_transcripts(st.session_state.transcripts)
                all_ckws: dict    = {}
                all_clabels: dict = {}
                for t in st.session_state.transcripts:
                    all_ckws.update(t.get("cluster_keywords", {}))
                    all_clabels.update(t.get("cluster_labels", {}))
            else:
                report_segs  = segs
                all_ckws     = active.get("cluster_keywords", {})
                all_clabels  = active.get("cluster_labels", {})

            rep_codebook = generate_codebook(report_segs)
            with st.spinner("Generating report… 20–40 seconds."):
                try:
                    rb = generate_ai_report(
                        coded_segments=report_segs,
                        codebook_df=rep_codebook,
                        cluster_keywords=all_ckws,
                        cluster_labels=all_clabels,
                        llm_provider=llm_provider,
                        llm_api_key=llm_api_key,
                        research_context=st.session_state.research_context,
                    )
                    st.session_state["report_bytes"] = rb
                    st.success("✅ Report ready!")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if "report_bytes" in st.session_state:
            st.download_button(
                "⬇ Download Report (.docx)",
                data=st.session_state["report_bytes"],
                file_name="thematic_analysis_report.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
                type="primary",
            )


# ═══════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ═══════════════════════════════════════════════════════════════════
with tab_export:
    _segs_coded_export = any(s["Code"].strip() for s in segs)
    if not _segs_coded_export:
        st.info("Complete coding first.")
    else:
        codebook_df = generate_codebook(segs)
        st.subheader("⬇️ Export Results")

        # Free
        st.markdown("#### 🆓 Free — Current Transcript")
        st.download_button(
            f"📄 {active['participant_id']} — Coded Segments (CSV)",
            data=export_coded_segments_to_csv(segs),
            file_name=f"{active['participant_id']}_coded.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Pro
        st.markdown("#### ⭐ Pro Exports")
        if not is_pro():
            pro_gate("Word, Excel & Batch Exports")
        else:
            st.markdown("**Current transcript**")
            p1, p2, p3 = st.columns(3)
            with p1:
                st.download_button(
                    "📘 Coded Transcript (Word)",
                    data=export_coded_segments_to_word(segs),
                    file_name=f"{active['participant_id']}_transcript.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".wordprocessingml.document"
                    ),
                    use_container_width=True,
                )
            with p2:
                st.download_button(
                    "📗 Codebook (Word)",
                    data=export_codebook_to_word(codebook_df, segs),
                    file_name=f"{active['participant_id']}_codebook.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".wordprocessingml.document"
                    ),
                    use_container_width=True,
                )
            with p3:
                st.download_button(
                    "📊 Results (Excel)",
                    data=export_to_excel(segs, codebook_df),
                    file_name=f"{active['participant_id']}_results.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )

            if n_loaded > 1:
                st.markdown("---")
                st.markdown(f"**All {n_loaded} Participants — Batch Exports**")
                st.caption(
                    "The Excel workbook includes: a cross-participant summary sheet "
                    "with pivot table and bar chart · a master 'All Segments' sheet · "
                    "one individual sheet per participant."
                )
                b1, b2 = st.columns(2)
                with b1:
                    st.download_button(
                        "📄 All Participants — Master CSV",
                        data=export_all_transcripts_csv(st.session_state.transcripts),
                        file_name="all_participants_coded.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with b2:
                    st.download_button(
                        f"📊 All Participants — Excel ({n_loaded} sheets)",
                        data=export_multi_transcript_excel(
                            st.session_state.transcripts
                        ),
                        file_name="all_participants_results.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet"
                        ),
                        use_container_width=True,
                    )

            st.markdown("---")
            st.markdown("**Individual Charts**")
            chart_map = {
                "Code Frequencies":       lambda: visualize_code_frequencies(codebook_df),
                "Theme Distribution":     lambda: plot_theme_distribution(segs),
                "Sentiment Distribution": lambda: plot_sentiment_distribution(segs),
                "Co-occurrence Network":  lambda: visualize_code_cooccurrence(segs),
                "Co-occurrence Heatmap":  lambda: plot_cooccurrence_heatmap(
                    compute_cooccurrence_matrix(segs)),
                "Code Timeline":          lambda: plot_code_timeline(segs),
            }
            for cname, cfn in chart_map.items():
                with st.expander(f"⬇ {cname}"):
                    fig = cfn()
                    if fig:
                        files = export_plot(fig, cname.lower().replace(" ", "_"))
                        cc1, cc2, cc3 = st.columns(3)
                        for col, (ext, buf) in zip([cc1, cc2, cc3], files.items()):
                            with col:
                                mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                                st.download_button(
                                    f"⬇ {ext.upper()}", data=buf,
                                    file_name=(
                                        f"{cname.lower().replace(' ','_')}.{ext}"
                                    ),
                                    mime=mime, use_container_width=True,
                                    key=f"exp_{cname}_{ext}",
                                )
                    else:
                        st.caption("Not enough data for this chart.")

            if st.session_state.memos:
                st.markdown("---")
                memo_df = pd.DataFrame(
                    [{"Code": k, "Memo": v}
                     for k, v in st.session_state.memos.items()]
                )
                st.download_button(
                    "📌 Download Memos (CSV)",
                    data=memo_df.to_csv(index=False).encode("utf-8"),
                    file_name="code_memos.csv",
                    mime="text/csv",
                )
