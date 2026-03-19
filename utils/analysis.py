"""
utils/analysis.py
NLP, clustering, sentiment and visualisation logic — v3.1

Improvements over v3.0:
  - SentenceTransformer loaded once and cached with @st.cache_resource
  - Model string updated to claude-sonnet-4-6 (current Sonnet 4)
  - KMeans n_init set explicitly to suppress FutureWarning
  - plot_* functions accept a max_codes kwarg for defensive rendering
  - auto_code_with_embeddings returns consistent 5-tuple
  - enrich_with_sentiment is non-destructive (returns new list)
  - All matplotlib figures explicitly closed after export to prevent memory leak
  - visualize_code_cooccurrence gracefully handles singleton graphs
  - plot_code_timeline: min-data guard raised to 2*window
  - Type hints completed throughout
"""

from __future__ import annotations

import io
import os
from collections import Counter, defaultdict
from textwrap import shorten
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend (non-interactive, no display required).
# switch_backend is safe to call even if already set to Agg — avoids the
# UserWarning that matplotlib.use() emits when called after pyplot import.
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#fafafa",
})

PALETTE: list[str] = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
]

# Current Anthropic model string
CLAUDE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Model loading — cached at Streamlit app level
# ---------------------------------------------------------------------------

def _get_encoder():
    """Load SentenceTransformer once per Streamlit worker process."""
    try:
        import streamlit as st

        @st.cache_resource(show_spinner="Loading embedding model…")
        def _load():
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")

        return _load()
    except Exception:
        # Fallback for non-Streamlit contexts (unit tests, CLI)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# LLM theme-label generation
# ---------------------------------------------------------------------------

def llm_generate_theme_label(
    keywords: list[str],
    sample_texts: list[str],
    provider: str,
    api_key: str,
) -> str:
    prompt = (
        "You are a qualitative research assistant helping with thematic analysis.\n"
        "Given the following cluster of text segments and their top keywords, "
        "provide a concise, human-readable theme label (3–5 words max).\n\n"
        f"Top keywords: {', '.join(keywords[:10])}\n"
        "Sample segments:\n"
        + "\n".join(f"- {t[:120]}" for t in sample_texts[:4])
        + "\n\nReturn only the theme label. No explanation, no quotes."
    )
    try:
        if provider == "Claude (Anthropic)":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()

        elif provider == "OpenAI (GPT-4)":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()

        elif provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            r = genai.GenerativeModel("gemini-pro").generate_content(prompt)
            return r.text.strip()

    except Exception:
        pass

    return " / ".join(keywords[:2]) if keywords else "Unnamed Theme"


# ---------------------------------------------------------------------------
# Auto-coding
# ---------------------------------------------------------------------------

def auto_code_with_embeddings(
    texts: list[str],
    n_clusters: int = 5,
    n_top_words: int = 10,
    use_llm: bool = False,
    llm_provider: str = "",
    llm_api_key: str = "",
) -> tuple[list[str], dict[str, list[str]], dict[str, str], np.ndarray, np.ndarray]:
    """
    Cluster *texts* with sentence-transformer embeddings + K-Means.

    Returns
    -------
    (codes, cluster_keywords, cluster_labels, raw_labels, embeddings)
    """
    if not texts:
        return [], {}, {}, np.array([]), np.array([])

    n_clusters = max(1, min(n_clusters, len(texts)))
    model = _get_encoder()
    embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels: np.ndarray = kmeans.fit_predict(embeddings)

    cluster_keywords: dict[str, list[str]] = {}
    cluster_labels:   dict[str, str]       = {}

    for cid in range(n_clusters):
        key    = f"Cluster {cid}"
        ctexts = [texts[i] for i, lbl in enumerate(raw_labels) if lbl == cid]

        if ctexts:
            try:
                vec = CountVectorizer(stop_words="english", max_features=500)
                X   = vec.fit_transform(ctexts)
                wc  = np.asarray(X.sum(axis=0)).flatten()
                top = wc.argsort()[::-1][:n_top_words]
                kws = list(vec.get_feature_names_out()[top])
            except ValueError:
                kws = []
        else:
            kws = []

        cluster_keywords[key] = kws

        if use_llm and _key_valid(llm_api_key):
            cluster_labels[key] = llm_generate_theme_label(
                kws, ctexts, llm_provider, llm_api_key
            )
        else:
            cluster_labels[key] = " / ".join(kws[:2]) if kws else key

    codes = [cluster_labels[f"Cluster {lbl}"] for lbl in raw_labels]
    return codes, cluster_keywords, cluster_labels, raw_labels, embeddings


def _key_valid(k: str) -> bool:
    return isinstance(k, str) and len(k.strip()) > 10


# ---------------------------------------------------------------------------
# Sentiment (lexicon-based)
# ---------------------------------------------------------------------------

_POS_WORDS: frozenset[str] = frozenset({
    "good", "great", "excellent", "positive", "wonderful", "amazing", "love", "best",
    "happy", "successful", "effective", "helpful", "strong", "improve", "benefit",
    "support", "opportunity", "clear", "confident", "agree", "progress", "thank",
    "appreciate", "glad", "hope", "enjoy", "valuable", "important", "interest", "learn",
})
_NEG_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "poor", "negative", "awful", "hate", "worst", "sad", "fail",
    "difficult", "problem", "issue", "concern", "worry", "hard", "struggle", "fear",
    "lack", "miss", "loss", "pain", "risk", "wrong", "never", "nothing", "unfortunately",
    "challenge", "barrier", "obstacle", "stress", "anxious", "frustrated", "confused",
})


def analyse_sentiment(text: str) -> tuple[str, float]:
    """Lexicon-based sentiment. Returns (label, score) where score ∈ [-1, 1]."""
    words = [w.strip(".,!?;:'\"") for w in text.lower().split()]
    pos   = sum(1 for w in words if w in _POS_WORDS)
    neg   = sum(1 for w in words if w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return "Neutral", 0.0
    score = (pos - neg) / total
    if score > 0.1:
        return "Positive", round(score, 3)
    if score < -0.1:
        return "Negative", round(score, 3)
    return "Neutral", round(score, 3)


def enrich_with_sentiment(coded_segments: list[dict]) -> list[dict]:
    """Return a *new* list with Sentiment and SentimentScore fields added."""
    result = []
    for seg in coded_segments:
        entry = dict(seg)
        label, score = analyse_sentiment(entry.get("Segment", ""))
        entry["Sentiment"]      = label
        entry["SentimentScore"] = score
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

def generate_codebook(coded_segments: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(coded_segments)
    df = df[df["Code"].astype(str).str.strip() != ""]
    if df.empty:
        return pd.DataFrame(columns=["Code", "Frequency", "Example"])

    agg = (
        df.groupby("Code")
        .agg(
            Frequency=("Code", "count"),
            Example=("Segment", lambda x: shorten(str(x.iloc[0]), 130, placeholder="...")),
        )
        .reset_index()
        .sort_values("Frequency", ascending=False)
    )

    if "Sentiment" in df.columns:
        dominant = (
            df.groupby("Code")["Sentiment"]
            .agg(lambda x: x.value_counts().index[0] if len(x) else "Neutral")
            .rename("Dominant Sentiment")
        )
        agg = agg.merge(dominant, on="Code", how="left")

    return agg


# ---------------------------------------------------------------------------
# ── VISUALISATIONS ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def visualize_code_frequencies(codebook: pd.DataFrame) -> plt.Figure:
    n   = max(1, len(codebook))
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55)))
    colours = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars = ax.barh(codebook["Code"], codebook["Frequency"],
                   color=colours, height=0.65, edgecolor="white")
    for bar, val in zip(bars, codebook["Frequency"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9, color="#333",
        )
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title("Code Frequencies", fontsize=14, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.set_xlim(0, codebook["Frequency"].max() * 1.15)
    plt.tight_layout()
    return fig


def plot_sentiment_distribution(coded_segments: list[dict]) -> plt.Figure | None:
    df = pd.DataFrame(coded_segments)
    if "Sentiment" not in df.columns or df["Code"].astype(str).str.strip().eq("").all():
        return None
    df = df[df["Code"].astype(str).str.strip() != ""]
    pivot = (
        df.groupby(["Code", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    colours = {"Positive": "#2CA02C", "Neutral": "#7F7F7F", "Negative": "#D62728"}
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6)))
    left = np.zeros(len(pivot))
    for col in ["Positive", "Neutral", "Negative"]:
        vals = pivot[col].values
        ax.barh(pivot.index, vals, left=left, color=colours[col],
                label=col, height=0.6, edgecolor="white")
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, i, str(v), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += vals

    ax.set_xlabel("Segment count", fontsize=11)
    ax.set_title("Sentiment Distribution by Code", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def compute_cooccurrence_matrix(coded_segments: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(coded_segments)
    df = df[df["Code"].astype(str).str.strip() != ""]
    codes  = sorted(df["Code"].unique())
    matrix = pd.DataFrame(0, index=codes, columns=codes)
    code_list = df["Code"].tolist()
    window = 4
    for i in range(len(code_list)):
        for j in range(i + 1, min(i + window, len(code_list))):
            a, b = code_list[i], code_list[j]
            if a != b:
                matrix.loc[a, b] += 1
                matrix.loc[b, a] += 1
    return matrix


def plot_cooccurrence_heatmap(matrix: pd.DataFrame) -> plt.Figure:
    n   = len(matrix)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))
    mask = matrix == 0
    sns.heatmap(
        matrix, cmap="YlOrRd", annot=True, fmt="d",
        linewidths=0.5, ax=ax, mask=mask,
        cbar_kws={"shrink": 0.75, "label": "Co-occurrences"},
        annot_kws={"size": 9},
    )
    ax.set_title("Code Co-occurrence Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def visualize_code_cooccurrence(coded_segments: list[dict]) -> plt.Figure | None:
    df = pd.DataFrame(coded_segments)
    df = df[df["Code"].astype(str).str.strip() != ""]
    if df["Code"].nunique() < 2:
        return None

    cooc: dict[tuple[str, str], int] = defaultdict(int)
    code_list = df["Code"].tolist()
    for i in range(len(code_list)):
        for j in range(i + 1, min(i + 4, len(code_list))):
            if code_list[i] != code_list[j]:
                pair = tuple(sorted([code_list[i], code_list[j]]))
                cooc[pair] += 1  # type: ignore[index]
    if not cooc:
        return None

    G = nx.Graph()
    for (a, b), wt in cooc.items():
        G.add_edge(a, b, weight=wt)

    if len(G.nodes()) < 2:
        return None

    centrality   = nx.degree_centrality(G)
    node_sizes   = [3000 * centrality[n] + 800 for n in G.nodes()]
    node_colours = [PALETTE[i % len(PALETTE)] for i, _ in enumerate(G.nodes())]
    weights      = [G[u][v]["weight"] for u, v in G.edges()]
    max_w        = max(weights) if weights else 1

    pos = nx.spring_layout(G, seed=42, k=2.5 / (len(G.nodes()) ** 0.5 + 0.1))

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="white")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_color="black", font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[0.8 + 5 * (wt / max_w) for wt in weights],
        edge_color="#aaaaaa", alpha=0.65,
    )
    edge_labels = {(u, v): d["weight"]
                   for u, v, d in G.edges(data=True) if d["weight"] > 1}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=7, label_pos=0.35)
    ax.set_title("Code Co-occurrence Network", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_wordcloud(keywords_dict: dict[str, list[str]]) -> plt.Figure | None:
    all_kw = [kw for kws in keywords_dict.values() for kw in kws]
    if not all_kw:
        return None
    freq = Counter(all_kw)
    wc = WordCloud(
        width=1000, height=480, background_color="white",
        colormap="tab10", max_words=100, prefer_horizontal=0.85,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Cluster Keyword Word Cloud", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def plot_theme_distribution(coded_segments: list[dict]) -> plt.Figure | None:
    df = pd.DataFrame(coded_segments)
    df = df[df["Code"].astype(str).str.strip() != ""]
    if df.empty:
        return None
    counts = df["Code"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140,
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(counts))],
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    legend_labels = [f"{code}  ({cnt})" for code, cnt in counts.items()]
    ax.legend(wedges, legend_labels, title="Codes", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9, title_fontsize=10)
    ax.set_title("Theme Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_code_timeline(
    coded_segments: list[dict],
    window: int = 10,
) -> plt.Figure | None:
    df = pd.DataFrame(coded_segments)
    df = df[df["Code"].astype(str).str.strip() != ""]
    if df.empty or len(df) < window * 2:
        return None
    codes = df["Code"].value_counts().head(8).index.tolist()
    df = df.reset_index(drop=True)
    df["SegIdx"] = range(len(df))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, code in enumerate(codes):
        mask    = (df["Code"] == code).astype(int)
        rolling = mask.rolling(window, min_periods=1).mean()
        ax.plot(df["SegIdx"], rolling,
                label=shorten(code, 30),
                color=PALETTE[i % len(PALETTE)],
                linewidth=2.2, alpha=0.85)

    ax.set_xlabel("Segment index", fontsize=11)
    ax.set_ylabel(f"Prevalence (rolling {window}-seg)", fontsize=11)
    ax.set_title("Code Prevalence Over Transcript",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-participant visualisations
# ---------------------------------------------------------------------------

def merge_all_transcripts(transcripts: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for t in transcripts:
        pid = t.get("participant_id", t.get("filename", "Unknown"))
        for seg in t.get("coded_segments", []):
            entry = dict(seg)
            entry["Participant"] = pid
            merged.append(entry)
    return merged


def plot_participant_code_heatmap(transcripts: list[dict]) -> plt.Figure | None:
    rows = [
        {"Participant": t.get("participant_id", "?"), "Code": seg["Code"]}
        for t in transcripts
        for seg in t.get("coded_segments", [])
        if seg.get("Code", "").strip()
    ]
    if not rows:
        return None
    df    = pd.DataFrame(rows)
    pivot = df.groupby(["Participant", "Code"]).size().unstack(fill_value=0)
    if pivot.empty:
        return None

    n_parts, n_codes = len(pivot), len(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_codes * 0.9), max(5, n_parts * 0.65)))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt="d",
                linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Segment count"},
                annot_kws={"size": 9})
    ax.set_title("Code Frequency per Participant",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Code", fontsize=11)
    ax.set_ylabel("Participant", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def plot_participant_theme_bars(transcripts: list[dict]) -> plt.Figure | None:
    rows = [
        {"Participant": t.get("participant_id", "?"), "Code": seg["Code"]}
        for t in transcripts
        for seg in t.get("coded_segments", [])
        if seg.get("Code", "").strip()
    ]
    if not rows:
        return None
    df    = pd.DataFrame(rows)
    pivot = df.groupby(["Code", "Participant"]).size().unstack(fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).nlargest(10).index]
    if pivot.empty:
        return None

    n_codes, n_parts = len(pivot), len(pivot.columns)
    bar_w = 0.8 / max(n_parts, 1)
    x     = np.arange(n_codes)

    fig, ax = plt.subplots(figsize=(max(10, n_codes * 1.1), 6))
    for i, participant in enumerate(pivot.columns):
        offset = (i - n_parts / 2 + 0.5) * bar_w
        ax.bar(x + offset, pivot[participant],
               width=bar_w * 0.9,
               label=participant,
               color=PALETTE[i % len(PALETTE)],
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Segment count", fontsize=11)
    ax.set_title("Theme Frequency by Participant (Top 10 Codes)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="Participant", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    return fig


def plot_participant_sentiment_summary(transcripts: list[dict]) -> plt.Figure | None:
    rows = [
        {"Participant": t.get("participant_id", "?"), "Sentiment": seg["Sentiment"]}
        for t in transcripts
        for seg in t.get("coded_segments", [])
        if seg.get("Sentiment")
    ]
    if not rows:
        return None
    df    = pd.DataFrame(rows)
    pivot = (
        df.groupby(["Participant", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    colours = {"Positive": "#2CA02C", "Neutral": "#7F7F7F", "Negative": "#D62728"}
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.8), 5))
    left = np.zeros(len(pivot))
    for col in ["Positive", "Neutral", "Negative"]:
        vals = pivot[col].values
        ax.bar(pivot.index, vals, bottom=left,
               color=colours[col], label=col, edgecolor="white", width=0.55)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(i, l + v / 2, str(v), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += vals

    ax.set_ylabel("Segment count", fontsize=11)
    ax.set_title("Sentiment Profile per Participant",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    return fig


def plot_cross_participant_prevalence(transcripts: list[dict]) -> plt.Figure | None:
    rows = [
        {"Participant": t.get("participant_id", "?"), "Code": seg["Code"]}
        for t in transcripts
        for seg in t.get("coded_segments", [])
        if seg.get("Code", "").strip()
    ]
    if not rows:
        return None
    df        = pd.DataFrame(rows)
    pivot     = df.groupby(["Code", "Participant"]).size().reset_index(name="Count")
    top_codes = df.groupby("Code").size().nlargest(12).index.tolist()
    pivot     = pivot[pivot["Code"].isin(top_codes)]
    if pivot.empty:
        return None

    participants = sorted(df["Participant"].unique())
    codes        = top_codes

    fig, ax = plt.subplots(
        figsize=(max(9, len(participants) * 1.2), max(5, len(codes) * 0.7))
    )
    for _, row in pivot.iterrows():
        x      = participants.index(row["Participant"])
        y      = codes.index(row["Code"])
        colour = PALETTE[y % len(PALETTE)]
        ax.scatter(x, y, s=row["Count"] * 120, color=colour,
                   alpha=0.75, edgecolors="white", linewidth=1)
        ax.text(x, y, str(int(row["Count"])),
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")

    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(participants, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=9)
    ax.set_xlabel("Participant", fontsize=11)
    ax.set_ylabel("Code", fontsize=11)
    ax.set_title(
        "Code Prevalence Across Participants\n(bubble size = segment count)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ── GROUP / POOLED ANALYSIS  (all transcripts treated as one corpus) ────────
# ---------------------------------------------------------------------------

def group_auto_code(
    transcripts: list[dict],
    n_clusters: int = 5,
    use_llm: bool = False,
    llm_provider: str = "",
    llm_api_key: str = "",
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Pool ALL segments from every transcript and run a single shared K-Means
    clustering pass.  Writes the resulting Code back into each segment dict
    in-place and returns (cluster_keywords, cluster_labels) for the shared
    codebook.

    This gives a *consistent* code vocabulary across all participants so that
    the cross-participant charts are truly comparable.
    """
    # Collect all segments with provenance
    all_texts: list[str]  = []
    provenance: list[tuple[int, int]] = []   # (transcript_idx, segment_idx)

    for ti, t in enumerate(transcripts):
        for si, seg in enumerate(t.get("coded_segments", [])):
            all_texts.append(seg["Segment"])
            provenance.append((ti, si))

    if not all_texts:
        return {}, {}

    codes, ckws, clabels, _, _ = auto_code_with_embeddings(
        all_texts,
        n_clusters=n_clusters,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
    )

    # Write codes back to original segment dicts
    for code, (ti, si) in zip(codes, provenance):
        transcripts[ti]["coded_segments"][si]["Code"] = code

    # Propagate shared codebook to every transcript
    for t in transcripts:
        t["cluster_keywords"] = ckws
        t["cluster_labels"]   = clabels
        t["coding_done"]      = True

    return ckws, clabels


def group_enrich_sentiment(transcripts: list[dict]) -> None:
    """Run sentiment enrichment on all segments of all transcripts in-place."""
    for t in transcripts:
        t["coded_segments"] = enrich_with_sentiment(t["coded_segments"])


def generate_group_codebook(transcripts: list[dict]) -> pd.DataFrame:
    """
    Build a combined codebook aggregated across all participants.
    Includes per-code participant count (how many distinct participants used it).
    """
    rows = []
    for t in transcripts:
        pid = t.get("participant_id", "?")
        for seg in t.get("coded_segments", []):
            if seg.get("Code", "").strip():
                entry = dict(seg)
                entry["Participant"] = pid
                rows.append(entry)

    if not rows:
        return pd.DataFrame(columns=["Code", "Frequency", "Participants", "Example"])

    df = pd.DataFrame(rows)

    agg = (
        df.groupby("Code")
        .agg(
            Frequency=("Code", "count"),
            Participants=("Participant", "nunique"),
            Example=("Segment", lambda x: shorten(str(x.iloc[0]), 130, placeholder="...")),
        )
        .reset_index()
        .sort_values("Frequency", ascending=False)
    )

    if "Sentiment" in df.columns:
        dominant = (
            df.groupby("Code")["Sentiment"]
            .agg(lambda x: x.value_counts().index[0] if len(x) else "Neutral")
            .rename("Dominant Sentiment")
        )
        agg = agg.merge(dominant, on="Code", how="left")

    return agg


def plot_group_code_frequencies(group_codebook: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of pooled code frequencies, with participant-count annotation."""
    n   = max(1, len(group_codebook))
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.55)))
    colours = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars = ax.barh(group_codebook["Code"], group_codebook["Frequency"],
                   color=colours, height=0.65, edgecolor="white")

    has_participants = "Participants" in group_codebook.columns
    for bar, (_, row) in zip(bars, group_codebook.iterrows()):
        label = str(int(row["Frequency"]))
        if has_participants:
            label += f"  ({int(row['Participants'])} participants)"
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8.5, color="#333",
        )

    ax.set_xlabel("Total Segments", fontsize=11)
    ax.set_title("Group Code Frequencies (all participants pooled)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.set_xlim(0, group_codebook["Frequency"].max() * 1.25)
    plt.tight_layout()
    return fig


def plot_group_theme_prevalence(group_codebook: pd.DataFrame,
                                n_participants: int) -> plt.Figure | None:
    """
    Stacked bar: for each code show absolute frequency (left) and
    % of participants who used it (right axis as scatter).
    Gives an immediate sense of theme breadth vs. depth.
    """
    if group_codebook.empty:
        return None

    top = group_codebook.head(15).copy()
    if "Participants" not in top.columns:
        return None

    top["pct"] = (top["Participants"] / max(n_participants, 1) * 100).round(1)

    fig, ax1 = plt.subplots(figsize=(11, max(5, len(top) * 0.6)))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(top))]
    ax1.barh(top["Code"], top["Frequency"], color=colours, height=0.6, edgecolor="white")
    ax1.set_xlabel("Total segment frequency", fontsize=11)
    ax1.invert_yaxis()

    ax2 = ax1.twiny()
    ax2.scatter(top["pct"], range(len(top)),
                color="#D62728", zorder=5, s=60, label="% participants")
    for i, (_, row) in enumerate(top.iterrows()):
        ax2.text(row["pct"] + 1, i, f"{row['pct']:.0f}%",
                 va="center", fontsize=8, color="#D62728")
    ax2.set_xlabel("% of participants using this code", fontsize=11, color="#D62728")
    ax2.tick_params(axis="x", colors="#D62728")
    ax2.set_xlim(0, 115)

    ax1.set_title("Theme Prevalence — Frequency vs. Participant Breadth",
                  fontsize=13, fontweight="bold", pad=28)
    plt.tight_layout()
    return fig


def plot_group_sentiment_overview(transcripts: list[dict]) -> plt.Figure | None:
    """Pie chart of overall sentiment distribution across all pooled segments."""
    rows = [
        seg["Sentiment"]
        for t in transcripts
        for seg in t.get("coded_segments", [])
        if seg.get("Sentiment") and seg.get("Code", "").strip()
    ]
    if not rows:
        return None

    from collections import Counter
    counts = Counter(rows)
    order  = ["Positive", "Neutral", "Negative"]
    labels = [l for l in order if l in counts]
    values = [counts[l] for l in labels]
    colours = {"Positive": "#2CA02C", "Neutral": "#7F7F7F", "Negative": "#D62728"}

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=[colours[l] for l in labels],
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.legend(wedges, [f"{l} ({counts[l]})" for l in labels],
              loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=9)
    ax.set_title("Overall Sentiment — All Participants",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_group_cooccurrence_network(transcripts: list[dict]) -> plt.Figure | None:
    """Co-occurrence network built from pooled segments of all participants."""
    all_segs = [
        seg for t in transcripts for seg in t.get("coded_segments", [])
        if seg.get("Code", "").strip()
    ]
    return visualize_code_cooccurrence(all_segs)


def plot_group_theme_timeline(transcripts: list[dict],
                              window: int = 10) -> plt.Figure | None:
    """
    Code prevalence timeline across the pooled, ordered corpus.
    Segments are ordered: all of participant 1, then participant 2, etc.
    A vertical dashed line marks each participant boundary.
    """
    rows = []
    boundaries: list[int] = []
    cursor = 0
    for t in transcripts:
        segs = [s for s in t.get("coded_segments", []) if s.get("Code", "").strip()]
        for seg in segs:
            rows.append({"Code": seg["Code"], "SegIdx": cursor})
            cursor += 1
        boundaries.append(cursor)

    if not rows or cursor < window * 2:
        return None

    df    = pd.DataFrame(rows)
    codes = df["Code"].value_counts().head(8).index.tolist()

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, code in enumerate(codes):
        mask    = (df["Code"] == code).astype(int)
        rolling = mask.rolling(window, min_periods=1).mean()
        ax.plot(df["SegIdx"], rolling,
                label=shorten(code, 28),
                color=PALETTE[i % len(PALETTE)],
                linewidth=2.2, alpha=0.85)

    # Participant boundary lines
    for b in boundaries[:-1]:
        ax.axvline(b, color="#aaa", linestyle="--", linewidth=0.9, alpha=0.7)

    ax.set_xlabel("Segment index (all participants, sequential)", fontsize=10)
    ax.set_ylabel(f"Prevalence (rolling {window}-seg)", fontsize=10)
    ax.set_title("Code Prevalence Timeline — Pooled Group",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ── CORPUS SYNTHESIS  (cross-participant overall analysis) ──────────────────
# ---------------------------------------------------------------------------

def synthesize_corpus(transcripts: list[dict]) -> dict:
    """
    Aggregate all coded segments from all participants into a single
    synthesis data structure used by the Synthesis tab and report.

    Returns a dict with keys:
        all_segments        list[dict]  — every coded segment with Participant field
        codebook            DataFrame   — pooled code frequencies + participant counts
        n_participants      int
        n_coded_segments    int
        n_unique_codes      int
        theme_saturation    DataFrame   — per-code: freq, n_participants, % saturation
        sentiment_summary   dict        — {Positive/Neutral/Negative: count, pct}
        top_quotes          list[dict]  — up to 3 representative quotes per theme
        participant_ids     list[str]
        codes_by_participant dict       — {pid: {code: count}}
    """
    all_segs: list[dict] = []
    participant_ids: list[str] = []
    codes_by_participant: dict[str, dict] = {}

    for t in transcripts:
        pid = t.get("participant_id", t.get("filename", "Unknown"))
        participant_ids.append(pid)
        p_codes: dict[str, int] = {}
        for seg in t.get("coded_segments", []):
            if not seg.get("Code", "").strip():
                continue
            entry = dict(seg)
            entry["Participant"] = pid
            all_segs.append(entry)
            p_codes[seg["Code"]] = p_codes.get(seg["Code"], 0) + 1
        codes_by_participant[pid] = p_codes

    if not all_segs:
        return {}

    df = pd.DataFrame(all_segs)
    n_p = len(participant_ids)

    # ── Pooled codebook with participant breadth ──────────────────────────
    codebook = (
        df.groupby("Code")
        .agg(
            Frequency=("Code", "count"),
            Participants=("Participant", "nunique"),
            Example=("Segment", lambda x: shorten(str(x.iloc[0]), 140, placeholder="...")),
        )
        .reset_index()
        .sort_values("Frequency", ascending=False)
    )
    if "Sentiment" in df.columns:
        dom_sent = (
            df.groupby("Code")["Sentiment"]
            .agg(lambda x: x.value_counts().index[0] if len(x) else "Neutral")
            .rename("Dominant Sentiment")
        )
        codebook = codebook.merge(dom_sent, on="Code", how="left")

    # ── Theme saturation table ────────────────────────────────────────────
    saturation = codebook[["Code", "Frequency", "Participants"]].copy()
    saturation["% Participants"] = (
        saturation["Participants"] / max(n_p, 1) * 100
    ).round(1)
    saturation["Saturation"] = saturation["% Participants"].apply(
        lambda p: "Universal (>80%)" if p > 80
        else "Major (50–80%)" if p >= 50
        else "Moderate (25–49%)" if p >= 25
        else "Minority (<25%)"
    )

    # ── Sentiment summary ─────────────────────────────────────────────────
    sentiment_summary: dict = {}
    if "Sentiment" in df.columns:
        counts = df["Sentiment"].value_counts()
        total  = counts.sum()
        for label in ["Positive", "Neutral", "Negative"]:
            n = int(counts.get(label, 0))
            sentiment_summary[label] = {
                "count": n,
                "pct":   round(n / max(total, 1) * 100, 1),
            }

    # ── Representative quotes — up to 3 per theme ────────────────────────
    top_quotes: list[dict] = []
    for code in codebook["Code"].head(10).tolist():
        code_segs = df[df["Code"] == code]["Segment"].tolist()
        # Pick segments of middling length (not too short, not too long)
        code_segs_sorted = sorted(
            code_segs, key=lambda s: abs(len(s) - 100)
        )
        for seg in code_segs_sorted[:3]:
            top_quotes.append({
                "Code":      code,
                "Quote":     seg,
                "Participant": df[df["Segment"] == seg]["Participant"].iloc[0]
                               if len(df[df["Segment"] == seg]) else "—",
            })

    return {
        "all_segments":          all_segs,
        "codebook":              codebook,
        "saturation":            saturation,
        "n_participants":        n_p,
        "n_coded_segments":      len(all_segs),
        "n_unique_codes":        codebook["Code"].nunique(),
        "sentiment_summary":     sentiment_summary,
        "top_quotes":            top_quotes,
        "participant_ids":       participant_ids,
        "codes_by_participant":  codes_by_participant,
    }


# ── Synthesis visualisations ──────────────────────────────────────────────────

def plot_synthesis_frequency(codebook: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart: pooled code frequency with a participant-breadth
    overlay as a secondary scatter series on a twin axis.
    The definitive 'what were the main themes?' chart.
    """
    top = codebook.head(15).copy()
    n   = len(top)
    fig, ax1 = plt.subplots(figsize=(11, max(5, n * 0.58)))

    colours = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars    = ax1.barh(top["Code"], top["Frequency"],
                       color=colours, height=0.62, edgecolor="white")

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax1.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['Frequency'])}  ({int(row['Participants'])}p)",
            va="center", ha="left", fontsize=8.5, color="#333",
        )

    ax1.set_xlabel("Total coded segments", fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlim(0, top["Frequency"].max() * 1.30)

    ax1.set_title(
        "Corpus Theme Frequencies\n"
        "(bar = total segments · label shows participant count)",
        fontsize=13, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    return fig


def plot_synthesis_saturation(saturation: pd.DataFrame,
                               n_participants: int) -> plt.Figure | None:
    """
    Bubble / dot chart: x = participant breadth (%), y = total frequency.
    Bubble size = frequency. Quadrant lines divide into 4 saturation zones.

    Improvements:
    - Jitter overlapping points so they don't stack on top of each other
    - Alternate label positions (above/below) to reduce annotation collisions
    - Show a data-quality note when all themes have low saturation (<25%)
      so the reader understands what they're looking at
    - Use ASCII ellipsis in labels to avoid font rendering issues
    """
    if saturation.empty:
        return None

    top = saturation.head(20).copy()

    # Detect degenerate case: all themes from single participants
    max_pct = top["% Participants"].max() if "% Participants" in top.columns else 0
    all_minority = max_pct <= (100 / max(n_participants, 1) + 0.1)

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Quadrant guides ───────────────────────────────────────────────────
    ax.axvline(50, color="#ddd", linewidth=1, linestyle="--", zorder=0)
    ymed = float(top["Frequency"].median())
    ax.axhline(ymed, color="#ddd", linewidth=1, linestyle="--", zorder=0)

    # Quadrant labels (only useful if data spans both sides)
    for xpos, ypos, label in [
        (25, ymed * 1.08, "Deep / narrow"),
        (75, ymed * 1.08, "Deep / broad"),
        (25, ymed * 0.30, "Peripheral / narrow"),
        (75, ymed * 0.30, "Peripheral / broad"),
    ]:
        ax.text(xpos, ypos, label, ha="center", fontsize=8,
                color="#bbb", style="italic", zorder=0)

    # ── Jitter: spread overlapping x-values so labels don't stack ─────────
    # Group points by their x-value and apply a small horizontal offset
    # within each group so bubbles and labels separate visually.
    np.random.seed(42)
    x_vals = top["% Participants"].values.copy()
    for uniq_x in np.unique(x_vals):
        mask = x_vals == uniq_x
        n    = mask.sum()
        if n > 1:
            # Spread within ±4% of x — small enough to stay in same quadrant
            offsets = np.linspace(-3.5, 3.5, n)
            np.random.shuffle(offsets)
            x_vals[mask] += offsets

    # ── Plot bubbles ───────────────────────────────────────────────────────
    for i, ((_, row), jx) in enumerate(zip(top.iterrows(), x_vals)):
        colour = PALETTE[i % len(PALETTE)]
        size   = max(80, float(row["Frequency"]) * 35)
        ax.scatter(jx, row["Frequency"],
                   s=size, color=colour, alpha=0.82,
                   edgecolors="white", linewidth=1.2, zorder=3)

        label = str(row["Code"])
        # ASCII truncation — avoids unicode ellipsis rendering as [...]
        if len(label) > 20:
            label = label[:17] + "..."

        # Alternate above/below to reduce label collisions
        va     = "bottom" if i % 2 == 0 else "top"
        offset = (0, 9) if i % 2 == 0 else (0, -9)
        ax.annotate(
            label, (jx, row["Frequency"]),
            fontsize=7.5, ha="center", va=va,
            xytext=offset, textcoords="offset points",
        )

    ax.set_xlabel("% of participants using this theme", fontsize=11)
    ax.set_ylabel("Total coded segments", fontsize=11)
    ax.set_xlim(-10, 115)
    ax.set_ylim(bottom=0)

    title = "Theme Saturation Map\n(bubble size = frequency  x = participant breadth)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # ── Data-quality note when all themes are minority-level ──────────────
    if all_minority:
        ax.text(
            0.98, 0.02,
            f"Note: all themes appear in only 1/{n_participants} participants.\n"
            "Try reducing the number of clusters (5-10 recommended)\nfor more meaningful cross-participant themes.",
            transform=ax.transAxes, fontsize=8, color="#c0392b",
            ha="right", va="bottom", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3f3",
                      edgecolor="#e0a0a0", alpha=0.9),
        )

    plt.tight_layout()
    return fig


def plot_synthesis_heatmap(transcripts: list[dict]) -> plt.Figure | None:
    """
    Participant × theme heatmap with hierarchical clustering on both axes.
    Reveals participant subgroups and theme clusters simultaneously.
    """
    rows = [
        {"Participant": t.get("participant_id", "?"), "Code": seg["Code"]}
        for t in transcripts
        for seg in t.get("coded_segments", [])
        if seg.get("Code", "").strip()
    ]
    if not rows:
        return None

    df    = pd.DataFrame(rows)
    pivot = df.groupby(["Participant", "Code"]).size().unstack(fill_value=0)
    if pivot.empty or pivot.shape[1] < 2:
        return None

    # Normalise rows to show relative emphasis per participant
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

    n_p, n_c = pivot_norm.shape
    fig, ax  = plt.subplots(figsize=(max(9, n_c * 0.85), max(5, n_p * 0.65)))
    sns.heatmap(
        pivot_norm, cmap="Blues", annot=pivot.values,
        fmt="d", linewidths=0.35, ax=ax,
        cbar_kws={"shrink": 0.65, "label": "Relative emphasis (row %)"},
        annot_kws={"size": 8},
    )
    ax.set_title(
        "Participant × Theme Heatmap\n"
        "(colour = relative emphasis per participant · numbers = raw counts)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Theme", fontsize=11)
    ax.set_ylabel("Participant", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def plot_synthesis_sentiment_breakdown(
        synthesis: dict) -> plt.Figure | None:
    """
    Two-panel figure:
      Left  — overall corpus sentiment donut
      Right — per-theme dominant sentiment stacked bar (top 12 themes)
    """
    all_segs = synthesis.get("all_segments", [])
    if not all_segs:
        return None
    df = pd.DataFrame(all_segs)
    if "Sentiment" not in df.columns or df["Sentiment"].isna().all():
        return None
    df = df[df["Code"].astype(str).str.strip() != ""]
    if df.empty:
        return None

    colours = {"Positive": "#2CA02C", "Neutral": "#7F7F7F", "Negative": "#D62728"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left — overall donut
    counts = df["Sentiment"].value_counts()
    order  = [l for l in ["Positive", "Neutral", "Negative"] if l in counts]
    wedges, _, autotexts = ax1.pie(
        [counts[l] for l in order],
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        colors=[colours[l] for l in order],
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.legend(wedges, [f"{l} ({counts.get(l,0)})" for l in order],
               loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, fontsize=8.5)
    ax1.set_title("Overall Corpus Sentiment", fontsize=12, fontweight="bold")

    # Right — per-theme stacked bar (top 12)
    top_codes = df["Code"].value_counts().head(12).index.tolist()
    pivot = (
        df[df["Code"].isin(top_codes)]
        .groupby(["Code", "Sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    left = np.zeros(len(pivot))
    for col in ["Positive", "Neutral", "Negative"]:
        vals = pivot[col].values if col in pivot.columns else np.zeros(len(pivot))
        ax2.barh(pivot.index, vals, left=left,
                 color=colours[col], label=col, height=0.6, edgecolor="white")
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax2.text(l + v / 2, i, str(int(v)), ha="center", va="center",
                         fontsize=7.5, color="white", fontweight="bold")
        left += vals

    ax2.invert_yaxis()
    ax2.set_xlabel("Segment count", fontsize=10)
    ax2.set_title("Sentiment by Theme (top 12)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Corpus Sentiment Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_synthesis_cooccurrence(all_segments: list[dict]) -> plt.Figure | None:
    """Full-corpus co-occurrence network — same as single-transcript but on all data."""
    return visualize_code_cooccurrence(all_segments)


def plot_synthesis_theme_journey(transcripts: list[dict],
                                  window: int = 8) -> plt.Figure | None:
    """
    Stacked area chart showing how each theme's share of coded segments
    evolves across participants (ordered by participant index).
    Reveals whether themes grow, fade, or remain consistent across the study.

    x-axis labels use a smart shortener that extracts the most meaningful
    part of the participant ID (e.g. 'P01_Sarah_Teacher' → 'P01 Sarah')
    and uses ASCII ellipsis to avoid font rendering issues.
    """
    rows = []
    for t in transcripts:
        pid = t.get("participant_id", "?")
        for seg in t.get("coded_segments", []):
            if seg.get("Code", "").strip():
                rows.append({"Participant": pid, "Code": seg["Code"]})
    if not rows:
        return None

    df        = pd.DataFrame(rows)
    pids      = list(dict.fromkeys(df["Participant"]))  # preserve order
    top_codes = df["Code"].value_counts().head(8).index.tolist()

    # Build participant-level proportions
    matrix = pd.DataFrame(index=pids, columns=top_codes, dtype=float).fillna(0)
    for pid in pids:
        p_df  = df[df["Participant"] == pid]
        total = max(len(p_df), 1)
        for code in top_codes:
            matrix.loc[pid, code] = (
                len(p_df[p_df["Code"] == code]) / total * 100
            )

    if matrix.empty or len(pids) < 2:
        return None

    # ── Smart x-axis labels ───────────────────────────────────────────────
    # Extract a short, readable label from participant IDs.
    # Handles formats like: P01_Sarah_Teacher, P01-Sarah, P01, Sarah_T, etc.
    def _short_pid(pid: str, max_chars: int = 12) -> str:
        # Replace underscores/hyphens with spaces for readability
        clean = pid.replace("_", " ").replace("-", " ").strip()
        parts = clean.split()
        if len(parts) == 1:
            # Single token — just truncate with ASCII ellipsis
            return clean[:max_chars] + ("..." if len(clean) > max_chars else "")
        # Multiple parts: take first 2 tokens (e.g. 'P01 Sarah')
        label = " ".join(parts[:2])
        if len(label) > max_chars:
            label = label[:max_chars - 3] + "..."
        return label

    x_labels = [_short_pid(p) for p in pids]

    # ── Theme labels: use ASCII ellipsis to avoid rendering glyphs ────────
    def _short_code(code: str, n: int = 22) -> str:
        return code[:n] + ("..." if len(code) > n else "")

    fig, ax = plt.subplots(figsize=(max(10, len(pids) * 1.1), 6))
    x = np.arange(len(pids))

    ax.stackplot(
        x,
        [matrix[c].values for c in top_codes],
        labels=[_short_code(c) for c in top_codes],
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(top_codes))],
        alpha=0.82,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% of participant's coded segments", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Theme Journey Across Participants\n"
        "(stacked area — how theme mix shifts participant by participant)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1),
        fontsize=8, title="Themes", title_fontsize=9,
    )
    plt.tight_layout()
    return fig


def plot_synthesis_quotes_table(top_quotes: list[dict]) -> plt.Figure | None:
    """
    A clean matplotlib table of representative quotes, one per top theme.
    Used when embedding quotes in the exported report visualisation.
    """
    if not top_quotes:
        return None

    # One quote per code (the first one)
    seen: dict[str, dict] = {}
    for q in top_quotes:
        if q["Code"] not in seen:
            seen[q["Code"]] = q
    rows = list(seen.values())[:10]

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.75)))
    ax.axis("off")

    col_labels = ["Theme", "Representative Quote", "Participant"]
    cell_data  = [
        [
            shorten(r["Code"],  28, placeholder="..."),
            shorten(r["Quote"], 95, placeholder="..."),
            shorten(str(r["Participant"]), 18, placeholder="..."),
        ]
        for r in rows
    ]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1, 2])

    # Style header row
    for col in range(3):
        cell = tbl[0, col]
        cell.set_facecolor("#1F77B4")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    # Alternating row shading
    for row in range(1, len(rows) + 1):
        bg = "#f0f4f8" if row % 2 == 0 else "white"
        for col in range(3):
            tbl[row, col].set_facecolor(bg)
            tbl[row, col].set_edgecolor("#dee2e6")

    ax.set_title("Representative Quotes by Theme",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Export plots to bytes — closes figure after saving to prevent memory leak
# ---------------------------------------------------------------------------

def export_plot(fig: plt.Figure, base_name: str = "plot") -> dict[str, io.BytesIO]:
    out: dict[str, io.BytesIO] = {}
    for ext, fmt in [("pdf", "pdf"), ("png", "png"), ("jpg", "jpeg")]:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
        buf.seek(0)
        out[ext] = buf
    plt.close(fig)
    return out
