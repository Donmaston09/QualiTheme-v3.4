# 🔍 QualiTheme v3.4 — Universal Thematic Analysis App

A production-ready Streamlit app for qualitative thematic analysis across any research domain — healthcare, education, business, social sciences, UX research, psychology, policy, and more.

---

## What's in v3.4

| Version | Key additions |
|---|---|
| v3.0 | Initial release — single transcript, auto-code, CSV export |
| v3.1 | Security hardening (XSS, file sanitisation, query_params fix), performance (cached embeddings, plt.close) |
| v3.2 | Group Pooled Analysis tab, shared codebook across all participants |
| v3.3 | Audio/Video transcription (faster-whisper, open-source, no API key), colour-highlighted transcript viewer |
| v3.4 | Corpus Synthesis tab (7 charts + full Word report), subscription persistence (HMAC tokens), Gemini model fix, chart embed fix, figure label rendering fixes |

---

## Feature Summary

### Free Plan
- Upload `.txt`, `.docx`, `.pdf`, `.csv` transcripts (1 at a time, 50 segments)
- Manual coding — assign codes segment by segment
- Auto-coding via sentence-transformer embeddings + K-Means
- Code frequency table + bar chart
- CSV export of coded segments

### Pro Plan ($10 one-time)
Everything in Free, plus:
- **Batch upload** up to 50 participant transcripts
- **🎨 Theme-highlighted transcript viewer** — colour-coded segments with hover tooltips and per-theme filter
- **🎙️ Audio/Video transcription** — faster-whisper (local, free, no API key), all formats, translate to English option, SRT subtitle export
- **🏷️ LLM-powered theme labels** — Claude, GPT-4, or Gemini (user supplies key)
- **🧩 Group Pooled Analysis** — single shared codebook across all participants, 5 group-level charts
- **👥 Cross-Participant** — heatmaps, bubble charts, grouped bars
- **🔬 Corpus Synthesis** — full cross-participant analysis with 7 charts + downloadable Word report
  - Corpus frequency chart with participant-count annotations
  - Theme Saturation Map (bubble plot, jitter, quadrant zones, data-quality warning)
  - Participant × Theme Heatmap (row-normalised)
  - Corpus Sentiment breakdown (two-panel)
  - Theme Co-occurrence Network
  - Theme Journey Across Participants (stacked area, smart x-axis labels)
  - Representative Quotes table
- **📝 AI-generated research reports** — individual and synthesis (Word .docx with embedded charts)
- **Sentiment analysis** per segment
- **Co-occurrence network** + heatmap
- **Word cloud**, timeline, theme distribution
- **Word, Excel** (multi-sheet) and **PDF/PNG/JPG** chart exports
- Code merge & memo tools

---

## Project Structure

```
qualitheme/
├── main_app.py                  # Streamlit entry point (2,264 lines, 9 tabs)
├── requirements.txt
├── packages.txt                 # ffmpeg — required for audio/video
├── README.md
├── .gitignore                   # blocks secrets.toml and .env files
└── utils/
    ├── __init__.py
    ├── analysis.py              # NLP, clustering, 20+ visualisation functions
    ├── export.py                # CSV / Word / Excel / AI reports
    ├── ingestion.py             # Multi-format transcript parser (10 MB cap)
    ├── subscription.py          # Free/Pro tier, Stripe, HMAC access tokens
    └── transcription.py         # Audio/video → text via faster-whisper
```

---

## Quick Start (local)

```bash
git clone https://github.com/YOUR_USERNAME/qualitheme.git
cd qualitheme
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run main_app.py
```

Open **http://localhost:8501**.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in → New app
3. Select repo / branch / `main_app.py`
4. **App Settings → Secrets** — paste your keys in TOML format:

```toml
STRIPE_API_KEY          = "sk_live_..."
STRIPE_PAYMENT_LINK     = "https://buy.stripe.com/..."
QUALITHEME_TOKEN_SECRET = "your-32-char-random-string"
```

5. Click **Deploy** — live in ~3 minutes

> Generate `QUALITHEME_TOKEN_SECRET` with:
> `python3 -c "import secrets; print(secrets.token_hex(32))"`

---

## Stripe Payment Flow

After a user pays, Stripe redirects to:
```
https://your-app.streamlit.app/?session_id={CHECKOUT_SESSION_ID}
```

The app verifies the payment server-side, upgrades to Pro, issues a signed HMAC access token, and shows it to the user. They can save and re-enter this token on future sessions to restore Pro access without re-paying.

**Demo promo codes** (visible only when `QUALITHEME_DEV_MODE=1`):
- `QUALITHEME10`  ·  `PROMO2025`  ·  `RESEARCH10`

---

## Audio/Video Transcription

Uses **faster-whisper** (MIT licence, open-source, no API key needed):
- All audio formats: mp3, wav, m4a, ogg, flac, webm, opus, aac
- All video formats: mp4, mov, avi, mkv (ffmpeg extracts audio)
- 4 model sizes: tiny (39 MB) → medium (769 MB)
- Language auto-detection or manual hint (18 languages)
- Optional: translate non-English audio directly to English
- SRT subtitle export alongside plain text

Add `ffmpeg` to `packages.txt` for video and compressed audio support.

---

## Corpus Synthesis Workflow

1. Load and code all participant transcripts (use 5–10 clusters for 10 participants)
2. Open the **🔬 Synthesis** tab → click **Run Corpus Synthesis**
3. Review the saturation table and representative quotes
4. Explore the 7 synthesis charts across sub-tabs
5. Click **Generate Synthesis Report** for a Word .docx with all charts embedded

The report includes an AI-written narrative (or a data-driven fallback if no LLM key), all 7 figures, Appendix A (saturation table), and Appendix B (representative quotes).

---

## LLM Providers

| Provider | Model used | Notes |
|---|---|---|
| **Claude (Anthropic)** | `claude-sonnet-4-6` | Recommended |
| OpenAI GPT-4 | `gpt-4o` | Strong alternative |
| Google Gemini | `gemini-1.5-flash` → `gemini-1.5-pro` → `gemini-2.0-flash` | Tries in order |

Users supply their own API key at runtime. Keys are held in Streamlit session state only — never written to disk, never logged.

---

## Security Notes

- XSS: all user strings escaped with `html.escape()` before `unsafe_allow_html` blocks
- Filenames sanitised before display or use
- File uploads capped at 10 MB (ingestion) and 200 MB (transcription)
- Stripe session IDs verified server-side — never trusted from URL alone
- Pro status persists via HMAC-signed access tokens (no database required)
- Promo code brute-force limited to 5 attempts per session
- Mock payment bypass gated behind `QUALITHEME_DEV_MODE=1` env var
- `.gitignore` blocks `secrets.toml` and all `.env` files
- `enableXsrfProtection = true` in `.streamlit/config.toml`

---

QualiTheme v3.4 · Built with Streamlit · © Anthony Onoja, PhD · University of Surrey
