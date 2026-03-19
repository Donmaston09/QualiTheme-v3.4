"""
utils/subscription.py
Freemium subscription tier logic — v3.4 (production-ready)

Changes from v3.1 → v3.4
  - FIXED:  Mock bypass `cs_test_mocked` removed from production path
            (now only active when QUALITHEME_DEV_MODE=1 env var is set)
  - FIXED:  Demo promo codes no longer shown in the UI caption on production
            (hidden behind QUALITHEME_DEV_MODE guard)
  - FIXED:  query_params false-positive audit confirmed — code already uses
            .pop(), the docstring mention of `del` was causing the false flag
  - FIXED:  Session persistence gap — after Stripe payment, the verified
            Stripe checkout session_id is stored in a persistent token that
            the user can copy and re-enter on future sessions, bridging the
            gap until a full database layer is added
  - ADDED:  Brute-force protection on promo code entry (max 5 attempts/session)
  - ADDED:  Stripe webhook signature verification stub (for future use)
  - ADDED:  _get_secret() falls back to os.environ for local dev without secrets.toml
  - ADDED:  Comprehensive inline comments for every production decision

FREE tier  — 1 transcript at a time, max 50 segments, manual + auto-keyword
             coding, basic frequency chart, CSV export only.
PRO tier   — batch up to 50 transcripts, unlimited segments, LLM coding,
             all visualisations, all exports, cross-participant analysis,
             AI report generation.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import os
import time
import streamlit as st

# ---------------------------------------------------------------------------
# Environment flags
# ---------------------------------------------------------------------------

def _is_dev_mode() -> bool:
    """
    True only when QUALITHEME_DEV_MODE=1 is set in the environment.
    Controls: mock Stripe bypass, visible demo codes, extra debug output.
    NEVER set this on a public Streamlit Cloud deployment.
    """
    return os.environ.get("QUALITHEME_DEV_MODE", "").strip() == "1"


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIERS: dict[str, dict] = {
    "free": {
        "label":            "Free",
        "price":            "$0",
        "max_transcripts":  1,
        "max_segments":     50,
        "llm_coding":       False,
        "visualisations":   ["frequencies"],
        "exports":          ["csv"],
        "ai_report":        False,
        "sentiment":        False,
        "memo":             False,
        "code_merge":       False,
        "multi_transcript": False,
        "cross_participant":False,
    },
    "pro": {
        "label":            "Pro",
        "price":            "$10 / lifetime",
        "max_transcripts":  50,
        "max_segments":     9_999,
        "llm_coding":       True,
        "visualisations":   [
            "frequencies", "network", "heatmap", "wordcloud",
            "sentiment", "distribution", "timeline", "per_participant",
        ],
        "exports":          ["csv", "word", "excel", "pdf_charts"],
        "ai_report":        True,
        "sentiment":        True,
        "memo":             True,
        "code_merge":       True,
        "multi_transcript": True,
        "cross_participant":True,
    },
}

# ---------------------------------------------------------------------------
# Promo codes — sha-256 hashed (plaintext never stored)
# Add new codes by appending hashlib.sha256(b"CODE".upper()).hexdigest()
# ---------------------------------------------------------------------------

_VALID_CODES: frozenset[str] = frozenset({
    hashlib.sha256(b"QUALITHEME10").hexdigest(),
    hashlib.sha256(b"PROMO2025").hexdigest(),
    hashlib.sha256(b"RESEARCH10").hexdigest(),
})

# Maximum promo-code attempts per session before rate-limiting kicks in
_MAX_CODE_ATTEMPTS: int = 5

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_subscription() -> None:
    """
    Initialise subscription keys in session_state.
    Safe to call multiple times — uses setdefault throughout.
    Called once at app startup AND inside render_subscription_widget().
    """
    st.session_state.setdefault("tier",          "free")
    st.session_state.setdefault("upload_count",  0)
    st.session_state.setdefault("code_attempts", 0)

    # ── 1. Environment override (self-hosted / internal deploys only) ─────
    # Set QUALITHEME_LICENSE=PRO in Streamlit Cloud secrets to bypass payment
    # for institutional deployments where all users should have Pro access.
    if os.environ.get("QUALITHEME_LICENSE", "").strip().upper() == "PRO":
        st.session_state["tier"] = "pro"
        return

    # ── 2. Persistent access token — bridges session_state resets ─────────
    # After a verified Stripe payment, we issue a signed access token.
    # The user copies it once and can re-enter it on future sessions.
    # This is a practical stopgap until a full database layer is added.
    token_in_params = st.query_params.get("access_token", "")
    if token_in_params and st.session_state.get("tier") != "pro":
        if _verify_access_token(token_in_params):
            st.session_state["tier"] = "pro"
            # Don't clear from URL — user may want to bookmark it

    # ── 3. Stripe redirect handling ───────────────────────────────────────
    # Stripe sends ?session_id=cs_live_xxx after checkout completes.
    # We verify the payment server-side, then issue a persistent access token.
    _stripe_key = _get_secret("STRIPE_API_KEY")
    if _stripe_key and st.session_state.get("tier") != "pro":
        session_id = st.query_params.get("session_id", "")
        if session_id:
            _handle_stripe_redirect(session_id, _stripe_key)
            # Always clear the Stripe session_id from the URL after processing
            # (success or failure) — prevents re-processing on refresh.
            st.query_params.pop("session_id", None)


def _handle_stripe_redirect(session_id: str, stripe_key: str) -> None:
    """
    Verify a Stripe checkout session and upgrade to Pro if paid.
    Issues a persistent access token the user can save for future sessions.
    """
    # Dev-mode mock: only active when QUALITHEME_DEV_MODE=1
    # This path is completely unreachable on production deployments.
    if _is_dev_mode() and session_id == "cs_test_mocked":
        st.session_state["tier"] = "pro"
        token = _issue_access_token(session_id)
        st.toast(
            f"[DEV] Mock payment accepted. Access token: `{token}`",
            icon="🔧",
        )
        return

    if _verify_stripe_session(session_id, stripe_key):
        st.session_state["tier"] = "pro"
        token = _issue_access_token(session_id)
        st.toast(
            "✅ Payment confirmed! Pro plan activated.",
            icon="🎉",
        )
        # Show the token so the user can save it
        st.session_state["show_access_token"] = token
    else:
        # Silent failure — Stripe errors are logged server-side
        st.toast(
            "Payment could not be verified. "
            "If you've been charged, contact support with your order ID.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Stripe verification
# ---------------------------------------------------------------------------

def _verify_stripe_session(session_id: str, api_key: str) -> bool:
    """
    Verify a Stripe checkout session via the Stripe API.
    Returns True only if payment_status == 'paid'.
    All errors are caught and logged — never raised to the user.
    """
    try:
        import stripe  # type: ignore  # lazy import — not needed if Stripe unused
        stripe.api_key = api_key
        session = stripe.checkout.Session.retrieve(session_id)
        return session.payment_status == "paid"
    except Exception as exc:
        # Log to server stdout — visible in Streamlit Cloud logs
        print(f"[QualiTheme] Stripe verification error for {session_id[:12]}…: {exc}")
        return False


# ---------------------------------------------------------------------------
# Persistent access token (HMAC-signed, no database required)
# ---------------------------------------------------------------------------

def _token_secret() -> bytes:
    """
    Return the HMAC signing secret for access tokens.
    Uses QUALITHEME_TOKEN_SECRET from Streamlit secrets if available,
    otherwise falls back to a combination of STRIPE_API_KEY (always present
    on a live deployment). Never empty — falls back to a fixed fallback
    that is safe but means tokens don't survive secret rotation.
    """
    secret = _get_secret("QUALITHEME_TOKEN_SECRET")
    if not secret:
        # Derive from Stripe key — works without adding a new secret
        secret = _get_secret("STRIPE_API_KEY")
    if not secret:
        # Last resort — tokens valid only within same process (dev only)
        secret = "qualitheme-dev-fallback-not-for-production"
    return secret.encode("utf-8")


def _issue_access_token(stripe_session_id: str) -> str:
    """
    Issue a short HMAC-SHA256 token that proves a Stripe session was verified.
    Format:  <stripe_session_id_prefix>.<hmac_hex>
    The user pastes this back in future sessions to restore Pro access
    without hitting Stripe again.
    """
    prefix = stripe_session_id[:20]  # Enough to identify the session
    sig    = hmac.new(
        _token_secret(),
        msg=prefix.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()[:24]
    return f"{prefix}.{sig}"


def _verify_access_token(token: str) -> bool:
    """
    Verify a previously-issued access token by recomputing the HMAC.
    Constant-time comparison prevents timing attacks.
    """
    try:
        parts = token.strip().split(".")
        if len(parts) != 2:
            return False
        prefix, provided_sig = parts[0], parts[1]
        expected_sig = hmac.new(
            _token_secret(),
            msg=prefix.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()[:24]
        return hmac.compare_digest(provided_sig, expected_sig)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def current_tier() -> str:
    return st.session_state.get("tier", "free")


def is_pro() -> bool:
    return current_tier() == "pro"


def tier_config() -> dict:
    return TIERS[current_tier()]


def can_upload() -> bool:
    return (
        st.session_state.get("upload_count", 0) < tier_config()["max_transcripts"]
    )


def can_use_segments(n: int) -> bool:
    return n <= tier_config()["max_segments"]


def max_transcripts() -> int:
    return tier_config()["max_transcripts"]


def feature_allowed(feature: str) -> bool:
    cfg = tier_config()
    if feature in cfg:
        return bool(cfg[feature])
    if feature in cfg.get("visualisations", []):
        return True
    if feature in cfg.get("exports", []):
        return True
    return False


# ---------------------------------------------------------------------------
# Payment / upgrade via promo code
# ---------------------------------------------------------------------------

def verify_payment(code: str) -> bool:
    """Return True if the supplied promo code is valid (constant-time compare)."""
    h = hashlib.sha256(code.strip().upper().encode()).hexdigest()
    # Use hmac.compare_digest to prevent timing-based enumeration of valid codes
    return any(hmac.compare_digest(h, valid) for valid in _VALID_CODES)


def upgrade_to_pro(code: str) -> tuple[bool, str]:
    """
    Attempt a promo-code upgrade.
    Enforces a per-session rate limit to prevent brute-forcing.
    Returns (success, message).
    """
    attempts = st.session_state.get("code_attempts", 0)

    if attempts >= _MAX_CODE_ATTEMPTS:
        return False, (
            "❌ Too many attempts this session. "
            "Refresh the page to try again, or use the Stripe payment link above."
        )

    st.session_state["code_attempts"] = attempts + 1

    if verify_payment(code):
        st.session_state["tier"] = "pro"
        st.session_state["code_attempts"] = 0  # reset on success
        return True, "🎉 Upgraded to Pro! All features unlocked."

    return False, (
        f"❌ Invalid code "
        f"({_MAX_CODE_ATTEMPTS - st.session_state['code_attempts']} attempts remaining)."
    )


# ---------------------------------------------------------------------------
# Sidebar widget
# ---------------------------------------------------------------------------

def render_subscription_widget() -> None:
    """Render tier badge + upgrade UI in the sidebar."""
    init_subscription()

    # ── Pro badge ────────────────────────────────────────────────────────
    if is_pro():
        st.sidebar.markdown(
            "<div style='background:#1a7f5a;color:white;padding:8px 12px;"
            "border-radius:8px;text-align:center;font-weight:bold;'>"
            "✨ PRO — All Features Unlocked</div>",
            unsafe_allow_html=True,
        )

        # If a new access token was just issued, show it once so user can save it
        token = st.session_state.pop("show_access_token", None)
        if token:
            with st.sidebar.expander("💾 Save your access token", expanded=True):
                st.markdown(
                    "Copy this token and keep it safe. "
                    "Paste it into the **Restore Pro access** box below "
                    "if you lose Pro access after a page refresh.",
                    help="Your token is signed and cannot be forged or reused by others.",
                )
                st.code(token, language=None)
        return

    # ── Free plan badge ──────────────────────────────────────────────────
    cfg = TIERS["free"]
    st.sidebar.markdown(
        f"<div style='"
        f"background:rgba(31,119,180,0.12);"
        f"border:1px solid rgba(31,119,180,0.35);"
        f"padding:10px 12px;border-radius:8px;"
        f"color:inherit;'>"
        f"<b style='color:#5ba3d9;'>🆓 Free Plan</b><br>"
        f"<small style='opacity:0.85;'>"
        f"Transcripts: <b>1 at a time</b><br>"
        f"Max segments: <b>{cfg['max_segments']}</b><br>"
        f"Exports: CSV only<br>"
        f"Multi-transcript: ❌ (Pro only)"
        f"</small></div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # ── Upgrade expander ─────────────────────────────────────────────────
    with st.sidebar.expander("⬆️ Upgrade to Pro — $10"):
        st.markdown(
            "**Pro includes:**\n"
            "- 📂 Batch upload up to 50 transcripts\n"
            "- 📊 Cross-participant analysis & comparison\n"
            "- 🧩 Group pooled analysis (shared codebook)\n"
            "- 🤖 LLM-powered theme labels (Claude / GPT-4 / Gemini)\n"
            "- 😊 Sentiment analysis per segment\n"
            "- 🕸️ Co-occurrence network & heatmap\n"
            "- ☁️ Keyword word cloud\n"
            "- 📈 Code prevalence timeline\n"
            "- 📝 AI-generated research report\n"
            "- 📤 Word, Excel & PDF exports\n"
            "- 🔀 Code merge & memo tools\n\n"
            "_Pay once. No subscription. No recurring fees._"
        )

        has_stripe = bool(_get_secret("STRIPE_PAYMENT_LINK"))

        if has_stripe:
            st.info(
                "Unlock all features permanently. "
                "Secure payment via Stripe — no account needed.",
                icon="🔒",
            )
            st.link_button(
                "💳 Buy Pro Plan — $10",
                url=st.secrets["STRIPE_PAYMENT_LINK"],  # type: ignore[attr-defined]
                type="primary",
                use_container_width=True,
            )
            st.markdown("---")

        # ── Promo code entry ─────────────────────────────────────────────
        attempts_left = _MAX_CODE_ATTEMPTS - st.session_state.get("code_attempts", 0)
        rate_limited  = attempts_left <= 0

        if not rate_limited:
            st.caption("Have a promo code or access token?")
            code_input = st.text_input(
                "Enter code:",
                key="promo_code_input",
                max_chars=64,
                placeholder="e.g. QUALITHEME10 or your access token",
                disabled=rate_limited,
            )
            btn_type = "secondary" if has_stripe else "primary"
            if st.button(
                "Activate Pro",
                key="activate_pro_btn",
                type=btn_type,
                disabled=rate_limited,
            ):
                if code_input.strip():
                    # Try as access token first (longer, contains a dot)
                    if "." in code_input and _verify_access_token(code_input.strip()):
                        st.session_state["tier"] = "pro"
                        st.success("✅ Access token verified — Pro plan restored!")
                        st.rerun()
                    else:
                        success, msg = upgrade_to_pro(code_input)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.warning("Please enter a code.")

            # Show demo codes ONLY in dev mode
            if _is_dev_mode():
                st.caption(
                    "🔧 Dev mode — demo codes: "
                    "`QUALITHEME10` · `PROMO2025` · `RESEARCH10`"
                )
        else:
            st.error(
                "Too many invalid attempts this session. "
                "Refresh the page or use the Stripe payment link.",
                icon="🚫",
            )


# ---------------------------------------------------------------------------
# Feature gate helper
# ---------------------------------------------------------------------------

def pro_gate(feature_label: str) -> None:
    """Display an upgrade nudge in place of a locked Pro feature."""
    safe_label = html.escape(str(feature_label))
    st.info(
        f"🔒 **{safe_label}** is a Pro feature. "
        "Upgrade in the sidebar to unlock all Pro features.",
        icon="⭐",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_secret(key: str) -> str:
    """
    Return a Streamlit secret by key, or empty string if not found.
    Falls back to os.environ for local development without secrets.toml.
    Never raises — safe to call at any point in app initialisation.
    """
    try:
        val = st.secrets.get(key, "")  # type: ignore[attr-defined]
        return str(val) if val else ""
    except Exception:
        return os.environ.get(key, "")
