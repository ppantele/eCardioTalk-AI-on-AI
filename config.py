"""
Central configuration for the eCardio Vodcast pipeline.
Edit this file to change model IDs, costs, iteration counts, and paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SECRETS_FILE = BASE_DIR / "secrets" / "api_keys.env"
QUESTIONS_FILE = BASE_DIR / "questions.txt"
OUTPUTS_DIR = BASE_DIR / "outputs"
DISTRIBUTIONS_DIR = OUTPUTS_DIR / "distributions"
RAW_DIR = OUTPUTS_DIR / "raw"
INTERVIEW_JSON = OUTPUTS_DIR / "interview.json"

# ---------------------------------------------------------------------------
# Interview metadata
# ---------------------------------------------------------------------------
TITLE = "'The Algorithm Will See You Now': AIs Talk on AI in Cardiovascular Healthcare"
HOST_NAME = "Pan G. Pantelidis"

# ---------------------------------------------------------------------------
# Model IDs  —  verify these against your provider dashboards before running.
#
# Flagship defaults as of May 2026 (from public announcements):
#   claude-opus-4-7   — Anthropic flagship  (https://anthropic.com)
#   gpt-5.5           — OpenAI flagship     (https://platform.openai.com/docs/models)
#   gemini-3.5-flash  — Google flagship     (https://ai.google.dev/gemini-api/docs/models)
#
# To use cheaper / faster models for testing, set e.g.:
#   "claude": "claude-sonnet-4-6"
#   "openai": "gpt-4o-mini"
#   "gemini": "gemini-2.0-flash"
# ---------------------------------------------------------------------------
MODELS = {
    "claude": "claude-opus-4-7",           # qualitative answers
    "claude_likert": "claude-sonnet-4-6",  # Likert only (supports temperature; Opus does not)
    "openai": "gpt-5.5",
    "gemini": "gemini-3.5-flash",
}

# ---------------------------------------------------------------------------
# Pricing (USD per 1 000 000 tokens)
# IMPORTANT: verify these before a full run — prices change frequently.
#   Claude Opus 4.7:  https://www.anthropic.com/pricing
#   GPT-5.5:          https://platform.openai.com/docs/models  ($5 in / $30 out per M)
#   Gemini 3.5 Flash: https://ai.google.dev/gemini-api/pricing
# ---------------------------------------------------------------------------
PRICES = {
    "claude":        {"in": 15.00, "out": 75.00},   # Opus 4.7 — verify
    "claude_likert": {"in":  3.00, "out": 15.00},   # Sonnet 4.6 — verify
    "openai":        {"in":  5.00, "out": 30.00},   # GPT-5.5  — from May 2026 announcement
    "gemini":        {"in":  0.30, "out":  2.50},   # Gemini 3.5 Flash estimate — verify
}

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------
# Number of Monte Carlo Likert iterations per question per model.
ITERATIONS = 100

# Temperature for Likert sampling: must be > 0 to produce a real distribution.
# Per-model temperatures — OpenAI and Gemini support up to 2.0, giving wider spreads.
# Anthropic caps at 1.0. Opus 4.7 ignores temperature entirely (uses Sonnet for Likert).
LIKERT_TEMPERATURE = 1.0   # default / fallback
LIKERT_TEMPERATURES = {
    "claude_likert": 1.0,  # Sonnet 4.6 — Anthropic hard cap is 1.0
    "openai":        1.5,  # GPT-5.5 — higher spread without being chaotic
    "gemini":        1.5,  # Gemini 3.5 Flash — same rationale
}

# Temperature for the one-off qualitative open-text answers.
QUALITATIVE_TEMPERATURE = 0.7

# Maximum output tokens for each call type.
LIKERT_MAX_TOKENS = 256    # reasoning models (gpt-5.5, gemini-3.5-flash) need budget for thinking + answer
QUALITATIVE_MAX_TOKENS = 2000 # headroom for reasoning models (gpt-5.5 reasons before writing)
                               # and for Gemini thinking tokens; actual visible output is bounded
                               # by the ~300-340 word prompt target (~480 tokens).

# ---------------------------------------------------------------------------
# Concurrency and resilience
# ---------------------------------------------------------------------------
MAX_WORKERS = 6            # parallel threads per Monte Carlo batch
MAX_RETRIES = 5            # exponential back-off retries on transient errors
MAX_PARSE_ATTEMPTS = 3     # re-sample attempts if Likert output is non-numeric

# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------
DEFAULT_MAX_COST = 12.0    # USD — hard abort ceiling; change with --max-cost
