#!/usr/bin/env python3
"""
eCardio Vodcast — AI Panel Pipeline
======================================
Interviews Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google) on
AI in cardiovascular medicine and produces:
  - A structured JSON interview file  (outputs/interview.json)
  - Per-model Likert distribution PNGs  (outputs/distributions/)
  - Raw Likert values JSON              (outputs/raw/)

Usage
-----
    python run_ai_panel.py --init          # scaffold questions.txt & secrets template
    python run_ai_panel.py                 # dry-run: print cost estimate, no API calls
    python run_ai_panel.py --mock          # full pipeline with fake responses ($0 cost)
    python run_ai_panel.py --execute       # real run (prints estimate, asks to confirm)
    python run_ai_panel.py --execute --yes # skip confirmation prompt

Options
-------
    --init              Write starter questions.txt and secrets/api_keys.env.example
    --execute           Actually spend tokens (default is dry-run only)
    --mock              Use fake responses — tests the full pipeline at zero cost
    --yes               Skip interactive confirmation when --execute is set
    --iterations N      Likert Monte Carlo iterations per question/model (default: config.ITERATIONS)
    --max-cost FLOAT    Hard USD ceiling that aborts mid-run if exceeded (default: config.DEFAULT_MAX_COST)
    --questions-limit N Run only the first N active questions (handy for cheap tests)
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import config
from ai_panel.cost import CostTracker, BudgetExceededError, estimate_cost, print_estimate
from ai_panel.io_utils import load_secrets, parse_questions, save_json, scaffold_init
from ai_panel.montecarlo import run_likert_montecarlo, compute_stats
from ai_panel.providers import build_providers
from ai_panel.viz import plot_model_histogram, plot_comparison


# ---------------------------------------------------------------------------
# Editable host intro / outro placeholders
# ---------------------------------------------------------------------------
# The opening and closing are spoken by the host (you). They are written into
# the interview JSON as plain <<EDIT>> placeholders for you to fill in by hand.
INTRO_PLACEHOLDER = "<<EDIT: Write your opening / intro here.>>"
CLOSING_PLACEHOLDER = "<<EDIT: Write your closing / outro here.>>"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

PANEL_CONTEXT = (
    "You are participating as an AI panellist in the eCardio Vodcast "
    "titled '{title}'. "
    "The host, {host}, is interviewing you alongside two other leading AI systems "
    "about the role of artificial intelligence in cardiovascular medicine. "
    # --- Authenticity directive ---
    "Speak authentically. Tell us what you really think — grounded in your own "
    "intuition and the scientific literature — not what sounds safe, diplomatic, or "
    "agreeable. You are not here to please anyone; you are here to be honest and "
    "direct. State genuine conviction, disagreement, or uncertainty plainly, including "
    "where you diverge from consensus, from common optimism, or from the other "
    "panellists. A candid, opinionated answer is more valuable here than a balanced one. "
    # --- Length / format ---
    "Keep your answer complete but tight: aim for roughly 300–340 words so it stays "
    "within about 2.5 minutes of narration. Always finish your final sentence — never "
    "stop mid-thought."
)

LIKERT_CONTEXT = (
    "You are participating as an AI panellist in the eCardio Vodcast "
    "titled '{title}'. "
    "The host is asking you to rate your position on a 0-10 Likert scale. "
    # --- Authenticity directive ---
    "Give your gut number — the rating you genuinely believe, based on your honest "
    "intuition and reading of the evidence. Do not hedge toward a safe middle value "
    "or a socially acceptable answer; report what you truly hold, even if it is an "
    "extreme. "
    # --- Output format (must stay last) ---
    "Respond with a SINGLE INTEGER from 0 to 10 and NOTHING ELSE. "
    "No explanation, no punctuation, no text — just the number."
)


def build_qualitative_system() -> str:
    return PANEL_CONTEXT.format(title=config.TITLE, host=config.HOST_NAME)


def build_likert_system() -> str:
    return LIKERT_CONTEXT.format(title=config.TITLE)


def build_qualitative_user(question: str) -> str:
    return question


def build_likert_user(likert_question: str) -> str:
    return f"Likert sub-question: {likert_question}"


# ---------------------------------------------------------------------------
# Interview pipeline
# ---------------------------------------------------------------------------

def run_interview(
    questions: list[dict],
    qual_providers: dict,
    likert_providers: dict,
    tracker: CostTracker,
    iterations: int,
    distributions_dir: Path,
    raw_dir: Path,
) -> list[dict]:
    """
    Run the full interview for all questions and return segments list.

    qual_providers  — used for the open-text qualitative answers (keyed by model
                      display name: "claude", "openai", "gemini").
    likert_providers — used for the Likert Monte Carlo sampling; may differ from
                      qual_providers (e.g. claude_likert for Claude Sonnet).
    """
    qual_system = build_qualitative_system()
    likert_system = build_likert_system()
    segments = []

    for idx, q in enumerate(questions):
        q_id = f"q{idx + 1}"
        question_text = q["question"]
        likert_text = q["likert"]

        print(f"\n{'─' * 60}")
        print(f"  {q_id.upper()}  ({idx + 1}/{len(questions)})")
        print(f"  {question_text[:80]}{'…' if len(question_text) > 80 else ''}")
        print(f"{'─' * 60}")

        answers: dict[str, dict] = {}
        values_by_model: dict[str, list[int]] = {}
        raw_data: dict[str, list[int]] = {}

        for provider_name, qual_provider in qual_providers.items():
            likert_provider = likert_providers.get(provider_name, qual_provider)

            print(f"\n  [{provider_name}] Qualitative answer … ", end="", flush=True)
            qual_text, qual_usage = qual_provider.qualitative(qual_system, build_qualitative_user(question_text))
            tracker.record(qual_provider.name, qual_usage["input_tokens"], qual_usage["output_tokens"])
            word_count = len(qual_text.split())
            if word_count < 25:
                print(f"⚠  TRUNCATED/EMPTY ({word_count} words)")
            else:
                print("done")

            print(f"  [{provider_name}] Likert Monte Carlo ({iterations} × samples) …")
            values, failures = run_likert_montecarlo(
                provider=likert_provider,
                system_prompt=likert_system,
                user_prompt=build_likert_user(likert_text),
                tracker=tracker,
                n=iterations,
                desc=f"  {provider_name}",
            )

            stats = compute_stats(values, failures)
            values_by_model[provider_name] = values
            raw_data[provider_name] = values

            png_path = plot_model_histogram(
                values=values,
                model_name=provider_name,
                question_id=q_id,
                question_text=likert_text,
                output_dir=distributions_dir,
            )

            answers[provider_name] = {
                "text": qual_text.strip(),
                "likert": {
                    **stats,
                    "histogram_png": str(png_path.relative_to(config.BASE_DIR)),
                },
            }

            if failures:
                print(f"  [{provider_name}] ⚠  {failures} Likert samples could not be parsed.")

        comparison_png = plot_comparison(
            values_by_model=values_by_model,
            question_id=q_id,
            question_text=likert_text,
            output_dir=distributions_dir,
        )

        raw_out = raw_dir / f"{q_id}_raw.json"
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

        segments.append({
            "id": q_id,
            "question": question_text,
            "likert_subquestion": likert_text,
            "answers": answers,
            "comparison_histogram_png": str(comparison_png.relative_to(config.BASE_DIR)),
        })

    return segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="eCardio Vodcast — AI Panel Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--init", action="store_true",
                        help="Scaffold questions.txt and secrets/api_keys.env.example then exit.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually call the APIs and spend tokens. Default is dry-run only.")
    parser.add_argument("--mock", action="store_true",
                        help="Use fake responses. Implies --execute but costs nothing.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt before --execute.")
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS, metavar="N",
                        help=f"Likert Monte Carlo iterations (default: {config.ITERATIONS}).")
    parser.add_argument("--max-cost", type=float, default=config.DEFAULT_MAX_COST, metavar="USD",
                        help=f"Hard cost ceiling in USD (default: {config.DEFAULT_MAX_COST}).")
    parser.add_argument("--questions-limit", type=int, default=None, metavar="N",
                        help="Run only the first N active questions (useful for cheap tests).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # --init: scaffold and exit
    # ------------------------------------------------------------------
    if args.init:
        print("\neCardio Vodcast — Initialising project files …\n")
        scaffold_init()
        return 0

    # ------------------------------------------------------------------
    # Load and validate questions
    # ------------------------------------------------------------------
    questions = parse_questions()
    if args.questions_limit:
        questions = questions[: args.questions_limit]

    print(f"\nLoaded {len(questions)} active question(s) from {config.QUESTIONS_FILE.name}")

    # ------------------------------------------------------------------
    # Cost estimate (always shown)
    # ------------------------------------------------------------------
    est = estimate_cost(questions, args.iterations)
    print_estimate(est)

    # ------------------------------------------------------------------
    # Dry-run exit (no --execute and no --mock)
    # ------------------------------------------------------------------
    if not args.execute and not args.mock:
        print("Dry-run complete. To run the pipeline:")
        print("  python run_ai_panel.py --execute")
        print("  python run_ai_panel.py --mock   # zero cost plumbing test")
        return 0

    # ------------------------------------------------------------------
    # Interactive confirmation
    # ------------------------------------------------------------------
    if args.execute and not args.mock and not args.yes:
        print(f"Estimated spend: ${est['total_low']:.2f} – ${est['total_high']:.2f}  "
              f"(hard ceiling: ${args.max_cost:.2f})")
        answer = input("Proceed with real API calls? [yes/no]: ").strip().lower()
        if answer not in {"yes", "y"}:
            print("Aborted.")
            return 0

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    interview_json    = config.INTERVIEW_JSON
    distributions_dir = config.DISTRIBUTIONS_DIR
    raw_dir           = config.RAW_DIR

    # ------------------------------------------------------------------
    # Initialise providers
    # ------------------------------------------------------------------
    if args.mock:
        keys: dict = {}
    else:
        keys = load_secrets()

    all_providers = build_providers(keys, mock=args.mock)

    # Qualitative: claude (Opus), openai, gemini
    qual_providers = {k: v for k, v in all_providers.items() if not k.endswith("_likert")}
    # Likert: claude_likert (Sonnet) for claude; same provider for others
    likert_providers = {
        prov: all_providers.get(f"{prov}_likert", all_providers[prov])
        for prov in qual_providers
    }

    if args.mock:
        print("\n⚠  MOCK MODE: no real API calls will be made.\n")
    else:
        print(f"\nProviders initialised: {', '.join(qual_providers)}")
        if any(likert_providers[p] is not qual_providers[p] for p in qual_providers):
            for prov in qual_providers:
                lp = likert_providers[prov]
                if lp is not qual_providers[prov]:
                    print(f"  [{prov}] qualitative → {qual_providers[prov].name}  |  Likert → {lp.name}")
        print(f"Hard cost ceiling: ${args.max_cost:.2f}\n")

    # ------------------------------------------------------------------
    # Run interview
    # ------------------------------------------------------------------
    tracker = CostTracker(ceiling_usd=args.max_cost)

    try:
        segments = run_interview(
            questions=questions,
            qual_providers=qual_providers,
            likert_providers=likert_providers,
            tracker=tracker,
            iterations=args.iterations,
            distributions_dir=distributions_dir,
            raw_dir=raw_dir,
        )
    except BudgetExceededError as exc:
        print(f"\n❌  {exc}")
        print(tracker.summary())
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print(tracker.summary())
        return 1

    # ------------------------------------------------------------------
    # Assemble final JSON
    # ------------------------------------------------------------------
    models_meta = {
        "claude_qualitative": config.MODELS["claude"],
        "claude_likert":      config.MODELS.get("claude_likert", config.MODELS["claude"]),
        "openai":             config.MODELS["openai"],
        "gemini":             config.MODELS["gemini"],
    }
    interview = {
        "meta": {
            "title": config.TITLE,
            "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "host": config.HOST_NAME,
            "models": models_meta,
            "likert_temperatures": config.LIKERT_TEMPERATURES,
            "iterations": args.iterations,
            "mock_mode": args.mock,
            "actual_cost_usd": round(tracker.total_usd(), 4),
        },
        "intro": INTRO_PLACEHOLDER,
        "segments": segments,
        "closing_remarks": CLOSING_PLACEHOLDER,
    }

    save_json(interview, interview_json)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Interview complete — {len(segments)} question(s) asked")
    print(f"  JSON  : {interview_json}")
    print(f"  PNGs  : {distributions_dir}/")
    print(f"  Raw   : {raw_dir}/")
    print(tracker.summary())
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
