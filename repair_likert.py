#!/usr/bin/env python3
"""
Repair script: re-runs Likert sampling ONLY for providers whose
parse_failures == n (i.e., all samples failed), patches interview.json
in-place, and regenerates the affected PNGs.

Usage:
    python repair_likert.py            # dry-run: show what would be re-run
    python repair_likert.py --execute  # actually re-run and patch
"""

import argparse
import json
import sys
from pathlib import Path

import config
from ai_panel.cost import CostTracker, BudgetExceededError
from ai_panel.io_utils import load_secrets, parse_questions
from ai_panel.montecarlo import run_likert_montecarlo, compute_stats
from ai_panel.providers import build_providers
from ai_panel.viz import plot_model_histogram, plot_comparison
from run_ai_panel import build_likert_system, build_likert_user


def find_failed_batches(interview: dict, min_success_rate: float = 0.9) -> list[tuple[str, str, str]]:
    """Return (segment_id, provider_name, likert_question) for batches with too many failures.

    A batch is considered failed if fewer than min_success_rate of its samples parsed
    successfully (default: 90% of iterations must succeed).
    """
    iterations = interview.get("meta", {}).get("iterations", config.ITERATIONS)
    threshold = int(iterations * min_success_rate)
    failed = []
    for seg in interview["segments"]:
        for prov, ans in seg["answers"].items():
            lk = ans["likert"]
            if lk["n"] < threshold:
                failed.append((seg["id"], prov, seg["likert_subquestion"]))
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Repair failed Likert batches")
    parser.add_argument("--execute", action="store_true",
                        help="Actually re-run and patch (default: dry-run)")
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS)
    parser.add_argument("--max-cost", type=float, default=config.DEFAULT_MAX_COST)
    parser.add_argument("--run-id", type=str, default=None, metavar="ID",
                        help="Run ID to repair (matches the --run-id used in run_ai_panel.py). "
                             "Omit to repair the default outputs/interview.json.")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve paths for this run
    # ------------------------------------------------------------------
    suffix = f"_{args.run_id}" if args.run_id else ""
    interview_json    = config.OUTPUTS_DIR / f"interview{suffix}.json"
    distributions_dir = config.OUTPUTS_DIR / f"distributions{suffix}"

    # ------------------------------------------------------------------
    # Load existing interview
    # ------------------------------------------------------------------
    if not interview_json.exists():
        print(f"ERROR: {interview_json} not found. Run the main pipeline first.")
        return 1

    interview = json.loads(interview_json.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Find failed batches
    # ------------------------------------------------------------------
    failed = find_failed_batches(interview)
    if not failed:
        print("No failed Likert batches found — nothing to repair.")
        return 0

    print(f"\nFailed Likert batches ({len(failed)} total):")
    providers_needed = set()
    for seg_id, prov, _ in failed:
        print(f"  {seg_id}  [{prov}]")
        providers_needed.add(prov)

    # Match segment ids to question objects from questions.txt
    questions = parse_questions()
    q_map = {f"q{i+1}": q for i, q in enumerate(questions)}

    # Cost estimate (rough)
    n_batches = len(failed)
    print(f"\nWill re-run {n_batches} batches × {args.iterations} iterations each.")

    if not args.execute:
        print("\nDry-run. Add --execute to actually run.")
        return 0

    # ------------------------------------------------------------------
    # Load providers (only those needed)
    # ------------------------------------------------------------------
    keys = load_secrets()
    all_providers = build_providers(keys, mock=False)
    # For Likert repairs use the dedicated Likert provider when available
    # (e.g. claude_likert → Sonnet with temperature instead of Opus without it).
    providers = {
        p: all_providers.get(f"{p}_likert", all_providers[p])
        for p in providers_needed
    }

    tracker = CostTracker(ceiling_usd=args.max_cost)
    likert_system = build_likert_system()

    try:
        for seg_id, prov_name, likert_text in failed:
            provider = providers[prov_name]
            print(f"\n  Repairing {seg_id} [{prov_name}] — {likert_text[:60]}…")

            user_prompt = build_likert_user(likert_text)
            values, failures = run_likert_montecarlo(
                provider=provider,
                system_prompt=likert_system,
                user_prompt=user_prompt,
                tracker=tracker,
                n=args.iterations,
                desc=f"  {prov_name}",
            )

            if failures:
                print(f"  ⚠  {failures} parse failures remain after repair.")

            stats = compute_stats(values, failures)

            seg = next(s for s in interview["segments"] if s["id"] == seg_id)
            likert_q = seg["likert_subquestion"]
            png = plot_model_histogram(
                values=values,
                model_name=prov_name,
                question_id=seg_id,
                question_text=likert_q,
                output_dir=distributions_dir,
            )

            seg["answers"][prov_name]["likert"] = {
                **stats,
                "histogram_png": str(png.relative_to(config.BASE_DIR)),
            }
            print(f"  Patched: mean={stats['mean']}, n={stats['n']}, parse_failures={failures}")

        affected_segs = {seg_id for seg_id, _, _ in failed}
        for seg_id in affected_segs:
            seg = next(s for s in interview["segments"] if s["id"] == seg_id)
            values_by_model = {
                prov: seg["answers"][prov]["likert"]["values"]
                for prov in seg["answers"]
            }
            comp_png = plot_comparison(
                values_by_model=values_by_model,
                question_id=seg_id,
                question_text=seg["likert_subquestion"],
                output_dir=distributions_dir,
            )
            seg["comparison_histogram_png"] = str(comp_png.relative_to(config.BASE_DIR))

    except BudgetExceededError as exc:
        print(f"\n❌  {exc}")
        print(tracker.summary())
        interview["meta"]["actual_cost_usd"] = round(
            interview["meta"].get("actual_cost_usd", 0) + tracker.total_usd(), 4
        )
        interview_json.write_text(
            json.dumps(interview, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        print(tracker.summary())
        return 1

    # ------------------------------------------------------------------
    # Save patched JSON
    # ------------------------------------------------------------------
    interview["meta"]["actual_cost_usd"] = round(
        interview["meta"].get("actual_cost_usd", 0) + tracker.total_usd(), 4
    )
    interview_json.write_text(
        json.dumps(interview, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"  Repair complete — {len(failed)} batches fixed")
    print(f"  Updated: {interview_json}")
    print(tracker.summary())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
