"""Cost estimation and live spend tracking with a hard budget ceiling."""

import threading

import config


class BudgetExceededError(RuntimeError):
    pass


class CostTracker:
    """Thread-safe accumulator of token usage and USD spend."""

    def __init__(self, ceiling_usd: float = config.DEFAULT_MAX_COST):
        self.ceiling = ceiling_usd
        self._lock = threading.Lock()
        self._totals: dict[str, dict] = {}

    def record(self, provider: str, input_tokens: int, output_tokens: int) -> None:
        prices = config.PRICES[provider]
        cost = (input_tokens * prices["in"] + output_tokens * prices["out"]) / 1_000_000
        with self._lock:
            if provider not in self._totals:
                self._totals[provider] = {"input": 0, "output": 0, "usd": 0.0}
            self._totals[provider]["input"] += input_tokens
            self._totals[provider]["output"] += output_tokens
            self._totals[provider]["usd"] += cost
            total = sum(v["usd"] for v in self._totals.values())
            if total > self.ceiling:
                raise BudgetExceededError(
                    f"Hard cost ceiling ${self.ceiling:.2f} exceeded "
                    f"(actual spend so far: ${total:.4f}). Aborting."
                )

    def total_usd(self) -> float:
        with self._lock:
            return sum(v["usd"] for v in self._totals.values())

    def summary(self) -> str:
        with self._lock:
            lines = ["", "=== Cost Summary ==="]
            total = 0.0
            for provider, data in self._totals.items():
                lines.append(
                    f"  {provider:<8}  {data['input']:>9,} in / {data['output']:>7,} out"
                    f"  = ${data['usd']:.4f}"
                )
                total += data["usd"]
            lines.append(f"  {'TOTAL':<8}  {'':>9}   {'':>7}       = ${total:.4f}")
            return "\n".join(lines)


def estimate_cost(questions: list, iterations: int) -> dict:
    """
    Return a projected cost range (low = 85% cache hits, high = no caching).

    Qualitative calls use each model's own pricing.
    Likert calls use `<model>_likert` pricing when a separate Likert model exists
    (e.g. claude_likert for Claude Sonnet) and the base model pricing otherwise.
    """
    n_q = len(questions)
    per_provider: dict[str, dict] = {}
    total_low = total_high = 0.0

    base_providers = [p for p in ["claude", "openai", "gemini"] if p in config.PRICES]

    for provider in base_providers:
        q_prices = config.PRICES[provider]
        lk_key = f"{provider}_likert" if f"{provider}_likert" in config.PRICES else provider
        lk_prices = config.PRICES[lk_key]

        q_in = n_q * 300
        q_out = n_q * config.QUALITATIVE_MAX_TOKENS

        mc_in_high = n_q * iterations * 200
        mc_in_low  = n_q * iterations * 200 * 0.15
        mc_out     = n_q * iterations * config.LIKERT_MAX_TOKENS

        cost_high = (
            q_in * q_prices["in"] + q_out * q_prices["out"]
            + mc_in_high * lk_prices["in"] + mc_out * lk_prices["out"]
        ) / 1_000_000
        cost_low = (
            q_in * q_prices["in"] + q_out * q_prices["out"]
            + mc_in_low * lk_prices["in"] + mc_out * lk_prices["out"]
        ) / 1_000_000

        per_provider[provider] = {
            "low": cost_low,
            "high": cost_high,
            "model": config.MODELS[provider],
            "likert_model": config.MODELS[lk_key],
        }
        total_low  += cost_low
        total_high += cost_high

    return {
        "n_questions": n_q,
        "iterations": iterations,
        "per_provider": per_provider,
        "total_low": total_low,
        "total_high": total_high,
    }


def print_estimate(est: dict) -> None:
    n_q = est["n_questions"]
    iterations = est["iterations"]
    qualitative_calls = n_q * 3
    likert_calls = n_q * iterations * 3

    print()
    print("=" * 72)
    print("  eCardio Vodcast — Projected Cost Estimate")
    print("=" * 72)
    print(f"  Questions : {n_q}")
    print(f"  Iterations: {iterations} per question per model")
    print(f"  Calls     : {qualitative_calls} qualitative + {likert_calls:,} Likert = {qualitative_calls + likert_calls:,} total")
    print()
    print(f"  {'Provider':<8}  {'Qual model':<22}  {'Likert model':<22}  {'Low':>7}  {'High':>7}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*22}  {'-'*7}  {'-'*7}")
    for prov, data in est["per_provider"].items():
        same = data["model"] == data["likert_model"]
        likert_col = "(same)" if same else data["likert_model"]
        print(f"  {prov:<8}  {data['model']:<22}  {likert_col:<22}  ${data['low']:>6.2f}  ${data['high']:>6.2f}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*22}  {'-'*7}  {'-'*7}")
    print(f"  {'TOTAL':<8}  {'':22}  {'':22}  ${est['total_low']:>6.2f}  ${est['total_high']:>6.2f}")
    print()
    print("  (Low = 85% cache hits; High = no caching)")
    print("  Verify prices in config.py before running.")
    print("=" * 72)
    print()
