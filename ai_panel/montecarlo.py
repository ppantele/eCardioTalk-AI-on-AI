"""
Monte Carlo Likert sampling.

For each question × model, calls the provider's likert() method `n` times
concurrently, parses the 0-10 integer from the response, and returns a list
of valid integers plus a parse-failure count.

If a response cannot be parsed as 0-10, it is resampled up to
config.MAX_PARSE_ATTEMPTS times before being recorded as a failure and skipped.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from ai_panel.cost import CostTracker
from ai_panel.providers import ProviderBase


# Regex: find a standalone integer 0-10 anywhere in the response.
_LIKERT_RE = re.compile(r"\b(10|[0-9])\b")


def _parse_likert(text: str) -> int | None:
    """Extract the first 0-10 integer from text. Returns None if not found."""
    m = _LIKERT_RE.search(text.strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val
    return None


def run_single_likert(
    provider: ProviderBase,
    system_prompt: str,
    user_prompt: str,
    tracker: CostTracker,
) -> int | None:
    """
    Execute one Likert sample.  Returns parsed integer, or None on parse failure
    after MAX_PARSE_ATTEMPTS resampling attempts.
    """
    for _ in range(config.MAX_PARSE_ATTEMPTS):
        text, usage = provider.likert(system_prompt, user_prompt)
        tracker.record(provider.name, usage["input_tokens"], usage["output_tokens"])
        value = _parse_likert(text)
        if value is not None:
            return value
    return None  # parse failure


def run_likert_montecarlo(
    provider: ProviderBase,
    system_prompt: str,
    user_prompt: str,
    tracker: CostTracker,
    n: int = config.ITERATIONS,
    desc: str = "",
) -> tuple[list[int], int]:
    """
    Sample the Likert question n times concurrently.

    Returns:
        (values, parse_failures)
        where values is a list of valid integers (may be < n if some failed).
    """
    values: list[int] = []
    parse_failures = 0

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_single_likert, provider, system_prompt, user_prompt, tracker): i
            for i in range(n)
        }
        with tqdm(total=n, desc=desc or f"{provider.name} Likert", unit="sample", leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()  # propagates BudgetExceededError
                if result is None:
                    parse_failures += 1
                else:
                    values.append(result)
                pbar.update(1)

    return values, parse_failures


def compute_stats(values: list[int], parse_failures: int) -> dict:
    """Return summary statistics for a list of 0-10 integers."""
    import statistics

    n = len(values)
    counts = {str(i): values.count(i) for i in range(11)}

    if n == 0:
        return {
            "n": 0,
            "parse_failures": parse_failures,
            "mean": None,
            "median": None,
            "std": None,
            "mode": None,
            "min": None,
            "max": None,
            "counts": counts,
            "values": values,
        }

    return {
        "n": n,
        "parse_failures": parse_failures,
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
        "std": round(statistics.stdev(values) if n > 1 else 0.0, 3),
        "mode": statistics.mode(values),
        "min": min(values),
        "max": max(values),
        "counts": counts,
        "values": values,
    }
