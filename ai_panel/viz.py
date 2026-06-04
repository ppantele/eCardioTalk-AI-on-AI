"""
Histogram visualisation for Likert Monte Carlo distributions.

Produces:
  • One per-model histogram per question (outputs/distributions/q{n}_{model}.png)
  • One 3-model overlay/comparison plot per question  (q{n}_comparison.png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for server use
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config

# Consistent colour palette for the three models
MODEL_COLOURS = {
    "claude": "#da7756",   # Anthropic orange
    "openai": "#10a37f",   # OpenAI green
    "gemini": "#4285f4",   # Google blue
}

LIKERT_BINS = list(range(12))  # edges 0..11, giving 11 integer bins 0-10


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_histogram(
    values: list[int],
    model_name: str,
    question_id: str,
    question_text: str,
    output_dir: Path = config.DISTRIBUTIONS_DIR,
) -> Path:
    """
    Save a single-model Likert histogram.

    Args:
        values: List of 0-10 integers.
        model_name: "claude" | "openai" | "gemini".
        question_id: e.g. "q1".
        question_text: Full question string (used in title).
        output_dir: Directory for the PNG.

    Returns:
        Path to the saved PNG.
    """
    colour = MODEL_COLOURS.get(model_name, "#888888")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if values:
        sns.histplot(
            values,
            bins=LIKERT_BINS,
            kde=False,
            color=colour,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
        )
        mean_val = np.mean(values)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_val:.1f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No valid responses", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")

    title = f"{model_name.capitalize()} — {question_id.upper()}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Likert score (0–10)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xticks(range(11))
    ax.set_xlim(-0.5, 10.5)
    ax.tick_params(labelsize=9)

    # Subtitle: truncated question
    short_q = (question_text[:90] + "…") if len(question_text) > 90 else question_text
    fig.text(0.5, 0.01, short_q, ha="center", fontsize=7, color="#555555",
             wrap=True, style="italic")

    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = output_dir / f"{question_id}_{model_name}.png"
    _save(fig, out_path)
    return out_path


def plot_comparison(
    values_by_model: dict[str, list[int]],
    question_id: str,
    question_text: str,
    output_dir: Path = config.DISTRIBUTIONS_DIR,
) -> Path:
    """
    Save a 3-model side-by-side comparison plot for one question.

    Args:
        values_by_model: {"claude": [...], "openai": [...], "gemini": [...]}.
        question_id: e.g. "q1".
        question_text: Full question string.
        output_dir: Directory for the PNG.

    Returns:
        Path to the saved PNG.
    """
    models = list(values_by_model.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(14, 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        values = values_by_model[model_name]
        colour = MODEL_COLOURS.get(model_name, "#888888")

        if values:
            sns.histplot(
                values,
                bins=LIKERT_BINS,
                kde=False,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                ax=ax,
                alpha=0.88,
            )
            mean_val = np.mean(values)
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                       label=f"Mean={mean_val:.1f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="grey")

        ax.set_title(model_name.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Likert score (0–10)", fontsize=9)
        ax.set_xticks(range(11))
        ax.set_xlim(-0.5, 10.5)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Count", fontsize=9)

    main_title = f"Comparison — {question_id.upper()}"
    short_q = (question_text[:110] + "…") if len(question_text) > 110 else question_text
    fig.suptitle(main_title, fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02, short_q, ha="center", fontsize=7.5, color="#555555",
             style="italic")

    fig.tight_layout()

    out_path = output_dir / f"{question_id}_comparison.png"
    _save(fig, out_path)
    return out_path
