"""Figures logged to MLflow and saved for the report.

Each follows A4's "7 Commandments": a title, named axes with units or scale, a legend
where colour carries meaning, a colour-blind-safe palette (Okabe-Ito) and a one-line
caption saying what the reader should take away.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are only saved

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#555555"  # Okabe-Ito
WEIGHTING_COLOURS = {"uniform": BLUE, "fisher": ORANGE, "feedback": GREEN}


def _caption(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=8, color=GREY, ha="left", va="bottom", wrap=True)


def weights_figure(columns: list[str], weights, variant: str, source: str = "genre separation"):
    """Horizontal bars: the weight given to each feature."""
    order = sorted(range(len(columns)), key=lambda i: weights[i])
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(columns) + 1.6))
    ax.barh([columns[i] for i in order], [weights[i] for i in order], color=BLUE)
    ax.axvline(1.0, color=GREY, linestyle="--", linewidth=1, label="uniform weight (1.0)")
    ax.set_xlabel("weight (multiplies the standardised feature; mean square = 1)")
    ax.set_ylabel("feature")
    ax.set_title(f"Feature weights learned from {source} ({variant})")
    ax.legend(loc="lower right", fontsize=8)
    _caption(
        fig,
        "Longer bars: features where listeners like tracks close to their query."
        if source == "likes and skips"
        else "Longer bars: features whose values differ more between genres than within.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def variants_figure(table: pd.DataFrame, baseline: float, k: int, queries: int):
    """Horizontal bars: genre match@K per candidate, coloured by weighting."""
    table = table.sort_values("genre_match_at_k")
    colours = [WEIGHTING_COLOURS.get(w, GREY) for w in table["weighting"]]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.8))
    ax.barh(table["variant"], 100 * table["genre_match_at_k"], color=colours)
    ax.axvline(100 * baseline, color=GREY, linestyle="--", linewidth=1)
    handles = [
        *(Patch(color=c, label=f"{w} weights") for w, c in WEIGHTING_COLOURS.items()
          if w in set(table["weighting"])),
        Line2D([], [], color=GREY, linestyle="--", label=f"random tracks ({100 * baseline:.1f}%)"),
    ]  # fmt: skip
    ax.set_xlabel(f"genre match@{k} (% of recommendations sharing a genre with the seed)")
    ax.set_ylabel("candidate (feature set - weighting)")
    ax.set_title(f"Candidate models on {queries:,} held-out seed tracks")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _caption(fig, "Higher is better; the best candidate is registered as the challenger.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def feedback_figure(table: pd.DataFrame, random: dict, k: int, lists: int):
    """Horizontal bars: NDCG@k on held-out feedback per candidate (cold-start vs feedback)."""
    if "feedback_ndcg" not in table or table["feedback_ndcg"].isna().all():
        fig, ax = plt.subplots(figsize=(9, 2.2))
        ax.axis("off")
        ax.set_title("Candidate models on held-out feedback")
        ax.text(0.5, 0.45, "No held-out feedback yet: run the user simulator, then retrain.",
                ha="center", va="center", fontsize=11, color=GREY)  # fmt: skip
        return fig
    table = table.dropna(subset=["feedback_ndcg"]).sort_values("feedback_ndcg")
    colours = [WEIGHTING_COLOURS.get(w, GREY) for w in table["weighting"]]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.8))
    ax.barh(table["variant"], table["feedback_ndcg"], color=colours)
    names = {"feedback": "feedback weights (learned)"}
    handles = [
        Patch(color=c, label=names.get(w, f"{w} weights"))
        for w, c in WEIGHTING_COLOURS.items()
        if w in set(table["weighting"])
    ]
    if random.get("ndcg") is not None:
        ax.axvline(random["ndcg"], color=GREY, linestyle="--", linewidth=1)
        handles.append(
            Line2D([], [], color=GREY, linestyle="--", label=f"random order ({random['ndcg']:.2f})")
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"NDCG@{k} on held-out likes and skips (0 to 1, linear scale)")
    ax.set_ylabel("candidate (feature set - weighting)")
    ax.set_title(f"Cold-start vs feedback-trained models on {lists} held-out lists")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    _caption(fig, "Higher: the model puts the songs listeners liked nearer the top of each list.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
