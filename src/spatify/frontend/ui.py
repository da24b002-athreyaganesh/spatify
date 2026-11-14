"""Shared pieces of the pages: sliders, messages, the pipeline diagram and the charts.

Charts follow A4's "7 Commandments": a title, a one-line description, named axes with
units and scale, a legend whenever colour carries meaning, and the colour-blind-safe
Okabe-Ito palette.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st

from spatify.frontend.api import GatewayError
from spatify.frontend.metrics import ERRORS_SHOWN, PAGE_VIEWS

# Okabe-Ito colours: distinguishable with every common kind of colour blindness.
BLUE, ORANGE, GREEN, VERMILLION, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999"

# Pipeline stage status -> (fill, text colour) in the diagram.
STATUS_COLOURS = {
    "ok": (GREEN, "white"),
    "failed": (VERMILLION, "white"),
    "running": (BLUE, "white"),
    "not run": ("#E6E6E6", "#333333"),
}
STATUS_LEGEND = (
    ":green[■] last run succeeded · :red[■] failed · :blue[■] running · "
    ":gray[■] not run yet · grey cylinders: input data"
)


@dataclass(frozen=True)
class Slider:
    key: str  # the gateway's name for the feature
    label: str
    hint: str  # what the two ends mean, in plain words
    low: float
    high: float
    step: float
    start: float


SLIDERS = [
    Slider("energy", "Energy", "calm → intense", 0.0, 1.0, 0.05, 0.7),
    Slider(
        "danceability", "Danceability", "hard to dance to → made for dancing", 0.0, 1.0, 0.05, 0.7
    ),
    Slider("valence", "Mood", "sad → happy", 0.0, 1.0, 0.05, 0.6),
    Slider("acousticness", "Acoustic", "electric → acoustic", 0.0, 1.0, 0.05, 0.3),
    Slider("tempo", "Tempo", "beats per minute", 50.0, 200.0, 5.0, 120.0),
    Slider("instrumentalness", "Instrumental", "with vocals → no vocals", 0.0, 1.0, 0.05, 0.1),
]
MORE_SLIDERS = [
    Slider("speechiness", "Spoken words", "all music → all speech", 0.0, 1.0, 0.05, 0.1),
    Slider("liveness", "Live feel", "studio → live recording", 0.0, 1.0, 0.05, 0.2),
    Slider(
        "loudness", "Loudness", "decibels; most songs sit between -15 and -3", -30.0, 0.0, 1.0, -8.0
    ),
]

LINKS = [
    ("MLflow", os.environ.get("MLFLOW_UI_URL", "http://localhost:5000"),
     "experiment runs, model versions, aliases and each model's lineage"),
    ("Grafana", os.environ.get("GRAFANA_URL", "http://localhost:3000"),
     "live dashboards for the service, the model and the data"),
    ("Prometheus alerts", os.environ.get("PROMETHEUS_URL", "http://localhost:9090") + "/alerts",
     "the alerts that are firing"),
    ("Ray Dashboard", os.environ.get("RAY_DASHBOARD_URL", "http://localhost:8265"),
     "the data pipeline's Ray jobs, while a Ray cluster is running"),
    ("Gateway API docs", os.environ.get("GATEWAY_DOCS_URL", "http://localhost:8000/docs"),
     "every endpoint, with a form to try each one"),
]  # fmt: skip


def page_view(page: str) -> None:
    PAGE_VIEWS.labels(page).inc()


def show_error(err: GatewayError, kind: str) -> None:
    """Show a gateway failure as a friendly message (never a stack trace)."""
    ERRORS_SHOWN.labels(kind).inc()
    if err.status is None or err.status >= 500:
        st.error(str(err), icon=":material/cloud_off:")
    else:
        st.warning(str(err), icon=":material/info:")


def md(text: str | None) -> str:
    """Escape Markdown characters, so a title like "*NSYNC" shows as written."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", text or "")


def track_label(track: dict) -> str:
    return f"{track.get('track_name') or 'Untitled'} — {track.get('artists') or 'unknown artist'}"


def slider(spec: Slider) -> float | None:
    """A switch and a slider: the value when switched on, None (no preference) when off."""
    switch, value = st.columns([1, 3], vertical_alignment="center")
    on = switch.toggle(spec.label, key=f"use-{spec.key}")
    chosen = value.slider(
        spec.hint,
        min_value=spec.low,
        max_value=spec.high,
        value=spec.start,
        step=spec.step,
        key=f"value-{spec.key}",
        disabled=not on,
    )
    return float(chosen) if on else None


def key_choice() -> int | None:
    """Major or minor key, or no preference."""
    switch, value = st.columns([1, 3], vertical_alignment="center")
    on = switch.toggle("Key", key="use-mode")
    chosen = value.segmented_control(
        "minor (darker) or major (brighter)",
        ["Minor", "Major"],
        default="Major",
        key="value-mode",
        disabled=not on,
    )
    return (1 if chosen == "Major" else 0) if on and chosen else None


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def dag_dot(dag: dict) -> str:
    """The pipeline as Graphviz DOT: stages coloured by their last result, inputs as data."""
    lines = [
        "digraph pipeline {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [fontname="Helvetica", fontsize=11]; edge [color="#555555"];',
    ]
    for path in dag["inputs"]:
        lines.append(
            f'  "{_quote(path)}" [shape=cylinder, style=filled, fillcolor="#F2F2F2", '
            f'color="{GREY}", fontcolor="#333333", fontsize=9];'
        )
    for stage in dag["stages"]:
        fill, text = STATUS_COLOURS.get(stage["status"], STATUS_COLOURS["not run"])
        label = stage["name"]
        if "rows_per_s" in stage["metrics"]:
            label += f"\\n{stage['metrics']['rows_per_s']:,.0f} rows/s"
        tooltip = f"{stage['status']}: {stage['command']}"
        lines.append(
            f'  "{_quote(stage["name"])}" [shape=box, style="rounded,filled", '
            f'fillcolor="{fill}", fontcolor="{text}", color="{fill}", '
            f'label="{_quote(label)}", tooltip="{_quote(tooltip)}"];'
        )
    for source, target in dag["edges"]:
        lines.append(f'  "{_quote(source)}" -> "{_quote(target)}";')
    lines.append("}")
    return "\n".join(lines)


def throughput_chart(stages: list[dict]) -> alt.Chart | None:
    """Bars: rows per second of each stage that reports it (one series, so no legend)."""
    rows = [
        {"stage": s["name"], "rows_per_s": s["metrics"]["rows_per_s"]}
        for s in stages
        if "rows_per_s" in s["metrics"]
    ]
    if not rows:
        return None
    return (
        alt.Chart(
            pd.DataFrame(rows),
            title=alt.Title(
                "Throughput of the data stages, last run",
                subtitle="Rows each stage processed per second; higher is faster.",
            ),
        )
        .mark_bar(color=BLUE)
        .encode(
            x=alt.X(
                "rows_per_s:Q",
                title="throughput (rows per second, linear scale)",
                axis=alt.Axis(format=",.0f"),
            ),
            y=alt.Y("stage:N", title="pipeline stage", sort=None),
            tooltip=[alt.Tooltip("stage:N"), alt.Tooltip("rows_per_s:Q", format=",.0f")],
        )  # fmt: skip
        .properties(height=40 * len(rows) + 60)
    )


ROLE_COLOURS = {"champion": GREEN, "previous": ORANGE, "challenger": BLUE, "other": GREY}


def like_rate_chart(versions: list[dict]) -> alt.Chart | None:
    """Bars: like rate per model version with feedback, coloured by its alias."""
    rows = []
    for v in versions:
        if v["likes"] + v["skips"] == 0:
            continue
        role = next(
            (a for a in ("champion", "previous", "challenger") if a in v["aliases"]), "other"
        )
        rows.append(
            {
                "version": f"version {v['version']}",
                "like_rate": 100 * (v["like_rate"] or 0.0),
                "role": role,
                "feedback": v["likes"] + v["skips"],
            }
        )
    if not rows:
        return None
    return (
        alt.Chart(
            pd.DataFrame(rows),
            title=alt.Title(
                "Like rate per model version",
                subtitle="Likes as a share of likes and skips; more feedback, steadier numbers.",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "like_rate:Q",
                title="like rate (% of feedback, linear scale)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("version:N", title="model version", sort="-x"),
            color=alt.Color(
                "role:N",
                title="alias",
                scale=alt.Scale(domain=list(ROLE_COLOURS), range=list(ROLE_COLOURS.values())),
            ),
            tooltip=["version", alt.Tooltip("like_rate:Q", format=".1f"), "feedback", "role"],
        )  # fmt: skip
        .properties(height=40 * len(rows) + 60)
    )


LEVEL_COLOURS = {"stable": BLUE, "moderate": ORANGE, "major": VERMILLION}


def drift_chart(batch: dict, thresholds: dict) -> alt.Chart:
    """Bars: PSI per feature for one batch, coloured by level, with the threshold lines."""
    rows = pd.DataFrame(batch["features"])
    bars = (
        alt.Chart(rows)
        .mark_bar()
        .encode(
            x=alt.X("psi:Q", title="PSI against the training data (linear scale)"),
            y=alt.Y("feature:N", title="audio feature", sort="-x"),
            color=alt.Color(
                "level:N",
                title="level",
                scale=alt.Scale(domain=list(LEVEL_COLOURS), range=list(LEVEL_COLOURS.values())),
            ),
            tooltip=[
                "feature",
                alt.Tooltip("psi:Q", format=".3f"),
                "level",
                alt.Tooltip("ks_pvalue:Q", title="KS p-value", format=".2g"),
            ],
        )  # fmt: skip
    )
    lines = pd.DataFrame(
        [
            {"psi": thresholds.get("psi_moderate", 0.1), "label": "moderate"},
            {"psi": thresholds.get("psi_major", 0.2), "label": "major"},
        ]
    )
    rules = alt.Chart(lines).mark_rule(color=GREY, strokeDash=[4, 3]).encode(x="psi:Q")
    labels = (
        alt.Chart(lines)
        .mark_text(align="left", dx=3, dy=-4, color=GREY, fontSize=10)
        .encode(x="psi:Q", y=alt.value(0), text="label:N")
    )
    return (bars + rules + labels).properties(
        title=alt.Title(
            f"Drift of {batch['name']}, by feature",
            subtitle="How far each feature moved from the training data; dashed lines: "
            "0.1 moderate, 0.2 major.",
        ),
        height=28 * len(rows) + 60,
    )
