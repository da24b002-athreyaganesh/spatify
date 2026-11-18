"""Operator console: the pipeline and its runs, the model versions with rollback, data
drift, alerts and live Grafana panels, and links to every tool, so an operator reaches all
of it from one place.

Admin actions (rollback, starting a run) need the admin key, typed into the sidebar. It
stays in this browser session; the frontend's container never holds it (D32).
"""

import pandas as pd
import streamlit as st

from spatify.frontend import api, ui

ui.page_view("console")


def admin_key() -> str:
    return st.session_state.get("admin_key", "").strip()


def start_run() -> None:
    """Button callback: queue a pipeline run."""
    try:
        run = api.trigger(admin_key())
        st.session_state.notice = ("success", f"Pipeline run {run['id']} queued.")
    except api.GatewayError as err:
        st.session_state.notice = ("error", err)


def roll_back(version: int) -> None:
    """Button callback: make `version` the champion."""
    try:
        st.session_state.notice = ("success", api.rollback(version, admin_key())["detail"])
    except api.GatewayError as err:
        st.session_state.notice = ("error", err)
    st.session_state.rollback_confirm = False


def show_notice() -> None:
    kind, payload = st.session_state.pop("notice", (None, None))
    if kind == "success":
        st.success(payload, icon=":material/check_circle:")
    elif kind == "error":
        ui.show_error(payload, "admin")


def run_button(label: str, key: str) -> None:
    st.button(
        label,
        icon=":material/play_arrow:",
        key=key,
        disabled=not admin_key(),
        on_click=start_run,
        help="Needs the admin key (sidebar).",
    )


def pipeline_tab() -> None:
    try:
        dag = api.pipeline_dag()
    except api.GatewayError as err:
        ui.show_error(err, "pipeline")
        dag = None
    if dag:
        st.subheader("The pipeline", help="DVC runs these stages in order: dvc repro.")
        st.graphviz_chart(ui.dag_dot(dag), width="stretch")
        st.caption(ui.STATUS_LEGEND)
        chart = ui.throughput_chart(dag["stages"])
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        timings = pd.DataFrame(
            [
                {
                    "stage": s["name"],
                    "last result": s["status"],
                    "rows": s["metrics"].get("rows"),
                    "seconds": s["metrics"].get("seconds"),
                    "rows per second": s["metrics"].get("rows_per_s"),
                }
                for s in dag["stages"]
                if s["metrics"]
            ]
        )
        if not timings.empty:
            st.dataframe(timings, hide_index=True, width="stretch")

    st.subheader("Run history")
    try:
        counts = api.pipeline_status()["counts"]
        runs = api.pipeline_runs(20)
    except api.GatewayError as err:
        ui.show_error(err, "runs")
        return
    columns = st.columns(4)
    for column, status in zip(
        columns, ("succeeded", "failed", "running", "requested"), strict=True
    ):
        column.metric(status.capitalize(), counts.get(status, 0), border=True)
    if runs:
        table = pd.DataFrame(runs)
        started = pd.to_datetime(table["started_at"])
        table["seconds"] = (pd.to_datetime(table["finished_at"]) - started).dt.total_seconds()
        st.dataframe(
            table[
                ["id", "trigger", "status", "requested_at", "seconds", "model_version",
                 "promoted", "message"]
            ],
            hide_index=True,
            width="stretch",
        )  # fmt: skip
        latest = runs[0]
        if latest.get("stages"):
            with st.expander(f"Stages of run {latest['id']} ({latest['status']})"):
                stages = pd.DataFrame(
                    [{"stage": n, **s} for n, s in latest["stages"].items()]
                ).rename(columns={"status": "result"})
                st.dataframe(stages, hide_index=True, width="stretch")
    else:
        st.info(
            "No pipeline runs recorded yet. Runs started here, or by new data, appear once "
            "the pipeline worker is running."
        )
    run_button("Run the pipeline now", "run-pipeline")


def models_tab() -> None:
    try:
        data = api.models()
    except api.GatewayError as err:
        ui.show_error(err, "models")
        return
    champion, versions = data["champion"], data["versions"]
    st.metric("Serving now", f"version {champion}" if champion else "no model yet", border=True)
    if not versions:
        st.info("No model versions yet: run the pipeline to train one.")
        return
    table = pd.DataFrame(versions)
    table["aliases"] = table["aliases"].map(", ".join)
    table["like rate %"] = (100 * table["like_rate"]).round(1)
    table["dataset"] = table["dataset_version"].str[:8]
    table["commit"] = table["git_commit"].str[:8]
    st.dataframe(
        table[["version", "aliases", "variant", "created_at", "likes", "skips", "like rate %",
               "dataset", "commit", "description"]],
        hide_index=True,
        width="stretch",
    )  # fmt: skip
    chart = ui.like_rate_chart(versions)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No feedback yet: like rates appear once listeners like or skip songs.")

    st.subheader("Roll back", help="Serve an earlier version again, e.g. after a bad promotion.")
    others = [v["version"] for v in versions if v["version"] != champion]
    if not others:
        st.info("Only one version is registered, so there's nothing to roll back to.")
        return
    default = data["previous"] if data["previous"] in others else others[0]
    target = st.selectbox(
        "Make this version the champion",
        others,
        index=others.index(default),
        format_func=lambda v: f"version {v}" + (" (previous)" if v == data["previous"] else ""),
        key="rollback_target",
    )
    confirmed = st.checkbox(
        f"Listeners will get recommendations from version {target} within about 15 seconds.",
        key="rollback_confirm",
    )
    if not admin_key():
        st.caption("Enter the admin key in the sidebar to roll back.")
    st.button(
        "Roll back",
        type="primary",
        icon=":material/undo:",
        key="rollback",
        disabled=not (admin_key() and confirmed),
        on_click=roll_back,
        args=(target,),
    )


def drift_tab() -> None:
    try:
        report = api.drift()
    except api.GatewayError as err:
        ui.show_error(err, "drift")
        return
    if report is None or not report["batches"]:
        st.info(
            "No drift report yet. When a batch of new tracks lands in data/incoming, the "
            "worker compares each of its audio features with the training data, and the "
            "results appear here.",
            icon=":material/hourglass_empty:",
        )
    else:
        drift_report(report)
    if report is not None and report.get("feedback"):
        feedback_drift(report["feedback"])
    st.subheader("Retrain", help="Start a pipeline run, e.g. after drift shows up.")
    run_button("Retrain now", "retrain")


STATUS_WORDS = {
    "processed": "checked",
    "quarantined": "rejected (quarantined)",
    "failed": "pipeline failed",
    "removed": "removed",
    "new": "waiting",
    "unknown": "checked",
}


def drift_report(report: dict) -> None:
    """The batches table, then one batch's PSI per feature."""
    batches = report["batches"]
    summary = pd.DataFrame(
        [
            {
                "batch": b["name"],
                "arrived": b["first_seen"],
                "result": STATUS_WORDS.get(b["status"], b["status"]),
                "drifted": {True: "yes", False: "no"}.get(b["drifted"], "—"),
                "largest PSI": max((f["psi"] for f in b["features"]), default=None),
                "rows": b["rows"],
                "note": b["message"],
            }
            for b in batches
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    scored = [b for b in batches if b["features"]]
    if not scored:
        return
    chosen = st.selectbox(
        "Batch", scored, format_func=lambda b: b["name"], key="drift_batch"
    )  # newest first, so the latest batch is shown by default
    if chosen["drifted"]:
        st.warning(
            f"{chosen['name']} has drifted from the training data.", icon=":material/warning:"
        )
    st.altair_chart(ui.drift_chart(chosen, report["thresholds"]), width="stretch")
    st.caption(
        "PSI (Population Stability Index) compares the batch with the training data over ten "
        "bins: below 0.1 is stable, 0.1 to 0.2 moderate, 0.2 or more a major shift. A batch "
        "has drifted when any feature is major or two are moderate; drift triggers retraining."
    )
    details = pd.DataFrame(chosen["features"]).rename(
        columns={"ks_stat": "KS statistic", "ks_pvalue": "KS p-value"}
    )
    st.dataframe(details, hide_index=True, width="stretch")


def feedback_drift(report: dict) -> None:
    """The like rate of the latest likes and skips against the ones before them (D37)."""
    st.subheader(
        "Listener feedback",
        help="Has the like rate fallen? A drop while the audio features stay stable means "
        "listeners react differently to the same kind of songs, or a new model serves them "
        "worse (concept drift).",
    )
    recent, before = report["recent_like_rate"], report["baseline_like_rate"]
    now, earlier = st.columns(2)
    now.metric(
        f"Like rate, latest {report['recent_events']:,} ratings",
        "—" if recent is None else f"{100 * recent:.0f}%",
        delta=None if report["drop"] is None else f"{-100 * report['drop']:.0f} points",
        border=True,
    )
    earlier.metric(
        f"Like rate, the {report['baseline_events']:,} before",
        "—" if before is None else f"{100 * before:.0f}%",
        border=True,
    )
    if report["drifted"]:
        st.warning(
            f"The like rate has dropped by {100 * report['drop']:.0f} points. If a new model "
            "was promoted recently, consider rolling back (Models tab).",
            icon=":material/warning:",
        )
    elif not report["enough_feedback"]:
        st.caption("Feedback drift is judged once enough likes and skips have come in.")
    st.caption(
        f"Feedback drift: a drop of {100 * report['threshold']:.0f} points or more. It is "
        "shown and alerted on, but doesn't retrain by itself."
    )


def monitoring_tab() -> None:
    """Firing alerts and their history (D41), and live panels from Grafana (D42)."""
    try:
        alerts = api.alerts()
    except api.GatewayError as err:
        ui.show_error(err, "alerts")
        alerts = None
    if alerts is not None:
        firing = [a for a in alerts if a["status"] == "firing"]
        for a in firing:
            show = st.error if a["severity"] == "critical" else st.warning
            show(
                f"**{ui.md(a['summary'] or a['alertname'])}**  \n{ui.md(a['description'])}",
                icon=":material/notifications_active:",
            )
        if not firing:
            st.success("No alerts are firing.", icon=":material/check_circle:")
        if alerts:
            with st.expander(f"Alert history ({len(alerts)})"):
                table = pd.DataFrame(alerts)[
                    ["alertname", "status", "severity", "starts_at", "ends_at", "summary"]
                ]
                st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "Prometheus checks its alert rules every 5 seconds and Alertmanager sends each "
            "alert here when it starts and when it ends. The rules are listed under Tools → "
            "Prometheus alerts."
        )

    st.subheader(
        "Live from Grafana",
        help="The last 30 minutes, refreshed every 10 seconds. Open Tools → Grafana for the "
        "four full dashboards.",
    )
    for pair in range(0, len(ui.GRAFANA_PANELS), 2):
        for column, (title, uid, panel_id) in zip(
            st.columns(2), ui.GRAFANA_PANELS[pair : pair + 2], strict=False
        ):
            with column:
                st.caption(title)
                st.iframe(ui.grafana_panel(uid, panel_id), height=260)
    st.caption(
        "Blank panels mean Grafana isn't running yet: `docker compose up -d grafana`. "
        "They show the data Prometheus has collected, so they fill in as Spatify is used."
    )


def tools_tab() -> None:
    st.write("Every tool behind Spatify, one click away (each opens in a new tab):")
    for name, url, what in ui.LINKS:
        link, text = st.columns([1, 3], vertical_alignment="center")
        link.link_button(name, url, icon=":material/open_in_new:", width="stretch")
        text.write(what)
    st.caption(
        "The Ray Dashboard opens only while a Ray cluster is running on this laptop "
        "(`ray start --head`); the pipeline's own Ray instances run inside the worker."
    )


st.title("Operator console")
st.write("The pipeline, the models and the data behind Spatify, and links to every tool.")
with st.sidebar:
    st.text_input(
        "Admin key",
        type="password",
        key="admin_key",
        help="Needed to roll back a model or start the pipeline. It's SPATIFY_API_KEY in the "
        "project's .env file, and it stays in this browser tab.",
    )

try:
    health = api.health()
except api.GatewayError as err:
    ui.show_error(err, "health")
    st.stop()
checks = health["checks"]
database, model_server, registry = st.columns(3)
database.metric("Database", "OK" if checks.get("postgres") == "ok" else "Down", border=True)
model_server.metric(
    "Model server", "OK" if checks.get("model_server") == "ok" else "Down", border=True
)
registry.metric("Registry", checks.get("registry", "unknown").capitalize(), border=True)
if health["status"] != "ok":
    st.warning(
        "Recommendations are unavailable until the database and the model server are both up.",
        icon=":material/warning:",
    )
show_notice()

pipeline, models, drift, monitoring, tools = st.tabs(
    ["Pipeline", "Models", "Data drift", "Monitoring", "Tools"]
)
with pipeline:
    pipeline_tab()
with models:
    models_tab()
with drift:
    drift_tab()
with monitoring:
    monitoring_tab()
with tools:
    tools_tab()
