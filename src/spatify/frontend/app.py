"""Spatify's web app: page set-up, navigation and the frontend's metrics server.

Run (as the Compose service does):  python -m spatify.frontend
Settings: API_BASE_URL (the gateway), METRICS_PORT (0 turns /metrics off), and the tool
links MLFLOW_UI_URL, GRAFANA_URL, PROMETHEUS_URL, RAY_DASHBOARD_URL, GATEWAY_DOCS_URL.
"""

import os

import streamlit as st

from spatify.frontend.metrics import start_metrics_server

st.set_page_config(page_title="Spatify", page_icon=":material/music_note:", layout="wide")


@st.cache_resource
def metrics_server() -> bool:
    """Start /metrics once per server process (st.cache_resource), not once per page view.
    `python -m spatify.frontend` has already started it; this covers `streamlit run`."""
    port = int(os.environ.get("METRICS_PORT", "0"))
    return start_metrics_server(port) if port else False


metrics_server()
st.navigation(
    [
        st.Page("views/discover.py", title="Discover", icon=":material/queue_music:", default=True),
        st.Page("views/console.py", title="Operator console", icon=":material/monitoring:"),
        st.Page("views/help.py", title="Help", icon=":material/help:"),
    ]
).run()
