"""Start the Streamlit frontend:  python -m spatify.frontend

Streamlit needs the app's file path, which inside the Docker image sits in site-packages;
this finds it and starts Streamlit with production settings.

It also starts the metrics server first. Streamlit runs app.py only when a browser opens
the app, so a metrics server started there would leave Prometheus seeing the frontend
down (and ServiceDown firing) after every restart until someone visited the page.
"""

import os
import sys
from pathlib import Path

from streamlit.web import cli

from spatify.frontend.metrics import start_metrics_server


def main() -> None:
    port = int(os.environ.get("METRICS_PORT", "0"))
    if port:
        start_metrics_server(port)  # app.py's own call then does nothing
    app = Path(__file__).with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app),
        "--server.address=0.0.0.0",
        f"--server.port={os.environ.get('PORT', '8501')}",
        "--server.headless=true",
        "--server.fileWatcherType=none",  # no reloading in production
        "--browser.gatherUsageStats=false",  # no calls home
        "--theme.primaryColor=#0072B2",  # Okabe-Ito blue
    ]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
