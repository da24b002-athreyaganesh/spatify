"""The monitoring configuration (D40-D42), checked without running the stack.

- Every scrape target is a Compose service; every service has a memory limit and publishes
  its ports on localhost only.
- Every PromQL query (dashboards and rules) is attributed, says what it computes, and names
  only metrics that Spatify's code, a recording rule or a pinned exporter provides.
- The dashboards are up to date with build_dashboards.py and follow A4's 7 commandments.
- Prometheus and Alertmanager accept their config, the alert rules pass their unit tests,
  and every dashboard query parses: promtool and amtool run in the pinned images, and these
  checks are skipped where Docker or the image isn't available.
"""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from spatify.frontend import ui

ROOT = Path(__file__).parents[1]
MONITORING = ROOT / "monitoring"
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
PROMETHEUS = yaml.safe_load((MONITORING / "prometheus/prometheus.yml").read_text())
RULE_FILES = sorted((MONITORING / "prometheus/rules").glob("*.yml"))
DASHBOARDS = {
    p.name: json.loads(p.read_text()) for p in (MONITORING / "grafana/dashboards").glob("*.json")
}
PROMETHEUS_IMAGE = COMPOSE["services"]["prometheus"]["image"]
ALERTMANAGER_IMAGE = COMPOSE["services"]["alertmanager"]["image"]

OKABE_ITO = {"#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#000000"}

# Metrics from the pinned exporters and Prometheus itself that the queries use.
EXTERNAL = {
    "up",  # written by Prometheus for every target
    "prometheus_tsdb_head_series",  # Prometheus
    "process_resident_memory_bytes",  # every Python and Go exporter
    "node_cpu_seconds_total",  # node-exporter
    "node_memory_MemTotal_bytes",
    "node_memory_MemAvailable_bytes",
    "node_filesystem_avail_bytes",
    "node_filesystem_size_bytes",
    "pg_up",  # postgres-exporter
    "pg_stat_user_tables_n_live_tup",
    "pg_database_size_bytes",
    "mlflow_http_request_total",  # mlflow server --expose-prometheus (checked on 3.17.1)
}
KEYWORDS = {"by", "without", "or", "and", "unless", "on", "ignoring", "bool", "offset"}
LOOK = ("color", "custom.lineStyle")  # what tells two series in a panel apart


def exported_metrics() -> set[str]:
    """Every series Spatify's own code exports, from its metric definitions."""
    names = set()
    for path in (ROOT / "src/spatify").rglob("*.py"):
        text = path.read_text()
        found = re.findall(r"\b(Counter|Gauge|Histogram|Summary|Info)\(\s*\"([a-z_:]+)\"", text)
        found += [
            (kind, f"spatify_frontend_{name}")  # the frontend's namespaced helper
            for kind, name in re.findall(r"_metric\(\s*(\w+),\s*\"([a-z_]+)\"", text)
        ]
        for kind, name in found:
            names |= {
                "Counter": {name.removesuffix("_total") + "_total"},
                "Gauge": {name},
                "Histogram": {f"{name}_bucket", f"{name}_sum", f"{name}_count"},
                "Summary": {name, f"{name}_sum", f"{name}_count"},
                "Info": {f"{name}_info"},
            }[kind]
    return names


def rules() -> list[dict]:
    return [
        r for f in RULE_FILES for g in yaml.safe_load(f.read_text())["groups"] for r in g["rules"]
    ]


def metric_names(expr: str) -> set[str]:
    """The metric names a PromQL expression reads (not functions, labels or keywords)."""
    expr = re.sub(r"#[^\n]*", "", expr)  # comments
    expr = re.sub(r'"[^"]*"', '""', expr)  # strings
    expr = re.sub(r"\{[^}]*\}", "", expr)  # label matchers
    expr = re.sub(r"\b(by|without)\s*\([^)]*\)", "", expr)  # grouping labels
    expr = re.sub(r"\[[^\]]*\]", "", expr)  # range selectors
    names = re.findall(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?![a-zA-Z0-9_:])(?!\s*\()", expr)
    return set(names) - KEYWORDS


def panels():
    for name, board in DASHBOARDS.items():
        for panel in board["panels"]:
            if panel["type"] != "row":
                yield name, panel


def dashboard_queries() -> list[str]:
    return [t["expr"] for _, p in panels() for t in p["targets"]]


# --- Compose and Prometheus -----------------------------------------------------------


def test_every_scrape_target_is_a_compose_service():
    services = set(COMPOSE["services"])
    jobs = {}
    for job in PROMETHEUS["scrape_configs"]:
        for target in job["static_configs"][0]["targets"]:
            host = target.split(":")[0]
            assert host in services | {"localhost"}, f"{job['job_name']}: {target}"
            jobs[job["job_name"]] = host
    # Every application service is scraped (Postgres through its exporter).
    scraped = set(jobs.values())
    if "postgres-exporter" in scraped:
        scraped.add("postgres")
    for service in ("gateway", "frontend", "worker", "model-server", "mlflow", "postgres"):
        assert service in scraped


def test_every_service_has_a_hard_memory_limit():
    for name, service in COMPOSE["services"].items():
        assert "mem_limit" in service, name
        assert service.get("memswap_limit") == service["mem_limit"], name  # no swap


def test_published_ports_are_on_localhost_only():
    for name, service in COMPOSE["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name}: {port}"


def test_alertmanager_posts_to_the_gateways_alerts_endpoint():
    config = yaml.safe_load((MONITORING / "alertmanager/alertmanager.yml").read_text())
    [receiver] = config["receivers"]
    [webhook] = receiver["webhook_configs"]
    assert webhook["url"] == "http://gateway:8000/alerts" and webhook["send_resolved"]
    assert webhook["http_config"]["authorization"]["type"] == "Bearer"
    from spatify.gateway.app import create_app
    from spatify.gateway.settings import Settings

    app = create_app(Settings(), store=object(), registry=object(), model=object())
    assert any(r.path == "/alerts" and "POST" in r.methods for r in app.routes)


# --- PromQL ---------------------------------------------------------------------------


def test_every_query_is_attributed_and_says_what_it_computes():
    for expr in dashboard_queries():
        assert expr.startswith("# AI-generated PromQL"), expr
        assert "Computes:" in expr.splitlines()[0], expr
    for path in RULE_FILES:
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if re.match(r"\s+- (record|alert):", line):
                comment = []
                for previous in reversed(lines[:i]):
                    if not previous.strip().startswith("#"):
                        break
                    comment.append(previous)
                text = " ".join(comment)
                assert "AI-generated PromQL" in text and "Computes:" in text, (path.name, line)


def test_every_metric_a_query_reads_exists():
    known = exported_metrics() | {r["record"] for r in rules() if "record" in r} | EXTERNAL
    queries = dashboard_queries() + [r["expr"] for r in rules()]
    unknown = {name for q in queries for name in metric_names(q)} - known
    assert not unknown, f"no service exports {sorted(unknown)}"


def test_the_metric_name_reader_ignores_functions_labels_and_strings():
    expr = (
        "# a comment naming not_a_metric\nsum by (le, kind) (label_replace(rate("
        'spatify_x_bucket{endpoint="/r"}[5m]), "kind", "http error", "kind", "http_.*")) '
        "or spatify:y:rate5m * 0"
    )
    assert metric_names(expr) == {"spatify_x_bucket", "spatify:y:rate5m"}


# --- Dashboards -----------------------------------------------------------------------


def builder():
    path = MONITORING / "grafana/build_dashboards.py"
    spec = importlib.util.spec_from_file_location("build_dashboards", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks itself up there
    spec.loader.exec_module(module)
    return module


def test_the_dashboards_are_up_to_date_with_their_builder():
    rendered = builder().render()
    assert set(rendered) == set(DASHBOARDS)
    for name, text in rendered.items():
        stored = (MONITORING / "grafana/dashboards" / name).read_text()
        assert text == stored, f"{name} is stale: run monitoring/grafana/build_dashboards.py"


def colours(node) -> set[str]:
    """Every colour a panel sets (fixed colours, thresholds, value mappings)."""
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("fixedColor", "color") and isinstance(value, str):
                found.add(value)
            found |= colours(value)
    elif isinstance(node, list):
        for item in node:
            found |= colours(item)
    return found


def test_panels_follow_the_seven_commandments():
    ids = {}
    for name, panel in panels():
        where = f"{name}: {panel['title']}"
        assert panel["title"] and panel["description"], where  # 6 and 7
        assert colours(panel) <= OKABE_ITO | {"transparent"}, where  # 1
        defaults = panel["fieldConfig"]["defaults"]
        assert defaults.get("unit"), where  # 4
        if panel["type"] == "timeseries":
            assert defaults["custom"]["axisLabel"].endswith("(linear)"), where  # 3, 4
            overrides = panel["fieldConfig"]["overrides"]
            looks = [
                json.dumps([p["value"] for p in o["properties"] if p["id"] in LOOK])
                for o in overrides
            ]
            assert len(set(looks)) == len(looks), f"{where}: two series look alike"  # 2
            assert panel["options"]["legend"]["showLegend"] == (len(overrides) > 1), where  # 5
        ids.setdefault(name, set())
        assert panel["id"] not in ids[name], f"{where}: duplicate panel id"
        ids[name].add(panel["id"])


def test_the_console_embeds_panels_that_exist():
    by_uid = {board["uid"]: board for board in DASHBOARDS.values()}
    for title, uid, panel_id in ui.GRAFANA_PANELS:
        board = by_uid[uid]
        assert any(p["id"] == panel_id and p["type"] != "row" for p in board["panels"]), title


# --- promtool and amtool, in the pinned images ----------------------------------------


def run_tool(image: str, entrypoint: str, args: list[str], mounts: dict[Path, str]):
    if shutil.which("docker") is None:
        pytest.skip("Docker isn't available")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip(f"{image} isn't pulled (docker pull {image})")
    command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", entrypoint]
    for source, target in mounts.items():
        command += ["-v", f"{source}:{target}:ro"]
    return subprocess.run(command + [image, *args], capture_output=True, text=True, timeout=120)


def test_prometheus_accepts_its_config_and_rules():
    result = run_tool(
        PROMETHEUS_IMAGE, "promtool", ["check", "config", "/etc/prometheus/prometheus.yml"],
        {MONITORING / "prometheus": "/etc/prometheus"},
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_alert_rules_pass_their_unit_tests():
    result = run_tool(
        PROMETHEUS_IMAGE, "promtool", ["test", "rules", "/etc/prometheus/tests/rules_test.yml"],
        {MONITORING / "prometheus": "/etc/prometheus"},
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_dashboard_query_parses(tmp_path):
    """Each query, wrapped as a recording rule, so promtool's parser checks it."""
    queries = dashboard_queries()
    group = {
        "name": "dashboard-queries",
        "rules": [{"record": f"dashboard:query_{i}", "expr": q} for i, q in enumerate(queries)],
    }
    (tmp_path / "queries.yml").write_text(yaml.safe_dump({"groups": [group]}))
    tmp_path.chmod(0o755)  # promtool runs as nobody in its container
    (tmp_path / "queries.yml").chmod(0o644)
    result = run_tool(
        PROMETHEUS_IMAGE, "promtool", ["check", "rules", "/queries/queries.yml"],
        {tmp_path: "/queries"},
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr


def test_alertmanager_accepts_its_config():
    result = run_tool(
        ALERTMANAGER_IMAGE, "amtool", ["check-config", "/etc/alertmanager/alertmanager.yml"],
        {MONITORING / "alertmanager": "/etc/alertmanager"},
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
