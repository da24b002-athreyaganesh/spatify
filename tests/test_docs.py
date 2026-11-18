"""The documents agree with the code and with each other.

- The LLD's API reference and docs/openapi.json match the gateway's OpenAPI spec and the
  packaged model's signature (scripts/export_api_docs.py).
- Every relative link in the documents points to a file that exists. Images are left out:
  the user manual's screenshots are taken last (REMAINING_PHASES 8.2).
- Every decision the HLD and the architecture cite exists in the decision log.
"""

import importlib.util
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]


def exporter():
    path = ROOT / "scripts/export_api_docs.py"
    spec = importlib.util.spec_from_file_location("export_api_docs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_lld_api_reference_and_openapi_json_are_up_to_date():
    spec_text, lld_text = exporter().render()
    assert (ROOT / "docs/openapi.json").read_text() == spec_text, (
        "docs/openapi.json is stale: run scripts/export_api_docs.py"
    )
    assert (ROOT / "docs/lld.md").read_text() == lld_text, (
        "docs/lld.md's generated sections are stale: run scripts/export_api_docs.py"
    )


def test_the_lld_covers_every_endpoint_the_gateway_serves():
    from spatify.gateway.app import create_app
    from spatify.gateway.settings import Settings

    app = create_app(Settings(), store=object(), registry=object(), model=object())
    lld = (ROOT / "docs/lld.md").read_text()
    for route in app.routes:
        if getattr(route, "include_in_schema", False):
            for method in route.methods - {"HEAD"}:
                assert f"### `{method} {route.path}`" in lld, f"{method} {route.path}"


def test_every_relative_link_in_the_documents_resolves():
    broken = []
    for doc in DOCS:
        text = doc.read_text()
        for match in re.finditer(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", text):  # not images
            target = match.group(1).split("#")[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (doc.parent / unquote(target)).exists():
                broken.append(f"{doc.name}: {target}")
    assert not broken, broken


def test_every_cited_decision_exists():
    logged = set(re.findall(r"^## (D\d+)\.", (ROOT / "docs/decision_log.md").read_text(), re.M))
    for name in ("hld.md", "architecture.md", "lld.md", "security.md", "test_plan.md"):
        cited = set(re.findall(r"\bD\d+\b", (ROOT / "docs" / name).read_text()))
        assert cited <= logged, f"{name} cites {sorted(cited - logged)}"
    assert {f"D{n}" for n in range(1, 45)} <= logged
