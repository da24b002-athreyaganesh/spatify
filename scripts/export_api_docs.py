"""Export the gateway's OpenAPI spec and write the LLD's interface reference from it.

    uv run python scripts/export_api_docs.py            # write docs/openapi.json and docs/lld.md
    uv run python scripts/export_api_docs.py --check    # fail if either is out of date

Two sections of docs/lld.md are generated, between their BEGIN/END GENERATED markers:
- "api": every gateway endpoint and schema, from FastAPI's OpenAPI spec (the same spec
  /docs shows), so the LLD can't drift from the code;
- "invocations": the model server's /invocations contract, from the signature every
  packaged model is logged with (spatify.model.pyfunc.signature).
The rest of lld.md is written by hand. tests/test_docs.py runs --check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
SPEC = ROOT / "docs/openapi.json"
LLD = ROOT / "docs/lld.md"


def openapi() -> dict:
    """The gateway's spec, from the real app built with stand-ins for its services."""
    from spatify.gateway.app import create_app
    from spatify.gateway.settings import Settings

    app = create_app(Settings(), store=object(), registry=object(), model=object())
    return app.openapi()


def type_of(schema: dict) -> str:
    """A field's type in words: `string (uuid)`, `array of Track`, `integer or null`."""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[1]
        return f"[{name}](#{name.lower()})"
    if "anyOf" in schema:
        return " or ".join(type_of(s) for s in schema["anyOf"])
    if "const" in schema:
        return f'`"{schema["const"]}"`'
    if "enum" in schema:
        return "one of " + ", ".join(f'`"{v}"`' for v in schema["enum"])
    kind = schema.get("type", "any")
    if kind == "array":
        items = schema.get("items") or {}
        if "prefixItems" in schema:
            return "pair of " + ", ".join(type_of(s) for s in schema["prefixItems"])
        return f"array of {type_of(items)}"
    if kind == "object" and isinstance(schema.get("additionalProperties"), dict):
        return f"object of {type_of(schema['additionalProperties'])}"
    return f"{kind} ({schema['format']})" if "format" in schema else kind


def limits(schema: dict) -> str:
    """Constraints and the default, as written in the schema."""
    parts = []
    for key, word in (("minimum", "≥"), ("maximum", "≤"), ("exclusiveMinimum", ">")):
        if key in schema:
            parts.append(f"{word} {schema[key]}")
    for key, word in (("minLength", "length ≥"), ("maxLength", "length ≤"),
                      ("maxItems", "items ≤")):  # fmt: skip
        if key in schema:
            parts.append(f"{word} {schema[key]}")
    for option in schema.get("anyOf", []):
        parts += [p for p in limits(option).split("; ") if p and p not in parts]
    if "default" in schema and schema["default"] not in (None, [], {}):
        parts.append(f"default `{json.dumps(schema['default'])}`")
    return "; ".join(parts)


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def api_section(spec: dict) -> str:
    out = []
    for path, operations in spec["paths"].items():
        for method, op in operations.items():
            out.append(f"### `{method.upper()} {path}`\n")
            purpose = (op.get("description") or op.get("summary") or "").strip()
            out.append(f"{cell(purpose)} *Tag: {', '.join(op.get('tags', []))}.*\n")
            params = op.get("parameters", [])
            auth = [p for p in params if p["in"] == "header"]
            if auth:
                header = "X-API-Key" if auth[0]["name"] == "x-api-key" else "Authorization: Bearer"
                out.append(f"**Needs the admin key** in `{header}`.\n")
            query = [p for p in params if p["in"] != "header"]
            if query:
                out.append("| Parameter | In | Type | Required | Limits | Description |")
                out.append("|---|---|---|---|---|---|")
                for p in query:
                    s = p.get("schema", {})
                    out.append(
                        f"| `{p['name']}` | {p['in']} | {type_of(s)} | "
                        f"{'yes' if p.get('required') else 'no'} | {limits(s)} | "
                        f"{cell(p.get('description') or s.get('description', ''))} |"
                    )
                out.append("")
            body = op.get("requestBody")
            if body:
                schema = body["content"]["application/json"]["schema"]
                optional = "" if body.get("required") else " (optional)"
                out.append(f"**Request body**{optional}: {type_of(schema)}\n")
            out.append("| Status | Meaning | Body |")
            out.append("|---|---|---|")
            for code, answer in op["responses"].items():
                content = answer.get("content", {}).get("application/json", {})
                shape = type_of(content["schema"]) if content.get("schema") else "none"
                out.append(f"| {code} | {cell(answer.get('description', ''))} | {shape} |")
            out.append("")
    out.append("### Schemas\n")
    for name, schema in spec["components"]["schemas"].items():
        out.append(f"#### {name}\n")
        if schema.get("description"):
            out.append(f"{cell(schema['description'])}\n")
        required = set(schema.get("required", []))
        out.append("| Field | Type | Required | Limits | Description |")
        out.append("|---|---|---|---|---|")
        for field, s in schema.get("properties", {}).items():
            out.append(
                f"| `{field}` | {type_of(s)} | {'yes' if field in required else 'no'} | "
                f"{limits(s)} | {cell(s.get('description', ''))} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def invocations_section() -> str:
    """The serving contract from the signature every packaged model carries."""
    from spatify.model.pyfunc import signature
    from spatify.model.recommender import DETAILS, Recommender

    columns = ["energy", "valence"]
    catalogue = pd.DataFrame(
        {"track_id": ["a", "b"], "track_name": ["A", "B"], "artists": ["X", "Y"],
         "album_name": ["P", "Q"], "genres": ["pop", "rock"], "popularity": [50, 60],
         "energy": [0.2, 0.8], "valence": [0.3, 0.7]}
    )[DETAILS + columns]  # fmt: skip
    rec = Recommender(catalogue, columns, np.array([0.5, 0.5]), np.array([0.2, 0.2]), np.ones(2))
    example = rec.recommend(pd.DataFrame([{"track_id": "a"}]), 1).assign(model_run_id="run")
    sig = signature(example, 10)
    out = [
        "**Input columns** (one row per query; send `dataframe_records` or `dataframe_split`):\n"
    ]
    out.append("| Column | Type | Required |")
    out.append("|---|---|---|")
    for col in sig.inputs.to_dict():
        out.append(
            f"| `{col['name']}` | {col['type']} | {'yes' if col.get('required') else 'no'} |"
        )
    out.append("\n**Parameters** (in `params`):\n")
    out.append("| Name | Type | Default |")
    out.append("|---|---|---|")
    for param in sig.params.to_dict():
        out.append(f"| `{param['name']}` | {param['type']} | {param['default']} |")
    out.append("\n**Output columns** (in `predictions`, one row per recommended track):\n")
    out.append("| Column | Type |")
    out.append("|---|---|")
    for col in sig.outputs.to_dict():
        out.append(f"| `{col['name']}` | {col['type']} |")
    return "\n".join(out) + "\n"


def fill(text: str, name: str, content: str) -> str:
    """Replace what's between `name`'s BEGIN and END GENERATED markers."""
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED: {name} .*?-->\n).*?(<!-- END GENERATED: {name} -->)", re.S
    )
    if not pattern.search(text):
        raise SystemExit(f"docs/lld.md has no '{name}' markers")
    return pattern.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(2), text)


def render() -> tuple[str, str]:
    """(openapi.json text, lld.md text) as they should be."""
    spec = openapi()
    lld = fill(LLD.read_text(), "api", api_section(spec))
    lld = fill(lld, "invocations", invocations_section())
    return json.dumps(spec, indent=2) + "\n", lld


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the OpenAPI spec and the LLD tables.")
    parser.add_argument("--check", action="store_true", help="fail if the files are stale")
    args = parser.parse_args()
    spec_text, lld_text = render()
    stale = [p.name for p, text in ((SPEC, spec_text), (LLD, lld_text))
             if not p.exists() or p.read_text() != text]  # fmt: skip
    if args.check:
        if stale:
            sys.exit(
                f"Out of date: {', '.join(stale)}. Run: uv run python scripts/export_api_docs.py"
            )
        print("docs/openapi.json and docs/lld.md are up to date")
        return
    SPEC.write_text(spec_text)
    LLD.write_text(lld_text)
    print(f"Wrote {SPEC.relative_to(ROOT)} and the generated sections of {LLD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
