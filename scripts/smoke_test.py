"""End-to-end check of the running Compose stack, through the gateway only.

Run once `docker compose up -d` is up and the registry has a champion:
    python scripts/smoke_test.py [--url http://localhost:8000]

Waits for /health, then exercises search, recommend, feedback, the operator endpoints and
the error paths. Prints each step (the evidence log) and exits non-zero on the first
failure.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAILED: {message}")
        sys.exit(1)


def show(recs: list[dict]) -> None:
    for r in recs:
        name, artists = (r["track_name"] or "")[:36], (r["artists"] or "")[:22]
        genres = ";".join(r["genres"])[:26]
        print(f"   {r['rank']:>2}. {name:36} | {artists:22} | {genres:26} | {r['score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the running stack.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=180, help="seconds to wait for /health")
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=15)

    print(f"== 1. /health (waiting up to {args.wait:.0f} s)")
    deadline = time.monotonic() + args.wait
    while True:
        try:
            health = api.get("/health")
            if health.status_code == 200:
                break
        except httpx.HTTPError:
            health = None
        check(time.monotonic() < deadline, f"not healthy: {health and health.json()}")
        time.sleep(2)
    print(f"   200 {health.json()['checks']}")

    print("== 2. GET /tracks/search?q=unholy")
    found = api.get("/tracks/search", params={"q": "unholy", "limit": 3})
    check(found.status_code == 200 and found.json()["tracks"], f"search: {found.text}")
    seed = found.json()["tracks"][0]
    print(f"   {len(found.json()['tracks'])} hits; seed: {seed['track_name']} ({seed['artists']})")

    print("== 3. POST /recommend with the seed track, k=5")
    shown = api.post("/recommend", json={"track_id": seed["track_id"], "k": 5})
    check(shown.status_code == 200, f"recommend: {shown.status_code} {shown.text}")
    body = shown.json()
    check(isinstance(body["model_version"], int), "response names no model version")
    print(f"   model_version {body['model_version']}, request_id {body['request_id']}")
    show(body["recommendations"])

    print("== 4. POST /recommend with sliders only (energetic, danceable, happy)")
    sliders = {"energy": 0.9, "danceability": 0.85, "valence": 0.9}
    r = api.post("/recommend", json={"sliders": sliders, "k": 5})
    check(r.status_code == 200, f"sliders: {r.text}")
    show(r.json()["recommendations"])

    print("== 5. POST /recommend: the seed, but acoustic and calm")
    override = {"track_id": seed["track_id"], "sliders": {"acousticness": 0.9, "energy": 0.2}}
    r = api.post("/recommend", json={**override, "k": 5})
    check(r.status_code == 200, f"override: {r.text}")
    show(r.json()["recommendations"])

    print("== 6. Bad requests are refused with a reason")
    cases = [
        ("unknown seed track", {"track_id": "not-a-track"}, 404),
        ("no seed and no slider", {}, 422),
        ("slider out of range", {"sliders": {"energy": 2}}, 422),
        ("k too large", {"track_id": seed["track_id"], "k": 500}, 422),
    ]
    for label, payload, expected in cases:
        r = api.post("/recommend", json=payload)
        check(r.status_code == expected, f"{label}: expected {expected}, got {r.status_code}")
        print(f"   {label}: {r.status_code}")

    print("== 7. POST /feedback")
    first, second = body["recommendations"][0], body["recommendations"][1]
    for track, action in ((first, "like"), (second, "skip")):
        r = api.post(
            "/feedback",
            json={
                "request_id": body["request_id"],
                "track_id": track["track_id"],
                "action": action,
            },
        )
        check(r.status_code == 201, f"feedback: {r.text}")
        print(
            f"   {action} {track['track_name']!r}: 201, model_version {r.json()['model_version']}"
        )
    r = api.post(
        "/feedback",
        json={"request_id": body["request_id"], "track_id": seed["track_id"], "action": "like"},
    )
    check(r.status_code == 400, "feedback on a track that wasn't shown should be refused")
    print("   like on a track that wasn't in the list: 400")

    print("== 8. GET /models")
    models = api.get("/models").json()
    check(models["champion"] == body["model_version"], "champion and served version differ")
    for v in models["versions"]:
        print(
            f"   v{v['version']} {v['aliases']} {v['variant']} likes={v['likes']} "
            f"skips={v['skips']} like_rate={v['like_rate']}"
        )

    print("== 9. Admin endpoints need the API key")
    for path in ("/models/rollback", "/pipeline/trigger"):
        r = api.post(path)
        check(r.status_code == 401, f"{path} without a key: {r.status_code}")
        print(f"   POST {path} without X-API-Key: 401")

    print("== 10. GET /pipeline/status, /drift and /metrics")
    r = api.get("/pipeline/status")
    check(r.status_code == 200, f"pipeline status: {r.text}")
    print(f"   /pipeline/status: {r.json()['counts']}")
    print(f"   /drift: {api.get('/drift').status_code}")
    metrics = api.get("/metrics").text
    check("spatify_recommendations_total" in metrics, "metrics missing")
    lines = [m for m in metrics.splitlines() if m.startswith("spatify_recommendations_total")]
    print(f"   /metrics: {'; '.join(lines)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
