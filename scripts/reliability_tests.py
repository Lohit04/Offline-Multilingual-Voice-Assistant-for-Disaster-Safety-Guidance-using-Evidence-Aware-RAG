"""Reliability tests against a RUNNING server: repeatability, concurrency, soak.

Usage: python scripts/reliability_tests.py [--base http://localhost:8001] [--soak 30]
Writes experiments/results/reliability_run1.json
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

ANSWERABLE = [
    "What should I do if floodwater enters my house?",
    "What should I do during an earthquake?",
    "How should I prepare for a cyclone?",
    "How can I stay safe during a cold wave?",
    "What should I do if there is a tsunami warning?",
]
OFF_TOPIC = [
    "Who won the cricket match yesterday?",
    "How do I cook chicken biryani?",
    "What is the capital of France?",
]


def ask(base: str, q: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base}/api/ask", data={"question": q, "language": "auto"}, timeout=240)
        body = r.json()
        return {"q": q, "http": r.status_code, "abstained": body.get("abstained"), "text": body.get("text", ""),
                "s": round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {"q": q, "http": None, "abstained": None, "text": f"{type(e).__name__}: {e}", "s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--soak", type=int, default=30)
    args = ap.parse_args()
    report: dict = {}
    failures = []

    print("== 1. repeatability (same question x3 must give identical text) ==", flush=True)
    rep = {}
    for q in ANSWERABLE[:3]:
        outs = [ask(args.base, q) for _ in range(3)]
        same = len({o["text"] for o in outs}) == 1
        rep[q] = same
        print(f"  {'IDENTICAL' if same else 'DIFFERS  '} {q}", flush=True)
        if not same:
            failures.append(f"non-deterministic: {q}")
    report["repeatability"] = rep

    print("== 2. concurrency (8 simultaneous requests) ==", flush=True)
    qs = (ANSWERABLE + OFF_TOPIC)[:8]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(lambda q: ask(args.base, q), qs))
    wall = round(time.perf_counter() - t0, 1)
    ok = all(o["http"] == 200 for o in outs)
    expect_ok = all((o["abstained"] is True) == (o["q"] in OFF_TOPIC) for o in outs if o["http"] == 200)
    for o in outs:
        print(f"  http={o['http']} abstained={o['abstained']} {o['s']}s  {o['q'][:50]}", flush=True)
    print(f"  wall time {wall}s; all 200: {ok}; answer/abstain as expected: {expect_ok}", flush=True)
    report["concurrency"] = {"wall_s": wall, "all_200": ok, "expected_behaviour": expect_ok}
    if not ok or not expect_ok:
        failures.append("concurrency")

    print(f"== 3. soak ({args.soak} sequential mixed requests) ==", flush=True)
    seq = [(ANSWERABLE + OFF_TOPIC)[i % 8] for i in range(args.soak)]
    res = [ask(args.base, q) for q in seq]
    errs = [r for r in res if r["http"] != 200]
    lat = [r["s"] for r in res]
    first, last = lat[: len(lat) // 3], lat[-(len(lat) // 3):]
    behaviour_ok = all((r["abstained"] is True) == (r["q"] in OFF_TOPIC) for r in res if r["http"] == 200)
    print(f"  errors: {len(errs)}; behaviour as expected: {behaviour_ok}; "
          f"median latency first third {statistics.median(first)}s -> last third {statistics.median(last)}s", flush=True)
    report["soak"] = {"n": len(res), "errors": len(errs), "behaviour_ok": behaviour_ok,
                      "median_first": statistics.median(first), "median_last": statistics.median(last)}
    if errs or not behaviour_ok:
        failures.append("soak")

    (ROOT / "experiments" / "results" / "reliability_run1.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nRESULT:", "ALL PASSED" if not failures else f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
