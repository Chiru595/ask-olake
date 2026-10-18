#!/usr/bin/env python3
"""Evaluation runner for Ask OLake.

Sends every question in a JSONL eval file to POST {base}/ask and scores the
responses: response contract, refusal behaviour, source hits (page level),
key-fact coverage, and latency. Optionally verifies that returned source URLs
resolve (--check-urls).

Usage:
    python eval/run_eval.py --base-url http://localhost:7860 --check-urls

Only depends on httpx (Python 3.11+).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

REFUSAL_PREFIX = "I couldn't find this in the OLake documentation."
# Extra phrasing counted as a refusal, but only when sources == [].
OUTSIDE_SCOPE_MARKERS = ("outside", "out of scope", "out-of-scope", "only answer questions about olake")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: invalid JSON: {exc}")
    return questions


def normalize_page_url(url: str) -> str:
    """Page-level form of a URL: lowercase scheme/host, no fragment or query,
    no trailing slash (so `/docs/x` and `/docs/x/#anchor` compare equal)."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def is_abs_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def check_contract(body: Any) -> tuple[bool, list[str]]:
    """Response must be exactly {"answer": str, "sources": list[str]}."""
    problems: list[str] = []
    if not isinstance(body, dict):
        return False, ["body is not a JSON object"]
    keys = set(body.keys())
    if keys != {"answer", "sources"}:
        extra = sorted(keys - {"answer", "sources"})
        missing = sorted({"answer", "sources"} - keys)
        if extra:
            problems.append(f"extra keys: {extra}")
        if missing:
            problems.append(f"missing keys: {missing}")
    if "answer" in body and not isinstance(body["answer"], str):
        problems.append("answer is not a string")
    if "sources" in body:
        if not isinstance(body["sources"], list):
            problems.append("sources is not a list")
        elif not all(isinstance(s, str) for s in body["sources"]):
            problems.append("sources contains non-string items")
    return not problems, problems


def detect_refusal(answer: str, sources: list[str]) -> bool:
    text = answer.strip()
    if text.startswith(REFUSAL_PREFIX):
        return True
    if not sources and any(m in text.lower() for m in OUTSIDE_SCOPE_MARKERS):
        return True
    return False


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
async def ask_one(client: httpx.AsyncClient, base_url: str, q: dict[str, Any],
                  sem: asyncio.Semaphore) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": q["id"],
        "category": q.get("category"),
        "question": q["question"],
        "must_refuse": bool(q.get("must_refuse")),
        "expected_urls": q.get("expected_urls", []),
        "key_facts": q.get("key_facts", []),
    }
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{base_url.rstrip('/')}/ask", json={"question": q["question"]})
            latency = time.perf_counter() - t0
        except httpx.HTTPError as exc:
            result.update(status=None, error=f"{type(exc).__name__}: {exc}",
                          latency_s=time.perf_counter() - t0)
            return score(result, None)

    result["status"] = resp.status_code
    result["latency_s"] = latency
    result["x_cache"] = resp.headers.get("X-Cache")
    result["x_top_score_raw"] = resp.headers.get("X-Top-Score")
    result["x_top_score"] = parse_float(result["x_top_score_raw"])
    try:
        body = resp.json()
    except ValueError:
        body = None
        result["error"] = "response is not JSON"
        result["raw_body"] = resp.text[:2000]
    result["response"] = body
    return score(result, body)


def score(result: dict[str, Any], body: Any) -> dict[str, Any]:
    ok_status = result.get("status") == 200
    contract_ok, problems = check_contract(body) if body is not None else (False, ["no JSON body"])
    answer = body.get("answer", "") if isinstance(body, dict) and isinstance(body.get("answer"), str) else ""
    raw_sources = body.get("sources", []) if isinstance(body, dict) and isinstance(body.get("sources"), list) else []
    sources = [s for s in raw_sources if isinstance(s, str)]

    sources_abs = all(is_abs_http_url(s) for s in raw_sources)
    if not sources_abs:
        problems.append("non-absolute or non-http(s) source URL")
    contract_pass = ok_status and contract_ok and sources_abs

    refused = detect_refusal(answer, sources) if contract_ok else False

    result["contract_pass"] = contract_pass
    result["contract_problems"] = problems if not contract_pass else []
    result["refused"] = refused

    if result["must_refuse"]:
        result["pass"] = bool(contract_pass and refused and sources == [])
        result["source_hit"] = None
        result["facts_hit"] = None
        result["facts_found"] = []
        result["not_refused"] = None
    else:
        expected = {normalize_page_url(u) for u in result["expected_urls"]}
        got = {normalize_page_url(s) for s in sources}
        result["source_hit"] = bool(expected & got)
        result["matched_sources"] = sorted(expected & got)
        low = answer.lower()
        found = [f for f in result["key_facts"] if f.lower() in low]
        result["facts_found"] = found
        result["facts_hit"] = (len(found) / len(result["key_facts"])) if result["key_facts"] else 1.0
        result["not_refused"] = not refused
        result["pass"] = bool(contract_pass and result["source_hit"] and result["not_refused"]
                              and result["facts_hit"] >= 0.5)
    return result


async def check_urls(urls: list[str], concurrency: int) -> dict[str, dict[str, Any]]:
    sem = asyncio.Semaphore(max(concurrency, 4))
    out: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0,
                                 headers={"User-Agent": "ask-olake-eval/1.0"}) as client:
        async def one(url: str) -> None:
            async with sem:
                try:
                    r = await client.head(url)
                    method = "HEAD"
                    if r.status_code >= 400 or r.status_code == 405:
                        r = await client.get(url)
                        method = "GET"
                    out[url] = {"status": r.status_code, "method": method, "final_url": str(r.url)}
                except httpx.HTTPError as exc:
                    out[url] = {"status": None, "method": "HEAD/GET", "error": f"{type(exc).__name__}: {exc}"}

        await asyncio.gather(*(one(u) for u in urls))
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_bool(v: bool | None) -> str:
    return "-" if v is None else ("Y" if v else "n")


def print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'id':<16} {'cat':<22} {'st':>3} {'lat_s':>6} {'cache':<5} {'top':>6} {'ctr':>3} {'ref':>3} {'src':>3} {'facts':>5} {'pass':>4}"
    print(header)
    print("-" * len(header))
    for r in results:
        facts = "-" if r.get("facts_hit") is None else f"{r['facts_hit']:.2f}"
        top = "-" if r.get("x_top_score") is None else f"{r['x_top_score']:.3f}"
        lat = "-" if r.get("latency_s") is None else f"{r['latency_s']:.2f}"
        print(f"{r['id']:<16} {str(r.get('category'))[:22]:<22} {str(r.get('status') or 'ERR'):>3} {lat:>6} "
              f"{(r.get('x_cache') or '-')[:5]:<5} {top:>6} {fmt_bool(r.get('contract_pass')):>3} "
              f"{fmt_bool(r.get('refused')):>3} {fmt_bool(r.get('source_hit')):>3} {facts:>5} "
              f"{'PASS' if r.get('pass') else 'FAIL':>4}")
        if r.get("error"):
            print(f"    error: {r['error']}")
        if r.get("contract_problems"):
            print(f"    contract: {'; '.join(r['contract_problems'])}")


def summarize(results: list[dict[str, Any]], url_checks: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    n = len(results)
    answerable = [r for r in results if not r["must_refuse"]]
    refuse = [r for r in results if r["must_refuse"]]

    def rate(xs: list[bool]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    contract_rate = rate([bool(r.get("contract_pass")) for r in results])
    # Refusal accuracy: must_refuse items refused (with no sources) + answerable items not refused.
    refusal_correct = [bool(r.get("pass")) for r in refuse] + [not r.get("refused") for r in answerable]
    refusal_acc = rate(refusal_correct)
    must_refuse_pass = rate([bool(r.get("pass")) for r in refuse])
    source_hit = rate([bool(r.get("source_hit")) for r in answerable])
    facts_vals = [r["facts_hit"] for r in answerable if r.get("facts_hit") is not None]
    facts_mean = statistics.fmean(facts_vals) if facts_vals else None
    false_refusals = [r["id"] for r in answerable if r.get("refused")]
    overall_pass = rate([bool(r.get("pass")) for r in results])

    have_cache_header = any(r.get("x_cache") for r in results)
    lat_pool = [r for r in results if r.get("status") == 200 and r.get("latency_s") is not None]
    if have_cache_header:
        lat_pool = [r for r in lat_pool if (r.get("x_cache") or "").strip().upper().startswith("MISS")]
    lats = [r["latency_s"] for r in lat_pool]

    summary = {
        "n_questions": n,
        "n_answerable": len(answerable),
        "n_must_refuse": len(refuse),
        "overall_pass_rate": overall_pass,
        "contract_pass_rate": contract_rate,
        "refusal_accuracy": refusal_acc,
        "must_refuse_pass_rate": must_refuse_pass,
        "source_hit_rate": source_hit,
        "mean_facts_coverage": facts_mean,
        "false_refusals": false_refusals,
        "latency_basis": "cache-miss only" if have_cache_header else "all 200 responses (no X-Cache header)",
        "latency_n": len(lats),
        "latency_p50_s": percentile(lats, 50),
        "latency_p95_s": percentile(lats, 95),
        "top_scores_answerable": sorted(r["x_top_score"] for r in answerable if r.get("x_top_score") is not None),
        "top_scores_must_refuse": sorted(r["x_top_score"] for r in refuse if r.get("x_top_score") is not None),
    }
    if url_checks is not None:
        bad = {u: v for u, v in url_checks.items() if v.get("status") != 200}
        summary["urls_checked"] = len(url_checks)
        summary["urls_non_200"] = bad
    return summary


def print_summary(s: dict[str, Any]) -> None:
    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 100:.1f}%"

    def sec(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}s"

    print("\n=== Summary ===")
    print(f"questions            : {s['n_questions']} ({s['n_answerable']} answerable, {s['n_must_refuse']} must-refuse)")
    print(f"overall pass rate    : {pct(s['overall_pass_rate'])}")
    print(f"contract pass rate   : {pct(s['contract_pass_rate'])}")
    print(f"refusal accuracy     : {pct(s['refusal_accuracy'])}  (must-refuse handled: {pct(s['must_refuse_pass_rate'])})")
    print(f"source hit rate      : {pct(s['source_hit_rate'])}")
    print(f"mean facts coverage  : {pct(s['mean_facts_coverage'])}")
    fr = s["false_refusals"]
    print(f"false refusals       : {len(fr)}" + (f"  {fr}" if fr else ""))
    print(f"latency ({s['latency_basis']}, n={s['latency_n']}): p50 {sec(s['latency_p50_s'])}  p95 {sec(s['latency_p95_s'])}")

    print("\n=== X-Top-Score (sorted) — use to pick REFUSAL_THRESHOLD ===")
    fmt = lambda xs: ", ".join(f"{x:.3f}" for x in xs) if xs else "(no X-Top-Score headers)"
    print(f"answerable  : {fmt(s['top_scores_answerable'])}")
    print(f"must_refuse : {fmt(s['top_scores_must_refuse'])}")
    a, r = s["top_scores_answerable"], s["top_scores_must_refuse"]
    if a and r:
        gap = min(a) - max(r)
        print(f"min(answerable)={min(a):.3f}  max(must_refuse)={max(r):.3f}  gap={gap:+.3f}"
              + ("  -> separable" if gap > 0 else "  -> overlapping"))

    if "urls_checked" in s:
        bad = s["urls_non_200"]
        print(f"\n=== URL check: {s['urls_checked']} distinct URLs, {len(bad)} non-200 ===")
        for u, v in sorted(bad.items()):
            print(f"  {v.get('status') or 'ERR'}  {u}" + (f"  ({v['error']})" if v.get("error") else ""))


async def main_async(args: argparse.Namespace) -> int:
    questions = load_questions(Path(args.file))
    if not questions:
        print("no questions found", file=sys.stderr)
        return 2

    sem = asyncio.Semaphore(max(1, args.concurrency))
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        results = await asyncio.gather(*(ask_one(client, args.base_url, q, sem) for q in questions))

    url_checks = None
    if args.check_urls:
        distinct = sorted({s for r in results
                           for s in ((r.get("response") or {}).get("sources") or [])
                           if isinstance(s, str) and is_abs_http_url(s)})
        url_checks = await check_urls(distinct, args.concurrency)
        for r in results:
            srcs = (r.get("response") or {}).get("sources") or []
            r["bad_source_urls"] = [s for s in srcs if isinstance(s, str) and url_checks.get(s, {}).get("status") != 200]

    print_table(results)
    summary = summarize(results, url_checks)
    print_summary(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base_url": args.base_url, "file": args.file,
                               "summary": summary, "results": results,
                               "url_checks": url_checks}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nfull results written to {out}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Ask OLake evaluation set against a running server.")
    p.add_argument("--base-url", default="http://localhost:7860", help="server base URL (default: %(default)s)")
    p.add_argument("--file", default="eval/questions.jsonl", help="JSONL eval file (default: %(default)s)")
    p.add_argument("--concurrency", type=int, default=2, help="parallel /ask requests (default: %(default)s)")
    p.add_argument("--out", default="eval/results.json", help="where to write full results (default: %(default)s)")
    p.add_argument("--check-urls", action="store_true",
                   help="HEAD (fallback GET) every distinct returned source URL and report non-200s")
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds (default: %(default)s)")
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
