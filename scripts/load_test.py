"""
Load-test the /predict endpoint: fires a pack of concurrent requests and
reports measured throughput (req/s) plus latency percentiles. Every run is
also appended as a row to results/benchmarks.csv so results stay comparable
across targets (local, Cloud Run, RunPod) and over time.

Usage:
  python scripts/load_test.py \
      --url http://localhost:8000/predict \
      --image data/camvid/test/0001TP_008550.png \
      --requests 100 --concurrency 10 --tag local

Point --url at a deployed Cloud Run or RunPod URL to benchmark that target instead.
"""
import argparse
import asyncio
import csv
import os
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BENCHMARKS_CSV = RESULTS_DIR / "benchmarks.csv"
CSV_FIELDS = [
    "timestamp", "tag", "url", "image", "requests", "concurrency",
    "total_elapsed_s", "throughput_req_s", "success", "failure",
    "latency_min_s", "latency_mean_s", "latency_median_s",
    "latency_p95_s", "latency_p99_s", "latency_max_s",
    "cpu_count", "platform",
]


async def _fire_request(client, url, image_bytes, filename, sem, results):
    async with sem:
        files = {"file": (filename, image_bytes, "image/png")}
        start = time.perf_counter()
        try:
            resp = await client.post(url, files=files)
            status = resp.status_code
        except httpx.HTTPError:
            status = None
        results.append((time.perf_counter() - start, status))


async def run_load_test(url, image_path, total_requests, concurrency, timeout):
    image_bytes = Path(image_path).read_bytes()
    filename = Path(image_path).name
    sem = asyncio.Semaphore(concurrency)
    results = []

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        start = time.perf_counter()
        tasks = [
            _fire_request(client, url, image_bytes, filename, sem, results)
            for _ in range(total_requests)
        ]
        await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

    return results, total_elapsed


def summarize(results, total_elapsed, total_requests, concurrency):
    """Compute throughput/latency stats. Returns the stats dict (also printed)."""
    latencies = sorted(r[0] for r in results)
    successes = [r for r in results if r[1] == 200]
    failures = [r for r in results if r[1] != 200]

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    stats = {
        "requests": total_requests,
        "concurrency": concurrency,
        "total_elapsed_s": total_elapsed,
        "throughput_req_s": total_requests / total_elapsed,
        "success": len(successes),
        "failure": len(failures),
        "latency_min_s": latencies[0],
        "latency_mean_s": statistics.mean(latencies),
        "latency_median_s": statistics.median(latencies),
        "latency_p95_s": pct(0.95),
        "latency_p99_s": pct(0.99),
        "latency_max_s": latencies[-1],
    }

    print(f"Requests          : {stats['requests']}")
    print(f"Concurrency (pack): {stats['concurrency']}")
    print(f"Total wall time   : {stats['total_elapsed_s']:.2f}s")
    print(f"Throughput        : {stats['throughput_req_s']:.2f} req/s")
    print(f"Success / Failure : {stats['success']} / {stats['failure']}")
    print(
        f"Latency min/mean/median: "
        f"{stats['latency_min_s']:.3f}s / {stats['latency_mean_s']:.3f}s / {stats['latency_median_s']:.3f}s"
    )
    print(
        f"Latency p95/p99/max    : "
        f"{stats['latency_p95_s']:.3f}s / {stats['latency_p99_s']:.3f}s / {stats['latency_max_s']:.3f}s"
    )

    if failures:
        sample = {status for _, status in failures}
        print(f"\nWARNING: {len(failures)} requests failed (status codes seen: {sample})")

    return stats


def record(stats: dict, tag: str, url: str, image_path: str) -> None:
    """Append one row to results/benchmarks.csv, creating it with a header if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "tag": tag,
        "url": url,
        "image": Path(image_path).name,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        **stats,
    }
    is_new = not BENCHMARKS_CSV.exists()
    with open(BENCHMARKS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"Recorded to {BENCHMARKS_CSV}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Target /predict URL")
    parser.add_argument("--image", required=True, help="Path to an image file to upload on every request")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max requests in flight at once (the 'pack' size)"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout, seconds")
    parser.add_argument(
        "--tag", default="local", help="Label for this run in results/benchmarks.csv (e.g. local, cloud-run, runpod)"
    )
    parser.add_argument(
        "--no-record", action="store_true", help="Print results only, don't append to results/benchmarks.csv"
    )
    args = parser.parse_args()

    results, total_elapsed = asyncio.run(
        run_load_test(args.url, args.image, args.requests, args.concurrency, args.timeout)
    )
    stats = summarize(results, total_elapsed, args.requests, args.concurrency)

    if not args.no_record:
        record(stats, args.tag, args.url, args.image)


if __name__ == "__main__":
    main()
