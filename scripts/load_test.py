"""
Load-test the /predict endpoint: fires a pack of concurrent requests and
reports measured throughput (req/s) plus latency percentiles.

Usage:
  python scripts/load_test.py \
      --url http://localhost:8000/predict \
      --image data/camvid/test/0001TP_008550.png \
      --requests 100 --concurrency 10

Point --url at a deployed Cloud Run URL to benchmark the live service instead.
"""
import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


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

    async with httpx.AsyncClient(timeout=timeout) as client:
        start = time.perf_counter()
        tasks = [
            _fire_request(client, url, image_bytes, filename, sem, results)
            for _ in range(total_requests)
        ]
        await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

    return results, total_elapsed


def summarize(results, total_elapsed, total_requests, concurrency):
    latencies = sorted(r[0] for r in results)
    successes = [r for r in results if r[1] == 200]
    failures = [r for r in results if r[1] != 200]

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    print(f"Requests          : {total_requests}")
    print(f"Concurrency (pack): {concurrency}")
    print(f"Total wall time   : {total_elapsed:.2f}s")
    print(f"Throughput        : {total_requests / total_elapsed:.2f} req/s")
    print(f"Success / Failure : {len(successes)} / {len(failures)}")
    print(
        f"Latency min/mean/median: "
        f"{latencies[0]:.3f}s / {statistics.mean(latencies):.3f}s / {statistics.median(latencies):.3f}s"
    )
    print(f"Latency p95/p99/max    : {pct(0.95):.3f}s / {pct(0.99):.3f}s / {latencies[-1]:.3f}s")

    if failures:
        sample = {status for _, status in failures}
        print(f"\nWARNING: {len(failures)} requests failed (status codes seen: {sample})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Target /predict URL")
    parser.add_argument("--image", required=True, help="Path to an image file to upload on every request")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max requests in flight at once (the 'pack' size)"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout, seconds")
    args = parser.parse_args()

    results, total_elapsed = asyncio.run(
        run_load_test(args.url, args.image, args.requests, args.concurrency, args.timeout)
    )
    summarize(results, total_elapsed, args.requests, args.concurrency)


if __name__ == "__main__":
    main()
