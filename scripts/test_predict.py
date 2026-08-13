"""
Send one real image to /predict and save the decoded segmentation mask as a
viewable PNG — a reproducible smoke test, not a load test (see load_test.py
for concurrent-throughput testing).

Usage:
  # against a local server
  python scripts/test_predict.py --image data/camvid/test/0001TP_008550.png

  # against the live Cloud Run deployment (auto-discovers the URL via gcloud)
  python scripts/test_predict.py --image data/camvid/test/0001TP_008550.png --cloud-run

  # against an explicit URL
  python scripts/test_predict.py --image data/camvid/test/0001TP_008550.png \
      --url https://pspnet-serving-xyz.a.run.app
"""
import argparse
import base64
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def discover_cloud_run_url(service: str, region: str) -> str:
    result = subprocess.run(
        ["gcloud", "run", "services", "describe", service, "--region", region, "--format", "value(status.url)"],
        capture_output=True,
        text=True,
        check=False,
    )
    url = result.stdout.strip()
    if result.returncode != 0 or not url:
        print(f"ERROR: could not discover Cloud Run URL for service '{service}' in '{region}':", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return url


def run_smoke_test(base_url: str, image_path: str, output_path: str, timeout: float) -> None:
    with httpx.Client(timeout=timeout) as client:
        print(f"GET  {base_url}/health")
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        print(f"  -> {health.json()}")

        image_bytes = Path(image_path).read_bytes()
        files = {"file": (Path(image_path).name, image_bytes, "image/png")}
        print(f"POST {base_url}/predict  (image: {image_path})")
        response = client.post(f"{base_url}/predict", files=files)
        response.raise_for_status()
        body = response.json()

    mask_bytes = base64.b64decode(body["mask_png_base64"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(mask_bytes)

    print(f"  -> inference_time_ms={body['inference_time_ms']}  size={body['width']}x{body['height']}")
    print(f"Saved mask to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to an image file to send")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API (no path)")
    parser.add_argument(
        "--cloud-run", action="store_true", help="Auto-discover the URL from the deployed Cloud Run service"
    )
    parser.add_argument("--service", default="pspnet-serving", help="Cloud Run service name (with --cloud-run)")
    parser.add_argument("--region", default="us-central1", help="Cloud Run region (with --cloud-run)")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the decoded mask PNG (default: results/predict_<timestamp>.png)",
    )
    parser.add_argument("--open", action="store_true", help="Open the saved mask after saving (macOS `open`)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout, seconds")
    args = parser.parse_args()

    base_url = discover_cloud_run_url(args.service, args.region) if args.cloud_run else args.url
    print(f"Target: {base_url}")

    output_path = args.output or RESULTS_DIR / f"predict_{datetime.now():%Y%m%d_%H%M%S}.png"  # noqa: DTZ005 -- local filename label, not time-sensitive logic
    run_smoke_test(base_url, args.image, output_path, args.timeout)

    if args.open:
        subprocess.run(["open", str(output_path)], check=False)


if __name__ == "__main__":
    main()
