import argparse
import json
import mimetypes
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    raise ValueError("Either --text or --text-file must be provided")


def _build_multipart_payload(audio_path: str, text: str, args: argparse.Namespace) -> tuple[bytes, str]:
    boundary = f"----ctc-align-{uuid.uuid4().hex}"
    line_break = "\r\n"

    def field(name: str, value: str) -> bytes:
        head = (
            f"--{boundary}{line_break}"
            f"Content-Disposition: form-data; name=\"{name}\"{line_break}{line_break}"
        )
        return (head + value + line_break).encode("utf-8")

    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    audio_name = Path(audio_path).name
    audio_bytes = Path(audio_path).read_bytes()

    parts = [
        field("text", text),
        field("language", args.language),
        field("romanize", str(args.romanize).lower()),
        field("split_size", args.split_size),
        field("star_frequency", args.star_frequency),
        field("merge_threshold", str(args.merge_threshold)),
    ]

    if args.batch_size is not None:
        parts.append(field("batch_size", str(args.batch_size)))
    if args.window_size is not None:
        parts.append(field("window_size", str(args.window_size)))
    if args.context_size is not None:
        parts.append(field("context_size", str(args.context_size)))

    file_head = (
        f"--{boundary}{line_break}"
        f"Content-Disposition: form-data; name=\"audio\"; filename=\"{audio_name}\"{line_break}"
        f"Content-Type: {content_type}{line_break}{line_break}"
    ).encode("utf-8")

    end = f"{line_break}--{boundary}--{line_break}".encode("utf-8")
    payload = b"".join(parts) + file_head + audio_bytes + end
    return payload, f"multipart/form-data; boundary={boundary}"


def _single_request(url: str, payload: bytes, content_type: str, timeout: float) -> dict:
    start = time.perf_counter()
    req = Request(url=url, data=payload, method="POST")
    req.add_header("Content-Type", content_type)

    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            latency = (time.perf_counter() - start) * 1000.0
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "latency_ms": latency,
                "error": "",
                "bytes": len(body),
            }
    except HTTPError as error:
        latency = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "status": error.code,
            "latency_ms": latency,
            "error": f"HTTPError: {error.reason}",
            "bytes": 0,
        }
    except URLError as error:
        latency = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "status": 0,
            "latency_ms": latency,
            "error": f"URLError: {error.reason}",
            "bytes": 0,
        }
    except Exception as error:
        latency = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "status": 0,
            "latency_ms": latency,
            "error": f"Exception: {error}",
            "bytes": 0,
        }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight benchmark for /align endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000/align")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--text", default="", help="Transcript text")
    parser.add_argument("--text-file", default="", help="Path to transcript text file")
    parser.add_argument("--language", default="en")
    parser.add_argument("--romanize", action="store_true")
    parser.add_argument("--split-size", default="word", choices=["sentence", "word", "char"])
    parser.add_argument("--star-frequency", default="edges", choices=["segment", "edges"])
    parser.add_argument("--merge-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=float, default=None)
    parser.add_argument("--context-size", type=float, default=None)
    parser.add_argument("--requests", type=int, default=20, help="Total benchmark requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per request timeout seconds")
    parser.add_argument("--output", default="", help="Optional json report path")

    args = parser.parse_args()

    if args.requests < 1:
        raise ValueError("--requests must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    text = _read_text(args)
    payload, content_type = _build_multipart_payload(args.audio, text, args)

    if args.warmup > 0:
        print(f"Warmup: {args.warmup} requests")
        for _ in range(args.warmup):
            _single_request(args.url, payload, content_type, args.timeout)

    print(
        f"Benchmark start: url={args.url}, requests={args.requests}, "
        f"concurrency={args.concurrency}, audio={args.audio}"
    )

    started_at = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_single_request, args.url, payload, content_type, args.timeout)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_seconds = max(1e-9, time.perf_counter() - started_at)
    success = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = sorted(r["latency_ms"] for r in results)

    summary = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "success_count": len(success),
        "failed_count": len(failed),
        "success_rate": len(success) / len(results),
        "throughput_rps": len(results) / total_seconds,
        "latency_ms": {
            "avg": mean(latencies),
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "min": latencies[0],
            "max": latencies[-1],
        },
        "errors": [r["error"] for r in failed[:5]],
    }

    print("\n=== Benchmark Summary ===")
    print(f"Requests      : {summary['requests']}")
    print(f"Concurrency   : {summary['concurrency']}")
    print(f"Success/Fail  : {summary['success_count']}/{summary['failed_count']}")
    print(f"Success Rate  : {summary['success_rate'] * 100:.2f}%")
    print(f"Throughput    : {summary['throughput_rps']:.2f} req/s")
    print(f"Latency Avg   : {summary['latency_ms']['avg']:.2f} ms")
    print(f"Latency P50   : {summary['latency_ms']['p50']:.2f} ms")
    print(f"Latency P90   : {summary['latency_ms']['p90']:.2f} ms")
    print(f"Latency P95   : {summary['latency_ms']['p95']:.2f} ms")
    print(f"Latency P99   : {summary['latency_ms']['p99']:.2f} ms")
    print(f"Latency Min/Max: {summary['latency_ms']['min']:.2f}/{summary['latency_ms']['max']:.2f} ms")

    if failed:
        print("\nSample errors:")
        for item in summary["errors"]:
            print(f"- {item}")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
