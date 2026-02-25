import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_urls(path: str) -> list[str]:
    urls: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


_NON_ID_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def _public_id_from_url(url: str, index: int) -> str:
    """
    Keep IDs stable and S3-key-friendly.
    Example: ".../I%20am%20the%20danger.mp4" -> "i_am_the_danger_00"
    """
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name)
    stem = Path(name).stem or f"video_{index:02d}"
    stem = _NON_ID_CHARS.sub("_", stem).strip("._-").lower() or f"video_{index:02d}"
    return f"{stem}_{index:02d}"


def _atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate videos sequentially using local pipeline and upload results to S3 bucket."
    )
    parser.add_argument(
        "--config",
        default="test_input.json",
        help="Path to JSON config with top-level key 'input' (default: test_input.json).",
    )
    parser.add_argument(
        "--links",
        default="links_to_test_videos.txt",
        help="Text file with one video URL per line (default: links_to_test_videos.txt).",
    )
    parser.add_argument(
        "--bucket",
        default="langswap-public",
        help="Bucket to upload outputs to (sets BUCKET env var). Default: langswap-public.",
    )
    parser.add_argument(
        "--out",
        default="translation_results.json",
        help="Where to write a summary JSON (updated after each video).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="S3 key prefix for outputs (e.g. 'stretch_whole'). "
        "If set, passed as input.output_prefix to the pipeline.",
    )
    parser.add_argument(
        "--dubbing-algo",
        default=None,
        choices=["speedup", "pause_based", "stretch_whole"],
        help="Override dubbing algorithm (passed as input.dubbing_algo).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and exit without running translation.",
    )

    args = parser.parse_args()

    base = _load_json(args.config)
    if not isinstance(base, dict) or "input" not in base or not isinstance(base["input"], dict):
        raise ValueError(f"{args.config} must be JSON with top-level object key 'input'.")

    urls = _read_urls(args.links)
    if not urls:
        print(f"No URLs found in {args.links}", file=sys.stderr)
        return 2

    start = max(0, args.start)
    end = args.end if args.end is not None else len(urls)
    end = min(end, len(urls))
    if start >= end:
        print(f"Invalid range: start={start}, end={end}, total_urls={len(urls)}", file=sys.stderr)
        return 2

    os.environ["BUCKET"] = args.bucket

    jobs: list[dict[str, Any]] = []
    for i in range(start, end):
        url = urls[i]
        public_id = _public_id_from_url(url, i)
        jobs.append({"index": i, "public_id": public_id, "url": url})

    if args.dry_run:
        plan: dict[str, Any] = {"bucket": args.bucket, "jobs": jobs}
        if args.output_prefix is not None:
            plan["output_prefix"] = args.output_prefix
        if args.dubbing_algo is not None:
            plan["dubbing_algo"] = args.dubbing_algo
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    from langswap.api import process_translation  # lazy import (heavy deps)

    summary: dict[str, Any] = {
        "bucket": args.bucket,
        "config_path": os.path.abspath(args.config),
        "links_path": os.path.abspath(args.links),
        "started_at": int(time.time()),
        "jobs": [],
    }

    _atomic_write_json(args.out, summary)

    for job in jobs:
        i = job["index"]
        url = job["url"]
        public_id = job["public_id"]

        payload = copy.deepcopy(base["input"])
        payload["s3_video_url"] = url
        payload["public_id"] = public_id
        payload["name"] = Path(urlparse(url).path).name
        if args.output_prefix is not None:
            payload["output_prefix"] = args.output_prefix
        if args.dubbing_algo is not None:
            payload["dubbing_algo"] = args.dubbing_algo

        print(f"[{i}] public_id={public_id} url={url}")
        try:
            result = process_translation(payload)
            record = {**job, "status": "completed", "result": result}
            print(f"[{i}] done: {result.get('s3_result_video_url')}")
        except Exception as e:
            record = {**job, "status": "failed", "error": repr(e)}
            print(f"[{i}] failed: {record['error']}", file=sys.stderr)

        summary["jobs"].append(record)
        summary["updated_at"] = int(time.time())
        _atomic_write_json(args.out, summary)

    return 0 if all(j.get("status") == "completed" for j in summary["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())


