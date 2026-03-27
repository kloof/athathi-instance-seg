"""Multi-connection parallel file downloader (IDM-style).

Splits a file into N chunks, downloads each chunk via HTTP Range requests
in parallel threads, then merges into the final file. Supports resume.

Usage:
    python scripts/parallel_download.py <url>
    python scripts/parallel_download.py <url> -o output.zip -n 16
    python scripts/parallel_download.py --file urls.txt -n 16 -o ./downloads/

    # Download all Structured3D panorama zips (16 connections each):
    python scripts/parallel_download.py --file DOWNLOAD_S3D.txt --filter panorama -n 16
"""

import urllib.request
import urllib.error
import os
import sys
import time
import argparse
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading


def get_file_size(url: str) -> int | None:
    """Get file size via HEAD request. Returns None if server doesn't report it."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        size = resp.headers.get("Content-Length")
        # Check if server accepts range requests
        accept_ranges = resp.headers.get("Accept-Ranges", "none")
        if size and accept_ranges != "none":
            return int(size)
        elif size:
            # Server reports size but doesn't explicitly accept ranges — try anyway
            return int(size)
    except Exception as e:
        print(f"  HEAD request failed: {e}")
    return None


def download_chunk(url: str, start: int, end: int, chunk_path: str,
                   progress: dict, lock: threading.Lock, chunk_id: int) -> bool:
    """Download a byte range to a file. Returns True on success."""
    headers = {"Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(url, headers=headers)

    # Resume: if chunk file exists and is partially downloaded
    existing_size = 0
    if os.path.exists(chunk_path):
        existing_size = os.path.getsize(chunk_path)
        expected = end - start + 1
        if existing_size == expected:
            with lock:
                progress["downloaded"] += existing_size
            return True
        elif existing_size > 0:
            # Resume from where we left off
            headers["Range"] = f"bytes={start + existing_size}-{end}"
            req = urllib.request.Request(url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        mode = "ab" if existing_size > 0 else "wb"
        with open(chunk_path, mode) as f:
            while True:
                data = resp.read(1024 * 256)  # 256KB reads
                if not data:
                    break
                f.write(data)
                with lock:
                    progress["downloaded"] += len(data)
        # Verify chunk size matches expected range
        actual = os.path.getsize(chunk_path)
        expected = end - start + 1
        if actual != expected:
            # Corrupt chunk — delete so retry starts fresh
            os.remove(chunk_path)
            with lock:
                progress["downloaded"] -= actual
            return False
        return True
    except Exception as e:
        # Retry logic is handled by caller
        return False


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.1f}MB"
    else:
        return f"{n / 1024**3:.2f}GB"


def format_speed(bps: float) -> str:
    if bps < 1024:
        return f"{bps:.0f} B/s"
    elif bps < 1024 ** 2:
        return f"{bps / 1024:.1f} KB/s"
    else:
        return f"{bps / 1024**2:.1f} MB/s"


def download_file(url: str, output_path: str, num_connections: int = 16,
                  max_retries: int = 3) -> bool:
    """Download a file using parallel connections.

    Args:
        url: download URL
        output_path: where to save the file
        num_connections: number of parallel download threads
        max_retries: retries per chunk on failure

    Returns:
        True on success
    """
    filename = os.path.basename(output_path)
    print(f"\n{'='*60}")
    print(f"Downloading: {filename}")
    print(f"URL: {url}")
    print(f"Connections: {num_connections}")

    # Check if already downloaded
    if os.path.exists(output_path):
        existing = os.path.getsize(output_path)
        remote_size = get_file_size(url)
        if remote_size and existing == remote_size:
            print(f"Already downloaded ({format_size(existing)}), skipping.")
            return True

    # Get file size
    file_size = get_file_size(url)
    if file_size is None:
        print("Cannot determine file size or server doesn't support range requests.")
        print("Falling back to single-connection download...")
        return _download_single(url, output_path)

    print(f"Size: {format_size(file_size)}")

    # Create temp directory for chunks (clear stale parts from prior runs)
    temp_dir = output_path + ".parts"
    if os.path.exists(temp_dir):
        existing_chunks = [f for f in os.listdir(temp_dir) if f.startswith("chunk_")]
        if len(existing_chunks) != num_connections:
            shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Split into chunks
    chunk_size = file_size // num_connections
    chunks = []
    for i in range(num_connections):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < num_connections - 1 else file_size - 1
        chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}")
        chunks.append((i, start, end, chunk_path))

    # Progress tracking
    progress = {"downloaded": 0}
    lock = threading.Lock()
    t0 = time.time()

    # Progress display thread
    stop_progress = threading.Event()

    def show_progress():
        while not stop_progress.is_set():
            with lock:
                done = progress["downloaded"]
            elapsed = time.time() - t0
            speed = done / max(elapsed, 0.01)
            pct = done / file_size * 100
            bar_len = 30
            filled = int(bar_len * done / file_size)
            bar = "#" * filled + "-" * (bar_len - filled)
            if speed > 1024 and done > 0:
                remaining = (file_size - done) / speed
                eta_str = f"ETA {remaining:.0f}s"
            else:
                eta_str = "ETA --"
            print(
                f"\r  {bar} {pct:5.1f}% | "
                f"{format_size(done)}/{format_size(file_size)} | "
                f"{format_speed(speed)} | "
                f"{eta_str}   ",
                end="", flush=True,
            )
            stop_progress.wait(0.5)

    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()

    # Download chunks in parallel
    success = True
    with ThreadPoolExecutor(max_workers=num_connections) as executor:
        futures = {}
        for chunk_id, start, end, chunk_path in chunks:
            f = executor.submit(
                download_chunk, url, start, end, chunk_path,
                progress, lock, chunk_id,
            )
            futures[f] = (chunk_id, start, end, chunk_path)

        for future in as_completed(futures):
            chunk_id, start, end, chunk_path = futures[future]
            result = future.result()
            if not result:
                # Retry
                for retry in range(max_retries):
                    print(f"\n  Retrying chunk {chunk_id} ({retry + 1}/{max_retries})...")
                    result = download_chunk(
                        url, start, end, chunk_path, progress, lock, chunk_id,
                    )
                    if result:
                        break
                if not result:
                    print(f"\n  FAILED: chunk {chunk_id} after {max_retries} retries")
                    success = False

    stop_progress.set()
    progress_thread.join(timeout=1)

    elapsed = time.time() - t0
    speed = file_size / max(elapsed, 0.01)
    print(f"\r  {'#' * 30} 100.0% | {format_size(file_size)} | "
          f"{format_speed(speed)} | {elapsed:.1f}s total   ")

    if not success:
        print("Download FAILED — some chunks could not be retrieved.")
        return False

    # Merge chunks
    print("  Merging chunks...", end="", flush=True)
    with open(output_path, "wb") as out:
        for chunk_id, start, end, chunk_path in chunks:
            with open(chunk_path, "rb") as chunk_f:
                shutil.copyfileobj(chunk_f, out, length=1024 * 1024)
    print(" done.")

    # Verify size
    final_size = os.path.getsize(output_path)
    if final_size != file_size:
        print(f"  WARNING: size mismatch! Expected {file_size}, got {final_size}")
        return False

    # Clean up chunks
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Saved: {output_path} ({format_size(final_size)})")
    return True


def _download_single(url: str, output_path: str) -> bool:
    """Fallback single-connection download."""
    try:
        t0 = time.time()
        urllib.request.urlretrieve(url, output_path)
        elapsed = time.time() - t0
        size = os.path.getsize(output_path)
        print(f"  Downloaded {format_size(size)} in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def parse_urls_file(filepath: str, filter_str: str | None = None) -> list[str]:
    """Parse a text file for URLs. Optionally filter by substring."""
    urls = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("http://") or line.startswith("https://"):
                if filter_str is None or filter_str in line:
                    urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Multi-connection parallel file downloader (IDM-style)"
    )
    parser.add_argument("url", nargs="?", help="URL to download")
    parser.add_argument("--file", "-f", help="Text file containing URLs (one per line)")
    parser.add_argument("--filter", help="Only download URLs containing this string")
    parser.add_argument("-o", "--output", default=".", help="Output file or directory")
    parser.add_argument("-n", "--connections", type=int, default=16,
                        help="Number of parallel connections (default: 16)")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.file:
        urls = parse_urls_file(args.file, args.filter)
        if not urls:
            print(f"No URLs found in {args.file}" +
                  (f" matching '{args.filter}'" if args.filter else ""))
            return
        print(f"Found {len(urls)} URLs to download")

        # Output must be a directory for batch downloads
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for url in urls:
            filename = url.split("/")[-1]
            output_path = str(out_dir / filename)
            ok = download_file(url, output_path, args.connections, args.retries)
            results.append((filename, ok))

        print(f"\n{'='*60}")
        print("Summary:")
        for filename, ok in results:
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {filename}")

    elif args.url:
        if os.path.isdir(args.output):
            filename = args.url.split("/")[-1]
            output_path = os.path.join(args.output, filename)
        else:
            output_path = args.output
        download_file(args.url, output_path, args.connections, args.retries)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
