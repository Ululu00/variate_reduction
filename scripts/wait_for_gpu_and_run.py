#!/usr/bin/env python3
"""Wait for GPU memory headroom, then exec a command."""

import argparse
import os
import subprocess
import sys
import time


def query_free_mib(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    first = output.strip().splitlines()[0].strip()
    return int(first)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument("--interval-sec", type=int, default=120)
    parser.add_argument("--max-wait-sec", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")

    start = time.time()
    print(
        f"[wait_for_gpu] gpu={args.gpu} min_free_mib={args.min_free_mib} "
        f"interval_sec={args.interval_sec}",
        flush=True,
    )
    while True:
        try:
            free_mib = query_free_mib(args.gpu)
        except Exception as exc:
            print(f"[wait_for_gpu] query failed: {exc}", flush=True)
            free_mib = -1
        elapsed = int(time.time() - start)
        print(f"[wait_for_gpu] elapsed_sec={elapsed} free_mib={free_mib}", flush=True)
        if free_mib >= args.min_free_mib:
            break
        if args.max_wait_sec > 0 and elapsed >= args.max_wait_sec:
            raise SystemExit(
                f"timed out waiting for {args.min_free_mib} MiB free on GPU {args.gpu}"
            )
        time.sleep(args.interval_sec)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[wait_for_gpu] starting: {' '.join(command)}", flush=True)
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
