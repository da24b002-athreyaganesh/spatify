"""Run a command and report its peak memory use; kill it if the machine runs low.

Prints the peak RSS of the command's process tree (shared memory, such as Ray's object
store, is counted once per process that maps it, so this overstates), the peak rise in
the machine's used memory, and the lowest available memory seen. If available memory
falls below --floor-mb, the whole process tree is killed, so a measurement can't freeze
the laptop.

Run:  python scripts/peak_memory.py [--floor-mb 1024] -- <command ...>
e.g.  python scripts/peak_memory.py -- dvc repro
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psutil

MB = 2**20


def main() -> None:
    parser = argparse.ArgumentParser(description="Peak memory of a command, with a safety floor.")
    parser.add_argument("--floor-mb", type=int, default=1024)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give a command after --")

    base = psutil.virtual_memory().used
    proc = subprocess.Popen(command)
    root = psutil.Process(proc.pid)
    peak_tree = peak_rise = 0
    lowest = psutil.virtual_memory().available
    start = time.monotonic()
    while proc.poll() is None:
        try:
            tree = [root, *root.children(recursive=True)]
            rss = sum(p.memory_info().rss for p in tree if p.is_running())
        except psutil.Error:
            tree, rss = [], 0
        vm = psutil.virtual_memory()
        peak_tree, peak_rise = max(peak_tree, rss), max(peak_rise, vm.used - base)
        lowest = min(lowest, vm.available)
        if vm.available < args.floor_mb * MB:
            print(f"ABORTED: available memory fell to {vm.available / MB:.0f} MB", file=sys.stderr)
            for p in reversed(tree):
                try:
                    p.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(0.1)
    proc.wait()
    print(
        f"[peak_memory] exit {proc.returncode} after {time.monotonic() - start:.0f} s; "
        f"peak rise in used memory {peak_rise / MB:.0f} MB; lowest available "
        f"{lowest / MB:.0f} MB; peak process-tree RSS {peak_tree / MB:.0f} MB"
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
