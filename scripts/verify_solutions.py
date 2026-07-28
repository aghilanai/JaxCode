#!/usr/bin/env python3
"""Run all reference solutions against the jax_judge web engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jax_solutions import SOLUTIONS  # noqa: E402
from jax_judge.tasks import TASKS  # noqa: E402
from jax_judge.web_engine import execute_code  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Optional task ids to verify (default: all)",
    )
    args = parser.parse_args()

    task_ids = list(args.tasks) if args.tasks else sorted(TASKS)
    missing = [t for t in task_ids if t not in SOLUTIONS]
    if missing:
        print(f"Missing solutions for: {missing}", file=sys.stderr)
        return 2

    failed: list[str] = []
    for tid in task_ids:
        result = execute_code(tid, SOLUTIONS[tid])
        passed = result.get("passed", 0)
        total = result.get("total", 0)
        ok = passed == total and total > 0 and not result.get("error")
        status = "PASS" if ok else "FAIL"
        print(f"{status} {tid} {passed}/{total}")
        if not ok:
            failed.append(tid)
            if result.get("error"):
                print(f"  error: {result['error']}")
            for test in result.get("tests", []):
                if not test.get("passed"):
                    msg = test.get("error_msg") or ""
                    print(f"  - {test.get('name')}: {msg[:300]}")

    print()
    print(f"Summary: {len(task_ids) - len(failed)}/{len(task_ids)} passed")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
