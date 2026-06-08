from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(script: str, extra: list[str]) -> int:
    cmd = [PYTHON, str(ROOT / "scripts" / script), *extra]
    print("[rpp-stage]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R++ stage scripts.")
    parser.add_argument("--only", choices=["imdb", "arxiv", "products", "small", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    common = ["--seed", str(args.seed), "--epochs", str(args.epochs)]
    if args.skip_existing:
        common.append("--skip-existing")
    jobs = []
    if args.only in {"imdb", "all"}:
        jobs.append(("run_rpp_imdb_rescue.py", common))
    if args.only in {"arxiv", "all"}:
        jobs.append(("run_rpp_arxiv_refine.py", common + (["--download"] if args.download else [])))
    if args.only in {"products", "all"}:
        jobs.append(("run_rpp_products_streaming_diffusion.py", common + (["--download"] if args.download else [])))
    if args.only in {"small", "all"}:
        jobs.append(("run_rpp_small_nonregression.py", common))

    failed = []
    for script, extra in jobs:
        code = _run(script, extra)
        if code != 0:
            failed.append((script, code))
    if failed:
        raise SystemExit(f"R++ stage failed scripts: {failed}")


if __name__ == "__main__":
    main()
