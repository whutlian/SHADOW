from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t2_safe_block_selection import main


if __name__ == "__main__":
    sys.argv.extend(["--datasets", "ogbn-arxiv", "ogbn-products", "--output", "experiments/tables/t2_sft_medium_seed42.csv", "--report", "experiments/reports/t2_sft_medium_summary.md", "--log-dir", "experiments/logs/t2_sft_medium_seed42"])
    main()
