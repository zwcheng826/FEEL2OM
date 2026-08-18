"""Train and evaluate A/C/G/U-specific FEEL2OM models."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feel2om.config import SUBSETS, specific_configs
from feel2om.test import evaluate_locked_test
from feel2om.train import run_cross_validation
from feel2om.utils import write_rows


def main() -> None:
    all_rows = []
    for cfg in specific_configs():
        run_cross_validation(cfg, include_test=True)
        rows = evaluate_locked_test(cfg)
        all_rows.extend(rows)
        write_rows(Path(cfg.output_root) / "summary" / f"{cfg.trial_id}_test_detail.csv", rows)
    matched = [
        row for row in all_rows
        if row["subset"] in SUBSETS and row["trial_id"] == f"specific_{row['subset']}_default"
    ]
    write_rows(Path(specific_configs()[0].output_root) / "summary" / "specific_matched_test_detail.csv", matched)
    print(f"[Specific] outputs written to {Path(specific_configs()[0].output_root).resolve()}")


if __name__ == "__main__":
    main()
