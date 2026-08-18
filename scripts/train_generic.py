"""Train and evaluate the generic FEEL2OM model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feel2om.config import generic_config
from feel2om.test import evaluate_locked_test
from feel2om.train import run_cross_validation
from feel2om.utils import write_rows


def main() -> None:
    cfg = generic_config()
    run_cross_validation(cfg, include_test=True)
    test_rows = evaluate_locked_test(cfg)
    write_rows(Path(cfg.output_root) / "summary" / "generic_test_detail.csv", test_rows)
    print(f"[Generic] outputs written to {Path(cfg.output_root).resolve()}")


if __name__ == "__main__":
    main()
