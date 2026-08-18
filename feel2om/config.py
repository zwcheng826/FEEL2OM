"""Default configuration for reproducible FEEL2OM training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


SUBSETS: Tuple[str, ...] = ("A", "U", "C", "G")


@dataclass
class ExperimentConfig:
    """Training and evaluation settings for the final FEEL2OM model."""

    mode: str = "generic"
    data_root: str = field(default_factory=lambda: os.environ.get("FEEL_DATA_ROOT", "dataset"))
    rna_fm_template: str = field(
        default_factory=lambda: os.environ.get(
            "FEEL_RNA_FM_TEMPLATE",
            "dataset/rna_fm_features_H2Opred_{subset}",
        )
    )
    output_root: str = field(default_factory=lambda: os.environ.get("FEEL_OUTPUT_ROOT", "outputs"))
    seed: int = 2002
    k_folds: int = 5
    seq_len: int = 41
    rna_fm_dim: int = 640
    hidden_dim: int = 192
    cnn_kernel_sizes: Tuple[int, int, int] = (3, 5, 7)
    num_heads: int = 8
    transformer_layers: int = 1
    batch_size: int = 64
    accumulate_grad_batches: int = 1
    dropout: float = 0.28
    epochs: int = 50
    max_optimizer_steps: int = 3750
    validation_interval_steps: int = 250
    lr: float = 1.5e-3
    weight_decay: float = 8.0e-3
    early_stop_patience: int = 999
    ema_decay: float = 0.997
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    msd_samples: int = 5
    eca_kernel_size: int = 9
    num_workers: int = 0
    use_amp: bool = True
    strict_features: bool = True
    subsets: Tuple[str, ...] = SUBSETS
    trial_id: str = "generic_default"
    run_test: bool = True
    train_csv_map: Dict[str, str] = field(init=False, repr=False)
    test_csv_map: Dict[str, str] = field(init=False, repr=False)
    rna_fm_dir_map: Dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.train_csv_map = {
            subset: str(Path(self.data_root) / f"{subset}_H2Opred_Tr.csv")
            for subset in self.subsets
        }
        self.test_csv_map = {
            subset: str(Path(self.data_root) / f"{subset}_H2Opred_Te.csv")
            for subset in self.subsets
        }
        self.rna_fm_dir_map = {
            subset: self.rna_fm_template.format(subset=subset)
            for subset in self.subsets
        }

    def public_dict(self) -> dict:
        excluded = {"train_csv_map", "test_csv_map", "rna_fm_dir_map"}
        return {key: value for key, value in asdict(self).items() if key not in excluded}


def generic_config() -> ExperimentConfig:
    """Return the default generic FEEL2OM configuration."""
    return ExperimentConfig(mode="generic", trial_id="generic_default", subsets=SUBSETS)


def specific_configs() -> tuple[ExperimentConfig, ...]:
    """Return the final A/C/G/U-specific FEEL2OM configurations."""
    settings = {
        "A": {"batch_size": 64, "lr": 0.0010, "dropout": 0.25, "weight_decay": 0.007, "focal_alpha": 0.50},
        "C": {"batch_size": 32, "lr": 0.0012, "dropout": 0.24, "weight_decay": 0.006, "focal_alpha": 0.48},
        "G": {"batch_size": 32, "lr": 0.0008, "dropout": 0.32, "weight_decay": 0.012, "focal_alpha": 0.52},
        "U": {"batch_size": 32, "lr": 0.0008, "dropout": 0.28, "weight_decay": 0.010, "focal_alpha": 0.52},
    }
    configs = []
    for subset in SUBSETS:
        configs.append(
            ExperimentConfig(
                mode="specific",
                trial_id=f"specific_{subset}_default",
                subsets=(subset,),
                **settings[subset],
            )
        )
    return tuple(configs)
