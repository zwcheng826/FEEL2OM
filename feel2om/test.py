"""Locked test evaluation for FEEL2OM."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from .model import build_model
from .train import build_loss, calculate_metrics, evaluate_epoch, load_all_samples, make_loader, write_predictions
from .utils import seed_dir

def load_best_checkpoint_paths(cfg) -> List[Path]:
    paths = sorted(seed_dir(cfg).glob("fold_*/best.pt"))
    if len(paths) != cfg.k_folds:
        raise FileNotFoundError(f"Expected {cfg.k_folds} checkpoints, found {len(paths)} under {seed_dir(cfg)}")
    return paths


def evaluate_locked_test(cfg):
    if not cfg.run_test:
        return []
    _train, test_by_subset = load_all_samples(cfg, include_test=True)
    checkpoints = load_best_checkpoint_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss(cfg).to(device)
    output_rows = []
    for subset, samples in test_by_subset.items():
        loader = make_loader(samples, cfg, False, cfg.seed + 50000)
        fold_probs = []
        labels = ids = subset_names = None
        inference_seconds = 0.0
        for checkpoint_path in checkpoints:
            model = build_model(cfg).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            _metrics, labels, probs, ids, subset_names = evaluate_epoch(model, loader, criterion, device, cfg)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            fold_probs.append(probs)
        ensemble = np.mean(np.vstack(fold_probs), axis=0)
        metrics = calculate_metrics(labels, ensemble, 0.5)
        output_rows.append(
            {
                "mode": cfg.mode,
                "trial_id": cfg.trial_id,
                "seed": cfg.seed,
                "subset": subset,
                "threshold_rule": "fixed_0_5",
                "ensemble_seconds_per_sample": inference_seconds / max(len(samples), 1),
                **metrics,
            }
        )
        write_predictions(
            Path(cfg.output_root) / "predictions" / cfg.trial_id / f"{subset}_predictions.csv",
            ids,
            subset_names,
            labels,
            ensemble,
            cfg.mode,
            cfg.seed,
            -1,
        )
    return output_rows
