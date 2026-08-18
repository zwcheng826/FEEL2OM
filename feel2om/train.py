"""Training and data-loading utilities for FEEL2OM."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .utils import atomic_json, atomic_torch_save, canonical_hash, capture_rng_state, fold_dir, restore_rng_state, seed_dir, set_seed, write_rows

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        base_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha * (1.0 - pt).pow(self.gamma) * base_loss).mean()


def build_loss(cfg) -> BinaryFocalLoss:
    return BinaryFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)


def calculate_binary_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    acc = (tp + tn) / max(len(y), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return {
        "acc": float(acc),
        "mcc": float(mcc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def calculate_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    metrics = calculate_binary_metrics(y, pred)
    metrics["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    metrics["auprc"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    metrics["threshold"] = float(threshold)
    metrics["n"] = int(len(y))
    return metrics


def macro_auprc(labels, probabilities, subsets: Sequence[str]) -> float:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    subset_names = np.asarray(subsets)
    values = []
    for subset in sorted(set(subsets)):
        mask = subset_names == subset
        if len(np.unique(labels[mask])) == 2:
            values.append(average_precision_score(labels[mask], probabilities[mask]))
    return float(np.mean(values)) if values else float("nan")


try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

@dataclass(frozen=True)
class Sample:
    sample_id: str
    sequence: str
    label: int
    subset: str
    row_index: int
    split: str


def _sample_id(subset: str, split: str, row_index: int, sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("ascii")).hexdigest()[:12]
    return f"{subset}_{split}_{row_index:06d}_{digest}"


def read_samples(path: str, subset: str, split: str) -> List[Sample]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.reader(handle)):
            if len(row) < 2:
                continue
            try:
                sequence = row[0].strip().upper().replace("T", "U")
                label = int(float(row[1]))
            except (TypeError, ValueError):
                continue
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label} in {path}, row {row_index + 1}")
            samples.append(Sample(_sample_id(subset, split, row_index, sequence), sequence, label, subset, row_index, split))
    if not samples:
        raise ValueError(f"No valid samples were read from {path}")
    return samples


def load_all_samples(cfg, include_test: bool = False):
    train: List[Sample] = []
    test: Dict[str, List[Sample]] = {}
    for subset in cfg.subsets:
        train.extend(read_samples(cfg.train_csv_map[subset], subset, "train"))
        if include_test:
            test[subset] = read_samples(cfg.test_csv_map[subset], subset, "test")
    return train, test


def dataset_manifest(cfg, samples: Sequence[Sample]) -> dict:
    payload = {
        "subsets": list(cfg.subsets),
        "n_samples": len(samples),
        "n_positive": int(sum(sample.label for sample in samples)),
        "n_negative": int(sum(1 - sample.label for sample in samples)),
        "files": {subset: cfg.train_csv_map[subset] for subset in cfg.subsets},
    }
    payload["data_hash"] = canonical_hash(
        {
            "samples": [
                {
                    "id": sample.sample_id,
                    "sequence": sample.sequence,
                    "label": sample.label,
                    "subset": sample.subset,
                }
                for sample in samples
            ]
        }
    )
    return payload


def preflight(cfg, include_test: bool = False) -> dict:
    train, test = load_all_samples(cfg, include_test=include_test)
    for subset in cfg.subsets:
        fm_dir = Path(cfg.rna_fm_dir_map[subset])
        if cfg.strict_features and not fm_dir.is_dir():
            raise FileNotFoundError(f"RNA-FM feature directory not found: {fm_dir}")
    manifest = dataset_manifest(cfg, train)
    if include_test:
        manifest["test_counts"] = {subset: len(rows) for subset, rows in test.items()}
    return manifest


def create_fold_manifest(samples: Sequence[Sample], k_folds: int, seed: int, output_root: str):
    labels = np.asarray([sample.label for sample in samples])
    groups = np.asarray([sample.sequence for sample in samples])
    if StratifiedGroupKFold is not None and len(np.unique(groups)) < len(groups):
        splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        folds = np.empty(len(samples), dtype=int)
        for fold, (_train, val) in enumerate(splitter.split(np.zeros(len(samples)), labels, groups)):
            folds[val] = fold
    else:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        folds = np.empty(len(samples), dtype=int)
        for fold, (_train, val) in enumerate(splitter.split(np.zeros(len(samples)), labels)):
            folds[val] = fold
    rows = [
        {
            "sample_id": sample.sample_id,
            "subset": sample.subset,
            "label": sample.label,
            "fold": int(folds[index]),
        }
        for index, sample in enumerate(samples)
    ]
    split_hash = canonical_hash({"folds": rows})
    write_rows(Path(output_root) / "fold_manifest.csv", rows)
    return folds, split_hash


def pad_or_truncate(sequence: str, seq_len: int):
    sequence = sequence.upper().replace("T", "U")
    if len(sequence) >= seq_len:
        start = (len(sequence) - seq_len) // 2
        return sequence[start:start + seq_len], np.ones(seq_len, dtype=np.float32)
    pad_left = (seq_len - len(sequence)) // 2
    padded = "N" * pad_left + sequence + "N" * (seq_len - len(sequence) - pad_left)
    mask = np.asarray([1.0 if char != "N" else 0.0 for char in padded], dtype=np.float32)
    return padded, mask


def one_hot(sequence: str) -> np.ndarray:
    alphabet = "AUCGN"
    matrix = np.zeros((len(alphabet), len(sequence)), dtype=np.float32)
    for index, char in enumerate(sequence):
        matrix[alphabet.index(char if char in alphabet else "N"), index] = 1.0
    return matrix


class RNADataset(Dataset):
    def __init__(self, samples: Sequence[Sample], cfg):
        self.samples = list(samples)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        sequence, mask = pad_or_truncate(sample.sequence, self.cfg.seq_len)
        fm_dir = Path(self.cfg.rna_fm_dir_map[sample.subset])
        candidates = (
            fm_dir / f"{sequence}.npy",
            fm_dir / f"{sample.row_index}.npy",
            fm_dir / f"{sample.sample_id}.npy",
        )
        fm_path = next((path for path in candidates if path.exists()), None)
        if fm_path is None:
            raise FileNotFoundError(f"RNA-FM feature not found for {sample.sample_id} in {fm_dir}")
        fm = np.load(fm_path).astype(np.float32)
        if fm.ndim == 1:
            fm = np.tile(fm[None, :], (self.cfg.seq_len, 1))
        if fm.shape[0] != self.cfg.seq_len:
            padded = np.zeros((self.cfg.seq_len, fm.shape[1]), dtype=np.float32)
            length = min(self.cfg.seq_len, fm.shape[0])
            padded[:length] = fm[:length]
            fm = padded
        return {
            "fm": torch.from_numpy(fm),
            "onehot": torch.from_numpy(one_hot(sequence)),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
            "sample_id": sample.sample_id,
            "subset": sample.subset,
        }


def make_loader(samples, cfg, shuffle: bool, loader_seed: int):
    generator = torch.Generator()
    generator.manual_seed(loader_seed)
    return DataLoader(
        RNADataset(samples, cfg),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        generator=generator,
    )


def evaluate_epoch(model, loader, criterion, device, cfg):
    model.eval()
    labels, probabilities, sample_ids, subsets = [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            fm = batch["fm"].to(device)
            onehot = batch["onehot"].to(device)
            mask = batch["mask"].to(device)
            target = batch["label"].to(device)
            logits = model(fm, onehot, mask).view(-1)
            total_loss += float(criterion(logits, target).item())
            labels.extend(target.cpu().numpy().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            sample_ids.extend(batch["sample_id"])
            subsets.extend(batch["subset"])
    metrics = calculate_metrics(labels, probabilities, 0.5)
    metrics["macro_auprc"] = macro_auprc(labels, probabilities, subsets)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, np.asarray(labels), np.asarray(probabilities), sample_ids, subsets


def write_predictions(path: Path, ids, subsets, labels, probabilities, mode, seed, fold):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "subset", "label", "probability", "mode", "seed", "fold"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in zip(ids, subsets, labels, probabilities):
            row = dict(zip(fields[:4], [values[0], values[1], int(values[2]), float(values[3])]))
            row.update({"mode": mode, "seed": seed, "fold": fold})
            writer.writerow(row)




class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        import copy

        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module):
        source_state = source.state_dict()
        for key, value in self.model.state_dict().items():
            incoming = source_state[key].detach()
            if value.dtype.is_floating_point:
                value.copy_(value * self.decay + incoming * (1.0 - self.decay))
            else:
                value.copy_(incoming)


def get_autocast(device, enabled):
    try:
        return autocast(device_type=device.type, enabled=enabled)
    except TypeError:  # pragma: no cover
        return autocast(enabled=enabled)


def train_update_block(model, loader, optimizer, scheduler, criterion, scaler, ema, device, cfg, optimizer_updates):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    total_loss = 0.0
    micro_batches = 0
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    for _ in range(optimizer_updates):
        for _micro in range(cfg.accumulate_grad_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            fm = batch["fm"].to(device)
            onehot = batch["onehot"].to(device)
            mask = batch["mask"].to(device)
            target = batch["label"].to(device)
            with get_autocast(device, use_amp):
                logits = model(fm, onehot, mask).view(-1)
                raw_loss = criterion(logits, target)
                loss = raw_loss / cfg.accumulate_grad_batches
            scaler.scale(loss).backward()
            total_loss += float(raw_loss.item())
            micro_batches += 1
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        ema.update(model)
    return total_loss / max(micro_batches, 1)


def train_fold(cfg, samples, train_idx, val_idx, data_hash, split_hash, fold, device):
    run_hash = f"{data_hash}_{split_hash}"
    current_fold_dir = fold_dir(cfg, fold)
    status_path = current_fold_dir / "status.json"
    best_path = current_fold_dir / "best.pt"
    last_path = current_fold_dir / "last.pt"
    prediction_path = current_fold_dir / "val_predictions.csv"
    force_retrain = os.environ.get("FEEL_FORCE_RETRAIN", "0") == "1"
    if not force_retrain and status_path.exists() and best_path.exists() and prediction_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete" and status.get("run_hash") == run_hash:
            print(f"[Resume] complete: {cfg.mode} fold={fold + 1}")
            return best_path, status

    current_fold_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed * 100 + fold)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg).to(device)
    ema = ModelEMA(model, cfg.ema_decay)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    train_loader = make_loader(train_samples, cfg, True, cfg.seed * 1000 + fold)
    val_loader = make_loader(val_samples, cfg, False, cfg.seed * 1000 + fold + 99)
    total_steps = cfg.max_optimizer_steps if cfg.max_optimizer_steps > 0 else max(1, cfg.epochs * math.ceil(len(train_loader) / cfg.accumulate_grad_batches))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    criterion = build_loss(cfg).to(device)
    best_score = -float("inf")
    best_epoch = 0
    stale_evals = 0
    completed_steps = 0
    history = []

    if not force_retrain and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("run_hash") == run_hash:
            model.load_state_dict(checkpoint["model"])
            ema.model.load_state_dict(checkpoint["ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            completed_steps = int(checkpoint.get("completed_optimizer_steps", 0))
            best_score = float(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
            stale_evals = int(checkpoint["stale_evals"])
            history = list(checkpoint.get("history", []))
            restore_rng_state(checkpoint.get("rng_state"))

    started = time.time()
    interval = cfg.validation_interval_steps or min(250, total_steps)
    while completed_steps < total_steps:
        updates = min(interval, total_steps - completed_steps)
        train_loss = train_update_block(model, train_loader, optimizer, scheduler, criterion, scaler, ema, device, cfg, updates)
        completed_steps += updates
        val_metrics, labels, probs, ids, subset_names = evaluate_epoch(ema.model, val_loader, criterion, device, cfg)
        score = float(val_metrics["macro_auprc"])
        row = {"optimizer_steps": completed_steps, "train_loss": train_loss, **val_metrics}
        history.append(row)
        if score > best_score + 1e-12:
            best_score, best_epoch, stale_evals = score, len(history), 0
            atomic_torch_save(
                best_path,
                {
                    "run_hash": run_hash,
                    "model": ema.model.state_dict(),
                    "step": completed_steps,
                    "score": score,
                    "metrics": val_metrics,
                    "parameter_count": parameter_count(model),
                },
            )
            write_predictions(prediction_path, ids, subset_names, labels, probs, cfg.mode, cfg.seed, fold)
        else:
            stale_evals += 1
        atomic_torch_save(
            last_path,
            {
                "run_hash": run_hash,
                "model": model.state_dict(),
                "ema": ema.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_evals": stale_evals,
                "history": history,
                "completed_optimizer_steps": completed_steps,
                "rng_state": capture_rng_state(),
            },
        )
        write_rows(current_fold_dir / "history.csv", history)
        print(f"[{cfg.mode} f{fold + 1}] steps={completed_steps:04d} macroAP={score:.5f}")
        if stale_evals >= cfg.early_stop_patience:
            break
    status = {
        "status": "complete",
        "run_hash": run_hash,
        "mode": cfg.mode,
        "trial_id": cfg.trial_id,
        "seed": cfg.seed,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_macro_auprc": best_score,
        "completed_optimizer_steps": completed_steps,
        "parameter_count": parameter_count(model),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
    }
    atomic_json(status_path, status)
    return best_path, status


def build_oof(cfg, samples, statuses):
    rows = []
    for status in statuses:
        path = fold_dir(cfg, int(status["fold"])) / "val_predictions.csv"
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    if len(rows) != len(samples) or len({row["sample_id"] for row in rows}) != len(samples):
        raise RuntimeError(f"Incomplete OOF predictions: {len(rows)} rows for {len(samples)} samples")
    order = {sample.sample_id: index for index, sample in enumerate(samples)}
    rows.sort(key=lambda row: order[row["sample_id"]])
    labels = np.asarray([int(row["label"]) for row in rows])
    probs = np.asarray([float(row["probability"]) for row in rows])
    subsets = [row["subset"] for row in rows]
    fixed = calculate_metrics(labels, probs, 0.5)
    fixed["macro_auprc"] = macro_auprc(labels, probs, subsets)
    current_seed_dir = seed_dir(cfg)
    write_rows(current_seed_dir / "oof_predictions.csv", rows)
    summary = {
        "mode": cfg.mode,
        "trial_id": cfg.trial_id,
        "seed": cfg.seed,
        "fixed_0_5": fixed,
        "fold_status": statuses,
    }
    atomic_json(current_seed_dir / "oof_summary.json", summary)
    return summary


def run_cross_validation(cfg, include_test: bool = False):
    manifest = preflight(cfg, include_test=include_test)
    samples, _test = load_all_samples(cfg, include_test=False)
    fold_ids, split_hash = create_fold_manifest(samples, cfg.k_folds, cfg.seed, cfg.output_root)
    current_seed_dir = seed_dir(cfg)
    atomic_json(current_seed_dir / "resolved_config.json", cfg.public_dict())
    atomic_json(
        current_seed_dir / "provenance.json",
        {
            "data_hash": manifest["data_hash"],
            "split_hash": split_hash,
            "python": os.sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] device={device} mode={cfg.mode} seed={cfg.seed} folds={cfg.k_folds}")
    statuses = []
    for fold in range(cfg.k_folds):
        train_idx = np.flatnonzero(fold_ids != fold)
        val_idx = np.flatnonzero(fold_ids == fold)
        _path, status = train_fold(cfg, samples, train_idx, val_idx, manifest["data_hash"], split_hash, fold, device)
        statuses.append(status)
    return build_oof(cfg, samples, statuses)
