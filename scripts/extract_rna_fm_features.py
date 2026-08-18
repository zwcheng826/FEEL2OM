"""Extract RNA-FM features for FEEL2OM datasets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SUBSETS = ("A", "C", "G", "U")


def normalize_sequence(raw_sequence: str, seq_len: int = 41) -> str | None:
    sequence = (raw_sequence or "").strip().upper().replace("T", "U")
    if not sequence or any(base not in "AUCGN" for base in sequence):
        return None
    if len(sequence) < seq_len:
        return sequence + "N" * (seq_len - len(sequence))
    return sequence[:seq_len]


def load_sequences(csv_path: Path, seq_len: int) -> list[str]:
    sequences = set()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            sequence = normalize_sequence(row[0], seq_len=seq_len)
            if sequence is not None:
                sequences.add(sequence)
    return sorted(sequences)


def extract_features(sequences, output_dir: Path, model, batch_converter, device, repr_layer: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sequence in tqdm(sequences, desc=f"RNA-FM {output_dir.name}"):
            save_path = output_dir / f"{sequence}.npy"
            if save_path.exists():
                continue
            _, _, batch_tokens = batch_converter([("sequence", sequence)])
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[repr_layer])
            representation = results["representations"][repr_layer]
            features = representation[0, 1 : len(sequence) + 1].cpu().numpy()
            if features.shape != (len(sequence), 640):
                raise ValueError(f"Unexpected feature shape for {sequence}: {features.shape}")
            np.save(save_path, features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RNA-FM features for FEEL2OM CSV files.")
    parser.add_argument("--data-root", default="dataset", help="Directory containing A/U/C/G_H2Opred_Tr/Te.csv files.")
    parser.add_argument(
        "--output-template",
        default="dataset/rna_fm_features_H2Opred_{subset}",
        help="Output directory template. Use {subset} for A/C/G/U.",
    )
    parser.add_argument("--model-path", required=True, help="Path to RNA-FM_pretrained.pt.")
    parser.add_argument("--rna-fm-root", default=None, help="Optional path to the RNA-FM repository.")
    parser.add_argument("--seq-len", type=int, default=41)
    parser.add_argument("--repr-layer", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rna_fm_root:
        sys.path.insert(0, args.rna_fm_root)

    import fm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data_root = Path(args.data_root)
    for subset in SUBSETS:
        csv_paths = [
            data_root / f"{subset}_H2Opred_Te.csv",
            data_root / f"{subset}_H2Opred_Tr.csv",
        ]
        missing = [path for path in csv_paths if not path.exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            print(f"Skip {subset}: missing {missing_text}")
            continue

        sequences = []
        for csv_path in csv_paths:
            sequences.extend(load_sequences(csv_path, seq_len=args.seq_len))
        sequences = sorted(set(sequences))
        output_dir = Path(args.output_template.format(subset=subset))
        extract_features(sequences, output_dir, model, batch_converter, device, args.repr_layer)

    print("Feature extraction completed.")


if __name__ == "__main__":
    main()
