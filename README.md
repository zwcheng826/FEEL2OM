# FEEL2OM

FEEL2OM is a reproducible model for RNA 2'-O-methylation site prediction. The final architecture combines RNA language-model features, a local one-hot sequence branch, efficient channel attention, BiLSTM encoding, Transformer fusion, and Binary Focal Loss.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Set the dataset and RNA feature locations with environment variables:

```bash
export FEEL_DATA_ROOT=/path/to/H2Opred_data
export FEEL_RNA_FM_TEMPLATE=/path/to/rna_fm_features_H2Opred_{subset}
export FEEL_OUTPUT_ROOT=/path/to/outputs
```

See `dataset/README.md` for expected file names and feature layout.

To extract RNA-FM features from the CSV files:

```bash
python scripts/extract_rna_fm_features.py \
  --data-root /path/to/H2Opred_data \
  --output-template /path/to/rna_fm_features_H2Opred_{subset} \
  --model-path /path/to/RNA-FM_pretrained.pt
```

## Train Generic Model

```bash
python scripts/train_generic.py
```

## Train A/C/G/U-Specific Models

```bash
python scripts/train_specific.py
```

## Outputs

Training outputs are written under `FEEL_OUTPUT_ROOT`:

```text
outputs/
runs/
predictions/
summary/
```

The default configuration reproduces the final FEEL2OM model settings used for the generic and nucleotide-specific experiments.
