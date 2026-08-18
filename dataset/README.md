# Dataset

This directory contains the CSV files used by the default FEEL2OM configuration.

Expected CSV names:

```text
A_H2Opred_Tr.csv
U_H2Opred_Tr.csv
C_H2Opred_Tr.csv
G_H2Opred_Tr.csv
A_H2Opred_Te.csv
U_H2Opred_Te.csv
C_H2Opred_Te.csv
G_H2Opred_Te.csv
```

Each CSV should contain at least two columns:

```text
sequence,label
```

Labels must be `0` or `1`.

The current code also requires precomputed RNA feature arrays. Feature directories are resolved with `FEEL_RNA_FM_TEMPLATE`, whose default is:

```text
dataset/rna_fm_features_H2Opred_{subset}
```

For example, features for subset `A` should be placed in `dataset/rna_fm_features_H2Opred_A/`.

The feature files are not included in this folder yet. If the feature directories are stored elsewhere, set:

```bash
export FEEL_RNA_FM_TEMPLATE=/path/to/rna_fm_features_H2Opred_{subset}
```
