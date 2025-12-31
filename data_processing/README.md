# Data Preprocessing for HiC2Self

This directory contains scripts for converting raw Hi-C data into diagonally cropped submatrices that can be used for HiC2Self training.

**Notes**
* Submatrices are extracted along the diagonal with a sliding window.
* Cropping is performed symmetrically to preserve matrix structure.
* This preprocessing step is required once per chromosome.
* Output files can be reused across experiments and model configurations.


## 1. Prepare input Hi-C data

HiC2Self accepts **sparse Hi-C contact data** in a 3-column format: `bin1 bin2 count`.

Below are two common ways to generate this format.

---

### (A) From `.pairs` files (e.g. Micro-C)

Example dataset: **H1ESC Micro-C**

```bash
# Download example dataset
wget https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/e7bef8ed-d59a-4761-955c-d98643fc0ddf/4DNFIH3CQYRW.pairs.gz
gunzip 4DNFIH3CQYRW.pairs.gz
```
Extract intra-chromosomal contacts (example: chr3):
```
chrom=chr3
tail -n +231 4DNFIH3CQYRW.pairs \
  | awk '$2=="'${chrom}'" && $4=="'${chrom}'" && $8=="UU"' \
  > Extracted_5k_${chrom}.txt
```

### (B) From .hic files (Juicer format)

Example dataset: GSE63525 GM12878
```
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%5F30.hic
```

Convert .hic → sparse contact format using hicstraw:
```
import hicstraw
import pandas as pd
import numpy as np

chromosome = "chr3"
RESOLUTION = 5000

HIC_PATH = "/data/hic_files/GSE63525_GM12878_insitu_primary_30.hic"
OUT_DIR = "/data/GM12878/hic_extraction"

result = hicstraw.straw(
    "observed", "NONE",
    HIC_PATH,
    chromosome, chromosome,
    "BP", RESOLUTION
)

rows = [[r.binX, r.binY, r.counts] for r in result]
df = pd.DataFrame(rows, columns=["bin0", "bin1", "counts"])
df.to_csv(f"{OUT_DIR}/Extracted_5k_{chromosome}.txt", sep="\t", index=False)

```

## 2. Convert to trainable format

This step converts sparse Hi-C contacts into diagonally cropped submatrices, which serve as training inputs.

Prepare directories
```
mkdir -p data/GM12878/processed
mkdir -p data/assembly
```

Download chromosome sizes:
```
cd data/assembly
wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
```

## 3. Run preprocessing script

```
cd HiC2Self/data_processing

chrom=chr3

python input_data_preparation_full_chr_split.py \
  --input_dir data/GM12878/hic_extraction \
  --subset_file Extracted_5k_${chrom}.txt \
  --save_dir data/GM12878/processed \
  --save_file Extracted_noisy_mat_5kb_size400_${chrom}.pkl \
  --chrom ${chrom} \
  --assembly hg38 \
  --resolution 5000 \
  --mat_size 400

```

## 4. Output

The output is a serialized NumPy array stored as a .pkl file:

```
data/GM12878/processed/
└── Extracted_noisy_mat_5kb_size400_chr3.pkl
```
