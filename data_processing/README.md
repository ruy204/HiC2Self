# Data Pre-processing for HiC2Self

## 1. Bulk Hi-C data

a) If you have .paris file, 
- Example data: H1ES Micro-C in .pairs format

```
# Download example dataset
wget https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/e7bef8ed-d59a-4761-955c-d98643fc0ddf/4DNFIH3CQYRW.pairs.gz

# Extract specific chromosome 
chrom=chr3
tail -n +231 4DNFIH3CQYRW.pairs | awk '$2=="${chrom}" && $4=="${chrom}" && $8=="UU"' > Extracted_5k_"${chrom}".txt

```

b) If you have .hic file, 
- Example data: GSE63525_GM12878_insitu_primary_30.hic

```
# Download example dataset
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%5F30.hic

# Extract counts from .hic file (in python)
# Bin the data at 5kb resolution
# Extract chromosome 3 as example

chromosome = 3
RESOLUTION = 5000

import numpy as np
import hicstraw
import pandas as pd
HIC_PTH = '/data/hic_files'
EXTRACT_PTH = "/data/GM12878/hic_extraction"
HIC_FILENAME = 'GSE63525_GM12878_insitu_primary_30.hic'
result = hicstraw.straw("observed", 'NONE', f'{HIC_PTH}/{HIC_FILENAME}', f'{chromosome}', f'{chromosome}', 'BP', RESOLUTION)
row_list = []
for i in range(len(result)):
    row_list.append([result[i].binX, result[i].binY, result[i].counts])
control_df = pd.DataFrame(np.array(row_list))
control_df.columns = ['bin0','bin1','counts']
control_df.to_csv(f'{EXTRACT_PTH}/Extracted_5k_chr3.txt',sep='\t',index=None)
```

## 2. Convert to trainable format

```
mkdir -p /data/GM12878/processed
mkdir -p /data/assembly
cd /data/assembly
wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
cd /HiC2Self/data_processing

chrom='chr3'
python /HiC2Self/data_processing/bulk_retrain_preparation_fullchr_split.py \
--input_dir /data/GM12878/hic_extraction \
--subset_file Extracted_5k_"${chrom}".txt \
--save_dir /data/GM12878/processed \
--save_file Extracted_noisy_mat_5kb_size400_"${chrom}".txt \
--chrom "${chrom}" \
--assembly hg38 \
--resolution 5000 \
--mat_size 400
```