#hicdc+ for normalization
# Calculate significant interactions in
# a) low-coverage Hi-C data
# b) high-coverage Hi-C data
# c) high-coverage replicte data
# d) HiC2Self recovered data
#
# Usage
# screen
# bsub -n 2 -W 10:00 -R 'span[hosts=1] rusage[mem=64]' -Is /bin/bash
# bsub -q gpuqueue -gpu - -W 10:00 -n 2 -R 'span[hosts=1] rusage[mem=256]' -Is /bin/bash #or using gpu # nolint
# bsub -J run_R -W 10:00 -sla llSC2 -q gpuqueue -R 'A100 rusage[mem=256]' -n 1 -gpu 'num=1:mode=exclusive_process'  -Is /bin/bash
# source /home/yangr2/miniconda3/etc/profile.d/conda.sh #nolint
# conda activate chromafold_env

# Rscript /data/leslie/yangr2/hic2self/scripts/hicdcplus/hicdcplus_normalization_ppl.R \
# 10000 \
# /data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/predictions/predictions.txt
# '/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/features/' \
# '/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/intermediate/' \
# 'hg19' \
# 3

library(HiCDCPlus)
library(BSgenome.Hsapiens.UCSC.hg19)

set.seed(1010)
args <- commandArgs(trailing = TRUE)
print(args)

#1. Initial set-up

resolution <- as.integer(args[1]) #resolution
input_file <- args[2]
outdir <- args[3]
outpth <- args[4]
assembly <- args[5]
chromosome <- as.integer(args[6])


resolution <- 5000 #resolution
outdir <- '/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/features/'
outpth <- '/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/intermediate/'
assembly <- 'hg19'
chromosome <- 3
input_file <- '/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/predictions/GM12878_hic001_predictions_3_log22.01_maskwidth3_symmetric_grid_predictions_v2.hic'
# save_hic_pth='/data/leslie/yangr2/hic2self/data/bulk_retrain/predictions/save_to_hic/all_chromosomes/' #nolint

########################################
#      Step 1. Construct features      #
########################################

if (assembly %in% c("mm9", "mm10")) {
  species <- "Mmusculus"
  chrom_length <- 19
} else if (assembly %in% c("hg19", "hg38")) {
  species <- "Hsapiens"
  chrom_length <- 22
} else {
  species <- "Unknown"
  print("Species not match")
}

#Step 1. construct features (resolution, chrom specific)
chromosomes <- c(paste("chr", chromosome, sep = ""))
construct_features(output_path = paste0(outdir, assembly ,"_", as.integer(resolution/1000),"kb_GATC_GANTC", sep = ""), # nolint
gen = species, gen_ver = assembly, sig = c("GATC", "GANTC"),
bin_type = "Bins-uniform", binsize = resolution, chrs = chromosomes)

#######################################################################
#      Step 2a. Generate instances and calculate -- Low-coverage      #
#######################################################################

# hicfile_path <- paste0(save_hic_pth, "GM12878_hic001_", chromosome, "_low.hic")
hicfile_path <- "/data/leslie/yangr2/hic2self/data/bulk_retrain/hic_files/GSM1551550_HIC001_30.hic"
#generate gi_list instance
gi_list <- generate_bintolen_gi_list(
  bintolen_path = paste0(outdir, assembly, "_", as.integer(resolution/1000),"kb_GATC_GANTC_bintolen.txt.gz"))
#add .hic counts
gi_list <- add_hic_counts(gi_list, hic_path = hicfile_path, chrs = names(gi_list)) #nolint
#expand features for modeling
gi_list <- expand_1D_features(gi_list)
#run HiC-DC+ on 2 cores
gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2)
#write results to a text file
gi_list_write(gi_list,
fname = paste0(outpth, "GM12878_hic001_", chromosome, "_low_normalized.txt.gz"))
for (chrom in names(gi_list)) {
  print(chrom)
  # Construct the file name
  file_name <- paste0(outpth, "GM12878_hic001_low_normalized_", chrom, ".txt")
  # Extract the data for the current chromosome
  data <- gi_list[[chrom]]
  # Save the data to a text file
  write.table(data, file = file_name, sep = "\t",
  row.names = FALSE, col.names = TRUE, quote = FALSE)
}

########################################################################
#      Step 2b. Generate instances and calculate -- High-coverage      #
########################################################################

# hicfile_path <- paste0(save_hic_pth, "GM12878_hic001_", chromosome, "_high.hic")
hicfile_path <- "/data/leslie/yangr2/hic2self/data/bulk_retrain/hic_files/GSE63525_GM12878_insitu_primary_30.hic"
#generate gi_list instance
gi_list <- generate_bintolen_gi_list(
  bintolen_path = paste0(outdir, assembly, "_", as.integer(resolution/1000),"kb_GATC_GANTC_bintolen.txt.gz"))
#add .hic counts
gi_list <- add_hic_counts(gi_list, hic_path = hicfile_path, chrs = names(gi_list)) #nolint
#expand features for modeling
gi_list <- expand_1D_features(gi_list)
#run HiC-DC+ on 2 cores
gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2)
#write results to a text file
gi_list_write(gi_list,
fname = paste0(outpth, "GM12878_hic001_", chromosome,
"_high_normalized.txt.gz"))
for (chrom in names(gi_list)) {
  print(chrom)
  # Construct the file name
  file_name <- paste0(outpth, "GM12878_hic001_high_normalized_", chrom, ".txt")
  # Extract the data for the current chromosome
  data <- gi_list[[chrom]]
  # Save the data to a text file
  write.table(data, file = file_name, sep = "\t",
  row.names = FALSE, col.names = TRUE, quote = FALSE)
}

####################################################################
#      Step 2c. Generate instances and calculate -- Replicate      #
####################################################################

# hicfile_path <- paste0(save_hic_pth, "GM12878_hic001_", chromosome, "_replicate.hic")
hicfile_path <- "/data/leslie/yangr2/hic2self/data/bulk_retrain/hic_files/GSE63525_GM12878_insitu_replicate_30.hic"
#generate gi_list instance
gi_list <- generate_bintolen_gi_list(
  bintolen_path = paste0(outdir, assembly, "_", as.integer(resolution/1000),"kb_GATC_GANTC_bintolen.txt.gz"))
#add .hic counts
gi_list <- add_hic_counts(gi_list, hic_path = hicfile_path, chrs = names(gi_list)) #nolint
#expand features for modeling
gi_list <- expand_1D_features(gi_list)
#run HiC-DC+ on 2 cores
gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2)
#write results to a text file
gi_list_write(gi_list,
fname = paste0(outpth, "GM12878_hic001_", chromosome,
"_replicate_normalized.txt.gz"))
for (chrom in names(gi_list)) {
  print(chrom)
  # Construct the file name
  file_name <- paste0(outpth,
  "GM12878_hic001_replicate_normalized_", chrom, ".txt")
  # Extract the data for the current chromosome
  data <- gi_list[[chrom]]
  # Save the data to a text file
  write.table(data, file = file_name, sep = "\t",
  row.names = FALSE, col.names = TRUE, quote = FALSE)
}

######################################################################
#      Step 2d. Generate instances and calculate -- Predictions      #
######################################################################

hicfile_path <- input_file
#generate gi_list instance
gi_list <- generate_bintolen_gi_list(
  bintolen_path = paste0(outdir, assembly, "_", as.integer(resolution/1000),"kb_GATC_GANTC_bintolen.txt.gz"))
#add .hic counts
gi_list <- add_hic_counts(gi_list, hic_path = hicfile_path, chrs = names(gi_list)) #nolint
#expand features for modeling
gi_list <- expand_1D_features(gi_list)
#run HiC-DC+ on 2 cores
gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2)
#write results to a text file
gi_list_write(gi_list,
fname = paste0(outpth,
"GM12878_hic001_", chromosome, "_pred_normalized.txt.gz"))
for (chrom in names(gi_list)) {
  print(chrom)
  # Construct the file name
  file_name <- paste0(outpth, "GM12878_hic001_pred_normalized_", chrom, ".txt")
  # Extract the data for the current chromosome
  data <- gi_list[[chrom]]
  # Save the data to a text file
  write.table(data, file = file_name, sep = "\t",
  row.names = FALSE, col.names = TRUE, quote = FALSE)
}