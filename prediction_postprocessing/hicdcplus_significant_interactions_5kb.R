#!/usr/bin/env Rscript

# HiC-DC+ normalization / significant interaction calling
# Runs HiC-DC+ on:
#   a) low-coverage Hi-C
#   b) high-coverage Hi-C
#   c) high-coverage replicate
#   d) HiC2Self recovered predictions (.hic)
#
# Run from repo root (example):
# Rscript prediction_postprocessing/hicdcplus_significant_interactions_5kb.R \
#   5000 \
#   outputs/predictions/GM12878_predictions_for_hic.hic \
#   outputs/hicdcplus/features/ \
#   outputs/hicdcplus/intermediate/ \
#   hg38 \
#   3 \
#   data/hic_files/GSM1551550_HIC001_30.hic \
#   data/hic_files/GSE63525_GM12878_insitu_primary_30.hic \
#   data/hic_files/GSE63525_GM12878_insitu_replicate_30.hic \
#   2 \
#   GM12878_hic001

suppressPackageStartupMessages({
  library(HiCDCPlus)
})

set.seed(1010)
args <- commandArgs(trailingOnly = TRUE)
print(args)

# -------------------------
# Argument parsing
# -------------------------
# Required (keep your original 6, in the same order)
resolution  <- as.integer(args[1])  # e.g. 5000 or 10000
input_file  <- args[2]             # predictions .hic (or any .hic)
outdir      <- args[3]             # features dir
outpth      <- args[4]             # intermediate/results dir
assembly    <- args[5]             # hg19 / hg38 / mm9 / mm10
chromosome  <- as.integer(args[6]) # e.g. 3

# Optional (portable replacements for your hard-coded paths)
hic_low     <- ifelse(length(args) >= 7,  args[7],  NA)
hic_high    <- ifelse(length(args) >= 8,  args[8],  NA)
hic_rep     <- ifelse(length(args) >= 9,  args[9],  NA)
ncore       <- ifelse(length(args) >= 10, as.integer(args[10]), 2)
prefix      <- ifelse(length(args) >= 11, args[11], "GM12878_hic001")

# Make output dirs if needed
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
dir.create(outpth, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# Assembly / species setup
# -------------------------
if (assembly %in% c("mm9", "mm10")) {
  species <- "Mmusculus"
  chrom_length <- 19
} else if (assembly %in% c("hg19", "hg38")) {
  species <- "Hsapiens"
  chrom_length <- 22
} else {
  stop("Assembly not supported: ", assembly)
}

chromosomes <- c(paste0("chr", chromosome))

# -------------------------
# Step 1. Construct features
# -------------------------
feature_prefix <- paste0(assembly, "_", as.integer(resolution/1000), "kb_GATC_GANTC")
feature_out_path <- file.path(outdir, feature_prefix)

construct_features(
  output_path = feature_out_path,
  gen = species,
  gen_ver = assembly,
  sig = c("GATC", "GANTC"),
  bin_type = "Bins-uniform",
  binsize = resolution,
  chrs = chromosomes
)

bintolen_path <- paste0(feature_out_path, "_bintolen.txt.gz")

# Helper: run HiC-DC+ for one hic and write outputs
run_hicdcplus_one <- function(hicfile_path, tag) {
  if (is.na(hicfile_path) || hicfile_path == "") {
    message("[Skip] ", tag, " hic path not provided.")
    return(invisible(NULL))
  }
  if (!file.exists(hicfile_path)) {
    stop("[Error] File not found for ", tag, ": ", hicfile_path)
  }

  gi_list <- generate_bintolen_gi_list(bintolen_path = bintolen_path)
  gi_list <- add_hic_counts(gi_list, hic_path = hicfile_path, chrs = names(gi_list))
  gi_list <- expand_1D_features(gi_list)
  gi_list <- HiCDCPlus_parallel(gi_list, ncore = ncore)

  # gz summary
  out_gz <- file.path(outpth, paste0(prefix, "_", chromosome, "_", tag, "_normalized.txt.gz"))
  gi_list_write(gi_list, fname = out_gz)

  # per-chrom txt (matches your original behavior)
  for (chrom in names(gi_list)) {
    message(chrom)
    out_txt <- file.path(outpth, paste0(prefix, "_", tag, "_normalized_", chrom, ".txt"))
    write.table(
      gi_list[[chrom]],
      file = out_txt,
      sep = "\t",
      row.names = FALSE,
      col.names = TRUE,
      quote = FALSE
    )
  }

  invisible(gi_list)
}

# -------------------------
# Step 2a–2d (same logic)
# -------------------------
run_hicdcplus_one(hic_low,  "low")
run_hicdcplus_one(hic_high, "high")
run_hicdcplus_one(hic_rep,  "replicate")

# Predictions: your original "input_file" slot
run_hicdcplus_one(input_file, "pred")
