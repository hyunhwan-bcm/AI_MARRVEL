#!/usr/bin/env Rscript
# genemap2_v2022.rds -> the table HPO_SIM (bin/phenoSim.R) merges on, as TSV for the Rust port:
#
#     Rscript rust/tools/export_genemap.R <data_dir> <refs_dir>
#
# Writes <refs_dir>/omim_annotate/<ref>/genemap2_pheno.tsv: unique(genemap2[, c("Pheno_ID",
# "Approved_Gene_Symbol", "Ensembl_Gene_ID", "Entrez_Gene_ID")]) in R's row order. Numbers are
# written with 17 significant digits (exact), missing values as empty fields.
args <- commandArgs(trailingOnly = TRUE)
for (ref in c("hg19", "hg38")) {
  g <- readRDS(file.path(args[1], "omim_annotate", ref, "genemap2_v2022.rds"))
  u <- unique(g[, c("Pheno_ID", "Approved_Gene_Symbol", "Ensembl_Gene_ID", "Entrez_Gene_ID")])
  stopifnot(is.numeric(u$Pheno_ID), is.character(u$Approved_Gene_Symbol),
            is.character(u$Ensembl_Gene_ID), is.numeric(u$Entrez_Gene_ID))
  for (s in c("Approved_Gene_Symbol", "Ensembl_Gene_ID")) {
    v <- u[[s]]
    stopifnot(!any(v[!is.na(v)] == ""), !any(grepl("[\t\n]", v[!is.na(v)])))
  }
  num <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))
  chr <- function(x) ifelse(is.na(x), "", x)
  out <- data.frame(Pheno_ID = num(u$Pheno_ID), Approved_Gene_Symbol = chr(u$Approved_Gene_Symbol),
                    Ensembl_Gene_ID = chr(u$Ensembl_Gene_ID), Entrez_Gene_ID = num(u$Entrez_Gene_ID))
  dir <- file.path(args[2], "omim_annotate", ref)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  write.table(out, file.path(dir, "genemap2_pheno.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  cat(ref, ": ", nrow(out), " rows\n", sep = "")
}
