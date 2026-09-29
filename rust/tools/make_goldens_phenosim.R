#!/usr/bin/env Rscript
# Golden HPO_SIM outputs (bin/phenoSim.R itself) for more patient phenotype sets than the
# fixtures, and for a synthetic HGMD phenotype table (the public one is empty):
#
#     Rscript rust/tools/make_goldens_phenosim.R <repo_root> <data_dir> <out_dir>
#
# Each <out_dir>/<case>/ gets input.hpo.txt, HGMD_phen.tsv, expected_hgmd_sim.tsv and
# expected_omim_sim.tsv.gz (hg38 inputs otherwise).
args <- commandArgs(trailingOnly = TRUE)
repo <- args[1]; data <- args[2]; out <- args[3]
set.seed(20260929)
ann <- file.path(data, "omim_annotate")
omim <- read.table(file.path(ann, "hg38", "HPO_OMIM.tsv"), sep = "\t", header = TRUE,
                   stringsAsFactors = FALSE, comment.char = "", fill = TRUE, quote = "\"")
terms <- unique(omim$HPO_ID)
hgmd <- data.frame(
  acc_num = sprintf("CM%06d", sample(c(7, 1234, 99999, 50), 60, replace = TRUE)),
  gene_sym = sample(c("BRCA1", "TP53", "MSH2", "SCN1A", "CFTR", "FBN1"), 60, replace = TRUE),
  phen_id = sample(c(101, 2, 33, 4000), 60, replace = TRUE),
  hpo_id = sample(c(terms[1:200], NA), 60, replace = TRUE))
cases <- list(
  hpo3 = sample(terms, 3),
  hpo12 = sample(terms, 12),
  dup_and_other = c(sample(terms, 4), "not-a-term", terms[5], terms[5]))
for (name in names(cases)) {
  d <- file.path(out, name)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  writeLines(cases[[name]], file.path(d, "input.hpo.txt"))
  write.table(hgmd, file.path(d, "HGMD_phen.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  omim_out <- file.path(d, "expected_omim_sim.tsv")
  st <- system2("Rscript", c(file.path(repo, "bin", "phenoSim.R"), file.path(d, "input.hpo.txt"),
                             file.path(d, "HGMD_phen.tsv"), file.path(ann, "hp.obo"),
                             file.path(ann, "hg38", "genemap2_v2022.rds"),
                             file.path(ann, "hg38", "HPO_OMIM.tsv"), file.path(d, "expected_hgmd_sim.tsv"), omim_out))
  stopifnot(st == 0)
  system2("gzip", c("-nf9", omim_out))
  cat(name, ": ", length(cases[[name]]), " terms\n", sep = "")
}
