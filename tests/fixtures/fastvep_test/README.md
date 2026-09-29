# fastVEP test fixture

- `test.vcf`, `test.gff3`: from https://github.com/Huang-lab/fastVEP (tag v0.4.0, commit d728c46), Apache-2.0.
  `test.vcf` has its data lines stably sorted by chromosome and position (upstream's are not
  sorted, which AIM's `tabix` step rejects, #34); `test.gff3` is unchanged.
  8 GRCh38 variants (7 BRCA1, 1 TP53); `test.gff3` is a trimmed BRCA1/TP53 gene model (not a complete Ensembl release).
- `test.aim.vcf`: `test.vcf` plus a GT column (sample `PROBAND`; all 0/1 except `rs_cds_tp53` = 1/1),
  because AIM derives zygosity features from genotypes, with data lines stably sorted by
  chromosome and position (the same order as `test.vcf`).
- `test.hpo.txt`: HP:0003002 (Breast carcinoma), HP:0100615 (Ovarian neoplasm).
- Coding consequences need GRCh38 chr17 FASTA (Ensembl release 115 `Homo_sapiens.GRCh38.dna.chromosome.17.fa.gz`), not committed.
