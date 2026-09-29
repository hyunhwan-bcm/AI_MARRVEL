# ClinVar sample fixture (GRCh38)

`clinvar_sample.vcf`: every 1000th biallelic SNV/indel (REF/ALT ≤ 20 bp, chr1–22, X) of
`vep/hg38/clinvar_20220730.vcf.gz` from the AIM data bucket, position-sorted, with a seeded
genotype per variant (15% `1/1`, else `0/1`) for sample `PROBAND`. ClinVar data is public domain.
`clinvar_sample.hpo.txt`: same HPO terms as the fastVEP fixture (breast carcinoma, ovarian neoplasm).
