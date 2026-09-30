# VEP regulatory golden

VEP 104.3 (native, `pixi -e vep`, `PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0`) with
`--everything --af_gnomad --individual all` on `input.vcf` (17 lines, two samples), using
`cache/`, a synthetic offline cache: `info.txt` and two chromosome 21 regulatory chunks
(`*_reg.gz`, Perl Storable like the real cache). There is no real annotation data: the features,
binding matrices and reference bases are made up to hit each rule of VEP's regulatory and motif
code, and there are no transcripts, so the other rows are intergenic.

- `expected.txt.gz`: VEP's output. The test removes its RegulatoryFeature and MotifFeature rows
  and has `aim vep-annotate --regulatory` put them back.

`rust/tools/make_goldens_vep_regulatory.pl` writes `cache/` and `input.vcf` (in the VEP
environment, for Storable) and gives the commands.
