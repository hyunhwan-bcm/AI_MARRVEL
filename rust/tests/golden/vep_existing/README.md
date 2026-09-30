# VEP known-variant golden

VEP 104.3 (native, `pixi -e vep`, `PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0`) with
`--everything --af_gnomad --individual all` on `input.vcf` (20 lines, two samples), using
`cache/`, a synthetic offline cache: `info.txt` and one known-variant file, `21/all_vars.gz`,
CSI-indexed like the real cache. There is no real annotation data: the cache has no transcripts
(every row is intergenic), and its known variants are made up to hit each rule of the
co-located code.

- `expected.txt.gz`: VEP's output, co-located columns included. The test blanks those columns
  and has `aim vep-annotate --known-variants` recompute them.

`rust/tools/make_goldens_vep_existing.py` writes `cache/` and `input.vcf`, and gives the commands.
