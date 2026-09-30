# VEP lookup goldens

VEP 104.3 (native, `pixi -e vep`) on `input.vcf` (15 chr17 lines, two samples) with the
GRCh38 VEP 104 cache:

- `base.txt.gz`: `--everything` without `--custom`/`--plugin` (the input of `aim vep-annotate`)
- `expected.txt.gz`: the same with two `--custom` VCFs and the REVEL, SpliceAI, CADD and dbNSFP
  plugins
- `data/`: the lookup files, synthetic (no real annotation data), written by
  `rust/tools/make_goldens_vep_annotate.py`, which also gives the exact commands

Compare from `## Column descriptions:` on: the lines before it hold a timestamp and version lines
that VEP prints in random order.
