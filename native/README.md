# Native baseline (no containers)

Runs AIM v1.1.3 on macOS/Linux with [pixi](https://pixi.sh) environments that mirror the
production images, so its outputs can serve as the reference for the Rust port.

| pixi env | mirrors | notes |
|---|---|---|
| `py` (+ `tools`) | `zhandongliulab/aim-lite:1.2` | Python 3.8.20 and the image's `pip freeze`; bcftools 1.20, GNU coreutils/sed/find/grep/gawk |
| `r` | `zhandongliulab/aim-lite-r` | R 4.4.2, dplyr 1.1.4, ontologyIndex 2.12, ontologySimilarity 2.7 (built from CRAN) |
| `vep` | `ensemblorg/ensembl-vep:release_104.3` | VEP code extracted from the image, run on native perl 5.32 |

Known differences from the images (all reviewed as output-neutral for AIM, to be confirmed on a
container run): bedtools 2.31.1 (image 2.30.0), data.table 1.16.4 (1.16.2), numexpr 2.8.4 (2.8.6),
PyTables from conda-forge instead of PyPI, perl 5.32 instead of the image's system perl, and
DB_File 1.853 (conda's perl-db_file 1.858 segfaults on osx-arm64).

Two container paths are made overridable, with the image paths kept as defaults:
`ANNOTATE_BY_VEP` uses `$AIM_VEP_BIN`, and `simple_repeat_anno.py` falls back to `bedtools` on PATH.

## Setup

```bash
pixi install --all
pixi run -e vep fetch-vep
pixi run -e r install-ontologysimilarity
```

## Run the fixture

```bash
native/run_fixture.sh    # REF_DIR, FIXTURE, OUT overridable via env
```

## Known data issue: hg38 gnomAD genome index

The published `vep/hg38/gnomad.genomes.GRCh38.v3.1.2.sites.vcf.gz.tbi` does not match its data
file, so as deployed, hg38 runs get no gnomAD genome AF. `native/fix_gnomad_hg38_index.sh`
builds a correct index into a separate directory and an overlay data directory that uses it,
without touching the original data:

```bash
pixi run -e py bash native/fix_gnomad_hg38_index.sh DATA_DIR FIX_DIR OVERLAY_DIR
REF_DIR=OVERLAY_DIR OUT=... native/run_fixture.sh
```

The same file names its homozygote count `nhomalt`, while the pipeline requests
`controls_nhomalt` (the hg19 name), so `hom` stays 0 on hg38 even with the fixed index.
Decision (2026-09-28): leave `hom` = 0 on hg38; sourcing `nhomalt_controls_and_biobanks` from
full gnomAD v3.1.2 (~2.3 TiB) was judged too costly.
