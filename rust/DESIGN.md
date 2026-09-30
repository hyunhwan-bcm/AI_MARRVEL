# AIM Rust port: design decisions

Status: draft, agreed in discussion 2026-09-28. Nothing here removes existing behaviour; every
change is additive and opt-in until retrained models have been validated.

## Goals (in priority order when they conflict)

1. Identical results to AIM v1.1.3 (floating-point noise accepted; row order may differ only
   within exact ties).
2. Less memory.
3. Fewer dependencies (one Rust binary; VEP replaced later only behind a compatibility layer).
4. Faster.

## Feature sets: `v1` default, `v2` placeholder

|                                                                | `v1` (default)                                                                             | `v2` (placeholder, opt-in)                                           |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Missing scores                                                 | Filled exactly as `bin/fillna_tier.py` does (per-sample mean/median/min/max, row-weighted) | Left missing; XGBoost routes them with its learned default direction |
| Network score (`diffuse_Phrank_STRING`)                        | Percentile rank among the sample's rows                                                    | Raw diffused heat                                                    |
| Variant type                                                   | Implicit (indel vs SNV fill values)                                                        | Explicit feature from VEP `VARIANT_CLASS`                            |
| Same-gene tier / recessive pairs                               | Kept                                                                                       | Kept (separate gene-level pass)                                      |
| Derived features (`conservationScore*`, `curationScore*`, ...) | Kept                                                                                       | Kept for now; pruning decided after retraining                       |
| Models                                                         | Current `model_inputs/*` (unchanged)                                                       | None yet — requires retraining                                       |

How the placeholder works:

- Every per-variant value is carried as "missing or value" through annotation. Nothing is filled
  early.
- Filling and the network-score transform are the last step before the model, chosen by the
  feature set.
- Each model bundle carries a manifest naming the feature set it was trained on. The binary
  refuses to pair a bundle with a different feature set, so `v2` cannot silently run on `v1`
  models. Until `v2` models exist, `--feature-set v2` only runs with an explicit
  "untrained placeholder" flag, and its outputs are marked as not valid for interpretation.

## Libraries over hand-written code

Where a Rust library does the job, use it: Polars for reading/writing tables, joins and
group-bys (added 2026-09-29 at the user's request, despite ~340 crates and ~2 min extra
release build). Hand-written code is limited to AIM's own rules (fill order, tier logic,
diffusion) and to numpy details that decide exact numbers (pairwise-sum mean, linear
median, Python float formatting). Consequence: Polars parses floats with correct rounding
where pandas 1.4's parser can be one ulp off, so continuous features may differ from the
pipeline by about one ulp (below float32 model resolution); discrete features and model
outputs are still compared exactly.

## Data: nothing is deleted

Lineage tracing (see discussion) found inputs that are never read, e.g. `dbNSFP4.3c_grch37.gz`,
a byte-identical copy of the GRCh37 VEP cache under `vep/hg38/`, and
`mod5_diffusion/combined_score.hdf5`. They stay in place. Compact per-field stores built for the
Rust port are added next to the originals, never in place of them.

## Findings that affect "identical"

- VEP 104.3 is not deterministic run to run. Two runs of the same command on the same input
  (8,233 multi-allelic lines, 156,053 rows) put 43,870 rows in a different order and differ in
  `CDS_position`, `Protein_position`, `HGVSp` and `Amino_acids` on a few dozen rows (Perl hash
  order: a sample's two alternates, consequence terms of equal rank). Single-allelic 0/1 and 1/1
  input is stable. `aim vep-annotate` keeps the rows and order of the VEP run it is given, so its
  output is identical to the full VEP command whenever VEP itself is.
- VEP's lookups (`aim vep-annotate`) are reproduced with their v1.1.3 behaviour, not fixed:
  dbNSFP returns the whole matching row (every `;` list complete, not this transcript's entry)
  and overwrites VEP's `APPRIS`/`TSL`; REVEL takes the first row with the same alternate amino
  acid whatever the transcript; custom VCFs ignore FILTER and join all matching records;
  `gnomAD_AF` and `CLIN_SIG` come from the VEP cache, not the custom files.

- The published hg38 gnomAD genome index (`vep/hg38/gnomad.genomes.GRCh38.v3.1.2.sites.vcf.gz.tbi`)
  does not match its data file (every lookup fails with "Invalid BGZF header"; a freshly built
  index works), and that file names the field `nhomalt` while AIM requests `controls_nhomalt`.
  So on hg38, v1.1.3 as deployed gets no gnomAD genome AF and `hom` is always 0. The Rust port
  models data sources as configuration, so the v1.1.3 hg38 profile can say "no gnomAD genome
  source" instead of reproducing a broken index; the files in the bucket are left untouched.
- pandas 1.4's default CSV float parser is not exactly round-trip: each write/read of an
  intermediate CSV can shift values by one ulp (18 of 824 cells between `fixture.matrix.txt`
  and `fixture.default_prediction.csv`). To stay bit-identical the port must either reproduce
  that parser at the same round-trip points or accept one-ulp differences (below f32 model
  resolution).

- Decision (2026-09-28): `hom` (gnomAD genome homozygote count) stays 0 on hg38, in both the
  as-deployed and corrected profiles. The hg38 file only has `nhomalt` (all samples), not the
  `controls_nhomalt` the pipeline requests; the closest v3 field, `nhomalt_controls_and_biobanks`,
  needs ~2.3 TiB of gnomAD downloads, judged too costly. hg38 predictions therefore never use
  `hom`; hg19 is unaffected.

- phrank scores are sums over a Python set intersection, so their last digits depend on CPython's
  set iteration order: random per run in production (#33), fixed under `PYTHONHASHSEED=0` (the
  native config). The port reproduces that order (SipHash-2-4 with a zero key, CPython 3.8's set
  table: `crate::pyset`) and matches the seeded pipeline byte for byte; summing in any other
  order changes the last one or two digits of many scores.
- `location_to_gene.py`'s `binary_search` is not an overlap test: it keeps the entry where the
  bisection stopped even without a match, plus neighbours with the same coordinate. The gene
  list therefore contains genes near, not at, a variant. Reproduced as is for `v1`.

- HPO_SIM (`phenoSim.R`) prints similarities with R's `formatReal`, whose 15-digit scaling is
  platform dependent: in double arithmetic on arm64 macOS (ported, byte-identical there), in
  80-bit long double on x86-64 Linux, where a last digit can differ (e.g. `0.32398886838473` vs
  `0.323988868384731`). Values agree to ~1e-15 either way. The genemap2 table is read from an
  RDS file, exported once to TSV with `rust/tools/export_genemap.R`, so the `--rust` pipeline
  needs no R at run time. That export is not a Nextflow input: after updating the data bucket,
  re-run `export_genemap.R` (and use a clean run rather than `-resume`).
- HPO_SIM input quirks reproduced from `read.table`: blank HGMD fields are `NA` in numeric
  columns and empty strings in text columns; patient files are split into columns by the widest
  of the first 5 lines, and longer lines wrap. Not reproduced: an unmatched quote character in
  the patient file (R drops or merges terms depending on where it falls); `aim` warns instead.
  Duplicate OBO term ids are an error (R merges them).

- `feature.py` (ANNOTATE_BY_MODULES), reproduced as is for `v1`:
  - DECIPHER never matches: `(chrom, start, stop) in decipherSortedDf` tests column names, not
    the index, so `decipherVarFound` is always 0 (DECIPHER is not even read by the port).
  - `omim_alleric_variants.json` (35 MB) is loaded but never used.
  - `clinVarSymMatchFlag` is set on a row copy (`iterrows`) and never copied back: always 0 in
    `scores.csv`; only `curationScoreClinVar` sees it.
  - `clinvarCurate` recomputes `curationScoreHGMD`, overwriting `hgmdCurate`'s value.
  - The hg19 gene tables are used for hg38 as well.
- pandas types a column in chunks of rows (`low_memory`): 2,048 rows for VEP's 459 columns. A
  chunk with any non-numeric value keeps that chunk's original text; a numeric chunk is
  re-parsed with pandas' lossy parser and re-printed. So how a VEP number appears in
  `scores.csv` depends on the other values within its 2,048-row block. The port reproduces
  this (`pdread`), verified against `feature.py` on a 15,216-row table (8 chunks).

## Open items

- Training data (per-patient annotated files, pre-fill) for `v2` retraining.
- Exome-scale VCF for memory/speed benchmarks (the fastVEP fixture only exercises fixed costs).
