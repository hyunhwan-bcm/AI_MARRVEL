# AIM Rust port — bullet journal

Key: `•` task · `×` done · `>` migrated (moved later) · `<` scheduled · `–` note · `!` important · `o` event

Tracking issue: [#35](https://github.com/hyunhwan-bcm/AI_MARRVEL/issues/35)

## Future log

- • VCF preprocessing in Rust (bcftools → noodles)
- • VEP consequence engine in Rust (stages 4–7 of the VEP plan; decide after exome timing)
- • VEP: fastVEP behind a VEP-104 compatibility layer (v2 option)
- • `v2` feature set + retraining (needs training data)
- • Exome-scale benchmark

## 2026-09-28 (Mon)

- o Kick-off: port AIM v1.1.3 to Rust — less memory, fewer deps, faster, identical results
- × Worktrees: `AI_MARRVEL-baseline` (current version), `AI_MARRVEL-rust` (port)
- × Full data bucket download started (519 GiB)
- × Assessed fastVEP as VEP replacement
  - – VEP 104 lacks 3 splice terms fastVEP emits; models split on IMPACT heavily → needs compat layer
- × Native baseline with pixi (no Docker): py/r/vep envs mirror the production images
  - – conda `perl-db_file` segfaults on arm64 → use perl's bundled DB_File
- × Lineage trace: input fields → features → model splits
  - – 42 of the default model's 103 features never split on (corrected 2026-09-29; earlier note said 36 of 104); `nc_*` ClinVar features unused but row duplication matters
  - – ~47 GiB of data never read (kept, not deleted)
- × Decisions: `v1` identical default, `v2` placeholder; additive only, nothing deleted
- × Prediction stage in Rust — bit-identical (XGBoost, confidence, ranking, SHAP)
- ! hg38 gnomAD genome index broken in the bucket (#30); rebuilt index + overlay
- ! `hom` always 0 on hg38 (#31) — decided: keep 0, noted

## 2026-09-29 (Tue)

- × Baseline fixture run: 24/24 steps, 42 s warm, peak 1.5 GB
- × `PYTHONHASHSEED=0` — phrank not reproducible run to run (#33)
- × Fork set up: issues on, #30–#35 filed
- × Stacked PRs #36 (fork-safe CI) → #37 (baseline) → #38 (Rust core + CI) → #39 (goldens)
- × Prettier fixes restacked across #37–#39
- × Diffusion ported, 1.4 GB → 15.5 MB, 3,608 scores identical (#40)
- × CI green on #36–#38; prettier excluded from golden files (#39, #40)
- × Bigger fixture: 1,450 ClinVar hg38 variants — baseline 5 min 15 s, 111 tasks
  - – task time: VEP 32%, feature.py 29%, ClinVar join 19%, tier 8%
  - – `simple_repeat` never fires on chrX: variant ids use `23`, the BED uses `X`
- × Missing-value fill + simple repeats → full MERGE step (#42) — matrix identical to ≤ 1 ulp
  - – switched table I/O to Polars (user: use Rust alternatives); ~1-ulp float parse diffs accepted
- × Review of #36–#40: no blocking issues; fixed VEP fetch hardening, fixture paths, SHAP claim, early-stop guard
  - ! overrides script linked to itself on re-run — fixed, idempotent now
- × Review of #42: no blocking issues; fixed `isP/LP` int64 rule and O(N×M) tier lookup
- o Merged #36–#42 into fork `main` (fast-forward, reviewed commits unchanged)
- × JOIN_PHRANK (`add_c_nc.py` + `generate_new_matrix_2.py`) — 3 chromosome files, 255,675 cells identical, 1,001 within 1 ulp
  - ! non-coding ClinVar features never match anything: int vs str chromosome (#43) — the N×M grids compute nothing
- × Review of #44: no blocking issues; fixed pandas `.loc` label lookup, `_x`/`_y` merge suffixes, literal `NA` strings
- × Tier (`VarTierDiseaseDBFalse.R`) — `Tier.v2.tsv` byte-identical on 4 files (990 rows)
  - – R `merge()` reorders rows within a gene: `do_merge` uses an unstable Shell sort; reproduced
  - – readr guesses column types from the first 1,000 rows; reproduced (not triggered in the goldens)
- o Merged #44 into fork `main`
- × PREDICTION I/O (part 1): `run_final.py` + default/nd `extraModel` files + SHAP JSON — row order and cells match both runs
  - – pandas `sort_values` is numpy's unstable introsort: ported, verified on tie-heavy inputs
  - ! pandas' CSV parser keeps 17 digits incl. leading zeros → up to ~1e-14 rel on small values (tests use it as an oracle)
  - – Python repr breaks exact 17-digit ties half-to-even; Rust's shortest formatter didn't — fixed
- × Review of #45/#46: readr samples 999 spaced rows + last (fixed); float32/f64 repr ties half-even at any length; `sort_index` no-op when sorted
- × PREDICTION I/O (part 2): expanded matrix, recessive pairs, recessive + nd_recessive — all files match both runs; SHAP identical
- > phrank chain + HPO similarity (R) — both done below
- • Feature annotation (`feature.py`)
- × Review of #47: no blocking issues (40 randomized samples matched Python byte for byte)
- o Merged #45–#47 into fork `main`
  - ! deleted stacked base branches before GitHub recorded the merges → #46/#47 closed; restored branches, merged into bases, now MERGED
- × `aim` CLI (join-phrank, tier, merge, predict) + opt-in `--rust true` in Nextflow
  - – per step on ClinVar sample, Python/R → Rust: tier 1.01 s/207 MB → 0.01 s/53 MB; merge 4.14 s/1,592 MB → 0.51 s/223 MB; predict 7.83 s/676 MB → 2.08 s/344 MB; join chr2 2.64 s/947 MB → 0.93 s/183 MB (see below)
  - – float repr: exact expansion only when a tie is possible (predict 4.25 s → 2.08 s)
- ! first `--rust true` run failed at JOIN_PHRANK: stale release binary; rebuilt
- × End-to-end `--rust true` on the ClinVar sample: 111/111 tasks, 4 min 38 s (baseline 5 min 15 s)
  - – every output vs baseline, rows compared as sets: VCF/VEP identical; matrix, predictions, rankings, expanded ≤ 6.9e-13 rel; SHAP ≤ 1.8e-15 abs
  - – merged row order differs run to run in _both_ versions: Nextflow concatenates chromosomes in completion order
- × JOIN reads only the chromosome's ClinVar coding rows (streamed, dtypes from all rows): chr2 1.2–1.5 GB → 183 MB, output byte-identical
  - ! the earlier 643 MB join figure did not reproduce (full read measured 1.2–1.5 GB); corrected
- • Final report artifact
- × Review of #48: fixed HGMD columns lost when a chromosome's coding rows are all filtered, chrom-filter fallback, `--rust` param checks
  - ! one review suggestion (strip the index in `predict`) was wrong — caught by re-running `predict` against the run's outputs, reverted to `to_csv_no_index`
- × PHRANK_SCORING in Rust (`aim phrank`): VCF → genes → phrank ranking in one step, 4 processes → 1
  - – scores sum over a Python set: CPython 3.8 set layout + SipHash emulated → `phrank.txt` byte-identical (2 runs + 5 larger HPO sets); sorted order would change last digits
  - – `location_to_gene.py` bisection keeps a gene even without overlap — reproduced
- × End-to-end `--rust true` with phrank: 108/108 tasks, 4 min 03 s (baseline 5 min 15 s); model outputs identical; features ≤ 6.9e-13 rel (pandas parser truncation in the baseline)
- × Review of #49: no blocking issues (set emulation matched CPython on 1,200 random cases up to 90k entries); fixed SV alleles with `:` (the shell chain drops their genes)
- o Merged #48 into fork `main`
- × HPO_SIM in Rust (`aim hpo-sim`): OBO parse, descendant IC, Lin best-match average, dplyr groups, R `merge()` order, `write.table` numbers — OMIM/HGMD similarity tables byte-identical to `phenoSim.R` (pipeline run + 3 synthetic cases incl. a non-empty HGMD table)
  - ! R's `formatReal` scales in double on arm64 but 80-bit long double on x86-64: a last digit can differ on Linux (documented)
  - – genemap2 is an RDS: exported once to TSV (`export_genemap.R`); no R at run time with `--rust`
  - – 5.31 s / 575 MB (R) → 0.22 s / 141 MB
- o Merged #49 into fork `main`
- × Review of #50: no blocking issues (formatting matched R on 1.4M doubles, OBO tags on 61,743 lines, 51 clean fuzz cases byte-identical); fixed blank HGMD fields and `read.table` column wrapping
  - – unmatched quotes in the patient file: R drops/merges terms erratically — `aim` warns instead (documented)

## 2026-09-29 (Tue, later)

- o Merged #49, #50, #51 into fork `main`; status report printed
- × ANNOTATE_BY_MODULES (`feature.py`) in Rust (`aim features`) — the main goal of this phase
  - – `scores.csv` byte-identical on all 25 pipeline files (23 ClinVar-sample chromosomes, fixture, gnomAD-fix run)
  - – and against `feature.py` itself on derived inputs: 15,216 rows (8 chunks), `-enableLIT`, numeric chunks, hg19 DGV, non-empty HGMD (all curation levels)
  - ! pandas types columns per 2,048-row chunk: a VEP number's printed form depends on its neighbours — reproduced (`pdread`)
  - ! DECIPHER never matches; OMIM allele file unused; `clinVarSymMatchFlag` always 0 (documented, reproduced)
  - – 15,216 rows: 36 s / 1.2 GB (Python) → 1.4 s / 322 MB
- < Optimization review of all `aim` subcommands (running)

## 2026-09-29 (Tue, evening)

- o Merged #52 (`feature.py`), #53 (optimizations); status report updated
- × VEP reverse-engineered (consequence engine, cache, lookups, tab output) and measured
  - – `--regulatory` is 4.5 s / 870 MB of a 7 s / 1.1 GB task; plugins ~1.2 s
  - ! dropping `--regulatory` changes results (diffusion rank, ClinVar counts, top variant) — kept
  - ! VEP is non-deterministic for multi-allelic 1/2 samples (row order, some CDS/protein fields)
- × Stage 3: VEP lookups in Rust (`aim vep-annotate`; tabix via noodles)
  - – tabix reader = htslib on 8 files × 1,283 regions (6.9M records), incl. the broken gnomAD index
  - – byte-identical to full VEP: 23 ClinVar-sample chromosomes (15,216 rows); synthetic golden
    (every lookup rule); 156,053 multi-allelic rows (lookup columns all equal; VEP's own columns
    vary run to run)
  - – lookups 1.3 s → 0.28 s per chromosome task (chr19), 118 MB; parallel over variants
  - – 17,533 ClinVar variants (218,432 rows): 42 s / 255 MB after memoising per allele (was 381 s / 1.4 GB: a 24 kb deletion re-matched 72k CADD lines per transcript row)
  - – end-to-end `--rust` ClinVar sample: 108/108 tasks, 2 min 49 s; outputs as before (merged `CADD_phred` ≤ 1 ulp, row order)
