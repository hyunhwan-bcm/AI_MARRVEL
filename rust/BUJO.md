# AIM Rust port — bullet journal

Key: `•` task · `×` done · `>` migrated (moved later) · `<` scheduled · `–` note · `!` important · `o` event

Tracking issue: [#35](https://github.com/hyunhwan-bcm/AI_MARRVEL/issues/35)

## Future log

- • VCF preprocessing in Rust (bcftools → noodles)
- • VEP: fastVEP behind a VEP-104 compatibility layer (optional)
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
  - – 36 of 104 features never split on; `nc_*` ClinVar features unused but row duplication matters
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
- > phrank chain + HPO similarity (R) — phrank done below; HPO similarity next
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
