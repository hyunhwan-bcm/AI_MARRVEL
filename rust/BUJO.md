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
- • Tier (`VarTierDiseaseDBFalse.R`)
- • PREDICTION I/O: recessive pairs, expanded matrix, SHAP JSON writer
- • phrank chain + HPO similarity (R)
- • Feature annotation (`feature.py`)
- • `aim` CLI + Nextflow `-profile rust` end-to-end comparison
- • Final report artifact
