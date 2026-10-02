# AIM Rust port — bullet journal

Key: `•` task · `×` done · `>` migrated (moved later) · `<` scheduled · `–` note · `!` important · `o` event

Tracking issue: [#35](https://github.com/hyunhwan-bcm/AI_MARRVEL/issues/35)

## Future log

- • VCF preprocessing in Rust (bcftools → noodles)
- • VEP consequence engine in Rust (stages 4–7 of the VEP plan; decide after exome timing)
- • VEP: fastVEP behind a VEP-104 compatibility layer (option C, in progress), then a faithful port of VEP 104 (option B)
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
- × Review of #55 (each finding checked against native VEP): lowercase alleles now upper-cased as VEP does; `chr_synonyms.txt` used for custom files (chrM); structural variants and other unsupported input exit with status 3 and the pipeline falls back to VEP's own lookups; `--threads ${task.cpus}`; htslib 1.22 VCF record spans (SVLEN, gVCF LEN, END cannot shorten); per-plugin caches
  - ! a new per-line INFO search doubled SpliceAI's time (29 → 68 s on 17.5k variants); fixed by scanning key starts only

## 2026-09-30 (Wed)

- × Why VEP is slow, from its Perl: `--fork` children of `buffer_size/(2N)` variants each reload every 1 Mb cache chunk (regulatory chunk ~110 ms); 5,883 loads for 1,798 distinct chunks on 17,533 variants; interval tree ~2%
- × `vep_buffer_size` param (default 1000): identical output, 184 → 70 s on 17,533 variants (6 forks), 234 → 128 s (2 forks)
  - ! a fixed 1000 slowed small tasks (one fork idle: ClinVar sample VEP 79 → 126 s) → batch = task's variant count, 50..1000
  - ! review of #56: VEP's HGNC_ID can depend on batch/fork grouping (transcripts spanning two cache chunks: DGCR5, TMSB15B, HERC2P7); unused by AIM; sparse tasks use up to ~7 GB; the seed fixes order per Perl build
- × VEP seeded (`PERL_HASH_SEED=0`, #54): reproducible output
- × fastVEP 0.4.0 on the same input with Ensembl 104 GFF3: 3.6 s; same transcript rows; Consequence 99.906% / IMPACT 99.990% after mapping VEP 115 splice terms; no regulatory or motif rows
- ! the VEP cache's FASTA `.fai` does not match its `.fa.gz` (63 G bases listed; chr17 lookups return header text) — to check with the gnomAD investigation
- × VEP 115 attribution of fastVEP differences: 87 rows after term mapping = 64 fastVEP-only, 19 VEP 105+ rule changes (stop_retained, start_lost, start_retained, frameshift), 4 other
- o Decision: option C (fastVEP + VEP-104 compatibility layer) first, then B (faithful port of VEP 104)
- × C step 2: co-located known variants in Rust (`aim vep-annotate --known-variants <cache>`; CSI support in the tabix reader)
  - – byte-identical to seeded VEP: 218,432 ClinVar rows, 136,667 rows sampled from the cache, 156,053 multi-allelic rows, `chr` names, synthetic golden; 2 s for 17.5k variants
  - ! `MAX_AF_POPS` order is Perl hash order: fixed with `PERL_HASH_SEED=0` for a given Perl build (5.32 reproduced; 5.26 differs), random without
  - × Review of #58: `.csi` preferred over `.tbi` as htslib does; the cache's `chr_synonyms.txt` used when `--chr-synonyms` is not given (as VEP does); rows matching no VCF line exit 3 (pipeline falls back to VEP)
- × C step 3: regulatory / motif rows from the 104 cache (`aim vep-annotate --regulatory <cache>`; Storable read with the `chrysalis` crate)
  - – byte-identical to seeded VEP after removing and regenerating the rows: 218,432 ClinVar rows, 136,667 cache-sampled rows, 156,053 multi-allelic rows, 23 chromosome tasks, duplicate/unnamed lines, synthetic golden
  - ! 808 motifs are stored only in a neighbouring chunk (up to 14.9 kb away): VEP finds them only if the same fork child's slice loads that chunk (depends on buffer size and fork count); decided: always report an overlapping motif
  - × Review of #59: amplification terms for tandem duplications; lines with one ID at one position no longer merge (rows matched to the sample's genotype alleles); a parser panic no longer hangs (slot released, error instead); HIGH_INF_POS sums bits in Perl's seeded order
  - ! the variant grouping matched unnamed variants on `chr_pos_REF` only: rows of two lines at one position merged (harmless for lookups, wrong for inserted rows); now the whole allele string, and a repeated feature+allele starts a new variant
  - – AIM reads no `MOTIF_*` column; motif rows matter through IMPACT (`TFBS_ablation` = MODERATE: 2 of 17,533 ClinVar variants)
  - – 17.5k variants: 42 s / 2.3 GB with 4 parses at once, 77 s at 2 (chosen: a chr17 task 3.5 s / 1.3 GB)
- × C step 4a: fastVEP on VEP 104 cache transcripts (chrysalis) with VEP 104's predicates
  - – fastVEP alone (VEP 104 GFF3): 87 rows differ in Consequence of 211,014; on cache transcripts 81; the rest are VEP 105+ rules and fastVEP-only cases
  - ! VEP's coordinates follow Ensembl's mapper (insertions at exon/CDS edges map to one flank, so `coding` can be false); ported `TranscriptMapper`
  - ! VEP evaluates consequences before its HGVS 3' shift, which later rewrites the shared transcript-level coordinates (multi-allelic lines print shifted CDS positions)
  - – Consequence + IMPACT + 27 transcript columns identical on 545,158 rows and 23 chromosome tasks
  - ! Review of #60 (fuzzing ~660k rows): fastVEP's codons are not VEP's: no reference seq edits (HCK/WT1 start codons: start_lost vs synonymous), no codon across an intron (indels spanning one lose stop_lost/frameshift), no IUPAC complement; and VEP 104 calls splice_region per differing region and stretches exons near frameshift introns
  - × Ported `codon`/`peptide`/`_get_alternate_cds`/`display_codon` (BioPerl translation, seq edits, codon tables) and `_intron_effects`; fastVEP now gives only up/downstream and NMD terms
  - – all 29 columns identical on 1.28 M rows (review sets included) and 23 chromosome tasks, but the two known differences; ClinVar 17.5k in 5.3 s / 454 MB
- • C step 4b: HGVSc/HGVSp (VEP's genomic 3' shift, cache FASTA), SIFT/PolyPhen, DOMAINS, miRNA
  - × HGVSc + HGVSp + HGVS_OFFSET ported (shift, notation, protein type/peptides/format on VEP's codons): identical on 1.28 M rows and 23 chromosome tasks
  - ! VEP's cached CDS coordinates stay shifted after the allele's codon: the reference codon of an insertion at the start codon and `frameshift` read them (2 ClinVar variants, 13 rows)
  - ! Review of #61 (11.5 M fuzz rows identical): VEP's per-transcript cache carries one allele's HGVS shift to the next (multi-allelic insertions: HGVSp, and the CDS/Protein_position/Amino_acids quirk); Y PAR sequence comes from X; the default term is intergenic_variant; FASTA blocks decompressed per call
  - × Modelled the cache across alleles (`TvState`), PARs, default term, 64 kb FASTA window: 1.66 M multi-allelic rows identical, the quirk gone; ClinVar CPU 44 → 24 s
  - – miRNA: needs a `miRNA` attribute the cache lacks, so always empty (as VEP)
  - • SIFT/PolyPhen (cache ProteinFunctionPredictionMatrix), DOMAINS (protein_features): deferred, AIM reads dbNSFP's scores, not these columns
- × C step 4c: whole rows from the VCF (`aim vep`): skeleton rows (samples, transcripts within 5 kb, intergenic) streamed into the lookups; VEP 104.3 header
  - ! a sample's two alternate alleles come in Perl hash order (`keys %non_ref`): ported seeded Perl 5.32's SBOX32/STADTX hash and key order (3,000 random keys identical)
  - ! the plugins read the input's Consequence/Amino_acids/SYMBOL: now the recomputed ones
  - – whole files identical to seeded VEP on every set (ClinVar, cache-sampled, chr, multi, chr17 battery, g2s, MT, #60/#61 review sets, 23 exome chromosome tasks) but DGCR5 HGNC_ID; SIFT/PolyPhen/DOMAINS empty
  - – ClinVar 17.5k 95 s / 2.4 GB (regulatory chunk parsing dominates); exome chromosome task 1-6 s
  - ! Review of #62 (84,803-variant exome identical; 116 s vs VEP 164 s): lower-case alleles hashed before upper-casing; `validate_vf` skips (contigs not in the cache, no base in the allele string, a `-` reference off an insertion); a genotype index past the alleles; ID `0`; an ambiguous allele's codon made at output from the shifted cache
  - × Fixed; unsupported input exits 3 (VEP falls back): >5 alternates in one genotype (Perl's hash splits), duplicate sample names, malformed lines, no #CHROM line, a generator panic; `rust_vep` in nextflow_schema.json
- × C step 5: `--rust_vep true` (with `--rust`, hg38): ANNOTATE_BY_VEP runs `aim vep`; exit 3 falls back to VEP
  - – ClinVar sample end to end (108/108 tasks, 2 min 47 s): the 23 VEP tables identical to the VEP-based `--rust` run but SIFT/PolyPhen/DOMAINS; every other output identical as row sets but 1-ulp merge-order noise in one imputed column (two VEP-based runs differ the same way)
- × Native runs reserve 4 GB per task (`native/native.config`): `conf/base.config`'s 25 GB ran one task at a time on a 32 GB Mac; ClinVar sample `--rust --rust_vep` 2 min 47 s → 47 s, outputs unchanged

## 2026-10-01 (Thu)

- × GIAB HG002 whole genome, `--rust --rust_vep` vs seeded v1.1.3: 9 min 05 s vs 17 min 47 s (unseeded 22 min 22 s); VEP tables identical but 6 `HGNC_ID` cells; predictions of all 4 models identical
  - ! `diffuse_Phrank_STRING` differs on 25 of 132,185 variants (≤ 2.1e-4 relative): numpy's BLAS summation order, which no plain order reproduces (best 309/400 rows); near-equal heat swaps ranks
  - o Decision: keep the Rust diffusion, documented as precision noise (DESIGN.md); calling BLAS from Rust would cost the 1.4 GB dense matrix and a native library
- ! HG002 full diff of every output: the VEP tables also differ in SIFT/PolyPhen/DOMAINS (~4,900 cells, not ported, left out of the earlier comparison) besides the 6 `HGNC_ID` cells; final predictions identical
- × Lookup store (`aim store build`, `aim store check`, `--vep_store`): CADD, SpliceAI and dbNSFP as Parquet (zstd), one file per chromosome, the lines tabix returns rebuilt from the kept fields; 202 → 36.3 GiB for hg38 (CADD 80.6 → 23.6, SpliceAI 90.7 → 9.1, dbNSFP 30.5 → 3.5)
  - ! pandas types the VEP table in chunks of `2**20 // n_columns` rows: without dbNSFP's 338 other columns (459 → 120 columns) `scores.csv` could print differently; the table records VEP's width (`## AIM_VEP_COLUMNS=459`) and `aim features` uses it, refusing a table without a lookup column it reads
  - ! dbNSFP's 367 columns cost a query 22 ms as Parquet columns (one decode each); kept in one text column, and only the overlapping rows decoded: 8 µs (tabix 0.6 ms)
  - ! macOS keeps up to ~1.6 GB of freed large blocks in its allocator cache (`MallocLargeCache=0`: ~80 MB for a dbNSFP build)
  - – chr21: 20,000 random regions per source identical to tabix; ClinVar sample and HG002 with the store: `scores.txt` and all four models' predictions identical; HG002 8 min 14 s (9 min 05 s without)
  - – gnomAD (index broken on hg38, returns nothing), ClinVar, HGMD, REVEL stay tabix
  - × Review of #65 (0 differences on 2,700 synthetic edge regions and 40,500 real queries): a store without a field a lookup reads was accepted and its values went missing → refused per lookup; repeated/blank column names refused for left-out columns; `store check` samples chromosome ends
  - × No originals: with `--vep_store` PREPARE_DATA links the store under the original names; ClinVar sample on a data directory without them gives the same outputs; unsupported input (e.g. `<DEL>`) stops the task with a message (VEP's own lookups would need the originals)
  - – SVs in v1.1.3 (chr21 + one `<DEL>`): no lookup values (plugins skip structural variants), ranked last; new version with originals identical to v1.1.3 with and without it
  - o Decision (user): remove structural variants with `--vep_store` (FILTER_UNPASSED, `bin/drop_structural.awk`); store-only run with the `<DEL>` = v1.1.3 without it; other predictions unchanged, tied-last ranks and tier counts shift
- × hg19 through the Rust path (`--rust --rust_vep` on GRCh37) and its store
  - ! the GRCh37 cache has `ncRNA`/`miRNA` transcript attributes (GRCh38's none): ported the miRNA column and `mature_miRNA_variant` (which suppresses the non-coding terms); 2,400 miRNA-focused variants, 16,997 rows identical to VEP
  - – ClinVar sample on GRCh37 (same 1,450 variants) and GIAB HG002 GRCh37 vs the original pipeline: VEP tables identical but SIFT/PolyPhen/DOMAINS and 30 `HGNC_ID`; predictions identical (HG002: 15 diffusion cells); HG002 12 min 22 s vs 16 min 35 s
  - – hg19 store 35.7 GiB (CADD 23.0, SpliceAI 9.0, dbNSFP 4.3a 3.7) from 203.6 GiB; gnomAD r2.1 3.4 from 5.3
- ! hg38 gnomAD index proven to return no record for any query (4 of 36,283 chunks start at real blocks: header text or chr1 records under chr9/10/21 chunks); `aim store build --no-records` checks this
- o Decision (user): hg38 uses gnomAD genomes: store from the rebuilt index (8.9 GiB), `hom` from `nhomalt`; ClinVar sample: 600 variants gain real AF, 520 predictions change (4 by > 0.05), top 20 the same set
- × hg38 gnomAD in the original Python/VEP path (user): INDEX_GNOMAD_GENOMES (storeDir) replaces the bucket's mismatched .tbi, `nhomalt` for `hom`
  - ! first test run: the rebuilt index went into a non-existent 12th tuple slot (the indexes are the 11th); fixed
  - – with the hg38 originals re-downloaded for the test: the original pipeline, `--rust` (VEP rows + aim lookups) and `--rust_vep` with the store give identical predictions (ClinVar sample; HG002 but the 25 diffusion cells); original HG002 17 min 59 s
- × Review of #66: --no-records proof also covers htslib's linear-index starts (a forged index is refused) and refuses when another index exists; mature_miRNA_variant ends the term list (tier cutoff); miRNA re-checked on 44,354 rows against VEP
