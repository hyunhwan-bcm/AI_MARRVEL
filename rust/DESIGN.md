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

- ANNOTATE_BY_VEP runs VEP with `PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0` (#54), so its output is
  the same on every run; before, the order of a 1/2 sample's allele rows, of equal-rank consequence
  terms and a few HGVS/position cells followed Perl's random hash order. The seed fixes one of the
  orders VEP could produce, and only for a given Perl build (hash functions differ between Perl
  versions, so native Perl 5.32 and the container's Perl can still order differently); the models
  were trained on unseeded output, i.e. on a random mix.
- VEP's time is mostly reloading its cache: with `--fork N` each batch of `--buffer_size`
  variants is split into children of at most `buffer_size / (2N)` variants, and every child
  deserialises each 1 Mb cache chunk it touches (a regulatory chunk: ~7,000 Perl objects, ~110 ms)
  and then exits. ANNOTATE_BY_VEP now uses the task's variant count as the batch, at least 50 and
  at most `vep_buffer_size` (default 1000), so each child covers more variants per load. Not a
  fixed 1000: VEP gives each child at least ~50 variants, so on a 60-variant task a 1000 batch
  leaves one of two forks idle (ClinVar sample: VEP task time 79 → 126 s). Rows and every column
  AIM reads are the same for any size (checked seeded, forks 2 and 6, 17,533 and 8,233-line sets,
  and the pipeline end to end). One VEP quirk does depend on how variants are grouped into
  children, and so on the batch size and the fork count: when a transcript appears in two cache
  chunks, each child keeps the copy it loaded first and copies `_gene_hgnc_id` across transcripts
  with the same symbol (`AnnotationType/Transcript.pm:249-285`). Only `HGNC_ID` changes, only for
  a few genes (DGCR5, TMSB15B, HERC2P7 in the 104 cache), and AIM does not read it. Memory grows
  with how spread out a task's variants are: 923 SNVs across chr1 peaked at 6.9 GB over the task's
  processes at batch 923 (0.9 GB at 50).
- VEP 104.3 is not deterministic run to run without a fixed seed. Two runs of the same command on the same input
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
- VEP's co-located known variants (`Existing_variation`, `CLIN_SIG`, `SOMATIC`, `PHENO`,
  `PUBMED`, the 1000 Genomes / ESP / gnomAD exome frequencies, `MAX_AF`, `MAX_AF_POPS`) are
  reproduced by `aim vep-annotate --known-variants <cache>` from the cache's `all_vars.gz`
  (`vep_existing.rs`), for when something other than VEP produces the rows. It is not used by the
  pipeline yet. Matching uses the per-sample variant's whole allele set: when one rs ID sits on
  two cache lines, which line is kept depends on the sample's alleles (reproduced).
  `MAX_AF_POPS` lists tied populations in Perl's `keys %FREQUENCY_KEYS` order: with
  `PERL_HASH_SEED=0` it is fixed for a given Perl build (5.32, the native VEP environment: esp,
  exac, gnomad, af, 1kg, used here; bioconda's 5.26: af, exac, esp, 1kg, gnomad), and random
  without the seed. AIM does not read the column. A `CLIN_SIG` with
  several allele-specific values for one allele would be joined in hash order (written sorted);
  no such case occurs in the 104 cache sample checked. Checked byte-identical against seeded VEP
  on 218,432 ClinVar rows, 136,667 rows sampled from the cache (indels, multi-allelic, HGMD,
  COSMIC, failed entries), 156,053 multi-allelic rows, `chr`-named input and a synthetic golden.
- VEP's regulatory and motif rows are regenerated by `aim vep-annotate --regulatory <cache>` from
  the cache's `<chr>/<s>-<e>_reg.gz` chunks (`vep_regulatory.rs`; Perl Storable, read with the
  `chrysalis` crate), for when something other than VEP produces the rows. It is not used by the
  pipeline yet. Rows go after the variant's transcript rows and before an intergenic row:
  regulatory features by stable ID, then motif features by dbID compared as strings, each with
  the variant's alleles in VEP's order. A deletion covering a whole motif is `TFBS_ablation`,
  IMPACT MODERATE, the one way these rows reach AIM's features (the IMPACT maximum, the LIT
  filter, the tier's HIGH/MODERATE counts); AIM reads no `MOTIF_*` column. A tandem duplication
  covering a feature gets the `_amplification` term. **Deliberate difference:** 808 motifs of the
  104 cache are stored only in a neighbouring chunk, up to 14,913 bp outside it. VEP reports
  such a motif only when that chunk is loaded for the same fork child's slice of its batch, so
  its output depends on `--buffer_size` and `--fork`, and so on the machine (like `HGNC_ID`).
  The port always reports a motif that overlaps the variant. Parsing a large chunk takes
  ~0.5 GB for a moment (the crate builds the whole Perl tree); at most two parse at once, and a
  chunk is parsed again only after 64 others were used. Checked
  byte-identical against seeded VEP, regulatory rows removed and regenerated together with the
  known-variant columns: 218,432 ClinVar rows, 136,667 cache-sampled rows, 156,053 multi-allelic
  rows, 23 per-chromosome tasks, duplicated and unnamed VCF lines, and a synthetic golden.
- Transcript rows without VEP (`aim vep-annotate --transcripts <cache>`, step 4 of #57, for
  checking only so far): transcripts are read from the VEP 104 cache itself
  (`vep_transcripts.rs`), so the transcript set, symbols, cross-references, sequences, codon
  table and sequence edits are VEP's and no GFF3 or FASTA is needed. fastVEP 0.4.0 (a pinned
  git dependency, Apache-2.0) supplies the transcript model and only the upstream, downstream
  and NMD terms: it follows VEP 105+ (splice terms, stop/start rules) and builds codons
  differently (no reference sequence edits, no window across an intron, no IUPAC complement),
  and each of these changed Consequence or IMPACT somewhere (review of #60: e.g. a cached
  `initial_met` edit on HCK's start codon flips start_lost/synonymous). So the rest is ported
  from VEP 104: Ensembl's `TranscriptMapper` for coordinates (`vep_mapper.rs`; insertions at
  exon and CDS edges map to one flank), `codon` / `peptide` / `_get_alternate_cds` /
  `display_codon` with BioPerl's translation (`vep_codon.rs`; an allele whose length differs
  from its CDS span is spliced into a rebuilt CDS, so a reference spanning an intron
  translates intron bases, as VEP's does), `_intron_effects` per differing region with the
  12 bp exon stretch of transcripts with a frameshift intron, and the coding, UTR,
  non-coding, splice and intron predicates with their includes and tiers
  (`vep_consequence.rs`). Checked against seeded VEP on 1.28 M rows (ClinVar,
  cache-sampled, `chr` names, multi-allelic, a chr17 boundary battery, MT, seq-edit and
  frameshift-intron transcripts, MNVs, intron-spanning indels and IUPAC alleles) and 23
  chromosome tasks: all 29 columns identical but one known difference: HGNC_ID for a symbol
  whose ID-bearing transcripts are in chunks the variant does not load comes from the gene's
  chunks (VEP: only if its batch loads them; DGCR5 is the one primary-assembly case). (On
  multi-allelic insertion lines VEP prints some of these columns from another allele's HGVS
  shift; that is reproduced with HGVS, below.) A transcript row where no term applies gets
  VEP's default, `intergenic_variant`. Known limit: VEP visits overlapping introns
  in its interval tree's order, which decides splice_region only next to an exon of under
  ~20 bp; the port takes transcript order. VEP 104's `mature_miRNA_variant` needs a `miRNA`
  attribute, which the cache has none of, so the miRNA column stays empty.
- HGVSc, HGVSp and HGVS_OFFSET (`vep_hgvs.rs`, with the cache's FASTA, which VEP itself reads
  offline; noodles-fasta reads the bgzip with its `.fai`/`.gzi`): an insertion or deletion is
  first shifted 3' along the genome in the transcript's direction (`_genomic_shift`), then
  `hgvs_transcript` and `hgvs_protein` are ported with their helpers, the protein side on
  `vep_codon.rs` at the shifted position. VEP's transcript-variation cache (one per
  transcript, shared by a variant's alleles) matters here and is modelled (`TvState`): after
  the allele's shifted `codon` the translation and CDS coordinates and the reference peptide
  stay at its shift, so an insertion's reference codon and the `frameshift` test that picks
  `fs` read them (a deletion starting in an intron becomes `p.Ser2033Ter`), and on a
  multi-allelic insertion line the next allele in VEP's order prints its CDS_position,
  Protein_position, Amino_acids and an unshifted allele's HGVSp from them. On Y, the
  pseudoautosomal regions are read from X as VEP does (`FastaSequence.pm` `%PARS`). HGVS_OFFSET
  is printed only with an HGVSp; `=` is escaped as `%3D`. Checked byte-identical with all
  transcript columns blanked and recomputed: the same 1.28 M rows and 23 chromosome tasks, and
  the #61 review's sets (11.5 M edge, repeat, special-transcript and long-indel rows; 1.66 M
  multi-allelic rows; a Y PAR set). ClinVar 17.5k: 5.4 s / 531 MB (a 64 kb FASTA window per
  variant). Not yet: SIFT, PolyPhen, DOMAINS (VEP's columns; AIM reads dbNSFP's scores).

- VEP without VEP (`aim vep --vcf <vcf> --cache <dir_cache>/homo_sapiens/104_GRCh38`, step 4c
  of #57): `vep_skeleton.rs` writes the rows VEP would (per sample with a non-reference
  genotype, a row per transcript within 5 kb and allele, else an intergenic row per allele,
  with Uploaded_variation, Location, Allele, IND, ZYG and VARIANT_CLASS, and VEP 104.3's header)
  and streams them into `annotate` with every lookup on: transcript columns and HGVS, known
  variants, regulatory and motif rows, then `--custom` and `--plugin` as before (the plugins see
  the recomputed Consequence, Amino_acids and SYMBOL). A sample with two alternate alleles gets
  them in `keys %non_ref` order, so `perl_hash.rs` ports seeded Perl 5.32's hash (SBOX32 up to
  24 bytes, STADTX above; checked on 3,000 random keys) and its key order (buckets from the
  last, newest first), on the alleles as written (VEP upper-cases them later); an unseeded VEP,
  as in AIM's Docker image, orders them at random. VEP's `validate_vf` skips are reproduced
  (chromosomes the cache and its synonyms lack, allele strings without a base, a `-` reference
  off an insertion); input whose output depends on more of VEP's state exits 3 so the pipeline
  runs VEP: more than five alternates in one genotype (Perl's hash splits at the sixth key and
  VEP reuses it), duplicate sample names, malformed lines, a VCF without a #CHROM line.
  Checked whole files against seeded VEP 104 (every header line but the timestamp, the rows
  and their order, every cell but SIFT, PolyPhen and DOMAINS, which are not ported and which
  AIM does not read): ClinVar 17.5k, cache-sampled, `chr` names, multi-allelic, the chr17
  battery, intron-spanning indels, MT, the review sets of #60 and #61 (1.66 M multi-allelic
  rows, Y PAR, an 84,803-variant synthetic exome and the #62 review's edge cases) and 23
  per-chromosome tasks of a real exome, all identical but DGCR5's HGNC_ID.
  Time and memory are dominated by parsing regulatory chunks (ClinVar 17.5k: 95 s / 2.4 GB on
  8 threads, VEP ~180 s; a chromosome task 1-6 s / 0.5-1.8 GB). The pipeline uses it with
  `--rust true --rust_vep true` on hg38 (opt-in; exit status 3, e.g. structural variants,
  falls back to VEP); on the ClinVar sample every pipeline output matches the VEP-based run
  as row sets, but for 1-ulp noise in one imputed column that two VEP-based runs show too.

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
- Decision (2026-10-01): network diffusion (`diffuse_Phrank_STRING`) is accepted as precision
  noise rather than made bit-identical. `mod5_diffusion.py` multiplies a dense float32 matrix
  with numpy (`nn @ (0.5 * F) + 0.5 * y`, 100 times), whose BLAS sums in a blocked, CPU-specific
  order; the feature is `rankdata(heat, "max") / N`, so two genes whose heat is within float32
  rounding can swap ranks. On GIAB HG002 (whole genome, 132,185 variants after FILTER_PROBAND),
  against the seeded original: 25 variants differ, in 5 groups, by up to 2.1e-4 relative (e.g.
  0.369506 vs 0.369585); all 4 models' predicted probabilities are identical. The original gives
  the same result run to run on this Mac. No plain summation order reproduces its BLAS: of 400
  rows of the matrix product, sequential float32 sums match 115, f64 rounded once (the port)
  278, and the best of 4- to 64-lane pairwise sums with and without FMA 309. The alternative,
  calling the same BLAS from Rust, needs the 1.4 GB dense matrix in memory and a native library,
  and is exact only for one BLAS build, so the original itself likely differs between arm64
  macOS and x86-64 Linux (not checked). `diffusion.rs` sums each row in f64 and rounds once.

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
