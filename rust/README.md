# AIM Rust port

Rust implementation of AI-MARRVEL's Python/R stages, aiming for identical results to v1.1.3
with less memory, fewer dependencies and more speed. Design and decisions: [DESIGN.md](DESIGN.md).

```bash
cd rust
cargo test --release                     # unit tests (what CI runs)
```

Golden tests compare against the pipeline and need the production models exported locally
(no data dependencies are committed). With the baseline's pixi environment (`native/README.md`):

```bash
pixi run -e py python3.8 rust/tools/export_models.py <data>/model_inputs rust/models
cd rust && cargo test --release -- --include-ignored
```

## Running the pipeline with the Rust steps

`--rust true` swaps PHRANK_SCORING, HPO_SIM, ANNOTATE_BY_MODULES, JOIN_PHRANK, ANNOTATE_TIER, MERGE_SCORES_BY_CHROMOSOME and PREDICTION for
the `aim` binary, and on hg38 moves ANNOTATE_BY_VEP's `--custom` and plugin lookups (gnomAD,
ClinVar, HGMD, REVEL, SpliceAI, CADD, dbNSFP) from VEP to `aim vep-annotate`; VEP still computes
the rows and consequences (default off; the other steps are unchanged). With `--rust_vep true`
as well, ANNOTATE_BY_VEP runs `aim vep` instead of VEP: rows, consequences, HGVS, known
variants and regulatory rows from the VEP cache alone (VEP's SIFT, PolyPhen and DOMAINS columns,
which AIM does not read, stay empty; input it does not support falls back to VEP):

```bash
cargo build --release                    # rust/target/release/aim
pixi run -e py python3.8 rust/tools/export_refs.py <data> rust/refs
pixi run -e r Rscript rust/tools/export_genemap.R <data> rust/refs   # genemap2 RDS -> TSV
nextflow run main.nf ... --rust true --aim_bin $PWD/rust/target/release/aim \
    --rust_refs $PWD/rust/refs --rust_models $PWD/rust/models
```

On the 1,450-variant ClinVar sample every output matches the Python/R run (rows as sets:
the merged row order follows Nextflow's chromosome completion order in both versions).

### Lookup store (smaller CADD, SpliceAI and dbNSFP)

`aim store build` copies a tabix lookup file into a store directory: Parquet, zstd, one file
per chromosome, keeping only the fields AIM reads if asked. `aim vep` and `aim vep-annotate`
read a store directory wherever the file is given and return the same records (`aim store
check` compares the two); the original files are left as they are. For hg38, as AIM uses them
(36.3 GiB instead of 202 GiB):

```bash
D=<data>/vep/hg38; S=<store>          # S: a new directory for the store
aim store build $D/hg38_whole_genome_SNV.tsv.gz --out $S/hg38_whole_genome_SNV.tsv.gz --drop-column RawScore
aim store build $D/spliceai_scores.masked.snv.hg38.vcf.gz --out $S/spliceai_scores.masked.snv.hg38.vcf.gz --drop-spliceai-positions
aim store build $D/spliceai_scores.masked.indel.hg38.vcf.gz --out $S/spliceai_scores.masked.indel.hg38.vcf.gz --drop-spliceai-positions
aim store build $D/dbNSFP4.1a_grch38.gz --out $S/dbNSFP4.1a_grch38.gz --keep-columns \
  'pos(1-based),alt,aaref,aaalt,GERP++_RS,GERP++_NR,LRT_Omega,LRT_score,phyloP100way_vertebrate,DANN_score,FATHMM_pred,FATHMM_score,GTEx_V8_gene,GTEx_V8_tissue,Polyphen2_HDIV_score,Polyphen2_HVAR_score,REVEL_score,SIFT_score,clinvar_clnsig,fathmm-MKL_coding_score,M-CAP_score,MutationAssessor_score,MutationTaster_score,ESP6500_AA_AC,ESP6500_AA_AF,ESP6500_EA_AC,ESP6500_EA_AF,CADD_phred'
nextflow run main.nf ... --rust true --rust_vep true --vep_store $S   # an absolute path
```

The VEP table then lacks the columns left out (CADD_RAW, SpliceAI's delta positions in
SpliceAI_pred, dbNSFP's other columns) and has a `## AIM_VEP_COLUMNS=` line with the count
VEP writes, which `aim features` needs to type the table as pandas does (rust/DESIGN.md). The
features and predictions are the same as without the store.

| Crate      | Contents                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aim-core` | XGBoost `binary:logistic` evaluation (bit-identical to xgboost 2.1.4) and approximate SHAP (bit-identical to the osx-arm64 wheel; the x86-64 Linux wheel used in production differs by a few float32 ulps), percentile confidence, rankings |
| `aim-cli`  | `aim` binary: `phrank`, `hpo-sim`, `vep-annotate`, `features`, `join-phrank`, `tier`, `merge`, `predict` (one subcommand per Nextflow process)                                                                                              |
