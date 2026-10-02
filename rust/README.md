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

`--rust true` swaps FILTER_PROBAND (`bcftools isec` with gnomAD's blacklists), PHRANK_SCORING, HPO_SIM, ANNOTATE_BY_MODULES, JOIN_PHRANK, ANNOTATE_TIER, MERGE_SCORES_BY_CHROMOSOME and PREDICTION for
the `aim` binary, and moves ANNOTATE_BY_VEP's `--custom` and plugin lookups (gnomAD,
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

### Lookup store (smaller CADD, SpliceAI and dbNSFP; all lookup VCFs but HGMD's)

`aim store build` copies a tabix lookup file into a store directory: Parquet, zstd, one file
per chromosome, keeping only the fields AIM reads if asked. `aim vep` and `aim vep-annotate`
read a store directory wherever the file is given and return the same records (`aim store
check` compares the two); the original files are left as they are. For hg38, as AIM uses them
(36.3 GiB instead of 202 GiB; gnomAD below):

```bash
D=<data>/vep/hg38; S=<store>          # S: a new directory for the store
aim store build $D/hg38_whole_genome_SNV.tsv.gz --out $S/hg38_whole_genome_SNV.tsv.gz --drop-column RawScore
aim store build $D/spliceai_scores.masked.snv.hg38.vcf.gz --out $S/spliceai_scores.masked.snv.hg38.vcf.gz --drop-spliceai-positions
aim store build $D/spliceai_scores.masked.indel.hg38.vcf.gz --out $S/spliceai_scores.masked.indel.hg38.vcf.gz --drop-spliceai-positions
aim store build $D/dbNSFP4.1a_grch38.gz --out $S/dbNSFP4.1a_grch38.gz --keep-columns \
  'pos(1-based),alt,aaref,aaalt,GERP++_RS,GERP++_NR,LRT_Omega,LRT_score,phyloP100way_vertebrate,DANN_score,FATHMM_pred,FATHMM_score,GTEx_V8_gene,GTEx_V8_tissue,Polyphen2_HDIV_score,Polyphen2_HVAR_score,REVEL_score,SIFT_score,clinvar_clnsig,fathmm-MKL_coding_score,M-CAP_score,MutationAssessor_score,MutationTaster_score,ESP6500_AA_AC,ESP6500_AA_AF,ESP6500_EA_AC,ESP6500_EA_AF,CADD_phred'
# gnomAD genomes: hg38's bucket index does not match its file; build from a rebuilt index
# (tabix -p vcf on a copy, or aim-data/fixes/) so its records are used (rust/DESIGN.md)
aim store build $D/gnomad.genomes.GRCh38.v3.1.2.sites.vcf.gz --out $S/gnomad.genomes.GRCh38.v3.1.2.sites.vcf.gz \
  --drop-column QUAL --drop-column FILTER
# ClinVar, and gnomAD's blacklists (FILTER_PROBAND reads only their positions and alleles)
aim store build $D/clinvar_20220730.vcf.gz --out $S/clinvar_20220730.vcf.gz --drop-column QUAL --drop-column FILTER
for l in genomes exomes; do f=gnomad.hg38.blacklist.$l.vcf.gz
  aim store build <data>/filter_vep/hg38/$f --out $S/$f --drop-column ID --drop-column QUAL --drop-column FILTER --drop-column INFO
done
nextflow run main.nf ... --rust true --rust_vep true --vep_store $S   # an absolute path
```

hg19 the same way (35.7 GiB instead of 203.6 GiB, plus gnomAD 3.4 GiB instead of 5.3 GiB),
with dbNSFP 4.3a matched on its `hg19_pos(1-based)` column:

```bash
D=<data>/vep/hg19; S=<store-hg19>
aim store build $D/hg19_whole_genome_SNVs.tsv.gz --out $S/hg19_whole_genome_SNVs.tsv.gz --drop-column RawScore
aim store build $D/spliceai_scores.masked.snv.hg19.vcf.gz --out $S/spliceai_scores.masked.snv.hg19.vcf.gz --drop-spliceai-positions
aim store build $D/spliceai_scores.masked.indel.hg19.vcf.gz --out $S/spliceai_scores.masked.indel.hg19.vcf.gz --drop-spliceai-positions
aim store build $D/dbNSFP4.3a_grch37.gz --out $S/dbNSFP4.3a_grch37.gz --keep-columns \
  'alt,aaref,aaalt,GERP++_RS,GERP++_NR,LRT_Omega,LRT_score,phyloP100way_vertebrate,DANN_score,FATHMM_pred,FATHMM_score,GTEx_V8_gene,GTEx_V8_tissue,Polyphen2_HDIV_score,Polyphen2_HVAR_score,REVEL_score,SIFT_score,clinvar_clnsig,fathmm-MKL_coding_score,M-CAP_score,MutationAssessor_score,MutationTaster_score,ESP6500_AA_AC,ESP6500_AA_AF,ESP6500_EA_AC,ESP6500_EA_AF,CADD_phred'
aim store build $D/gnomad.genomes.r2.1.sites.grch37_noVEP.vcf.gz --out $S/gnomad.genomes.r2.1.sites.grch37_noVEP.vcf.gz \
  --drop-column QUAL --drop-column FILTER
aim store build $D/clinvar_20220730.vcf.gz --out $S/clinvar_20220730.vcf.gz --drop-column QUAL --drop-column FILTER
for l in genomes exomes; do f=gnomad.hg19.blacklist.$l.vcf.gz
  aim store build <data>/filter_vep/hg19/$f --out $S/$f --drop-column ID --drop-column QUAL --drop-column FILTER --drop-column INFO
done
nextflow run main.nf ... --ref_ver hg19 --rust true --rust_vep true --vep_store $S
```

The pipeline takes a store's copy of any lookup file it has (gnomAD, ClinVar, REVEL, CADD,
dbNSFP, SpliceAI, and the blacklists) and the original otherwise; CADD, SpliceAI and dbNSFP
must be in the store. The ClinVar and blacklist stores are about two thirds of the files' size
(ClinVar 41 MB instead of 59, the genome blacklist 80/71 MB instead of 130/121 on hg38/hg19,
the exome blacklist 1.3/1.1 MB instead of 1.8). `aim blacklist` refuses a blacklist record
with a symbolic ALT (gnomAD's have none), whose pairing would need its INFO.

With `--vep_store` the pipeline does not read the original files the store has copies of,
so they can be removed from the data directory. Structural variants (symbolic ALT alleles such
as `<DEL>`) are then removed from the input, since their lookups would need VEP and the original
files; the rest of the output is as if the input had none (rust/DESIGN.md).

The VEP table then lacks the columns left out (CADD_RAW, SpliceAI's delta positions in
SpliceAI_pred, dbNSFP's other columns) and has a `## AIM_VEP_COLUMNS=` line with the count
VEP writes, which `aim features` needs to type the table as pandas does (rust/DESIGN.md). The
features and predictions are the same as without the store.

| Crate      | Contents                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aim-core` | XGBoost `binary:logistic` evaluation (bit-identical to xgboost 2.1.4) and approximate SHAP (bit-identical to the osx-arm64 wheel; the x86-64 Linux wheel used in production differs by a few float32 ulps), percentile confidence, rankings |
| `aim-cli`  | `aim` binary: `phrank`, `hpo-sim`, `vep-annotate`, `features`, `join-phrank`, `tier`, `merge`, `predict` (one subcommand per Nextflow process)                                                                                              |
