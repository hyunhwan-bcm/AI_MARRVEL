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
the `aim` binary (default off; the other steps are unchanged):

```bash
cargo build --release                    # rust/target/release/aim
pixi run -e py python3.8 rust/tools/export_refs.py <data> rust/refs
pixi run -e r Rscript rust/tools/export_genemap.R <data> rust/refs   # genemap2 RDS -> TSV
nextflow run main.nf ... --rust true --aim_bin $PWD/rust/target/release/aim \
    --rust_refs $PWD/rust/refs --rust_models $PWD/rust/models
```

On the 1,450-variant ClinVar sample every output matches the Python/R run (rows as sets:
the merged row order follows Nextflow's chromosome completion order in both versions).

| Crate      | Contents                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aim-core` | XGBoost `binary:logistic` evaluation (bit-identical to xgboost 2.1.4) and approximate SHAP (bit-identical to the osx-arm64 wheel; the x86-64 Linux wheel used in production differs by a few float32 ulps), percentile confidence, rankings |
| `aim-cli`  | `aim` binary: `phrank`, `hpo-sim`, `features`, `join-phrank`, `tier`, `merge`, `predict` (one subcommand per Nextflow process)                                                                                                              |
