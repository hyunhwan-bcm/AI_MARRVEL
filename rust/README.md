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

| Crate | Contents |
|---|---|
| `aim-core` | XGBoost `binary:logistic` evaluation and approximate SHAP (bit-identical to xgboost 2.1.4), percentile confidence, rankings |
