# Nextflow fixture goldens

Outputs of AIM v1.1.3 (`baseline/v1.1.3` worktree, commit cd21a6f plus the native harness) run natively with
`native/run_fixture.sh` on `tests/fixtures/fastvep_test/test.aim.vcf` (GRCh38, run id `fixture`),
data `s3://aim-data-dependencies-2.4-public` as published (including its broken hg38 gnomAD
genome index, so gnomAD genome AF is empty for every variant), on 2026-09-28.

`prediction/` holds the PREDICTION process's inputs and outputs, copied unchanged:
- `fixture.matrix.txt`: MERGE_SCORES_BY_CHROMOSOME output, input of `run_final.py`
- `fixture.default_prediction.csv`: `run_final.py` output, input of `extraModel_main.py`
- `fixture.recessive_matrix.csv`: `conf_4Model/recessive_matrix/fixture.csv`
- `fixture_{default,nd,recessive,nd_recessive}_predictions.csv`: `conf_4Model/` outputs
- `fixture_*_shap_values.json`: `shap_outputs/`
