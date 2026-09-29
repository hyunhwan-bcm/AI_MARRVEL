#!/usr/bin/env bash
# Run AIM v1.1.3 natively on the fastVEP test fixture (GRCh38, 8 variants in BRCA1/TP53).
# Outputs, Nextflow trace/timeline/report land in $OUT; intermediate files stay in $OUT/work.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
REF_DIR="${REF_DIR:?set REF_DIR to the AIM data dependencies directory}"
FIXTURE="${FIXTURE:-$here/tests/fixtures/fastvep_test}"
OUT="${OUT:-$PWD/aim-fixture-run}"
mkdir -p "$OUT"
cd "$OUT"
NXF_VER="${NXF_VER:-24.10.5}" nextflow -c "$here/native/native.config" run "$here/main.nf" \
  -profile debug -work-dir "$OUT/work" -with-trace "$OUT/trace.txt" \
  -with-report "$OUT/report.html" -with-timeline "$OUT/timeline.html" \
  --ref_dir "$REF_DIR" --ref_ver hg38 \
  --input_vcf "$FIXTURE/test.aim.vcf" --input_hpo "$FIXTURE/test.hpo.txt" \
  --outdir "$OUT/out" --storedir "${STORE_DIR:-$OUT/store}" --run_id fixture "$@"
