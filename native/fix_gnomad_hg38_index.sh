#!/usr/bin/env bash
# The published hg38 gnomAD genome sites file's .tbi does not match the .vcf.gz (every lookup
# fails with "Invalid BGZF header"), so VEP silently finds no gnomAD genome AF on hg38.
# This builds a correct index into FIX_DIR and an OVERLAY_DIR that mirrors DATA_DIR with
# symlinks, except that this one index points at the rebuilt copy. DATA_DIR is not modified.
#
#   pixi run -e py bash native/fix_gnomad_hg38_index.sh DATA_DIR FIX_DIR OVERLAY_DIR
#   REF_DIR=OVERLAY_DIR native/run_fixture.sh
set -euo pipefail

data="${1:?usage: fix_gnomad_hg38_index.sh DATA_DIR FIX_DIR OVERLAY_DIR}"
fix="${2:?}"
overlay="${3:?}"
rel=vep/hg38/gnomad.genomes.GRCh38.v3.1.2.sites.vcf.gz
name="$(basename "$rel")"

mkdir -p "$fix"
if [[ ! -s "$fix/$name.tbi" ]]; then
  echo "Indexing $data/$rel"
  bcftools index -t --threads 4 -o "$fix/$name.tbi.tmp" "$data/$rel"
  mv "$fix/$name.tbi.tmp" "$fix/$name.tbi"
fi

link_all_except() {  # link_all_except SRC_DIR DEST_DIR SKIP_NAME
  local src="$1" dest="$2" skip="$3" entry
  mkdir -p "$dest"
  for entry in "$src"/*; do
    [[ "$(basename "$entry")" == "$skip" ]] || ln -sfn "$entry" "$dest/"
  done
}
link_all_except "$data" "$overlay" vep
link_all_except "$data/vep" "$overlay/vep" hg38
link_all_except "$data/vep/hg38" "$overlay/vep/hg38" "$name.tbi"
ln -sfn "$fix/$name.tbi" "$overlay/$rel.tbi"

echo "Check (TP53 R175H):"
tabix "$overlay/$rel" chr17:7675088-7675088 | cut -f1-5,8
