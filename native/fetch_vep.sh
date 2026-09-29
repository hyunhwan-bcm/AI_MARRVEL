#!/usr/bin/env bash
# Extract the pure-Perl VEP 104.3 tree (vep, modules/, Ensembl API + BioPerl under Bio/)
# from the production image ensemblorg/ensembl-vep:release_104.3
# (manifest sha256:62d189098d86cbee612ad761a2b7b88b664ebac36c5f77d10fc7c429e1d152ea),
# so the native baseline runs byte-identical VEP code. Compiled modules (Bio::DB::HTS,
# Set::IntervalTree, ...) come from the pixi "vep" environment instead.
set -euo pipefail

dest="${1:?usage: fetch_vep.sh <dest_dir>}"
repo="ensemblorg/ensembl-vep"
# Image layers touching opt/vep/src/ensembl-vep, in manifest order.
layers=(
  sha256:8a8d63aed153fc43ffd2ba1fdcc5ff4806b93e03a0698dff03746b078ac2f43d
  sha256:ee9870b4e46bc3da74015810da3d296d08ef9ad389ca875f6704a0e07f77c314
  sha256:d838e84a64e1af35f4a220f0a5712cb236cde0e501a61c3c56034ac152c42b39
)

if [[ -x "$dest/vep" ]]; then
  echo "VEP already present at $dest"; exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
token="$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["token"])')"

for digest in "${layers[@]}"; do
  echo "Fetching layer ${digest:7:12}"
  curl -fsSL -H "Authorization: Bearer $token" "https://registry-1.docker.io/v2/${repo}/blobs/${digest}" -o "$tmp/layer.tgz"
  echo "${digest#sha256:}  $tmp/layer.tgz" | shasum -a 256 -c - >/dev/null
  tar -xzf "$tmp/layer.tgz" -C "$tmp" 'opt/vep/src/ensembl-vep' 2>/dev/null || true
  # Apply OCI whiteouts: ".wh.NAME" deletes NAME from lower layers.
  find "$tmp/opt/vep/src/ensembl-vep" -name '.wh.*' 2>/dev/null | while read -r wh; do
    rm -rf "$(dirname "$wh")/$(basename "$wh" | sed 's/^\.wh\.//')" "$wh"
  done
  rm -f "$tmp/layer.tgz"
done

mkdir -p "$(dirname "$dest")"
mv "$tmp/opt/vep/src/ensembl-vep" "$dest"
"$dest/vep" --help >/dev/null 2>&1 || true
echo "VEP 104.3 extracted to $dest"
