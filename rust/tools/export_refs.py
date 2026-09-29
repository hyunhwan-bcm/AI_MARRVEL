#!/usr/bin/env python3.8
"""Export reference data the Rust port reads in its own formats, next to (not instead of) the originals.

    python3.8 rust/tools/export_refs.py <data_dir> <out_dir>

<out_dir>/mod5_diffusion/
  net.csr        sparse copy of net_norm_cor_GeneID.npz["net_norm"] (row-major CSR, little-endian):
                 b"AIMCSR1\\0", u64 n_rows, u64 n_cols, u64 nnz,
                 u64 indptr[n_rows + 1], u32 indices[nnz], f32 values[nnz]
  genes.txt      cor_GeneID_arr, one Ensembl gene id per line (row/column order of net.csr)
  manifest.json  shape, nnz, sha256 of the source
<out_dir>/annotate/feature_stats.csv                   copied unchanged
<out_dir>/merge_expand/<ref>/simpleRepeats.<ref>.bed   copied unchanged (hg19, hg38)
<out_dir>/merge_expand/<ref>/{clin,hgmd}_{c,nc}.tsv.gz  copied unchanged (hg19, hg38)
<out_dir>/var_tier/<ref>/genemap2.Inh.F.txt           copied unchanged (hg19, hg38)
"""
import shutil
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def export_diffusion(data, out):
    src = data / "mod5_diffusion" / "net_norm_cor_GeneID.npz"
    d = np.load(src, allow_pickle=True)
    net = d["net_norm"]
    assert net.dtype == np.float32 and net.ndim == 2
    genes = [str(g) for g in np.ravel(d["cor_GeneID_arr"])]
    assert len(genes) == net.shape[0] == net.shape[1]

    rows, cols = np.nonzero(net)  # row-major order
    values = net[rows, cols]
    indptr = np.zeros(net.shape[0] + 1, dtype=np.uint64)
    np.cumsum(np.bincount(rows, minlength=net.shape[0]), out=indptr[1:])

    dst = out / "mod5_diffusion"
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "net.csr", "wb") as fh:
        fh.write(b"AIMCSR1\0")
        fh.write(np.array([net.shape[0], net.shape[1], len(values)], dtype="<u8").tobytes())
        fh.write(indptr.astype("<u8").tobytes())
        fh.write(cols.astype("<u4").tobytes())
        fh.write(values.astype("<f4").tobytes())
    (dst / "genes.txt").write_text("\n".join(genes) + "\n")
    manifest = {
        "source": str(src.relative_to(data)),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "shape": list(net.shape),
        "nnz": int(len(values)),
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"diffusion: {net.shape} dense ({net.nbytes / 1e9:.2f} GB) -> {len(values)} non-zeros "
          f"({(dst / 'net.csr').stat().st_size / 1e6:.1f} MB)")


def copy_unchanged(data, out):
    files = ["annotate/feature_stats.csv"]
    files.append("omim_annotate/hp.obo")
    for r in ("hg19", "hg38"):
        files += [f"omim_annotate/{r}/HPO_OMIM.tsv", f"omim_annotate/{r}/HGMD_phen.tsv"]
        files.append(f"merge_expand/{r}/simpleRepeats.{r}.bed")
        files += [f"merge_expand/{r}/{t}.tsv.gz" for t in ("clin_c", "clin_nc", "hgmd_c", "hgmd_nc")]
        files.append(f"var_tier/{r}/genemap2.Inh.F.txt")
        files += [f"phrank/{r}/{t}.txt" for t in ("child_to_parent", "disease_to_pheno", "disease_to_gene",
                                                  "ensembl_to_symbol")]
        files.append(f"phrank/{r}/{'grch37' if r == 'hg19' else 'grch38'}_symbol_to_location.txt")
    for rel in files:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / rel, out / rel)
        print(f"copied {rel}")


# omim_annotate/<ref>/genemap2_pheno.tsv comes from the RDS: rust/tools/export_genemap.R


if __name__ == "__main__":
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    export_diffusion(data, out)
    copy_unchanged(data, out)
