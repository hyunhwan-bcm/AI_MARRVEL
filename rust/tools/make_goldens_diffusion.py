#!/usr/bin/env python3.8
"""Golden values for diffusion (bin/mod5_diffusion.py), computed with the pipeline's own code.

    python3.8 rust/tools/make_goldens_diffusion.py <repo_root> <data_dir> <merge_workdir> <out_dir>

<merge_workdir> is a MERGE_SCORES_BY_CHROMOSOME work directory (for its scores.txt.gz and
<id>.phrank.txt). Scenarios: "fixture" (that sample) and three synthetic samples. Each
<out_dir>/<scenario>/ gets:
  rows.tsv      varId, geneEnsId in merged-table row order
  phrank.tsv    the sample's phrank file (gene, score), as the pipeline reads it
  expected.tsv  varId, diffuse_Phrank_STRING (diffuseSample output: first row per varId)
  heat.tsv      final heat per network gene (float32, repr)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260929


def main(repo, data, merge_workdir, out):
    repo, data, merge_workdir, out = map(Path, (repo, data, merge_workdir, out))
    sys.path.insert(0, str(repo / "bin"))
    import mod5_diffusion

    net_file = data / "mod5_diffusion" / "net_norm_cor_GeneID.npz"
    d = np.load(net_file, allow_pickle=True)
    net_norm = d["net_norm"]
    net_genes = [str(g) for g in np.ravel(d["cor_GeneID_arr"])]

    scenarios = {}
    merged = pd.read_csv(merge_workdir / "scores.txt.gz", sep="\t")
    phrank_path = next(merge_workdir.glob("*.phrank.txt"))
    scenarios["fixture"] = (merged[["varId", "geneEnsId"]], phrank_path.read_text())

    rng = np.random.default_rng(SEED)
    for k in range(3):
        in_net = list(rng.choice(net_genes, size=400, replace=False))
        off_net = [f"ENSG9{k}{i:09d}" for i in range(40)]
        phrank_genes = in_net + off_net
        scores = rng.gamma(2.0, 3.0, size=len(phrank_genes))
        phrank_text = "".join(f"{g}\t{s!r}\n" for g, s in zip(phrank_genes, scores))
        other_net = list(rng.choice(net_genes, size=600, replace=False))
        nowhere = [f"ENSG8{k}{i:09d}" for i in range(30)]
        rows = []
        for v in range(1200):
            var = f"{rng.integers(1, 23)}_{rng.integers(1, 10**8)}_A_G"
            for _ in range(rng.integers(1, 5)):  # several transcript rows per variant
                u = rng.random()
                pool = phrank_genes if u < 0.4 else other_net if u < 0.8 else nowhere if u < 0.9 else ["-"]
                rows.append((var, pool[rng.integers(len(pool))]))
        scenarios[f"synthetic{k}"] = (pd.DataFrame(rows, columns=["varId", "geneEnsId"]), phrank_text)

    for name, (rows, phrank_text) in scenarios.items():
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            os.symlink(data / "mod5_diffusion", Path(tmp) / "mod5_diffusion")
            (Path(tmp) / "sample.phrank.txt").write_text(phrank_text)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = mod5_diffusion.diffuseSample("sample", rows.copy(), ".")
                # Heat exactly as diffuseSample builds its input (same steps as the function).
                phrank = pd.read_csv("sample.phrank.txt", sep="\t", header=None)
                phrank = phrank.rename({0: "Ensembl_Gene_ID", 1: "Score"}, axis="columns")
                phrank = phrank.sort_values("Score", ascending=False).drop_duplicates("Ensembl_Gene_ID").sort_index()
                simi = phrank.rename({"Score": "Similarity_Score"}, axis="columns")
                cor = pd.DataFrame(d["cor_GeneID_arr"], columns=["ID"])
                y = cor.merge(simi, left_on="ID", right_on="Ensembl_Gene_ID", how="left")
                y = y[["ID", "Similarity_Score"]].fillna(0).set_index("ID")
                heat = np.ravel(mod5_diffusion.diffusion(net_norm, y, 0.5, 100))
            finally:
                os.chdir(cwd)
        rows.to_csv(dst / "rows.tsv", sep="\t", index=False)
        (dst / "phrank.tsv").write_text(phrank_text)
        with open(dst / "expected.tsv", "w") as fh:
            fh.write("varId\tdiffuse_Phrank_STRING\n")
            for var, v in result["diffuse_Phrank_STRING"].items():
                fh.write(f"{var}\t{float(v)!r}\n")
        with open(dst / "heat.tsv", "w") as fh:
            fh.write("gene\theat\n")
            for g, h in zip(net_genes, heat):
                fh.write(f"{g}\t{float(np.float32(h))!r}\n")
        print(f"{name}: {len(rows)} rows, {len(result)} variants, heat dtype {heat.dtype}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
