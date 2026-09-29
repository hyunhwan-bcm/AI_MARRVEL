#!/usr/bin/env python3.8
"""Golden phrank rankings (bin/run_phrank.py with the phrank package) for richer phenotype sets
than the fixtures' two terms, so the Rust port's CPython set-order emulation is exercised.

    python3.8 rust/tools/make_goldens_phrank.py <repo_root> <data_dir> <genes.txt> <out_dir>

<genes.txt> is an ENSEMBL_TO_GENESYM output (<id>-gene.txt). run_phrank.py runs with
PYTHONHASHSEED=0, as in the native config. Each <out_dir>/<case>/ gets input.hpo.txt and
expected_phrank.txt; all cases share <out_dir>/genes.txt.
"""
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

SEED = 20260929
SIZES = {"hpo1": 1, "hpo5": 5, "hpo15": 15, "hpo40": 40}


def main(repo, data, genes, out):
    repo, data, genes, out = map(Path, (repo, data, genes, out))
    ph = data / "phrank" / "hg38"
    terms = sorted({l.split("\t")[0] for l in open(ph / "disease_to_pheno.txt")})
    rng = random.Random(SEED)
    cases = {k: rng.sample(terms, n) for k, n in SIZES.items()}
    # unknown term, duplicate and blank line: phrank still puts them in its sets
    cases["odd_lines"] = rng.sample(terms, 6) + ["HP:9999999", "", cases["hpo5"][0]]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(genes, out / "genes.txt")
    env = dict(os.environ, PYTHONHASHSEED="0")
    for name, hpos in cases.items():
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "input.hpo.txt").write_text("".join(h + "\n" for h in hpos))
        res = subprocess.run(
            [sys.executable, str(repo / "bin" / "run_phrank.py"), str(out / "genes.txt"),
             str(d / "input.hpo.txt"), str(ph / "child_to_parent.txt"),
             str(ph / "disease_to_pheno.txt"), str(ph / "gene_to_phenotype.txt"),
             str(ph / "disease_to_gene.txt")],
            env=env, check=True, capture_output=True, text=True)
        (d / "expected_phrank.txt").write_text(res.stdout)
        print(f"{name}: {len(hpos)} terms -> {res.stdout.count(chr(10))} genes")


if __name__ == "__main__":
    main(*sys.argv[1:5])
