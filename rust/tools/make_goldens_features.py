#!/usr/bin/env python3.8
"""Golden feature.py outputs for inputs the fixtures don't cover, made by running feature.py:

    python3.8 rust/tools/make_goldens_features.py <repo_root> <data_dir> <out_dir> <vep.txt>...

The VEP tables (e.g. a run's per-chromosome ANNOTATE_BY_MODULES inputs) are concatenated.
Cases, each <out_dir>/<case>/ with vep.txt.gz, args.txt and expected_scores.csv.gz:
  numeric_chunks  2,500 rows, rows with a CADD_PHRED value first, so the first 2,048-row chunk
                  of several columns is numeric (pandas prints its re-parsed numbers)
  lit             3,000 rows with -enableLIT
  hgmd            3,000 rows (half from the synthetic table's genes), some given HGMD accessions,
                  with a synthetic HGMD similarity table
  hg19            the numeric_chunks rows with -genomeRef hg19 (hg19 DGV)
The OMIM similarity table is <out_dir>/../nextflow_clinvar/hpo_sim/expected_omim_sim.tsv.gz.
"""
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main(repo, data, out, veps):
    repo, data, out = Path(repo).resolve(), Path(data).resolve(), Path(out).resolve()
    meta, header, rows = [], None, []
    for v in veps:
        for line in open(v):
            if line.startswith("##"):
                if not header:
                    meta.append(line)
            elif line.startswith("#Uploaded"):
                header = line
            else:
                rows.append(line)
    cols = header.rstrip("\n").split("\t")
    ci = {c: i for i, c in enumerate(cols)}
    has_cadd = lambda r: r.split("\t")[ci["CADD_PHRED"]] not in ("-", "")
    numeric = [r for r in rows if has_cadd(r)] + [r for r in rows if not has_cadd(r)]
    genes = {"BRCA1": "CM000050", "TP53": "CM001234", "MSH2": "CM099999", "SCN1A": "CM000007",
             "CFTR": "CM000050", "FBN1": "CM001234"}
    in_genes = lambda r: r.split("\t")[ci["SYMBOL"]] in genes
    hgmd_rows = []
    for k, r in enumerate([r for r in rows if in_genes(r)][:1500] + [r for r in rows if not in_genes(r)][:1500]):
        f = r.rstrip("\n").split("\t")
        if f[ci["SYMBOL"]] in genes and k % 3 == 0:
            f[ci["hgmd"]] = genes[f[ci["SYMBOL"]]]
        elif k % 97 == 0:
            f[ci["hgmd"]] = "CM424242"
        hgmd_rows.append("\t".join(f) + "\n")
    sims = out.parent / "nextflow_clinvar" / "hpo_sim"
    synthetic_hgmd = out.parent / "phenosim_cases" / "dup_and_other" / "expected_hgmd_sim.tsv"
    cases = {
        "numeric_chunks": (numeric[:2500], "hg38", [], sims / "expected_hgmd_sim.tsv"),
        "lit": (rows[:3000], "hg38", ["-enableLIT"], sims / "expected_hgmd_sim.tsv"),
        "hgmd": (hgmd_rows, "hg38", [], synthetic_hgmd),
        "hg19": (numeric[:2500], "hg19", [], sims / "expected_hgmd_sim.tsv"),
    }
    for name, (case_rows, ref, extra, hgmd) in cases.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "annotate").symlink_to((data / "annotate").resolve())
            with open(tmp / "in.vcf-vep.txt", "w") as o:
                o.writelines(meta + [header] + case_rows)
            with gzip.open(sims / "expected_omim_sim.tsv.gz", "rb") as i, open(tmp / "omim.tsv", "wb") as o:
                shutil.copyfileobj(i, o)
            args = extra + ["-genomeRef", ref]
            subprocess.run(
                [sys.executable, str(repo / "bin" / "feature.py"), "-diseaseInh", "AD", "-modules",
                 "curate,conserve", "-inFileType", "vepAnnotTab", "-patientFileType", "one",
                 "-patientHPOsimiOMIM", "omim.tsv", "-patientHPOsimiHGMD", str(hgmd.resolve()),
                 "-varFile", "in.vcf-vep.txt"] + args,
                cwd=tmp, check=True, capture_output=True,
                env=dict(os.environ, PYTHONPATH=str(repo / "bin"), PYTHONHASHSEED="0"))
            for src, dst in (("in.vcf-vep.txt", "vep.txt.gz"), ("scores.csv", "expected_scores.csv.gz")):
                with open(tmp / src, "rb") as i, gzip.GzipFile(d / dst, "wb", mtime=0) as o:
                    shutil.copyfileobj(i, o)
        if name == "hgmd":
            shutil.copyfile(hgmd, d / "hgmd_sim.tsv")
        (d / "args.txt").write_text(" ".join(args) + "\n")
        print(f"{name}: {len(case_rows)} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
