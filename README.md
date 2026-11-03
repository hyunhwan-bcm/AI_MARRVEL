<p align="center">
  <img width="50%" align="center" src="https://raw.githubusercontent.com/LiuzLab/AI_MARRVEL/main/docs/images/logo.v1.png" alt="logo">
</p>
  <h1 align="center">
  AI-MARRVEL
</h1>

<p align="center">
  <a style="text-decoration:none">
    <img src="https://img.shields.io/badge/AI_MARRVEL-v1.1.3-blue.svg"/>
  </a>
  <a href='https://ai-marrvel.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/ai-marrvel/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

**AI-MARRVEL (AIM)** is an AI system for rare genetic disease diagnosis.

It takes as input patient VCF and phenotype (formatted with HPO) to predict the causal variant(s).
In making prediction, it takes variant annotation from [MARRVEL](https://marrvel.org/) database and more,
and generates **prediction score** + **confidence score** as output.

You can use AI-MARRVEL from our [website](https://ai.marrvel.org/) or follow the [documentation](https://aimarrvel.readthedocs.io/en/main/) to run locally.

:new: Our paper is now published in [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300009)!

## Quick Start

This guide runs **version 1**, the Rust pipeline. It uses the same models as v1.1.3 and gives the
same predictions, apart from two deliberate changes (see
[What changed from v1.1.3](#what-changed-from-v113)). The annotation, scoring and prediction steps
run in one Rust program, `aim`. It needs 117 GiB of data instead of 519 GiB and runs without
Docker. The v1.1.3 instructions are under [Original pipeline (v1.1.3)](#original-pipeline-v113).

It needs Linux (x86-64) or macOS (Apple silicon), the two platforms its pixi environments support.
There are three steps: get the data, get the software, run.

### 1. Get the data

All the data is in one folder, `aim-rust-data-v1/`:

| Download           | Size         |
| ------------------ | ------------ |
| Both genome builds | 116.8 GiB    |
| hg38 only          | about 62 GiB |
| hg19 only          | about 55 GiB |

Put it on any local disk with enough space, for example at `/data/aim-rust-data-v1`. Keep the
folder as it is: the pipeline finds each file by its path inside the folder. You give the pipeline
the folder's absolute path.

There is no login. Download it with the AWS CLI or with rclone, in the folder that should hold it:

```bash
aws s3 sync s3://public/aim-rust-data-v1/ aim-rust-data-v1/ --endpoint-url https://s3.bioinfolder.com --no-sign-request
```

```bash
rclone copy ":s3,provider=Other,endpoint='https://s3.bioinfolder.com':public/aim-rust-data-v1" aim-rust-data-v1 -P
```

To download one build only, add these options to the command:

|           | AWS CLI                                           | rclone                                              |
| --------- | ------------------------------------------------- | --------------------------------------------------- |
| hg38 only | `--exclude 'vep/hg19/*' --exclude 'store/hg19/*'` | `--exclude 'vep/hg19/**' --exclude 'store/hg19/**'` |
| hg19 only | `--exclude 'vep/hg38/*' --exclude 'store/hg38/*'` | `--exclude 'vep/hg38/**' --exclude 'store/hg38/**'` |

rclone needs `**`, because its `*` does not match files in subfolders.

Then check the download. Run this inside the folder; it stops with an error if a file is missing
or differs:

```bash
cd aim-rust-data-v1
md5sum -c --quiet MD5SUMS                  # Linux
rclone checksum md5 MD5SUMS . --one-way    # macOS, or anywhere with rclone
```

If you downloaded one build, leave the other build out of the check (hg38 shown; for hg19, swap
the build names):

```bash
grep -v -e ' vep/hg19/' -e ' store/hg19/' MD5SUMS | md5sum -c --quiet                    # Linux
rclone checksum md5 MD5SUMS . --one-way --exclude 'vep/hg19/**' --exclude 'store/hg19/**'  # rclone
```

What the folder holds (it has its own `README.md` too):

| Path                                                                                                      | What                                                                                               | Size            |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------- |
| `store/hg38/`, `store/hg19/`                                                                              | Lookup stores (Parquet) of CADD, SpliceAI, dbNSFP, gnomAD genomes, ClinVar and gnomAD's blacklists | 45.3 / 39.2 GiB |
| `vep/hg38/`, `vep/hg19/`                                                                                  | Ensembl VEP 104 cache, REVEL, HGMD placeholder, VEP plugin config                                  | 16.4 / 15.2 GiB |
| `annotate/`, `merge_expand/`, `omim_annotate/`, `phrank/`, `var_tier/`, `bcf_annotate/`, `filter_exonic/` | Gene and phenotype tables, unchanged from v1.1.3, plus `omim_annotate/*/genemap2_pheno.tsv`        | 0.6 GiB         |
| `mod5_diffusion/`                                                                                         | Gene network for the diffusion score                                                               | 16 MB           |
| `models/`                                                                                                 | The four XGBoost models                                                                            | 32 MB           |
| `model_inputs/`, `predict_new/`                                                                           | Model inputs, unchanged from v1.1.3                                                                | 0.1 GiB         |

**Terms of use.** The data come from public resources, each with its own terms. CADD, SpliceAI's
precomputed scores, dbNSFP (academic version) and REVEL are free for **non-commercial use only**.
The other sources are gnomAD, ClinVar, Ensembl VEP, HPO and OMIM. HGMD is licensed, so the folder
holds a one-record placeholder in its place, as the v1.1.3 data did. You are responsible for
complying with each source's terms.

### 2. Get the software

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A523.10.1-23aa62.svg)](https://www.nextflow.io/)

You need:

- git.
- Java 17 and [Nextflow](https://www.nextflow.io/docs/latest/install.html). The run command below
  sets `NXF_VER=24.10.5`, the version it was tested with; Nextflow downloads that version itself.
- [pixi](https://pixi.sh). It installs the command-line tools the pipeline calls (bcftools,
  htslib, vcftools, GATK, GNU coreutils and others) inside the repository folder.
- [Rust](https://rustup.rs) (stable) and a C compiler, to build `aim`. There are no prebuilt
  binaries yet.

```bash
git clone https://github.com/hyunhwan-bcm/AI_MARRVEL
cd AI_MARRVEL
pixi install --all                    # the py, r, vep and tools environments
pixi run -e vep fetch-vep             # VEP 104.3, used only as a fallback (see below)
(cd rust && cargo build --release)    # builds rust/target/release/aim
```

The environments and the build take about 7 GB in the repository folder; pixi also keeps a package
cache in your home folder. Notes:

- Clone with git; a ZIP download does not work. The pipeline records the commit it runs with
  `git rev-parse HEAD`.
- `native/native.config` runs each step in a pixi environment: `py` for most steps, `r` for
  HPO_SIM and ANNOTATE_TIER, and `vep` for ANNOTATE_BY_VEP. With the Rust options, `aim` does the
  R and VEP work, but the config still activates those environments, so install all of them. You
  do not need `pixi run -e r install-ontologysimilarity`; only the original R step uses it.
- `fetch-vep` copies VEP's code from its Docker Hub image; Docker itself is not needed. `aim vep`
  passes the rare input it does not handle to VEP, for example a genotype with more than five
  alternate alleles. Without VEP, such a run stops with an error.

### 3. Run

**Inputs.** You need two files:

- A VCF for one person (the proband). Its name ends in `.vcf` or `.vcf.gz` (plain, gzip or bgzip),
  with no spaces in the path. It must be sorted by position and have a genotype (`GT`) column.
  Records with FILTER `PASS` are kept; if none pass, all are kept. Chromosomes can be named `1` or
  `chr1`; only 1-22, X and Y are analysed. A gVCF is genotyped with GATK first.
- An HPO file: the patient's phenotypes, one HPO term per line.

For example (columns separated by tabs; the full file is
[`tests/fixtures/clinvar_sample/clinvar_sample.vcf`](tests/fixtures/clinvar_sample/clinvar_sample.vcf)):

```text
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	PROBAND
1	926010	clinvar1632958	G	T	50	PASS	.	GT	0/1
1	1341820	clinvar1644819	G	A	50	PASS	.	GT	1/1
```

and `sample.hpo.txt`:

```text
HP:0003002
HP:0100615
```

**Command.** Run it from the `AI_MARRVEL` folder. Give absolute paths for the data folder, the
inputs and `aim`, because each step runs in its own work folder.

```bash
D=/abs/path/aim-rust-data-v1
NXF_VER=24.10.5 nextflow -c native/native.config run main.nf -profile debug \
  --ref_dir $D --ref_ver hg38 --input_vcf /abs/path/sample.vcf.gz --input_hpo /abs/path/sample.hpo.txt \
  --outdir out --storedir store --run_id sample \
  --rust true --rust_vep true --aim_bin $PWD/rust/target/release/aim \
  --vep_store $D/store/hg38 --rust_refs $D --rust_models $D/models
```

For hg19, use `--ref_ver hg19 --vep_store $D/store/hg19`.

To check your setup first, run it on the ClinVar sample in the repository (1,450 variants, hg38):

```bash
NXF_VER=24.10.5 nextflow -c native/native.config run main.nf -profile debug \
  --ref_dir $D --ref_ver hg38 \
  --input_vcf $PWD/tests/fixtures/clinvar_sample/clinvar_sample.vcf \
  --input_hpo $PWD/tests/fixtures/clinvar_sample/clinvar_sample.hpo.txt \
  --outdir out --storedir store --run_id clinvar_sample \
  --rust true --rust_vep true --aim_bin $PWD/rust/target/release/aim \
  --vep_store $D/store/hg38 --rust_refs $D --rust_models $D/models
```

| Option                         | What it does                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `-c native/native.config`      | Runs the steps in the pixi environments, without containers. Required.                                       |
| `-profile debug`               | Optional. Also writes Nextflow's `timeline-*.html`, `report-*.html` and `trace-*.txt` to the current folder. |
| `--ref_dir $D`                 | The data folder.                                                                                             |
| `--ref_ver`                    | `hg38` or `hg19`: the genome build of your VCF.                                                              |
| `--input_vcf`, `--input_hpo`   | Your input files.                                                                                            |
| `--outdir`, `--run_id`         | Results go to `<outdir>/<run_id>/`.                                                                          |
| `--storedir`                   | A cache that later runs reuse (the reference genome, below). Use the same one for every run.                 |
| `--rust true`                  | Runs the steps in Rust. Needs `--rust_refs` and `--rust_models`.                                             |
| `--rust_vep true`              | Annotates the variants with `aim vep` instead of VEP.                                                        |
| `--aim_bin`                    | The `aim` binary. Without it, `aim` must be on your `PATH`.                                                  |
| `--vep_store $D/store/<build>` | The lookup stores of the build. Must be an absolute path.                                                    |
| `--rust_refs $D`               | The data folder again: `aim` reads the gene network and gene tables from it.                                 |
| `--rust_models $D/models`      | The models.                                                                                                  |

`nextflow run main.nf --help` lists the other options.

The first run with a new `--storedir` downloads the UCSC reference genome of the build (about 1 GB;
it needs internet access) and keeps an indexed copy there. Later runs with the same `--storedir`
skip this.

Trio mode (`--input_ped`) still runs the original Python steps. It was not part of version 1's
checks.

**Output.** Results are in `<outdir>/<run_id>/`, for example `out/sample/`:

```text
out/sample/
├── prediction/
│   ├── conf_4Model/
│   │   ├── sample_default_predictions.csv        the main result: one row per variant
│   │   ├── sample_nd_predictions.csv             novel disease gene mode
│   │   ├── sample_recessive_predictions.csv      recessive mode: pairs of variants (only if there are pairs)
│   │   └── sample_nd_recessive_predictions.csv   novel disease gene + recessive (only if there are pairs)
│   ├── final_matrix_expanded/sample.expanded.csv.gz   default predictions with all annotations
│   └── shap_outputs/                             SHAP values of each model (JSON)
├── merged/      sample.matrix.txt (the feature matrix) and scores.txt.gz
├── vep/         the variant annotation of each chromosome (VEP's tab format)
├── phrank/      sample.phrank.txt (phenotype match score of each gene)
├── vcf/         the input after the filters (sample.filt.rmMT.vcf.gz) and after removing
│                gnomAD blacklist sites (sample.filt.rmBL.vcf.gz)
└── manifest/    pipeline version, git commit, and a list of the files in the data folder
```

Each prediction file has the variant's features and AIM's results: `predict` (the score),
`ranking`, `confidence` and `confidence level`. The columns are described in
[docs/source/output.rst](docs/source/output.rst).

Nextflow also writes its intermediate files to `work/` and its log to `.nextflow.log` in the
current folder. You can delete `work/` after a successful run; keep the `--storedir` folder.

### Expected time and resources

- **Time.** A whole genome (GIAB HG002, hg38) took 6 min 18 s on our test machine; v1.1.3 took
  17 min 59 s. The ClinVar sample takes about a minute.
- **Memory.** Each task reserves 4 GB and 2 CPUs (v1.1.3: 25 GB). Nextflow runs as many tasks at
  once as your CPUs and memory allow. Measured peaks are under 2 GB per task on the test inputs.
  For an unusually large input, reserve more with a second config file after
  `-c native/native.config`, for example `-c big.config` with the line `process.memory = 8.GB`.
- **Disk.** The data takes 117 GiB for both builds, or about 62 GiB (hg38) or 55 GiB (hg19) for
  one; v1.1.3 needs 519 GiB. Add about 7 GB for the software, about 3 GB for the reference genome
  in `--storedir`, and room for `work/` during a run.

### What changed from v1.1.3

Apart from the two deliberate changes below, version 1 gives the same answers as v1.1.3. The VEP
annotation columns AIM reads, the feature matrix, and all four models' predictions and SHAP values
are identical, on the ClinVar sample and on a whole genome (GIAB HG002), hg38 and hg19. There are
two exceptions: float noise in the last digit, and the diffusion score (`diffuse_Phrank_STRING`) of
25 HG002 variants on hg38 and 15 on hg19, which differ by at most 2.1e-4 (relative).

In the `vep/` tables, VEP's SIFT, PolyPhen and DOMAINS columns are empty. The tables also lack
what the stores leave out: CADD_RAW, most dbNSFP columns, and SpliceAI's delta positions in
`SpliceAI_pred`. AIM reads none of these.

It is also faster and needs less memory and disk; see
[Expected time and resources](#expected-time-and-resources).

Two changes are deliberate:

- **hg38 uses the gnomAD genome frequencies.** In v1.1.3 as deployed, the index of the hg38
  gnomAD genome file did not match the file, so hg38 runs got no gnomAD genome frequency (and the
  homozygote count `hom` was always 0). hg38 results now differ from v1.1.3 for variants found in
  gnomAD genomes. hg19 is unchanged.
- **Structural variants are removed from the input** when the stores are used (`--vep_store`):
  records with a symbolic ALT such as `<DEL>`, or a breakend. v1.1.3 scored them, but without any
  of the lookup values (gnomAD, ClinVar, CADD and the others).

The details are in [rust/DESIGN.md](rust/DESIGN.md).

## Original pipeline (v1.1.3)

The original Python, R and VEP pipeline still runs when you leave out the `--rust` options. It
needs the full data bucket (519 GiB); it cannot use `aim-rust-data-v1`.

1. **Data.** Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
   Then, in the folder you want the data in:

   ```bash
   aws s3 sync s3://aim-data-dependencies-2.4-public . --no-sign-request
   ```

   HGMD is not included. To add a licensed copy, see
   [docs/source/installation.rst](docs/source/installation.rst).

2. **Software.** Nextflow (23.10.1 or later) and Docker or Singularity, and a clone of this
   repository (`git clone https://github.com/hyunhwan-bcm/AI_MARRVEL`).

3. **Run.**

   ```bash
   nextflow run AI_MARRVEL/main.nf -profile docker \
     --ref_dir /path/to/dependencies --ref_ver hg38 \
     --input_vcf /path/to/input.vcf.gz --input_hpo /path/to/input.hpo.txt \
     --outdir /path/to/output --storedir /path/to/store --run_id sample
   ```

   Use `-profile singularity` for Singularity. Each task reserves 25 GB of memory. To run without
   containers, use the pixi setup in [native/README.md](native/README.md). For all options, run
   `nextflow run AI_MARRVEL/main.nf --help`.

Run from this repository, the original pipeline also uses the hg38 gnomAD genome frequencies: it
indexes the gnomAD file once and keeps the index in `--storedir`.

## For developers

- [rust/README.md](rust/README.md): building and testing `aim`, the lookup stores, and how the data
  folder is made.
- [rust/DESIGN.md](rust/DESIGN.md): design decisions and every known difference from v1.1.3.
- [native/README.md](native/README.md): the pixi environments and the native v1.1.3 baseline.

## License

AI-MARRVEL is licensed under GPL-3.0. You are welcomed to use it for research purpose.  
For business purpose, please contact us for licensing.

## Disclaimer

- Some of the data and software included in the distribution may be subject to third-party constraints. Users of the data and software are solely responsible for establishing the nature of and complying with any such restrictions.
- AI-MARRVEL provides this data and software in good faith, but make no warranty, express or implied, nor assume any legal liability or responsibility for any purpose for which they are used.

## Citing AI-MARRVEL

```bibtex
@article{doi:10.1056/AIoa2300009,
author = {Dongxue Mao  and Chaozhong Liu  and Linhua Wang  and Rami AI-Ouran  and Cole Deisseroth  and Sasidhar Pasupuleti  and Seon Young Kim  and Lucian Li  and Jill A. Rosenfeld  and Linyan Meng  and Lindsay C. Burrage  and Michael F. Wangler  and Shinya Yamamoto  and Michael Santana  and Victor Perez  and Priyank Shukla  and Christine M. Eng  and Brendan Lee  and Bo Yuan  and Fan Xia  and Hugo J. Bellen  and Pengfei Liu  and Zhandong Liu },
title = {AI-MARRVEL — A Knowledge-Driven AI System for Diagnosing Mendelian Disorders},
journal = {NEJM AI},
volume = {1},
number = {5},
pages = {AIoa2300009},
year = {2024},
doi = {10.1056/AIoa2300009},

URL = {https://ai.nejm.org/doi/abs/10.1056/AIoa2300009},
eprint = {https://ai.nejm.org/doi/pdf/10.1056/AIoa2300009}
,
    abstract = { AI-MARRVEL is an AI system for genetic diagnosis that improves diagnostic accuracy, surpassing state-of-the-art benchmarked methods. }
}
```
