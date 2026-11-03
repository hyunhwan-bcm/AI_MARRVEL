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

## Quick Start: version 1 (Rust)

Version 1 runs AIM's annotation, scoring and prediction in one Rust program, `aim`. It uses the
same models as v1.1.3 and gives the same predictions, apart from two deliberate changes
([below](#what-changed-from-v113)). It needs 117 GiB of data instead of 519 GiB, no Docker, and
about a third of the time: a whole genome (GIAB HG002, hg38) takes 6 minutes instead of 18.

It runs on Linux (x86-64) and macOS (Apple silicon). There are four steps.

### 1. Get the software

You need git, Java 17 or later, [Nextflow](https://www.nextflow.io/docs/latest/install.html),
[pixi](https://pixi.sh) and [Rust](https://rustup.rs) (to build `aim`; there are no prebuilt
binaries yet). Then:

```bash
git clone https://github.com/hyunhwan-bcm/AI_MARRVEL    # with git: a ZIP download does not run
cd AI_MARRVEL
pixi install --all                    # bcftools, htslib, GATK, Python and R, inside this folder
pixi run -e vep fetch-vep             # VEP 104.3, used only for the rare input aim does not handle
(cd rust && cargo build --release)    # builds rust/target/release/aim
```

### 2. Get the data

All the data is one folder, `aim-rust-data-v1`, in a public AWS bucket. No login is needed. Put
it on any local disk with enough space (here `/data`) and keep the folder as it is:

```bash
cd /data
aws s3 sync s3://aim-rust-data-store-v1/ aim-rust-data-v1/ --region us-east-2 --no-sign-request
```

| Download           | Size         | Add to the command                                |
| ------------------ | ------------ | ------------------------------------------------- |
| Both genome builds | 116.8 GiB    |                                                   |
| hg38 only          | about 62 GiB | `--exclude 'vep/hg19/*' --exclude 'store/hg19/*'` |
| hg19 only          | about 55 GiB | `--exclude 'vep/hg38/*' --exclude 'store/hg38/*'` |

The AWS CLI is [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
rclone works too: `rclone copy ":s3,provider=AWS,region=us-east-2:aim-rust-data-store-v1"
aim-rust-data-v1 -P`, with `**` in place of `*` in the excludes.

Check the download inside the folder. It reports any file that is missing or differs:

```bash
cd aim-rust-data-v1
md5sum -c --quiet MD5SUMS                  # Linux
rclone checksum md5 MD5SUMS . --one-way    # macOS
```

After a one-build download, check that build only, for example hg38:
`grep -v -e ' vep/hg19/' -e ' store/hg19/' MD5SUMS | md5sum -c --quiet`.

The folder's own `README.md` describes what it holds. **Terms:** CADD, SpliceAI, dbNSFP (academic
version) and REVEL are free for non-commercial use only; you are responsible for each source's terms.

### 3. Run

Run from the `AI_MARRVEL` folder, with absolute paths for the data and the inputs:

```bash
D=/data/aim-rust-data-v1
NXF_VER=24.10.5 nextflow -c native/native.config run main.nf \
  --ref_dir $D --ref_ver hg38 \
  --input_vcf /path/to/sample.vcf.gz --input_hpo /path/to/sample.hpo.txt \
  --outdir out --storedir /path/to/store --run_id sample \
  --rust true --rust_vep true --aim_bin $PWD/rust/target/release/aim \
  --vep_store $D/store/hg38 --rust_refs $D --rust_models $D/models
```

For hg19, use `--ref_ver hg19 --vep_store $D/store/hg19`.

| Option                                     | What it is                                                                 |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `-c native/native.config`                  | Runs each step in the pixi environments, without containers                |
| `--ref_dir $D`, `--rust_refs $D`           | The data folder (both options)                                             |
| `--vep_store $D/store/<build>`             | The build's lookup tables                                                  |
| `--rust_models $D/models`                  | The models                                                                 |
| `--ref_ver`                                | `hg38` or `hg19`: the build of your VCF                                    |
| `--input_vcf`                              | One person's VCF (`.vcf` or `.vcf.gz`), sorted, with a `GT` column         |
| `--input_hpo`                              | The patient's phenotypes, one HPO term per line (for example `HP:0003002`) |
| `--storedir`                               | A cache kept between runs; use the same one every time                     |
| `--outdir`, `--run_id`                     | Results go to `<outdir>/<run_id>/`                                         |
| `--rust true --rust_vep true`, `--aim_bin` | Run the Rust steps, with this `aim`                                        |

The first run with a new `--storedir` downloads the UCSC reference genome of the build (about
3 GB once indexed), so it needs internet access. Later runs reuse it.

To try your setup, run the ClinVar sample in the repository (1,450 variants, hg38; a few minutes):
use `--input_vcf $PWD/tests/fixtures/clinvar_sample/clinvar_sample.vcf` and
`--input_hpo $PWD/tests/fixtures/clinvar_sample/clinvar_sample.hpo.txt`.

### 4. Read the results

The main result is `<outdir>/<run_id>/prediction/conf_4Model/<run_id>_default_predictions.csv`:
one row per variant, with `predict` (the score), `ranking`, `confidence` and `confidence level`.
The same folder has the novel-disease-gene (`_nd_`) and recessive results. The columns are
described in [docs/source/output.rst](docs/source/output.rst).

Nextflow keeps its intermediate files in `work/`; you can delete it after a successful run.

### What changed from v1.1.3

- **hg38 uses the gnomAD genome frequencies.** In v1.1.3 as deployed, the hg38 gnomAD genome index
  did not match its file, so hg38 runs got no gnomAD genome frequency. hg38 results now differ for
  variants found in gnomAD genomes; hg19 is unchanged.
- **Structural variants are removed from the input** (symbolic ALTs such as `<DEL>`). v1.1.3
  scored them without any lookup values.

Otherwise the feature matrix and all four models' predictions match v1.1.3 on the ClinVar sample
and on a whole genome, hg38 and hg19, apart from float noise. Details: [rust/README.md](rust/README.md)
and [rust/DESIGN.md](rust/DESIGN.md).

## Original pipeline (v1.1.3)

The original Python, R and VEP pipeline runs when you leave out the `--rust` options. It needs the
full data bucket (519 GiB) and cannot use `aim-rust-data-v1`.

### Install Required Data Dependencies

AIM utilizes various databases for variant annotation, all of which have been compiled and are available for download. We use AWS S3 for data access, and the data can be downloaded by following these steps:

1. **Install the AWS CLI**: Follow the instructions provided in the [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
2. **Navigate to Your Desired Directory**: Change to the directory where you want your data dependencies downloaded. For example, in Ubuntu, use:
   ```bash
   $ cd <desired/folder/path>
   ```
3. Use the following command to sync the S3 bucket to your local directory:
   ```bash
   $ aws s3 sync s3://aim-data-dependencies-2.4-public . --no-sign-request
   ```

### Get the software

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A523.10.1-23aa62.svg)](https://www.nextflow.io/)

AIM is released as a Nextflow pipeline for easy distribution. To get it:

```bash
$ git clone https://github.com/LiuzLab/AI_MARRVEL
$ cd AI_MARRVEL
$ nextflow run main.nf --version
```

### Run with your sample

```bash
nextflow run /path/to/AI_MARRVEL/main.nf \\
          -profile <debug/docker/singularity/> \\
          --ref_dir /path/to/dependencies/ \\
          --input_vcf /path/to/input.vcf \\
          --input_hpo /path/to/input.hpos.txt \\
          --outdir /path/to/output \\
          --storedir /path/to/store \\
          --run_id <Sample_ID> \\
          --ref_ver <hg19/hg38>
```

For more information on usage and parameters which are open for modification, please use `--help` option as shown below.

```
$ nextflow run main.nf --help
```

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
