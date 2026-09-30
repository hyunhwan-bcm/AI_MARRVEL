//! Rust vs VEP 104.3: the `--custom` and plugin lookups of ANNOTATE_BY_VEP added to a VEP run
//! without them, compared with VEP run with them, byte for byte from `## Column descriptions:`
//! on (VEP's first header lines hold a timestamp and version lines in random order).
//!
//! `vep_annotate/` holds VEP's output on 15 chr17 variants (the fastVEP fixture positions plus
//! multi-allelic, per-sample and unnamed cases) and synthetic lookup files, not real data;
//! `rust/tools/make_goldens_vep_annotate.py` writes them and documents the VEP commands.

mod common;

use std::io::{BufReader, Read};
use std::path::Path;

use aim_core::vep_annotate::{annotate, Lookups};
use common::golden_dir;

fn read_gz(path: &Path) -> String {
    let mut s = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut s)
        .unwrap();
    s
}

fn from_descriptions(s: &str) -> &str {
    &s[s.find("## Column descriptions:")
        .expect("column descriptions")..]
}

#[test]
fn lookups_match_vep() {
    let dir = golden_dir().join("vep_annotate");
    let customs = [
        "cv.vcf.gz,cv,vcf,exact,0,CLNSIG,CLNREVSTAT",
        "gx.vcf.gz,gx,vcf,exact,0,AF,AC,X,FL,Z,E,MISSING",
    ]
    .map(String::from);
    let plugins = [
        "REVEL,revel.tsv.gz,ALL",
        "SpliceAI,snv=spliceai_snv.vcf.gz,indel=spliceai_indel.vcf.gz,cutoff=0.5",
        "CADD,cadd.tsv.gz,ALL",
        "dbNSFP,dbNSFP4.test.tsv.gz,ALL",
    ]
    .map(String::from);
    let lookups = Lookups::open(&customs, &plugins, &dir.join("data"), "GRCh38", None).unwrap();
    let base = read_gz(&dir.join("base.txt.gz"));
    let vcf = std::fs::File::open(dir.join("input.vcf")).unwrap();
    let mut out = Vec::new();
    annotate(base.as_bytes(), BufReader::new(vcf), &lookups, 2, &mut out).unwrap();
    let got = String::from_utf8(out).unwrap();
    let want = read_gz(&dir.join("expected.txt.gz"));
    let (got, want) = (from_descriptions(&got), from_descriptions(&want));
    if got != want {
        let line = got
            .lines()
            .zip(want.lines())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "differs at line {} after the descriptions:\nrust {:?}\nvep  {:?}",
            line + 1,
            got.lines().nth(line),
            want.lines().nth(line)
        );
    }
}
