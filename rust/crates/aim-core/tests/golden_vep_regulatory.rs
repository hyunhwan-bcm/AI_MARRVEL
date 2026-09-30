//! Rust vs VEP 104.3: the regulatory and motif rows (RegulatoryFeature / MotifFeature, with
//! their consequences and MOTIF_* columns) regenerated from the offline cache's `_reg.gz` chunks.
//!
//! `vep_regulatory/` holds VEP's output on 17 lines and a synthetic cache (no real data);
//! `rust/tools/make_goldens_vep_regulatory.pl` writes them and documents the VEP command. The
//! test removes those rows from VEP's output and compares the regenerated file with it, byte
//! for byte.

mod common;

use std::io::{BufReader, Read};

use aim_core::vep_annotate::{annotate, Lookups};
use common::golden_dir;

#[test]
fn regulatory_rows_match_vep() {
    let dir = golden_dir().join("vep_regulatory");
    let mut want = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(dir.join("expected.txt.gz")).unwrap())
        .read_to_string(&mut want)
        .unwrap();
    let mut ft = None;
    let mut base = String::new();
    let mut removed = 0;
    for l in want.lines() {
        if let Some(h) = l.strip_prefix('#').filter(|_| !l.starts_with("##")) {
            ft = h.split('\t').position(|c| c == "Feature_type");
        } else if !l.starts_with('#') {
            let t = l.split('\t').nth(ft.unwrap()).unwrap();
            if t == "RegulatoryFeature" || t == "MotifFeature" {
                removed += 1;
                continue;
            }
        }
        base.push_str(l);
        base.push('\n');
    }
    assert!(removed > 20, "regulatory rows not found");
    let lookups = Lookups::open(&[], &[], &dir, "GRCh38", None)
        .unwrap()
        .with_regulatory(&dir.join("cache/homo_sapiens/104_GRCh38"))
        .unwrap();
    let vcf = std::fs::File::open(dir.join("input.vcf")).unwrap();
    let mut out = Vec::new();
    annotate(base.as_bytes(), BufReader::new(vcf), &lookups, 2, &mut out).unwrap();
    let got = String::from_utf8(out).unwrap();
    if got != want {
        let line = got
            .lines()
            .zip(want.lines())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "differs at line {}:\nrust {:?}\nvep  {:?}",
            line + 1,
            got.lines().nth(line),
            want.lines().nth(line)
        );
    }
}
