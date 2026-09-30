//! Rust vs VEP 104.3: the co-located known-variant columns (`Existing_variation`, `CLIN_SIG`,
//! the frequencies, `MAX_AF`, ...) recomputed from the offline cache's `all_vars.gz`.
//!
//! `vep_existing/` holds VEP's output on 20 lines and a synthetic cache (no real data);
//! `rust/tools/make_goldens_vep_existing.py` writes them and documents the VEP command. The
//! test blanks the columns in VEP's output and compares the recomputed file with it, byte for
//! byte.

mod common;

use std::io::{BufReader, Read};

use aim_core::vep_annotate::{annotate, Lookups};
use common::golden_dir;

#[test]
fn known_variants_match_vep() {
    let dir = golden_dir().join("vep_existing");
    let mut want = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(dir.join("expected.txt.gz")).unwrap())
        .read_to_string(&mut want)
        .unwrap();
    // VEP's output with every co-located column set to "-"
    let known = aim_core::vep_existing::columns();
    let mut blank_ix: Vec<usize> = Vec::new();
    let mut base = String::new();
    for l in want.lines() {
        if let Some(h) = l.strip_prefix('#').filter(|_| !l.starts_with("##")) {
            blank_ix = h
                .split('\t')
                .enumerate()
                .filter(|(_, c)| known.iter().any(|k| k == c))
                .map(|(i, _)| i)
                .collect();
            base.push_str(l);
        } else if l.starts_with('#') {
            base.push_str(l);
        } else {
            let row: Vec<&str> = l
                .split('\t')
                .enumerate()
                .map(|(i, v)| if blank_ix.contains(&i) { "-" } else { v })
                .collect();
            base.push_str(&row.join("\t"));
        }
        base.push('\n');
    }
    assert!(blank_ix.len() > 20, "co-located columns not found");
    assert_ne!(base, want);
    let lookups = Lookups::open(&[], &[], &dir, "GRCh38", None)
        .unwrap()
        .with_known_variants(&dir.join("cache/homo_sapiens/104_GRCh38"))
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
