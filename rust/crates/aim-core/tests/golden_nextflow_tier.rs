//! Rust vs actual Nextflow runs of AIM v1.1.3, ANNOTATE_TIER (`VarTierDiseaseDBFalse.R`):
//! per-chromosome `scores.csv` -> `Tier.v2.tsv`, compared byte for byte.

mod common;

use aim_core::tier::{read_inheritance, tier};
use common::{golden_dir, refs_dir};

fn check(run: &str, chrom: &str) {
    let dir = golden_dir().join(run);
    let own = dir.join("tier").join(chrom).join("scores.csv.gz");
    let scores = if own.exists() {
        own
    } else {
        dir.join("join_phrank").join(chrom).join("scores.csv.gz")
    };
    let inheritance =
        read_inheritance(refs_dir().join("var_tier/hg38/genemap2.Inh.F.txt")).unwrap();
    let got = tier(&scores, &inheritance).unwrap();
    let want =
        std::fs::read_to_string(dir.join("tier").join(chrom).join("expected_Tier.v2.tsv")).unwrap();
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let first_diff = g.iter().zip(&w).position(|(a, b)| a != b);
    eprintln!(
        "{run}/{chrom}: {} rows (rust {}), first differing line: {first_diff:?}",
        w.len() - 1,
        g.len() - 1
    );
    if let Some(i) = first_diff {
        panic!(
            "{run}/{chrom}: line {i} differs\n rust:     {}\n pipeline: {}",
            g[i], w[i]
        );
    }
    assert_eq!(got, want, "{run}/{chrom}: Tier.v2.tsv not byte-identical");
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn tier_matches_nextflow() {
    check("nextflow_fixture", "chr17");
    check("nextflow_clinvar", "chr1");
    check("nextflow_clinvar", "chr2");
    check("nextflow_clinvar", "chr17");
}
