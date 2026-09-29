//! Rust vs actual Nextflow runs of AIM v1.1.3, MERGE_SCORES_BY_CHROMOSOME: missing-value fill
//! (`fillna_tier.feature_engineering`) from the merged scores + tier table, against the
//! features in the process's `matrix.txt`.

mod common;

use aim_core::diffusion::Network;
use aim_core::fill::{feature_engineering, FeatureStats};
use aim_core::pandas::Frame;
use aim_core::postprocess::{post_process, MergeRefs, SimpleRepeats};
use common::{golden_dir, parse, refs_dir, Table};

/// Polars parses floats with correct rounding, pandas 1.4 sometimes one ulp off; allow that.
const REL_TOL: f64 = 1e-14;

fn check(run: &str) {
    let dir = golden_dir().join(run).join("merge");
    let scores = Frame::read_path(dir.join("scores.txt.gz"), b'\t').unwrap();
    let tier = Frame::read_path(dir.join("Tier.v2.tsv"), b'\t').unwrap();
    let stats = FeatureStats::parse(
        &std::fs::read_to_string(refs_dir().join("annotate/feature_stats.csv")).unwrap(),
    );
    let got = feature_engineering(scores, &tier, &stats).unwrap();
    let want = Table::read(&dir.join("matrix.txt"), '\t');

    let row_of: std::collections::HashMap<&str, usize> = got
        .index
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i))
        .collect();
    let (mut compared, mut exact, mut max_rel, mut worst) = (0usize, 0usize, 0.0f64, String::new());
    let mut failures = Vec::new();
    for name in &want.header {
        if name == "diffuse_Phrank_STRING" || name == "simple_repeat" {
            continue; // not part of feature_engineering
        }
        let c = want.col(name);
        let col = got
            .get(name)
            .unwrap_or_else(|| panic!("{run}: Rust output lacks column {name}"));
        for (id, vals) in &want.rows {
            let i = row_of[id.as_str()];
            let (g, w) = (col.f64_at(i).unwrap(), parse(&vals[c]));
            compared += 1;
            if g == w || (g.is_nan() && w.is_nan()) {
                exact += 1;
                continue;
            }
            let rel = ((g - w) / w).abs();
            if rel > max_rel {
                max_rel = rel;
                worst = format!("{id} {name}: rust {g:e} pipeline {w:e}");
            }
            if rel > REL_TOL {
                failures.push(format!("{id} {name}: rust {g:e} pipeline {w:e}"));
            }
        }
    }
    eprintln!(
        "{run}: {} variants, {exact}/{compared} values exact, max rel diff {max_rel:e} {worst}",
        want.rows.len()
    );
    assert!(
        failures.is_empty(),
        "{run}: {} values differ, e.g. {:?}",
        failures.len(),
        &failures[..failures.len().min(8)]
    );
}

/// Whole MERGE step: every column of `matrix.txt`, including diffusion and simple repeats.
fn check_matrix(run: &str, refs: &MergeRefs) {
    let dir = golden_dir().join(run).join("merge");
    let scores = Frame::read_path(dir.join("scores.txt.gz"), b'\t').unwrap();
    let tier = Frame::read_path(dir.join("Tier.v2.tsv"), b'\t').unwrap();
    let phrank = std::fs::read_to_string(dir.join("phrank.txt")).unwrap();
    let got = post_process(scores, &tier, &phrank, refs).unwrap();
    let want = Table::read(&dir.join("matrix.txt"), '\t');
    assert_eq!(got.columns, want.header, "{run}: matrix columns/order");
    let want_ids: Vec<&str> = want.rows.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(got.index, want_ids, "{run}: matrix rows/order");
    let mut failures = Vec::new();
    let mut exact = 0usize;
    for (j, name) in want.header.iter().enumerate() {
        for (i, (id, vals)) in want.rows.iter().enumerate() {
            let (g, w) = (got.data[j].f64_at(i).unwrap(), parse(&vals[j]));
            if g == w {
                exact += 1;
            } else if ((g - w) / w).abs() > REL_TOL {
                failures.push(format!("{id} {name}: rust {g:e} pipeline {w:e}"));
            }
        }
    }
    eprintln!(
        "{run}: matrix {}x{}, {exact} values exact",
        want.rows.len(),
        want.header.len()
    );
    assert!(
        failures.is_empty(),
        "{run}: {} values differ, e.g. {:?}",
        failures.len(),
        &failures[..failures.len().min(8)]
    );
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn merge_step_matches_nextflow() {
    let refs = MergeRefs {
        network: Network::read(refs_dir().join("mod5_diffusion")).unwrap(),
        stats: FeatureStats::parse(
            &std::fs::read_to_string(refs_dir().join("annotate/feature_stats.csv")).unwrap(),
        ),
        repeats: SimpleRepeats::read(refs_dir().join("merge_expand/hg38/simpleRepeats.hg38.bed"))
            .unwrap(),
    };
    check_matrix("nextflow_fixture", &refs);
    check_matrix("nextflow_clinvar", &refs);
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn fill_matches_nextflow_fixture() {
    check("nextflow_fixture");
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn fill_matches_nextflow_clinvar_sample() {
    check("nextflow_clinvar");
}
