//! Rust vs the pipeline's `mod5_diffusion.diffuseSample` (`rust/tools/make_goldens_diffusion.py`).

mod common;

use std::collections::HashMap;

use aim_core::diffusion::{diffuse_sample, Network};
use common::{golden_dir, parse, refs_dir};

/// numpy sums the matrix product in BLAS order, so heat agrees to float32 rounding only.
const HEAT_REL_TOL: f64 = 1e-5;

fn tsv(path: &std::path::Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .skip(1)
        .map(|l| l.split('\t').map(str::to_owned).collect())
        .collect()
}

fn check(net: &Network, scenario: &str) {
    let dir = golden_dir().join("diffusion").join(scenario);
    let rows = tsv(&dir.join("rows.tsv"));
    let genes: Vec<&str> = rows.iter().map(|r| r[1].as_str()).collect();
    let phrank_text = std::fs::read_to_string(dir.join("phrank.tsv")).unwrap();
    let phrank: Vec<(&str, f64)> = phrank_text
        .lines()
        .map(|l| {
            let mut f = l.split('\t');
            (f.next().unwrap(), parse(f.next().unwrap()))
        })
        .collect();

    let (scores, heat) = diffuse_sample(net, &genes, &phrank);

    // Heat per network gene.
    let want_heat = tsv(&dir.join("heat.tsv"));
    let mut max_rel = 0.0f64;
    for ((gene, &h), w) in net.genes.iter().zip(&heat).zip(&want_heat) {
        assert_eq!(gene, &w[0]);
        let w = parse(&w[1]);
        let rel = if w == 0.0 {
            (h as f64).abs()
        } else {
            ((h as f64 - w) / w).abs()
        };
        max_rel = max_rel.max(rel);
    }

    // First row per varId, as drop_duplicates(subset=["varId"]).
    let mut first: HashMap<&str, f64> = HashMap::new();
    for (row, &s) in rows.iter().zip(&scores) {
        first.entry(row[0].as_str()).or_insert(s);
    }
    let expected = tsv(&dir.join("expected.tsv"));
    assert_eq!(expected.len(), first.len(), "{scenario}: variant count");
    let mismatches: Vec<String> = expected
        .iter()
        .filter(|e| first[e[0].as_str()] != parse(&e[1]))
        .map(|e| format!("{}: {} vs {}", e[0], first[e[0].as_str()], e[1]))
        .collect();
    eprintln!(
        "{scenario}: {} variants, {} score mismatches, heat max rel diff {max_rel:e}",
        expected.len(),
        mismatches.len()
    );
    assert!(
        max_rel <= HEAT_REL_TOL,
        "{scenario}: heat max rel diff {max_rel:e}"
    );
    assert!(
        mismatches.is_empty(),
        "{scenario}: {:?}",
        &mismatches[..mismatches.len().min(5)]
    );
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn diffusion_matches_pipeline() {
    let net = Network::read(refs_dir().join("mod5_diffusion")).unwrap();
    for scenario in ["fixture", "synthetic0", "synthetic1", "synthetic2"] {
        check(&net, scenario);
    }
}
