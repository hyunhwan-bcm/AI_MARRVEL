//! Rust vs actual Nextflow runs of AIM v1.1.3, ANNOTATE_BY_MODULES (`bin/feature.py`): one
//! chromosome's VEP table -> `scores.csv`, compared byte for byte. The phenotype-similarity
//! inputs are the HPO_SIM goldens (the same files the runs used). `feature_cases/` holds
//! `feature.py` itself run on derived inputs (`rust/tools/make_goldens_features.py`).

mod common;

use std::io::Read;
use std::path::Path;

use aim_core::features::{features, FeatureOptions, FeatureRefs};
use common::{golden_dir, refs_dir};

fn read_gz(path: &Path) -> String {
    let mut s = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut s)
        .unwrap();
    s
}

fn first_difference(got: &str, want: &str) -> String {
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let header: Vec<&str> = w
        .first()
        .map(|h| h.split(',').collect())
        .unwrap_or_default();
    match g.iter().zip(&w).position(|(a, b)| a != b) {
        Some(i) => {
            let (a, b): (Vec<&str>, Vec<&str>) =
                (g[i].split(',').collect(), w[i].split(',').collect());
            let col = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(0);
            format!(
                "line {}: column {:?}: rust {:?} python {:?}",
                i + 1,
                header.get(col),
                a.get(col),
                b.get(col)
            )
        }
        None => format!("{} vs {} lines", g.len(), w.len()),
    }
}

fn check(dir: &Path, refs: &FeatureRefs, omim: &Path, hgmd: &Path, genome_ref: &str, lit: bool) {
    let got = features(
        &dir.join("vep.txt.gz"),
        omim,
        hgmd,
        refs,
        &FeatureOptions {
            genome_ref,
            enable_lit: lit,
        },
    )
    .unwrap();
    let want = read_gz(&dir.join("expected_scores.csv.gz"));
    assert!(
        got == want,
        "{}: {}",
        dir.display(),
        first_difference(&got, &want)
    );
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn features_match_nextflow() {
    let refs = FeatureRefs::read(&refs_dir().join("annotate"), "hg38").unwrap();
    let sim = golden_dir().join("nextflow_clinvar/hpo_sim");
    let (omim, hgmd) = (
        sim.join("expected_omim_sim.tsv.gz"),
        sim.join("expected_hgmd_sim.tsv"),
    );
    let mut n = 0;
    for run in ["nextflow_clinvar", "nextflow_fixture"] {
        for e in std::fs::read_dir(golden_dir().join(run).join("features")).unwrap() {
            check(&e.unwrap().path(), &refs, &omim, &hgmd, "hg38", false);
            n += 1;
        }
    }
    assert_eq!(n, 4);
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn features_match_feature_py_on_derived_inputs() {
    let annotate = refs_dir().join("annotate");
    let hg38 = FeatureRefs::read(&annotate, "hg38").unwrap();
    let hg19 = FeatureRefs::read(&annotate, "hg19").unwrap();
    let sim = golden_dir().join("nextflow_clinvar/hpo_sim");
    let omim = sim.join("expected_omim_sim.tsv.gz");
    let mut n = 0;
    for e in std::fs::read_dir(golden_dir().join("feature_cases")).unwrap() {
        let dir = e.unwrap().path();
        let args = std::fs::read_to_string(dir.join("args.txt")).unwrap();
        let args: Vec<&str> = args.split_whitespace().collect();
        let genome_ref = args[args.iter().position(|a| *a == "-genomeRef").unwrap() + 1];
        let lit = args.contains(&"-enableLIT");
        let hgmd = if dir.join("hgmd_sim.tsv").exists() {
            dir.join("hgmd_sim.tsv")
        } else {
            sim.join("expected_hgmd_sim.tsv")
        };
        let refs = if genome_ref == "hg19" { &hg19 } else { &hg38 };
        check(&dir, refs, &omim, &hgmd, genome_ref, lit);
        n += 1;
    }
    assert_eq!(n, 4);
}
