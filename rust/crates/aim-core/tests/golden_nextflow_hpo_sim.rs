//! Rust vs `bin/phenoSim.R` (HPO_SIM): the pipeline run's HGMD / OMIM similarity tables and
//! `rust/tools/make_goldens_phenosim.R` cases (more phenotype sets, a synthetic HGMD table),
//! compared byte for byte.

mod common;

use std::io::Read;
use std::path::Path;

use aim_core::phenosim::{
    hgmd_similarity, omim_similarity, patient_terms, Genemap, Ontology, PatientSim,
};
use common::{golden_dir, refs_dir};

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_gz(path: &Path) -> String {
    let mut s = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut s)
        .unwrap();
    s
}

fn first_difference(got: &str, want: &str) -> String {
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    match g.iter().zip(&w).position(|(a, b)| a != b) {
        Some(i) => format!("line {}: rust {:?} R {:?}", i + 1, g[i], w[i]),
        None => format!("{} vs {} lines", g.len(), w.len()),
    }
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py + export_genemap.R (run with --include-ignored)"]
fn hpo_sim_matches_phenosim_r() {
    let ann = refs_dir().join("omim_annotate");
    let onto = Ontology::parse(&read(&ann.join("hp.obo"))).unwrap();
    let genemap = Genemap::parse(&read(&ann.join("hg38/genemap2_pheno.tsv"))).unwrap();
    let omim = read(&ann.join("hg38/HPO_OMIM.tsv"));

    let mut cases = vec![(
        golden_dir().join("nextflow_clinvar/hpo_sim"),
        ann.join("hg38/HGMD_phen.tsv"),
    )];
    for e in std::fs::read_dir(golden_dir().join("phenosim_cases")).unwrap() {
        let d = e.unwrap().path();
        cases.push((d.clone(), d.join("HGMD_phen.tsv")));
    }
    assert_eq!(cases.len(), 4);
    for (dir, hgmd) in cases {
        let patient = patient_terms(&read(&dir.join("input.hpo.txt")));
        let sim = PatientSim::new(&onto, &patient).unwrap();
        let omim_table = omim_similarity(&sim, &onto, &omim, &genemap).unwrap();
        let want = read_gz(&dir.join("expected_omim_sim.tsv.gz"));
        assert!(
            omim_table == want,
            "{}: OMIM table {}",
            dir.display(),
            first_difference(&omim_table, &want)
        );
        let hgmd_table = hgmd_similarity(&sim, &onto, &read(&hgmd)).unwrap();
        let want = read(&dir.join("expected_hgmd_sim.tsv"));
        assert!(
            hgmd_table == want,
            "{}: HGMD table {}",
            dir.display(),
            first_difference(&hgmd_table, &want)
        );
    }
}
