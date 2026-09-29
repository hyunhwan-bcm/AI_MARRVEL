//! Rust vs actual Nextflow runs of AIM v1.1.3 (with `PYTHONHASHSEED=0`), PHRANK_SCORING:
//! VCF -> gene list (`location_to_gene.py` + symbol joins) -> `run_phrank.py` ranking, compared
//! byte for byte; plus `run_phrank.py` itself on larger phenotype sets
//! (`rust/tools/make_goldens_phrank.py`).

mod common;

use std::collections::HashSet;
use std::io::BufReader;

use aim_core::phrank::{
    first_fields, genes_via_symbols, phrank_text, vcf_variants, GeneLocations, Phrank,
};
use common::{golden_dir, refs_dir};

fn read(path: std::path::PathBuf) -> String {
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn phrank() -> Phrank {
    let ph = refs_dir().join("phrank/hg38");
    Phrank::new(
        &read(ph.join("child_to_parent.txt")),
        &read(ph.join("disease_to_pheno.txt")),
        &read(ph.join("disease_to_gene.txt")),
    )
    .unwrap()
}

fn gene_set(text: &str) -> HashSet<String> {
    first_fields(text).into_iter().collect()
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn vcf_to_phrank_matches_nextflow() {
    let ph = refs_dir().join("phrank/hg38");
    let locations = GeneLocations::parse(&read(ph.join("grch38_symbol_to_location.txt"))).unwrap();
    let to_symbol = read(ph.join("ensembl_to_symbol.txt"));
    let mut p = phrank();
    for run in ["nextflow_fixture", "nextflow_clinvar"] {
        let dir = golden_dir().join(run).join("phrank");
        let vcf = flate2::read::MultiGzDecoder::new(
            std::fs::File::open(dir.join("input.vcf.gz")).unwrap(),
        );
        let variants = vcf_variants(BufReader::new(vcf)).unwrap();
        let mut ensembl = std::collections::BTreeSet::new();
        for (chrom, pos) in &variants {
            ensembl.extend(locations.genes_at(chrom, *pos));
        }
        let genes = genes_via_symbols(&ensembl, &to_symbol);
        let got: String = genes.iter().map(|g| format!("{g}\n")).collect();
        assert_eq!(
            got,
            read(dir.join("expected_genes.txt")),
            "{run}: gene list"
        );

        let hpo = first_fields(&read(dir.join("input.hpo.txt")));
        let ranked = p.rank_genes(&genes.into_iter().collect(), &hpo);
        assert_eq!(
            phrank_text(&ranked),
            read(dir.join("expected_phrank.txt")),
            "{run}: phrank"
        );
    }
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn run_phrank_matches_python_on_larger_phenotype_sets() {
    let dir = golden_dir().join("phrank_cases");
    let genes = gene_set(&read(dir.join("genes.txt")));
    let mut p = phrank();
    let mut cases = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let case = entry.unwrap().path();
        if !case.is_dir() {
            continue;
        }
        let hpo = first_fields(&read(case.join("input.hpo.txt")));
        let got = phrank_text(&p.rank_genes(&genes, &hpo));
        let want = read(case.join("expected_phrank.txt"));
        let diffs: Vec<(&str, &str)> = got
            .lines()
            .zip(want.lines())
            .filter(|(a, b)| a != b)
            .take(5)
            .collect();
        assert!(
            got == want,
            "{}: {} vs {} lines; first differences {diffs:?}",
            case.display(),
            got.lines().count(),
            want.lines().count()
        );
        cases += 1;
    }
    assert_eq!(cases, 5);
}
