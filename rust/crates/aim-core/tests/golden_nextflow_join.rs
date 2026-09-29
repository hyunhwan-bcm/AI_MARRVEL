//! Rust vs actual Nextflow runs of AIM v1.1.3, JOIN_PHRANK (`generate_new_matrix_2.py` +
//! `add_c_nc.py`): per-chromosome `scores.csv` + ClinVar tables + phrank -> `scores.txt.gz`,
//! compared cell by cell as text (numbers may differ by one ulp: Polars parses floats with
//! correct rounding, pandas 1.4 sometimes one ulp off).

mod common;

use aim_core::join::{chrom_filter, join_phrank, ClinVarTables};
use aim_core::pandas::{read_df, to_csv};
use common::{golden_dir, refs_dir};
use polars::prelude::*;

const REL_TOL: f64 = 1e-14;

/// All cells as text.
fn read_text_table(bytes: Vec<u8>) -> DataFrame {
    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(0))
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .into_reader_with_file_handle(std::io::Cursor::new(bytes))
        .finish()
        .unwrap()
}

fn check(run: &str, chrom: &str, tables: &ClinVarTables) {
    let dir = golden_dir().join(run).join("join_phrank").join(chrom);
    let score = read_df(dir.join("scores.csv.gz"), b',').unwrap();
    let phrank = std::fs::read_to_string(dir.join("phrank.txt")).unwrap();
    let got = join_phrank(&score, &phrank, tables).unwrap();
    let got = read_text_table(to_csv(&got, '\t').unwrap().into_bytes());
    let want = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(0))
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .try_into_reader_with_file_path(Some(dir.join("expected_scores.txt.gz")))
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(
        got.get_column_names(),
        want.get_column_names(),
        "{run}/{chrom}: columns"
    );
    assert_eq!(got.height(), want.height(), "{run}/{chrom}: rows");
    let (mut exact, mut near, mut failures) = (0usize, 0usize, Vec::new());
    for (g, w) in got.columns().iter().zip(want.columns()) {
        let (g, w) = (g.str().unwrap(), w.str().unwrap());
        for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            let (a, b) = (a.unwrap_or(""), b.unwrap_or(""));
            if a == b {
                exact += 1;
            } else if let (Ok(x), Ok(y)) = (a.parse::<f64>(), b.parse::<f64>()) {
                if ((x - y) / y).abs() <= REL_TOL {
                    near += 1;
                } else {
                    failures.push(format!("row {i} {}: rust {a:?} pipeline {b:?}", w.name()));
                }
            } else {
                failures.push(format!("row {i} {}: rust {a:?} pipeline {b:?}", w.name()));
            }
        }
    }
    eprintln!(
        "{run}/{chrom}: {} rows x {} cols, {exact} identical, {near} within one ulp",
        want.height(),
        want.width()
    );
    assert!(
        failures.is_empty(),
        "{run}/{chrom}: {} cells differ, e.g. {:?}",
        failures.len(),
        &failures[..failures.len().min(8)]
    );
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn join_phrank_matches_nextflow() {
    let tables = ClinVarTables::read(refs_dir().join("merge_expand/hg38")).unwrap();
    check("nextflow_fixture", "chr17", &tables);
    check("nextflow_clinvar", "chr1", &tables);
    check("nextflow_clinvar", "chr17", &tables);
}

#[test]
#[ignore = "needs exported refs: rust/tools/export_refs.py (run with --include-ignored)"]
fn join_phrank_with_per_chromosome_tables_matches_nextflow() {
    // The CLI reads only the chromosome's coding rows; the result must not change.
    for (run, chrom) in [
        ("nextflow_fixture", "chr17"),
        ("nextflow_clinvar", "chr1"),
        ("nextflow_clinvar", "chr17"),
    ] {
        let dir = golden_dir().join(run).join("join_phrank").join(chrom);
        let score = read_df(dir.join("scores.csv.gz"), b',').unwrap();
        let keep = chrom_filter(&score).unwrap();
        let tables =
            ClinVarTables::read_where(refs_dir().join("merge_expand/hg38"), &keep).unwrap();
        check(run, chrom, &tables);
    }
}
