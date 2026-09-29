//! Rust vs actual Nextflow runs, PREDICTION process files: `run_final.py`'s
//! `<id>.default_prediction.csv`, `extraModel_main.py`'s default / nd predictions (row order,
//! every cell; numbers may differ by one ulp from pandas' float parsing) and SHAP JSON.

mod common;

use std::path::Path;

use aim_core::predict_io::{extra_model, run_final, shap_json, Indexed, ModelOutput};
use aim_core::recessive::{expanded, recessive_matrix, recessive_model};
use aim_core::xgb::Booster;
use common::pandas_parse::numbers_agree;
use common::{golden_dir, model_dir, parse};
use polars::prelude::*;

const REL_TOL: f64 = 1e-14;

fn text_table(bytes: Vec<u8>, sep: u8) -> DataFrame {
    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(0))
        .with_parse_options(CsvParseOptions::default().with_separator(sep))
        .into_reader_with_file_handle(std::io::Cursor::new(bytes))
        .finish()
        .unwrap()
}

fn read_bytes(path: &Path) -> Vec<u8> {
    let raw = std::fs::read(path).unwrap();
    if path.extension().is_some_and(|e| e == "gz") {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .unwrap();
        out
    } else {
        raw
    }
}

fn compare(label: &str, got: String, want: &Path, sep: u8) {
    let (g, w) = (
        text_table(got.into_bytes(), sep),
        text_table(read_bytes(want), sep),
    );
    assert_eq!(
        g.get_column_names(),
        w.get_column_names(),
        "{label}: columns"
    );
    assert_eq!(g.height(), w.height(), "{label}: rows");
    let (mut exact, mut near, mut bad) = (0usize, 0usize, Vec::new());
    for (gc, wc) in g.columns().iter().zip(w.columns()) {
        for (i, (a, b)) in gc
            .str()
            .unwrap()
            .iter()
            .zip(wc.str().unwrap().iter())
            .enumerate()
        {
            let (a, b) = (a.unwrap_or(""), b.unwrap_or(""));
            if a == b {
                exact += 1;
            } else if numbers_agree(a, b, REL_TOL) {
                near += 1;
            } else {
                bad.push(format!("row {i} {}: rust {a:?} pipeline {b:?}", wc.name()));
            }
        }
    }
    eprintln!(
        "{label}: {} rows, {exact} cells identical, {near} differ only by pandas float parsing",
        w.height()
    );
    assert!(
        bad.is_empty(),
        "{label}: {} cells differ, e.g. {:?}",
        bad.len(),
        &bad[..bad.len().min(6)]
    );
}

fn booster(model: &str) -> (Booster, Vec<f64>) {
    let dir = model_dir(model);
    let panel = std::fs::read_to_string(dir.join("reference_panel.txt"))
        .unwrap()
        .lines()
        .map(parse)
        .collect();
    (
        Booster::from_json_file(dir.join("model.json")).unwrap(),
        panel,
    )
}

fn check_shap(run: &str, model: &str, b: &Booster, out: &ModelOutput, want: &Path) {
    let got: serde_json::Value =
        serde_json::from_str(&shap_json(b, &out.table.index, &out.rows, &out.data)).unwrap();
    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(want).unwrap()).unwrap();
    let (ga, wa) = (got.as_array().unwrap(), want.as_array().unwrap());
    assert_eq!(ga.len(), wa.len(), "{run} {model}: SHAP entries");
    let (mut exact, mut parsed) = (0usize, 0usize);
    for (g, w) in ga.iter().zip(wa) {
        assert_eq!(
            g["variant_id"], w["variant_id"],
            "{run} {model}: SHAP row order"
        );
        assert_eq!(
            g["base_value"], w["base_value"],
            "{run} {model}: base_value"
        );
        assert_eq!(
            g["model_output_score"], w["model_output_score"],
            "{run} {model}: SHAP values"
        );
        for (k, gv) in g["feature_values"].as_object().unwrap() {
            let (gs, ws) = (gv.to_string(), w["feature_values"][k].to_string());
            if gs == ws {
                exact += 1
            } else {
                assert!(
                    numbers_agree(&gs, &ws, 0.0),
                    "{run} {model}: feature_values[{k}] {gs} vs {ws}"
                );
                parsed += 1
            }
        }
    }
    eprintln!("{run} {model}: SHAP JSON — ids, base values and SHAP values identical; feature_values {exact} identical, {parsed} differ only by pandas float parsing");
}

/// `to_csv(index=False)`: drop the leading index field of each line.
fn without_index(text: String) -> String {
    text.lines()
        .map(|l| &l[l.find(',').map_or(0, |i| i + 1)..])
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn check_recessive(run: &str, id: &str, dir: &Path, merged_scores: &Path, shap: bool) {
    let gz = |name: &str| {
        let p = dir.join(format!("{name}.gz"));
        if p.exists() {
            p
        } else {
            dir.join(name)
        }
    };
    let dp = Indexed::read(gz(&format!("{id}.default_prediction.csv")), b',').unwrap();
    let merged = aim_core::pandas::read_df(merged_scores, b'\t').unwrap();
    let ex = expanded(&dp, &merged).unwrap();
    compare(
        &format!("{run} expanded"),
        without_index(aim_core::pandas::to_csv(&ex, ',').unwrap()),
        &gz(&format!("{id}.expanded.csv")),
        b',',
    );

    // process_sample reads the expanded matrix and default predictions the pipeline wrote.
    let ex_read = Indexed::read(gz(&format!("{id}.expanded.csv")), b',').unwrap();
    let default_pred = Indexed::read(gz(&format!("{id}_default_predictions.csv")), b',').unwrap();
    let rm = recessive_matrix(&dp, &ex_read, &default_pred)
        .unwrap()
        .expect("pairs");
    compare(
        &format!("{run} recessive matrix"),
        rm.to_csv(',').unwrap(),
        &gz(&format!("{id}.recessive_matrix.csv")),
        b',',
    );

    let rm_read = Indexed::read(gz(&format!("{id}.recessive_matrix.csv")), b',').unwrap();
    for model in ["recessive", "nd_recessive"] {
        let (b, panel) = booster(model);
        let out = recessive_model(&rm_read, &b, &panel).unwrap();
        compare(
            &format!("{run} {model}"),
            out.table.to_csv(',').unwrap(),
            &gz(&format!("{id}_{model}_predictions.csv")),
            b',',
        );
        if shap {
            check_shap(
                run,
                model,
                &b,
                &out,
                &dir.join(format!("{id}_{model}_shap_values.json")),
            );
        }
    }
}

fn check(run: &str, id: &str, matrix: &Path, dir: &Path, shap: bool) {
    let (default, _) = booster("default");
    let m = Indexed::read(matrix, b'\t').unwrap();
    let got = run_final(&m, &default, id).unwrap();
    let gz = |name: &str| {
        let p = dir.join(format!("{name}.gz"));
        if p.exists() {
            p
        } else {
            dir.join(name)
        }
    };
    compare(
        &format!("{run} run_final"),
        got.to_csv(',').unwrap(),
        &gz(&format!("{id}.default_prediction.csv")),
        b',',
    );

    // extraModel reads the default_prediction.csv the pipeline wrote.
    let dp = Indexed::read(gz(&format!("{id}.default_prediction.csv")), b',').unwrap();
    for model in ["default", "nd"] {
        let (b, panel) = booster(model);
        let out = extra_model(&dp, &b, &panel).unwrap();
        compare(
            &format!("{run} {model}"),
            out.table.to_csv(',').unwrap(),
            &gz(&format!("{id}_{model}_predictions.csv")),
            b',',
        );
        if shap {
            let got: serde_json::Value =
                serde_json::from_str(&shap_json(&b, &out.table.index, &out.rows, &out.data))
                    .unwrap();
            let want: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join(format!("{id}_{model}_shap_values.json")))
                    .unwrap(),
            )
            .unwrap();
            let (mut exact, mut parsed) = (0usize, 0usize);
            let (ga, wa) = (got.as_array().unwrap(), want.as_array().unwrap());
            assert_eq!(ga.len(), wa.len());
            for (g, w) in ga.iter().zip(wa) {
                assert_eq!(
                    g["variant_id"], w["variant_id"],
                    "{run} {model}: SHAP row order"
                );
                assert_eq!(
                    g["base_value"], w["base_value"],
                    "{run} {model}: base_value"
                );
                assert_eq!(
                    g["model_output_score"], w["model_output_score"],
                    "{run} {model}: SHAP values"
                );
                for (k, gv) in g["feature_values"].as_object().unwrap() {
                    let (gs, ws) = (gv.to_string(), w["feature_values"][k].to_string());
                    if gs == ws {
                        exact += 1
                    } else {
                        assert!(
                            numbers_agree(&gs, &ws, 0.0),
                            "{run} {model}: feature_values[{k}] {gs} vs {ws}"
                        );
                        parsed += 1
                    }
                }
            }
            eprintln!("{run} {model}: SHAP JSON — ids, base values and SHAP values identical; feature_values {exact} identical, {parsed} differ only by pandas float parsing");
        }
    }
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn prediction_files_match_nextflow_fixture() {
    let dir = golden_dir().join("nextflow_fixture/prediction");
    check(
        "fixture",
        "fixture",
        &dir.join("fixture.matrix.txt"),
        &dir,
        true,
    );
    check_recessive(
        "fixture",
        "fixture",
        &dir,
        &golden_dir().join("nextflow_fixture/merge/scores.txt.gz"),
        true,
    );
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn prediction_files_match_nextflow_clinvar() {
    let g = golden_dir().join("nextflow_clinvar");
    check(
        "clinvar",
        "clinvar",
        &g.join("merge/matrix.txt"),
        &g.join("prediction"),
        false,
    );
    check_recessive(
        "clinvar",
        "clinvar",
        &g.join("prediction"),
        &g.join("merge/scores.txt.gz"),
        false,
    );
}
