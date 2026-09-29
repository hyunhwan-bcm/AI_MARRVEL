//! Rust vs an actual Nextflow run of AIM v1.1.3 (fastVEP fixture, GRCh38): the PREDICTION
//! process's own inputs and outputs, from `rust/tests/golden/nextflow_fixture/prediction/`.
//!
//! - `run_final.py`: default model on `fixture.matrix.txt` -> `fixture.default_prediction.csv`
//! - `extraModel_main.py`: default/nd on that file, recessive/nd_recessive on the recessive
//!   pair matrix -> `fixture_<model>_predictions.csv` and `fixture_<model>_shap_values.json`
//!
//! Row ids are not unique in the recessive files (a variant pair appears once per gene), so
//! rows are matched on id plus their model feature values. `run_final.py`'s ids are unique, and
//! it cannot be matched on values: pandas 1.4's default CSV float parser is not exactly
//! round-trip, so reading `fixture.matrix.txt` shifts some values by one ulp before they are
//! written back (18 of 824 cells in this fixture).

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use aim_core::stats::{confidence_level, percentile_of_score, rank_predictions};
use common::{golden_dir, load_model, parse, Table};

/// See `golden_predict.rs`: exact on osx-arm64, a few f32 ulps elsewhere.
const SHAP_ABS_TOL: f64 = 1e-5;

fn dir() -> PathBuf {
    golden_dir().join("nextflow_fixture/prediction")
}

/// pandas writes float32 columns with the shortest repr that round-trips float32.
fn parse_f32(v: &str) -> f32 {
    v.parse::<f32>()
        .unwrap_or_else(|_| panic!("not a number: {v:?}"))
}

type Key = (String, Vec<u64>);

fn key(id: &str, values: &[f64], by_values: bool) -> Key {
    let bits = if by_values {
        values.iter().map(|v| v.to_bits()).collect()
    } else {
        Vec::new()
    };
    (id.to_owned(), bits)
}

/// One input row: id, original float64 values, float32 values the model sees.
struct Row {
    id: String,
    f64s: Vec<f64>,
    f32s: Vec<f32>,
}

/// Predict every row of `input` and compare with `expected` (predict, ranks, and confidence if
/// `with_confidence`). Returns the rows, for the SHAP check.
fn check_predictions(
    model: &str,
    input: &Table,
    expected: &Table,
    with_confidence: bool,
    match_on_values: bool,
) -> Vec<Row> {
    let (booster, panel) = load_model(model);
    let names = booster.feature_names();
    let rows: Vec<Row> = input
        .features_f64(names)
        .into_iter()
        .map(|(id, f64s)| {
            let f32s = f64s.iter().map(|&x| x as f32).collect();
            Row { id, f64s, f32s }
        })
        .collect();
    let predict: Vec<f32> = rows
        .iter()
        .map(|r| booster.predict_proba(&r.f32s))
        .collect();
    let ranks = rank_predictions(&predict);

    let mut exp: HashMap<Key, Vec<&Vec<String>>> = HashMap::new();
    for ((id, vals), (_, raw)) in expected.features_f64(names).iter().zip(&expected.rows) {
        exp.entry(key(id, vals, match_on_values))
            .or_default()
            .push(raw);
    }
    assert_eq!(
        expected.rows.len(),
        rows.len(),
        "{model}: row count differs from Nextflow output"
    );
    let (c_pred, c_min, c_max, c_rank) = (
        expected.col("predict"),
        expected.col("min_ranking"),
        expected.col("max_ranking"),
        expected.col("ranking"),
    );

    let mut mismatches = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let e = exp
            .get_mut(&key(&row.id, &row.f64s, match_on_values))
            .and_then(Vec::pop)
            .unwrap_or_else(|| {
                panic!(
                    "{model}: {} (with these features) not in Nextflow output",
                    row.id
                )
            });
        let want = parse_f32(&e[c_pred]);
        if predict[i].to_bits() != want.to_bits() {
            mismatches.push(format!(
                "{}: predict {:e} vs {:e}",
                row.id, predict[i], want
            ));
        }
        let (min_r, max_r) = (ranks[i].0.to_string(), ranks[i].1.to_string());
        if min_r != e[c_min] || max_r != e[c_max] || max_r != e[c_rank] {
            mismatches.push(format!(
                "{}: ranks {min_r}/{max_r} vs {}/{}/{}",
                row.id, e[c_min], e[c_max], e[c_rank]
            ));
        }
        if with_confidence {
            let conf = percentile_of_score(&panel, predict[i] as f64);
            let (c_conf, c_level) = (expected.col("confidence"), expected.col("confidence level"));
            if conf != parse(&e[c_conf]) || confidence_level(conf) != e[c_level] {
                mismatches.push(format!(
                    "{}: confidence {conf} vs {} ({})",
                    row.id, e[c_conf], e[c_level]
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{model}: {} mismatches: {:?}",
        mismatches.len(),
        mismatches
    );
    rows
}

fn check_shap(model: &str, rows: &[Row]) {
    let (booster, _) = load_model(model);
    let names = booster.feature_names();
    let path = dir().join(format!("fixture_{model}_shap_values.json"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let entries = json.as_array().unwrap();
    assert_eq!(entries.len(), rows.len(), "{model}: SHAP entry count");

    // `feature_values` holds the original float64 inputs, which identify the row.
    let mut by_key: HashMap<Key, Vec<&Row>> = HashMap::new();
    for r in rows {
        by_key.entry(key(&r.id, &r.f64s, true)).or_default().push(r);
    }
    let mut max_diff = 0.0f64;
    for entry in entries {
        let id = entry["variant_id"].as_str().unwrap();
        let values: Vec<f64> = names
            .iter()
            .map(|n| entry["feature_values"][n].as_f64().unwrap())
            .collect();
        let row = by_key
            .get_mut(&key(id, &values, true))
            .and_then(Vec::pop)
            .unwrap_or_else(|| panic!("{model}: SHAP entry {id} matches no input row"));
        let got = booster.approx_contributions(&row.f32s);
        max_diff =
            max_diff.max((got[names.len()] as f64 - entry["base_value"].as_f64().unwrap()).abs());
        for (j, name) in names.iter().enumerate() {
            max_diff = max_diff
                .max((got[j] as f64 - entry["model_output_score"][name].as_f64().unwrap()).abs());
        }
    }
    eprintln!(
        "{model}: {} SHAP entries, max |diff| vs Nextflow JSON {max_diff:e}",
        entries.len()
    );
    assert!(
        max_diff <= SHAP_ABS_TOL,
        "{model}: SHAP max |diff| {max_diff:e}"
    );
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn run_final_default_matches_nextflow() {
    let input = Table::read(&dir().join("fixture.matrix.txt"), '\t');
    let expected = Table::read(&dir().join("fixture.default_prediction.csv"), ',');
    check_predictions("default", &input, &expected, false, false);
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn extra_model_default_and_nd_match_nextflow() {
    let input = Table::read(&dir().join("fixture.default_prediction.csv"), ',');
    for model in ["default", "nd"] {
        let expected = Table::read(&dir().join(format!("fixture_{model}_predictions.csv")), ',');
        let rows = check_predictions(model, &input, &expected, true, true);
        check_shap(model, &rows);
    }
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn extra_model_recessive_models_match_nextflow() {
    let input = Table::read(&dir().join("fixture.recessive_matrix.csv"), ',');
    for model in ["recessive", "nd_recessive"] {
        let expected = Table::read(&dir().join(format!("fixture_{model}_predictions.csv")), ',');
        let rows = check_predictions(model, &input, &expected, true, true);
        check_shap(model, &rows);
    }
}
