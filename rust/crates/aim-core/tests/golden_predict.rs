//! Rust vs the AIM v1.1.3 pipeline, prediction stage.
//!
//! Expected values in `rust/tests/golden/predict/<model>/` are produced by
//! `rust/tools/make_goldens_predict.py`, which calls the pipeline's own code
//! (`XGBClassifier.predict_proba`, `extraModel.confidence.assign_confidence_score`,
//! `extraModel_main.assign_ranking`, `model_interpreter.ModelInterpreter`).
//! Models are read from `rust/models/<model>/` (`rust/tools/export_models.py`), or from
//! `$AIM_MODELS_DIR`.

mod common;

use std::collections::HashMap;

use aim_core::stats::{confidence_level, percentile_of_score, rank_predictions};
use common::{golden_dir, load_model, parse, Table};

/// SHAP matches bit for bit against goldens made with the osx-arm64 xgboost wheel; goldens made
/// on a build without fused multiply-add differ by a few f32 ulps (< 1e-6 observed), so allow
/// that. Predictions, confidence and rankings must match exactly.
const SHAP_ABS_TOL: f64 = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
    0.0 // goldens were made with the osx-arm64 wheel: must match bit for bit
} else {
    1e-5
};

fn check_model(model: &str) {
    let golden = golden_dir().join("predict").join(model);
    let (booster, panel) = load_model(model);

    let input = Table::read(&golden.join("input.csv"), ',');
    assert_eq!(
        input.header,
        booster.feature_names(),
        "{model}: feature order differs from the model"
    );

    // Python gives XGBoost float64 values; XGBoost rounds them to float32.
    let rows: Vec<(String, Vec<f32>)> = input
        .rows
        .iter()
        .map(|(id, vals)| (id.clone(), vals.iter().map(|v| parse(v) as f32).collect()))
        .collect();
    let predict: Vec<f32> = rows.iter().map(|(_, r)| booster.predict_proba(r)).collect();

    // --- predictions, confidence, rankings -------------------------------------------------
    let expected = Table::read(&golden.join("expected.csv"), ',');
    let (c_pred, c_conf, c_level, c_min, c_max) = (
        expected.col("predict"),
        expected.col("confidence"),
        expected.col("confidence level"),
        expected.col("min_ranking"),
        expected.col("max_ranking"),
    );
    let exp: HashMap<&str, &Vec<String>> = expected
        .rows
        .iter()
        .map(|(id, v)| (id.as_str(), v))
        .collect();
    let ranks = rank_predictions(&predict);

    let mut pred_mismatch = Vec::new();
    let mut other_mismatch = Vec::new();
    for (i, (id, _)) in rows.iter().enumerate() {
        let e = exp[id.as_str()];
        let want = parse(&e[c_pred]) as f32;
        if predict[i].to_bits() != want.to_bits() {
            pred_mismatch.push(format!("{id}: rust {:e} python {:e}", predict[i], want));
        }
        let conf = percentile_of_score(&panel, predict[i] as f64);
        if conf != parse(&e[c_conf]) || confidence_level(conf) != e[c_level] {
            other_mismatch.push(format!(
                "{id}: confidence {conf} vs {} ({})",
                e[c_conf], e[c_level]
            ));
        }
        let (min_r, max_r) = ranks[i];
        if min_r.to_string() != e[c_min] || max_r.to_string() != e[c_max] {
            other_mismatch.push(format!(
                "{id}: ranks {min_r}/{max_r} vs {}/{}",
                e[c_min], e[c_max]
            ));
        }
    }
    assert!(
        pred_mismatch.is_empty(),
        "{model}: {} of {} predictions differ, e.g. {:?}",
        pred_mismatch.len(),
        rows.len(),
        &pred_mismatch[..pred_mismatch.len().min(5)]
    );
    assert!(
        other_mismatch.is_empty(),
        "{model}: {} confidence/ranking mismatches, e.g. {:?}",
        other_mismatch.len(),
        &other_mismatch[..other_mismatch.len().min(5)]
    );

    // --- SHAP (approximate contributions) --------------------------------------------------
    let shap = Table::read(&golden.join("shap.csv"), ',');
    assert_eq!(shap.header[0], "base_value");
    assert_eq!(&shap.header[1..], booster.feature_names());
    let shap_rows: HashMap<&str, &Vec<String>> =
        shap.rows.iter().map(|(id, v)| (id.as_str(), v)).collect();
    let (mut max_diff, mut worst, mut exact) = (0.0f64, String::new(), 0usize);
    let mut total = 0usize;
    for (id, row) in &rows {
        let got = booster.approx_contributions(row);
        let want = shap_rows[id.as_str()];
        let n = booster.num_features();
        let pairs =
            std::iter::once((got[n], &want[0])).chain(got[..n].iter().copied().zip(&want[1..]));
        for (g, w) in pairs {
            let d = (g as f64 - parse(w)).abs();
            total += 1;
            if d == 0.0 {
                exact += 1;
            }
            if d > max_diff {
                max_diff = d;
                worst = id.clone();
            }
        }
    }
    eprintln!("{model}: SHAP {exact}/{total} values exact, max |diff| {max_diff:e} (row {worst})");
    assert!(
        max_diff <= SHAP_ABS_TOL,
        "{model}: SHAP max |diff| {max_diff:e} at {worst}"
    );
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn default_model_matches_pipeline() {
    check_model("default");
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn nd_model_matches_pipeline() {
    check_model("nd");
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn recessive_model_matches_pipeline() {
    check_model("recessive");
}

#[test]
#[ignore = "needs exported models: rust/tools/export_models.py (run with --include-ignored)"]
fn nd_recessive_model_matches_pipeline() {
    check_model("nd_recessive");
}
