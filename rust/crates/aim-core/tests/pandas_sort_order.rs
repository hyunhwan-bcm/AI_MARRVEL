//! Row order of pandas' unstable `sort_values` / `sort_index` (numpy introsort) on inputs with
//! many ties and NaNs, against orders recorded from pandas 1.4.3 / numpy 1.24.4.

mod common;

use aim_core::npsort::{sort_index_str, sort_values_f64};
use common::golden_dir;

#[test]
fn sort_orders_match_pandas() {
    let text = std::fs::read_to_string(golden_dir().join("npsort/pandas_sorts.json")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    for (k, case) in doc["float_cases"].as_array().unwrap().iter().enumerate() {
        // values were float32; ties and NaNs as pandas saw them
        let v: Vec<f64> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32 as f64)
            .collect();
        let order = |key: &str| -> Vec<usize> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        };
        assert_eq!(
            sort_values_f64(&v, true),
            order("asc"),
            "case {k} ascending (n={})",
            v.len()
        );
        assert_eq!(
            sort_values_f64(&v, false),
            order("desc"),
            "case {k} descending (n={})",
            v.len()
        );
    }
    let idx: Vec<String> = doc["str_case"]["index"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect();
    let want: Vec<usize> = doc["str_case"]["sorted"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(
        sort_index_str(&idx),
        want,
        "sort_index on a string index with duplicates"
    );
}
