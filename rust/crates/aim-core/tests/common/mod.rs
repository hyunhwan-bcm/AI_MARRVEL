//! Helpers shared by the golden tests.
#![allow(dead_code)]

pub mod pandas_parse;

use std::path::{Path, PathBuf};

use aim_core::xgb::Booster;

pub fn rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn golden_dir() -> PathBuf {
    rust_dir().join("tests/golden")
}

/// Exported model bundle: `rust/models/<model>/` or `$AIM_MODELS_DIR/<model>/`.
pub fn model_dir(model: &str) -> PathBuf {
    let base = std::env::var_os("AIM_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| rust_dir().join("models"));
    let dir = base.join(model);
    assert!(
        dir.join("model.json").exists(),
        "{} missing: run rust/tools/export_models.py <model_inputs> rust/models in the baseline py env",
        dir.join("model.json").display()
    );
    dir
}

/// Exported reference data: `rust/refs/` or `$AIM_REFS_DIR` (`rust/tools/export_refs.py`).
pub fn refs_dir() -> PathBuf {
    let dir = std::env::var_os("AIM_REFS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| rust_dir().join("refs"));
    assert!(
        dir.join("mod5_diffusion/net.csr").exists(),
        "{} missing: run rust/tools/export_refs.py <data_dir> rust/refs in the baseline py env",
        dir.display()
    );
    dir
}

pub fn load_model(model: &str) -> (Booster, Vec<f64>) {
    let dir = model_dir(model);
    let booster = Booster::from_json_file(dir.join("model.json")).unwrap();
    let panel = std::fs::read_to_string(dir.join("reference_panel.txt"))
        .unwrap()
        .lines()
        .map(parse)
        .collect();
    (booster, panel)
}

/// A delimited table whose first column is the row id (pandas `to_csv` with an index).
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<(String, Vec<String>)>,
}

impl Table {
    pub fn read(path: &Path, sep: char) -> Table {
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut lines = text.lines();
        let header = lines
            .next()
            .unwrap()
            .split(sep)
            .skip(1)
            .map(str::to_owned)
            .collect();
        let rows = lines
            .map(|line| {
                let mut fields = line.split(sep);
                let id = fields.next().unwrap().to_owned();
                (id, fields.map(str::to_owned).collect())
            })
            .collect();
        Table { header, rows }
    }

    pub fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name:?}"))
    }

    /// Rows as `f64` feature vectors in the given feature order.
    pub fn features_f64(&self, features: &[String]) -> Vec<(String, Vec<f64>)> {
        let idx: Vec<usize> = features.iter().map(|f| self.col(f)).collect();
        self.rows
            .iter()
            .map(|(id, v)| (id.clone(), idx.iter().map(|&i| parse(&v[i])).collect()))
            .collect()
    }

    /// Rows as `f32` feature vectors (Python passes float64 to XGBoost, which rounds to float32).
    pub fn features_f32(&self, features: &[String]) -> Vec<(String, Vec<f32>)> {
        self.features_f64(features)
            .into_iter()
            .map(|(id, v)| (id, v.into_iter().map(|x| x as f32).collect()))
            .collect()
    }
}

pub fn parse(v: &str) -> f64 {
    v.parse::<f64>()
        .unwrap_or_else(|_| panic!("not a number: {v:?}"))
}
