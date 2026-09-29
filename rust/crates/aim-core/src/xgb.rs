//! Evaluation of XGBoost `binary:logistic` gradient-boosted trees saved with
//! `Booster.save_model("*.json")`, reproducing XGBoost 2.1's CPU predictor:
//!
//! - inputs are rounded to `f32`; a split sends a row left when `value < split_condition`;
//! - a missing value (NaN) follows the node's learned default direction;
//! - the margin starts at the base score (as a margin) and adds each tree's leaf in tree
//!   order, in `f32`;
//! - probabilities use XGBoost's `common::Sigmoid`.
//!
//! [`Booster::approx_contributions`] reproduces `pred_contribs=True, approx_contribs=True`
//! (the per-feature contributions `shap.TreeExplainer(..., approximate=True)` returns).

use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum ModelError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Unsupported(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "cannot read model: {e}"),
            ModelError::Json(e) => write!(f, "cannot parse model JSON: {e}"),
            ModelError::Unsupported(what) => write!(f, "unsupported model: {what}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// One regression tree in structure-of-arrays form, indexed by node id.
#[derive(Debug, Clone)]
struct Tree {
    left: Vec<i32>,
    right: Vec<i32>,
    split_index: Vec<u32>,
    /// Split threshold for internal nodes, leaf value for leaves (as XGBoost stores them).
    split_condition: Vec<f32>,
    default_left: Vec<bool>,
    sum_hessian: Vec<f32>,
    /// Cover-weighted mean output below each node (XGBoost's `FillNodeMeanValues`).
    mean_value: Vec<f32>,
}

impl Tree {
    fn is_leaf(&self, node: usize) -> bool {
        self.left[node] == -1
    }

    fn next(&self, node: usize, row: &[f32]) -> usize {
        let value = row[self.split_index[node] as usize];
        let go_left = if value.is_nan() {
            self.default_left[node]
        } else {
            value < self.split_condition[node]
        };
        (if go_left {
            self.left[node]
        } else {
            self.right[node]
        }) as usize
    }

    fn leaf_value(&self, row: &[f32]) -> f32 {
        let mut node = 0;
        while !self.is_leaf(node) {
            node = self.next(node, row);
        }
        self.split_condition[node]
    }

    fn fill_mean_values(&mut self) {
        let n = self.left.len();
        self.mean_value = vec![0.0; n];
        if n > 0 {
            self.fill_mean_value(0);
        }
    }

    fn fill_mean_value(&mut self, node: usize) -> f32 {
        let result = if self.is_leaf(node) {
            self.split_condition[node]
        } else {
            let (l, r) = (self.left[node] as usize, self.right[node] as usize);
            // `result += mean(r) * cover(r)` in XGBoost's C++ is contracted into one fused
            // multiply-add by clang (the osx-arm64 wheel); this reproduces those roundings bit for
            // bit. A build without FMA (e.g. x86-64 manylinux) differs by a few f32 ulps.
            let mut result = self.fill_mean_value(l) * self.sum_hessian[l];
            result = self.fill_mean_value(r).mul_add(self.sum_hessian[r], result);
            result / self.sum_hessian[node]
        };
        self.mean_value[node] = result;
        result
    }

    /// XGBoost's `RegTree::CalculateContributionsApprox` (Saabas attribution).
    fn add_approx_contributions(&self, row: &[f32], out: &mut [f32]) {
        let bias = out.len() - 1;
        let mut node_value = self.mean_value[0];
        out[bias] += node_value;
        if self.is_leaf(0) {
            return;
        }
        let mut node = 0;
        let mut split = 0;
        while !self.is_leaf(node) {
            split = self.split_index[node] as usize;
            node = self.next(node, row);
            let new_value = self.mean_value[node];
            out[split] += new_value - node_value;
            node_value = new_value;
        }
        out[split] += self.split_condition[node] - node_value;
    }
}

/// A `binary:logistic` gbtree model.
#[derive(Debug, Clone)]
pub struct Booster {
    trees: Vec<Tree>,
    base_margin: f32,
    feature_names: Vec<String>,
}

impl Booster {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let text = std::fs::read_to_string(path).map_err(ModelError::Io)?;
        Self::from_json_str(&text)
    }

    pub fn from_json_str(text: &str) -> Result<Self, ModelError> {
        let doc: json::Document = serde_json::from_str(text).map_err(ModelError::Json)?;
        let learner = doc.learner;
        if learner.objective.name != "binary:logistic" {
            return Err(ModelError::Unsupported(format!(
                "objective {}",
                learner.objective.name
            )));
        }
        if learner.gradient_booster.name != "gbtree" {
            return Err(ModelError::Unsupported(format!(
                "booster {}",
                learner.gradient_booster.name
            )));
        }
        let model = learner.gradient_booster.model;
        if model.tree_info.iter().any(|&g| g != 0) {
            return Err(ModelError::Unsupported("more than one output group".into()));
        }
        let base_score = parse_f32(&learner.learner_model_param.base_score)?;
        let trees = model
            .trees
            .into_iter()
            .map(json::Tree::into_tree)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Booster {
            trees,
            base_margin: prob_to_margin(base_score),
            feature_names: learner.feature_names,
        })
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Raw score (log-odds) for one row of `f32` feature values in model order; NaN = missing.
    pub fn predict_margin(&self, row: &[f32]) -> f32 {
        debug_assert_eq!(row.len(), self.num_features());
        let mut margin = self.base_margin;
        for tree in &self.trees {
            margin += tree.leaf_value(row);
        }
        margin
    }

    /// Probability of the positive class, as `XGBClassifier.predict_proba(X)[:, 1]`.
    pub fn predict_proba(&self, row: &[f32]) -> f32 {
        sigmoid(self.predict_margin(row))
    }

    /// Per-feature contributions plus the bias as the last element (length `num_features + 1`),
    /// as `Booster.predict(..., pred_contribs=True, approx_contribs=True)`.
    pub fn approx_contributions(&self, row: &[f32]) -> Vec<f32> {
        let n = self.num_features() + 1;
        let mut total = vec![0.0f32; n];
        let mut per_tree = vec![0.0f32; n];
        for tree in &self.trees {
            per_tree.iter_mut().for_each(|v| *v = 0.0);
            tree.add_approx_contributions(row, &mut per_tree);
            for (t, v) in total.iter_mut().zip(&per_tree) {
                *t += *v;
            }
        }
        total[n - 1] += self.base_margin;
        total
    }
}

/// XGBoost's `LogisticRegression::ProbToMargin`.
fn prob_to_margin(base_score: f32) -> f32 {
    -(1.0f32 / base_score - 1.0f32).ln()
}

/// XGBoost's `common::Sigmoid`.
fn sigmoid(x: f32) -> f32 {
    const EPS: f32 = 1e-16;
    let x = (-x).min(88.7f32);
    1.0f32 / (x.exp() + 1.0f32 + EPS)
}

fn parse_f32(text: &str) -> Result<f32, ModelError> {
    text.trim()
        .parse::<f32>()
        .map_err(|_| ModelError::Unsupported(format!("not a number: {text:?}")))
}

/// Serde mirror of the parts of XGBoost's JSON model format that prediction needs.
mod json {
    use super::{parse_f32, ModelError};
    use serde::Deserialize;
    use serde_json::Number;

    #[derive(Deserialize)]
    pub struct Document {
        pub learner: Learner,
    }

    #[derive(Deserialize)]
    pub struct Learner {
        pub feature_names: Vec<String>,
        pub gradient_booster: GradientBooster,
        pub learner_model_param: LearnerModelParam,
        pub objective: Objective,
    }

    #[derive(Deserialize)]
    pub struct LearnerModelParam {
        pub base_score: String,
    }

    #[derive(Deserialize)]
    pub struct Objective {
        pub name: String,
    }

    #[derive(Deserialize)]
    pub struct GradientBooster {
        pub name: String,
        pub model: Model,
    }

    #[derive(Deserialize)]
    pub struct Model {
        pub trees: Vec<Tree>,
        pub tree_info: Vec<i64>,
    }

    #[derive(Deserialize)]
    pub struct Tree {
        pub left_children: Vec<i32>,
        pub right_children: Vec<i32>,
        pub split_indices: Vec<u32>,
        pub split_conditions: Vec<Number>,
        pub default_left: Vec<u8>,
        pub sum_hessian: Vec<Number>,
        pub split_type: Vec<u8>,
    }

    fn floats(numbers: &[Number]) -> Result<Vec<f32>, ModelError> {
        // With serde_json's arbitrary_precision, `Number` keeps the original text, so each
        // value is rounded once, straight to f32.
        numbers.iter().map(|n| parse_f32(&n.to_string())).collect()
    }

    impl Tree {
        pub fn into_tree(self) -> Result<super::Tree, ModelError> {
            if self.split_type.iter().any(|&t| t != 0) {
                return Err(ModelError::Unsupported("categorical splits".into()));
            }
            let mut tree = super::Tree {
                left: self.left_children,
                right: self.right_children,
                split_index: self.split_indices,
                split_condition: floats(&self.split_conditions)?,
                default_left: self.default_left.into_iter().map(|d| d != 0).collect(),
                sum_hessian: floats(&self.sum_hessian)?,
                mean_value: Vec::new(),
            };
            tree.fill_mean_values();
            Ok(tree)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stump: feature 0 < 0.5 goes left (leaf -1.0), otherwise right (leaf 2.0); missing goes right.
    const STUMP: &str = r#"{"learner":{
        "feature_names":["a"],
        "learner_model_param":{"base_score":"5E-1"},
        "objective":{"name":"binary:logistic"},
        "gradient_booster":{"name":"gbtree","model":{"tree_info":[0],"trees":[{
            "left_children":[1,-1,-1],"right_children":[2,-1,-1],"split_indices":[0,0,0],
            "split_conditions":[0.5,-1.0,2.0],"default_left":[0,0,0],
            "sum_hessian":[4.0,1.0,3.0],"split_type":[0,0,0]}]}}}}"#;

    #[test]
    fn split_is_strict_less_than() {
        let b = Booster::from_json_str(STUMP).unwrap();
        assert_eq!(b.predict_margin(&[0.4999]), -1.0);
        assert_eq!(b.predict_margin(&[0.5]), 2.0);
    }

    #[test]
    fn missing_follows_default_direction() {
        let b = Booster::from_json_str(STUMP).unwrap();
        assert_eq!(b.predict_margin(&[f32::NAN]), 2.0);
    }

    #[test]
    fn contributions_sum_to_margin() {
        let b = Booster::from_json_str(STUMP).unwrap();
        // Root mean = (-1*1 + 2*3) / 4 = 1.25; going left moves it by -2.25.
        let c = b.approx_contributions(&[0.0]);
        assert_eq!(c, vec![-2.25, 1.25]);
        assert_eq!(c.iter().sum::<f32>(), b.predict_margin(&[0.0]));
    }
}
