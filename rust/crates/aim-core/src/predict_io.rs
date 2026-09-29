//! PREDICTION process I/O: `bin/run_final.py` (default model on the feature matrix) and the
//! single-variant part of `bin/extraModel_main.py` (default / nd models with confidence,
//! rankings and SHAP JSON), writing the same files the pipeline writes.

use std::fmt::Write as _;
use std::path::Path;

use polars::prelude::*;

use crate::npsort::sort_values_f64;
use crate::pandas::{py_repr, read_df};
use crate::stats::{confidence_level, percentile_of_score, rank_predictions};
use crate::xgb::Booster;

/// Model inputs per row: float32 as the model sees them, and the float64 values read.
pub type FeatureRows = (Vec<Vec<f32>>, Vec<Vec<f64>>);

/// A table with a string row index, as pandas holds it after `read_csv(index_col=0)`.
#[derive(Debug, Clone)]
pub struct Indexed {
    pub index: Vec<String>,
    pub df: DataFrame,
}

impl Indexed {
    pub fn read(path: impl AsRef<Path>, sep: u8) -> PolarsResult<Indexed> {
        let mut df = read_df(path, sep)?;
        let first = df.get_column_names()[0].to_string();
        let idx = df.column(&first)?.cast(&DataType::String)?;
        let index = idx
            .str()?
            .iter()
            .map(|v| v.unwrap_or("").to_owned())
            .collect();
        df = df.drop(&first)?;
        Ok(Indexed { index, df })
    }

    fn take(&self, order: &[usize]) -> PolarsResult<Indexed> {
        let idx = IdxCa::from_vec("".into(), order.iter().map(|&i| i as IdxSize).collect());
        Ok(Indexed {
            index: order.iter().map(|&i| self.index[i].clone()).collect(),
            df: self.df.take(&idx)?,
        })
    }

    /// Feature rows as the model sees them (float64 values rounded to float32).
    pub fn features(&self, names: &[String]) -> PolarsResult<FeatureRows> {
        let cols: Vec<Vec<f64>> = names
            .iter()
            .map(|n| -> PolarsResult<Vec<f64>> {
                let c = self.df.column(n)?.cast(&DataType::Float64)?;
                Ok(c.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            })
            .collect::<PolarsResult<_>>()?;
        let n = self.index.len();
        let f64s: Vec<Vec<f64>> = (0..n)
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        let f32s = f64s
            .iter()
            .map(|r| r.iter().map(|&x| x as f32).collect())
            .collect();
        Ok((f32s, f64s))
    }

    /// `to_csv(sep=sep)`: index first (unnamed), pandas' number formatting.
    pub fn to_csv(&self, sep: char) -> PolarsResult<String> {
        let body = crate::pandas::to_csv(&self.df, sep)?;
        let mut out = String::with_capacity(body.len());
        for (i, line) in body.lines().enumerate() {
            let rest = &line[line.find(sep).unwrap_or(line.len())..];
            if i == 0 {
                out.push_str(rest);
            } else {
                out.push_str(&quote_field(&self.index[i - 1], sep));
                out.push_str(rest);
            }
            out.push('\n');
        }
        Ok(out)
    }
}

fn quote_field(v: &str, sep: char) -> String {
    if v.contains(sep) || v.contains('"') || v.contains('\n') {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_owned()
    }
}

/// numpy `str(np.float32(x))`: shortest float32 digits, scientific below 1e-4 or from 1e16.
pub fn py_repr_f32(x: f32) -> String {
    if !x.is_finite() || x == 0.0 {
        return py_repr(x as f64);
    }
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        if point <= 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(point as usize - digits.len())
            )
        } else {
            let (int, frac) = digits.split_at(point as usize);
            format!("{sign}{int}.{frac}")
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        format!(
            "{sign}{first}{rest}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    }
}

/// Writes a float32 column as pandas does (numpy float32 repr).
fn f32_column(name: &str, values: &[f32]) -> Column {
    Column::new(
        name.into(),
        values.iter().map(|&v| py_repr_f32(v)).collect::<Vec<_>>(),
    )
}

fn int_column(name: &str, values: impl Iterator<Item = usize>) -> Column {
    Column::new(name.into(), values.map(|v| v as i64).collect::<Vec<_>>())
}

/// Insert or replace a column, keeping the position of an existing one.
fn set_column(df: &mut DataFrame, col: Column) -> PolarsResult<()> {
    df.with_column(col)?;
    Ok(())
}

/// Min/max rankings of `predict` in the frame's current (sorted) order.
fn set_rankings(t: &mut Indexed, predict: &[f32]) -> PolarsResult<()> {
    let ranks = rank_predictions(predict);
    set_column(
        &mut t.df,
        int_column("min_ranking", ranks.iter().map(|r| r.0)),
    )?;
    set_column(
        &mut t.df,
        int_column("max_ranking", ranks.iter().map(|r| r.1)),
    )?;
    set_column(&mut t.df, int_column("ranking", ranks.iter().map(|r| r.1)))?;
    Ok(())
}

/// `run_final.py` / `predict_new.utilities.rank_patient`: default model on `<id>.matrix.txt`.
pub fn run_final(matrix: &Indexed, booster: &Booster, identifier: &str) -> PolarsResult<Indexed> {
    let (rows, _) = matrix.features(booster.feature_names())?;
    let predict: Vec<f32> = rows.iter().map(|r| booster.predict_proba(r)).collect();
    let mut t = matrix.clone();
    t.df.with_column(f32_column("predict", &predict))?;
    let order = sort_values_f64(
        &predict.iter().map(|&p| p as f64).collect::<Vec<_>>(),
        false,
    );
    let mut t = t.take(&order)?;
    let sorted: Vec<f32> = order.iter().map(|&i| predict[i]).collect();
    set_rankings(&mut t, &sorted)?;
    let n = t.index.len();
    t.df.with_column(Column::new("identifier".into(), vec![identifier; n]))?;
    Ok(t)
}

/// One model's output of `extraModel_main.AIM()` for default / nd: the predictions table and
/// the rows (in output order) for SHAP.
pub struct ModelOutput {
    pub table: Indexed,
    pub rows: Vec<Vec<f32>>,
    pub data: Vec<Vec<f64>>,
}

/// default / nd models on `<id>.default_prediction.csv` (read back from disk, as the pipeline).
pub fn extra_model(
    default_prediction: &Indexed,
    booster: &Booster,
    reference: &[f64],
) -> PolarsResult<ModelOutput> {
    let mut t = default_prediction.clone();
    if t.df
        .get_column_names()
        .iter()
        .any(|n| n.as_str() == "predict")
    {
        t.df = t.df.drop("predict")?;
    }
    let (rows, data) = t.features(booster.feature_names())?;
    let predict: Vec<f32> = rows.iter().map(|r| booster.predict_proba(r)).collect();
    // insert(loc=shape[1] - 1): before the last column
    let at = t.df.width().saturating_sub(1);
    t.df.insert_column(at, f32_column("predict", &predict))?;
    // assign_confidence_score
    let conf: Vec<f64> = predict
        .iter()
        .map(|&p| percentile_of_score(reference, p as f64))
        .collect();
    t.df.with_column(Column::new("confidence".into(), conf.clone()))?;
    t.df.with_column(Column::new(
        "confidence level".into(),
        conf.iter()
            .map(|&c| confidence_level(c))
            .collect::<Vec<_>>(),
    ))?;
    // sort_values("confidence", ascending=False), then assign_ranking's sort by predict
    let by_conf = sort_values_f64(&conf, false);
    let pred_after: Vec<f64> = by_conf.iter().map(|&i| predict[i] as f64).collect();
    let by_pred = sort_values_f64(&pred_after, false);
    let order: Vec<usize> = by_pred.iter().map(|&k| by_conf[k]).collect();
    let mut t = t.take(&order)?;
    let sorted: Vec<f32> = order.iter().map(|&i| predict[i]).collect();
    set_rankings(&mut t, &sorted)?;
    Ok(ModelOutput {
        table: t,
        rows: order.iter().map(|&i| rows[i].clone()).collect(),
        data: order.iter().map(|&i| data[i].clone()).collect(),
    })
}

/// Python `json.dumps` of a float.
fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        py_repr(x)
    }
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

/// `model_interpreter.bin.create_shap_json`: `json.dumps(entries, indent=2)`.
pub fn shap_json(
    booster: &Booster,
    ids: &[String],
    rows: &[Vec<f32>],
    data: &[Vec<f64>],
) -> String {
    let names = booster.feature_names();
    let mut out = String::from("[");
    for (k, (id, (row, values))) in ids.iter().zip(rows.iter().zip(data)).enumerate() {
        let contrib = booster.approx_contributions(row);
        out.push_str(if k == 0 { "\n" } else { ",\n" });
        let _ = write!(
            out,
            "  {{\n    \"variant_id\": {},\n    \"base_value\": {},\n",
            json_str(id),
            json_float(contrib[names.len()] as f64)
        );
        let block = |out: &mut String, key: &str, vals: &mut dyn Iterator<Item = f64>| {
            let _ = write!(out, "    \"{key}\": {{");
            for (j, (name, v)) in names.iter().zip(vals).enumerate() {
                out.push_str(if j == 0 { "\n" } else { ",\n" });
                let _ = write!(out, "      {}: {}", json_str(name), json_float(v));
            }
            out.push_str(if names.is_empty() { "}" } else { "\n    }" });
        };
        block(
            &mut out,
            "model_output_score",
            &mut contrib[..names.len()].iter().map(|&c| c as f64),
        );
        out.push_str(",\n");
        block(&mut out, "feature_values", &mut values.iter().copied());
        out.push_str("\n  }");
    }
    out.push_str(if ids.is_empty() { "]" } else { "\n]" });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float32_repr_matches_numpy() {
        assert_eq!(py_repr_f32(0.92790127), "0.92790127");
        assert_eq!(py_repr_f32(2.5413758e-05), "2.5413758e-05");
        assert_eq!(py_repr_f32(4.5339763e-07), "4.5339763e-07");
        assert_eq!(py_repr_f32(1.0), "1.0");
    }
}
