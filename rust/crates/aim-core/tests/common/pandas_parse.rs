//! Test oracle: pandas 1.4's default CSV float parser (`precise_xstrtod`, tokenizer.c, now in
//! `aim_core::pdread`). It keeps at most 17 digits including leading zeros, so small values lose
//! significant digits. The golden tests use it to accept a pipeline value that is exactly what
//! pandas makes of Rust's.
#![allow(dead_code)]

pub use aim_core::pdread::precise_xstrtod;

/// Rust's value `rust` (text) vs the pipeline's `pipeline` (text): equal, within `rel_tol`, or
/// the pipeline's is what pandas' parser makes of Rust's (possibly twice: read, write, read).
pub fn numbers_agree(rust: &str, pipeline: &str, rel_tol: f64) -> bool {
    let (Ok(x), Ok(y)) = (rust.parse::<f64>(), pipeline.parse::<f64>()) else {
        return false;
    };
    if x == y || ((x - y) / y).abs() <= rel_tol {
        return true;
    }
    let once = precise_xstrtod(rust);
    once == Some(y) || once.and_then(|v| precise_xstrtod(&aim_core::pandas::py_repr(v))) == Some(y)
}
