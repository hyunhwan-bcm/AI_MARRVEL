//! Test oracle: pandas 1.4's default CSV float parser (`precise_xstrtod`, tokenizer.c). It keeps
//! at most 17 digits including leading zeros, so small values lose significant digits. The
//! golden tests use it to accept a pipeline value that is exactly what pandas makes of Rust's.
#![allow(dead_code)]

pub fn precise_xstrtod(s: &str) -> Option<f64> {
    const MAX_DIGITS: usize = 17;
    let b = s.trim_matches(|c: char| c == ' ' || c == '\t').as_bytes();
    let mut p = 0;
    let negative = match b.first() {
        Some(b'-') => {
            p += 1;
            true
        }
        Some(b'+') => {
            p += 1;
            false
        }
        _ => false,
    };
    let (mut number, mut exponent, mut num_digits) = (0.0f64, 0i32, 0usize);
    while p < b.len() && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + f64::from(b[p] - b'0');
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < b.len() && b[p] == b'.' {
        p += 1;
        let mut num_decimals = 0;
        while num_digits < MAX_DIGITS && p < b.len() && b[p].is_ascii_digit() {
            number = number * 10.0 + f64::from(b[p] - b'0');
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        while p < b.len() && b[p].is_ascii_digit() {
            p += 1;
        }
        exponent -= num_decimals;
    }
    if num_digits == 0 {
        return None;
    }
    if negative {
        number = -number;
    }
    if p < b.len() && (b[p] == b'e' || b[p] == b'E') {
        let save = p;
        p += 1;
        let neg = match b.get(p) {
            Some(b'-') => {
                p += 1;
                true
            }
            Some(b'+') => {
                p += 1;
                false
            }
            _ => false,
        };
        let (mut n, mut d) = (0i32, 0);
        while d < MAX_DIGITS && p < b.len() && b[p].is_ascii_digit() {
            n = n * 10 + i32::from(b[p] - b'0');
            d += 1;
            p += 1;
        }
        if d == 0 {
            p = save;
        } else if neg {
            exponent -= n;
        } else {
            exponent += n;
        }
    }
    if p != b.len() {
        return None;
    }
    let pow10 = |e: i32| -> f64 { format!("1e{e}").parse().unwrap() };
    Some(if exponent > 308 {
        f64::INFINITY.copysign(number)
    } else if exponent > 0 {
        number * pow10(exponent)
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            number / pow10(-308 - exponent) / pow10(308)
        }
    } else {
        number / pow10(-exponent)
    })
}

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
