//! Tables are read with Polars; this module keeps only the pandas 1.4 / numpy 1.24 details that
//! decide exact numbers: cell types as the fill logic sees them, `Series.describe()` statistics
//! (numpy pairwise-sum mean, linear median) and Python float formatting.

use std::fmt::Write as _;

use polars::prelude::*;
use rayon::prelude::*;

/// One cell as pandas holds it after `read_csv`.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    /// NaN / missing.
    Na,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl Cell {
    pub fn is_na(&self) -> bool {
        match self {
            Cell::Na => true,
            Cell::Float(f) => f.is_nan(),
            _ => false,
        }
    }

    pub fn is_str(&self, s: &str) -> bool {
        matches!(self, Cell::Str(v) if v == s)
    }

    /// Python `str(x)` of the cell's value.
    pub fn to_py_str(&self) -> String {
        match self {
            Cell::Na => "nan".into(),
            Cell::Int(i) => i.to_string(),
            Cell::Float(f) => py_repr(*f),
            Cell::Bool(b) => if *b { "True" } else { "False" }.into(),
            Cell::Str(s) => s.clone(),
        }
    }

    /// `astype("float64")` of one cell: numbers as they are, strings through Python's `float()`.
    pub fn to_f64(&self) -> Result<f64, String> {
        match self {
            Cell::Na => Ok(f64::NAN),
            Cell::Int(i) => Ok(*i as f64),
            Cell::Float(f) => Ok(*f),
            Cell::Bool(b) => Ok(f64::from(u8::from(*b))),
            Cell::Str(s) => {
                py_float(s).ok_or_else(|| format!("could not convert string to float: {s:?}"))
            }
        }
    }
}

/// Python's `float(str)` (correctly rounded).
pub fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let lower = t.to_ascii_lowercase();
    match lower.trim_start_matches(['+', '-']) {
        "nan" => return Some(f64::NAN),
        "inf" | "infinity" => {
            return Some(if lower.starts_with('-') {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            })
        }
        _ => {}
    }
    if t.contains('_') {
        return None; // underscores never occur in the pipeline's numbers
    }
    t.parse::<f64>().ok()
}

/// pandas' default NA strings for `read_csv`.
pub(crate) const NA_STRINGS: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null",
];

/// A column-major table read with Polars, with cells typed as `pd.read_csv` would give them
/// (int64 without missing values, float64, bool, or strings; pandas' NA strings are missing).
#[derive(Debug, Clone)]
pub struct Frame {
    pub columns: Vec<String>,
    pub data: Vec<Vec<Cell>>,
}

impl Frame {
    /// Reads a delimited file (gzip is detected) with a header row.
    pub fn read_path(path: impl AsRef<std::path::Path>, sep: u8) -> PolarsResult<Frame> {
        let df = csv_options(sep)
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        Frame::from_polars(df)
    }

    pub fn read_str(text: &str, sep: u8) -> PolarsResult<Frame> {
        let df = CsvReader::new(std::io::Cursor::new(text.as_bytes().to_vec()))
            .with_options(csv_options(sep))
            .finish()?;
        Frame::from_polars(df)
    }

    /// Converts column by column, freeing each Polars column once converted (only one copy
    /// of the table is alive at a time).
    fn from_polars(df: DataFrame) -> PolarsResult<Frame> {
        let mut columns = Vec::new();
        let mut data = Vec::new();
        for (i, col) in df.into_columns().into_iter().enumerate() {
            let name = col.name().as_str();
            columns.push(if name.is_empty() {
                format!("Unnamed: {i}")
            } else {
                name.to_owned()
            });
            let s = col.as_materialized_series();
            let cells: Vec<Cell> = match s.dtype() {
                DataType::Int64 if s.null_count() == 0 => {
                    s.i64()?.into_no_null_iter().map(Cell::Int).collect()
                }
                DataType::Int64 => s
                    .i64()?
                    .iter()
                    .map(|v| v.map_or(Cell::Float(f64::NAN), |x| Cell::Float(x as f64)))
                    .collect(),
                DataType::Float64 => s
                    .f64()?
                    .iter()
                    .map(|v| Cell::Float(v.unwrap_or(f64::NAN)))
                    .collect(),
                DataType::Boolean => s
                    .bool()?
                    .iter()
                    .map(|v| v.map_or(Cell::Na, Cell::Bool))
                    .collect(),
                DataType::String => s
                    .str()?
                    .iter()
                    .map(|v| v.map_or(Cell::Na, |x| Cell::Str(x.to_owned())))
                    .collect(),
                DataType::Null => vec![Cell::Float(f64::NAN); s.len()],
                other => {
                    let s = s.cast(&DataType::String)?;
                    let _ = other;
                    s.str()?
                        .iter()
                        .map(|v| v.map_or(Cell::Na, |x| Cell::Str(x.to_owned())))
                        .collect()
                }
            };
            data.push(cells);
        }
        Ok(Frame { columns, data })
    }

    pub fn n_rows(&self) -> usize {
        self.data.first().map_or(0, Vec::len)
    }

    /// Moves a column out (leaving it empty).
    pub fn take_col(&mut self, name: &str) -> Vec<Cell> {
        let i = self.col_index(name);
        std::mem::take(&mut self.data[i])
    }

    pub fn col(&self, name: &str) -> &Vec<Cell> {
        let i = self.col_index(name);
        &self.data[i]
    }

    pub fn col_index(&self, name: &str) -> usize {
        self.columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("no column {name:?}"))
    }
}

/// `pd.read_csv(path, sep=sep)` as a Polars frame (pandas' NA strings; gzip detected).
pub fn read_df(path: impl AsRef<std::path::Path>, sep: u8) -> PolarsResult<DataFrame> {
    csv_options(sep)
        .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
        .finish()
}

/// `DataFrame.to_csv(sep=sep)` (with the default RangeIndex written first) for frames holding
/// what pandas would: int64 columns with missing values print as floats (pandas upcasts them),
/// missing values print empty, floats use Python's repr, fields are quoted only if needed.
pub fn to_csv(df: &DataFrame, sep: char) -> PolarsResult<String> {
    render_csv(df, sep, true)
}

/// `DataFrame.to_csv(sep=sep, index=False)`.
pub fn to_csv_no_index(df: &DataFrame, sep: char) -> PolarsResult<String> {
    render_csv(df, sep, false)
}

fn render_csv(df: &DataFrame, sep: char, index: bool) -> PolarsResult<String> {
    let quote = |v: &str| -> String {
        if v.contains(sep) || v.contains('"') || v.contains('\n') || v.contains('\r') {
            format!("\"{}\"", v.replace('"', "\"\""))
        } else {
            v.to_owned()
        }
    };
    let cols: Vec<&Series> = df
        .columns()
        .iter()
        .map(|c| c.as_materialized_series())
        .collect();
    let rendered: Vec<Vec<String>> = cols
        .par_iter()
        .map(|s| -> PolarsResult<Vec<String>> {
            let v: Vec<String> = match s.dtype() {
                DataType::Int64 if s.null_count() == 0 => s
                    .i64()?
                    .into_no_null_iter()
                    .map(|x| x.to_string())
                    .collect(),
                DataType::Int64 => s
                    .i64()?
                    .iter()
                    .map(|x| x.map_or(String::new(), |x| py_repr(x as f64)))
                    .collect(),
                DataType::Float64 => s
                    .f64()?
                    .iter()
                    .map(|x| match x {
                        Some(x) if !x.is_nan() => py_repr(x),
                        _ => String::new(),
                    })
                    .collect(),
                DataType::Boolean => s
                    .bool()?
                    .iter()
                    .map(|x| {
                        x.map_or(String::new(), |b| {
                            if b { "True" } else { "False" }.to_owned()
                        })
                    })
                    .collect(),
                DataType::String => s
                    .str()?
                    .iter()
                    .map(|x| x.map_or(String::new(), quote))
                    .collect(),
                DataType::Null => vec![String::new(); s.len()],
                _ => s
                    .cast(&DataType::String)?
                    .str()?
                    .iter()
                    .map(|x| x.map_or(String::new(), quote))
                    .collect(),
            };
            Ok(v)
        })
        .collect::<PolarsResult<_>>()?;
    let mut out = String::new();
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|n| quote(n.as_str()))
        .collect();
    if index {
        out.push(sep);
    }
    out.push_str(&names.join(&sep.to_string()));
    out.push('\n');
    for i in 0..df.height() {
        if index {
            out.push_str(&i.to_string());
        }
        for (j, col) in rendered.iter().enumerate() {
            if index || j > 0 {
                out.push(sep);
            }
            out.push_str(&col[i]);
        }
        out.push('\n');
    }
    Ok(out)
}

fn csv_options(sep: u8) -> CsvReadOptions {
    let na = NullValues::AllColumns(NA_STRINGS.iter().map(|s| PlSmallStr::from(*s)).collect());
    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(None)
        .with_parse_options(
            CsvParseOptions::default()
                .with_separator(sep)
                .with_null_values(Some(na)),
        )
}

/// numpy's pairwise summation (`pairwise_sum` in `loops_utils.h`, blocks of 128, 8 lanes).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `Series.describe()` of a float column (NaN = missing).
#[derive(Debug, Clone, Copy)]
pub struct Describe {
    pub count: usize,
    pub mean: f64,
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

pub fn describe(values: &[f64]) -> Describe {
    // pandas nanmean: NaN replaced by 0, summed pairwise over the whole array, / count.
    let filled: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    let mut present: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let count = present.len();
    present.sort_by(|a, b| a.total_cmp(b));
    let mean = if count == 0 {
        f64::NAN
    } else {
        pairwise_sum(&filled) / count as f64
    };
    Describe {
        count,
        mean,
        min: present.first().copied().unwrap_or(f64::NAN),
        median: percentile_linear(&present, 0.5),
        max: present.last().copied().unwrap_or(f64::NAN),
    }
}

/// numpy 1.24 `percentile(..., method="linear")` on sorted, NaN-free values.
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let virtual_index = q * (sorted.len() - 1) as f64;
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let lo = previous as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let (a, b) = (sorted[lo], sorted[hi]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// Python `repr(float)`: shortest round-trip digits; scientific below 1e-4 or from 1e16.
///
/// Rust's shortest formatter gives the shortest length; its digits are Python's unless two
/// shortest strings are equally close to the value (e.g. float32 values widened to f64), where
/// Python rounds the exact value half to even. [`no_tie`] rules that out cheaply for almost
/// every value; otherwise the digits are recomputed with [`round_half_even`].
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let mut sci = String::with_capacity(32);
    let _ = write!(sci, "{x:e}");
    repr_common(&sci, x.abs())
}

/// numpy `str(np.float32(x))` for finite non-zero `x` (same layout rules as [`py_repr`]).
pub(crate) fn py_repr_f32_nonzero(x: f32) -> String {
    let mut sci = String::with_capacity(32);
    let _ = write!(sci, "{x:e}");
    repr_common(&sci, f64::from(x.abs()))
}

const LOG10_5: f64 = 1.0 - std::f64::consts::LOG10_2;
/// True when `x`'s exact decimal expansion certainly has more than `n + 1` significant digits.
/// A tie at `n` digits needs exactly `n + 1` (the last a 5), so then the shortest digits are
/// the correctly rounded ones. `x = m 2^q` with `m` odd has `floor(log10(m 5^-q)) + 1` digits
/// for `q < 0`; the bound below is a lower estimate with half a digit of slack.
#[inline]
fn no_tie(x: f64, n: usize) -> bool {
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    let f = bits & ((1u64 << 52) - 1);
    let (mut m, mut q) = if e == 0 {
        (f, -1074)
    } else {
        (f | (1u64 << 52), e - 1075)
    };
    let tz = m.trailing_zeros();
    m >>= tz;
    q += tz as i32;
    if q >= 0 {
        return false;
    }
    let b = (64 - m.leading_zeros()) as f64;
    (b - 1.0) * std::f64::consts::LOG10_2 + (-q) as f64 * LOG10_5 >= n as f64 + 1.5
}

/// Python's float layout of `digits` with decimal exponent `exp`.
fn layout(out: &mut String, sign: &str, digits: &str, exp: i32) {
    out.push_str(sign);
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..(-point) {
                out.push('0');
            }
            out.push_str(digits);
        } else if point as usize >= digits.len() {
            out.push_str(digits);
            for _ in 0..(point as usize - digits.len()) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            let (int, frac) = digits.split_at(point as usize);
            out.push_str(int);
            out.push('.');
            out.push_str(frac);
        }
    } else {
        let (first, rest) = digits.split_at(1);
        out.push_str(first);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
}

/// Digits of a Rust `{:e}` string, corrected for ties, in Python's layout.
fn repr_common(sci: &str, xabs: f64) -> String {
    let (mantissa, e) = sci.split_once('e').unwrap();
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let mut buf = [0u8; 40];
    let mut k = 0;
    for &c in mantissa.as_bytes() {
        if c != b'.' {
            buf[k] = c;
            k += 1;
        }
    }
    let shortest = k;
    let mut out = String::with_capacity(24);
    if no_tie(xabs, shortest) {
        let digits = std::str::from_utf8(&buf[..k]).unwrap();
        layout(&mut out, sign, digits, e.parse().unwrap());
    } else {
        let (digits, exp) = round_half_even(xabs, shortest);
        layout(&mut out, sign, &digits, exp);
    }
    out
}

/// `n` significant digits of `x` (> 0), rounded half to even from its exact decimal expansion.
pub(crate) fn round_half_even(x: f64, n: usize) -> (String, i32) {
    // A tie needs the exact value to end in 5 right after the n-th digit. Check that with
    // cheap fixed-precision formats; only then expand exactly.
    let near = format!("{x:.*e}", n); // n + 1 significant digits, correctly rounded
    let (m, _) = near.split_once('e').unwrap();
    let digits: String = m.chars().filter(char::is_ascii_digit).collect();
    if !digits.ends_with('5') {
        let short = format!("{x:.*e}", n - 1);
        let (m, e) = short.split_once('e').unwrap();
        return (
            m.chars().filter(char::is_ascii_digit).collect(),
            e.parse().unwrap(),
        );
    }
    let exact = format!("{x:.800e}"); // exact: every double has a finite decimal expansion
    let (m, e) = exact.split_once('e').unwrap();
    let mut exp: i32 = e.parse().unwrap();
    let all: Vec<u8> = m
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| b - b'0')
        .collect();
    let (keep, rest) = all.split_at(n);
    let mut d = keep.to_vec();
    let tail_nonzero = rest[1..].iter().any(|&v| v != 0);
    let round_up = rest[0] > 5 || (rest[0] == 5 && (tail_nonzero || d[n - 1] % 2 == 1));
    if round_up {
        let mut i = n - 1;
        loop {
            if d[i] < 9 {
                d[i] += 1;
                break;
            }
            d[i] = 0;
            if i == 0 {
                d.insert(0, 1);
                d.pop();
                exp += 1;
                break;
            }
            i -= 1;
        }
    }
    (d.iter().map(|v| char::from(b'0' + v)).collect(), exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_repr_matches_python() {
        for (x, want) in [
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123.0, "123.0"),
            (0.2142857142857142, "0.2142857142857142"),
            (-1.5e-7, "-1.5e-07"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (f64::from(-0.170_764_92_f32), "-0.17076492309570312"), // exact tie: half to even, like Python
            (
                "-0.94886016845703125".parse::<f64>().unwrap(),
                "-0.9488601684570312",
            ), // tie at 16 digits
        ] {
            assert_eq!(py_repr(x), want);
        }
    }

    #[test]
    fn describe_median_and_mean() {
        let d = describe(&[3.0, f64::NAN, 1.0, 2.0, 10.0]);
        assert_eq!((d.count, d.min, d.max), (4, 1.0, 10.0));
        assert_eq!(d.median, 2.5);
        assert_eq!(d.mean, 4.0);
    }

    #[test]
    fn read_csv_infers_like_pandas() {
        let f = Frame::read_str("a,b,c,d,e\n1,1.5,x,True,\n2,,-,False,\n", b',').unwrap();
        assert_eq!(f.col("a"), &vec![Cell::Int(1), Cell::Int(2)]);
        assert!(matches!(f.col("b")[1], Cell::Float(v) if v.is_nan()));
        assert_eq!(
            f.col("c"),
            &vec![Cell::Str("x".into()), Cell::Str("-".into())]
        );
        assert_eq!(f.col("d"), &vec![Cell::Bool(true), Cell::Bool(false)]);
        assert!(f.col("e").iter().all(Cell::is_na));
    }

    #[test]
    fn to_csv_with_and_without_index() {
        let df = DataFrame::new(
            2,
            vec![
                Column::new("id".into(), ["a", "b\nc"]),
                Column::new("x".into(), [1.5, 2.0]),
            ],
        )
        .unwrap();
        assert_eq!(
            to_csv(&df, ',').unwrap(),
            ",id,x\n0,a,1.5\n1,\"b\nc\",2.0\n"
        );
        assert_eq!(
            to_csv_no_index(&df, ',').unwrap(),
            "id,x\na,1.5\n\"b\nc\",2.0\n"
        );
    }

    /// The tie shortcut in `py_repr` never changes a result: compare with always taking the
    /// exact half-even path, on random bit patterns and on float32 values widened to f64 (where
    /// ties actually occur).
    #[test]
    fn repr_shortcut_matches_exact_rounding() {
        fn exact(x: f64) -> String {
            let sci = format!("{x:e}");
            let (m, _) = sci.split_once('e').unwrap();
            let m = m.strip_prefix('-').unwrap_or(m);
            let n = m.chars().filter(char::is_ascii_digit).count();
            let (digits, exp) = round_half_even(x.abs(), n);
            let mut out = String::new();
            layout(&mut out, if x < 0.0 { "-" } else { "" }, &digits, exp);
            out
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200_000 {
            let x = f64::from_bits(next());
            if x.is_finite() && x != 0.0 {
                assert_eq!(py_repr(x), exact(x), "{x:e}");
            }
            let y = f64::from(f32::from_bits(next() as u32));
            if y.is_finite() && y != 0.0 {
                assert_eq!(py_repr(y), exact(y), "{y:e}");
            }
        }
    }
}
