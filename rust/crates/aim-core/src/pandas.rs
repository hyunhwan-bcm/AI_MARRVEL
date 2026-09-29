//! Tables are read with Polars; this module keeps only the pandas 1.4 / numpy 1.24 details that
//! decide exact numbers: cell types as the fill logic sees them, `Series.describe()` statistics
//! (numpy pairwise-sum mean, linear median) and Python float formatting.

use polars::prelude::*;

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
const NA_STRINGS: &[&str] = &[
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
        Frame::from_polars(&df)
    }

    pub fn read_str(text: &str, sep: u8) -> PolarsResult<Frame> {
        let df = CsvReader::new(std::io::Cursor::new(text.as_bytes().to_vec()))
            .with_options(csv_options(sep))
            .finish()?;
        Frame::from_polars(&df)
    }

    fn from_polars(df: &DataFrame) -> PolarsResult<Frame> {
        let mut columns = Vec::new();
        let mut data = Vec::new();
        for (i, col) in df.columns().iter().enumerate() {
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
    let sci = format!("{:e}", x); // e.g. "-2.142857142857142e-1"
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
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
}
