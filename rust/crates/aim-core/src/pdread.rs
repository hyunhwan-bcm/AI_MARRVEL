//! `pd.read_csv` (pandas 1.4, C engine, default options) for the columns a caller needs, with
//! the typing pandas gives them:
//!
//! - rows are converted in chunks (`low_memory=True`): the chunk length is the largest power of
//!   two below `2**20 // n_columns` (2,048 rows for VEP's 459 columns);
//! - each chunk of a column becomes int64 (no missing values), float64 (parsed with pandas'
//!   `precise_xstrtod`, which keeps only 17 digits counting leading zeros), bool, or strings,
//!   tried in that order;
//! - chunks of different types are concatenated as numpy does: int64 + float64 -> float64,
//!   otherwise object, holding each chunk's own values (e.g. a float chunk before a string
//!   chunk stays floats, the rest stay the original text);
//! - pandas' default NA strings are missing values.
//!
//! Tokenizing follows the C tokenizer for the files AIM reads: one-character separator, `"`
//! quoting with doubled quotes, blank lines skipped.

use std::collections::HashMap;
use std::io::{self, BufRead};

use crate::pandas::NA_STRINGS;
use crate::pyobj::{Col, Kind, Py};

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// pandas' `precise_xstrtod` (tokenizer.c): at most 17 digits are accumulated, leading zeros
/// included, then scaled by an exact power of ten.
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

/// A token as the C parser's float conversion reads it (`to_double`, then `inf` spellings).
fn parse_double(t: &str) -> Option<f64> {
    if let Some(v) = precise_xstrtod(t) {
        // an out-of-range value (HUGE_VAL) is a failed parse: the chunk stays text
        return v.is_finite().then_some(v);
    }
    let l = t.trim().to_ascii_lowercase();
    match l.as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// `str_to_int64`: optional surrounding spaces, sign, digits.
fn parse_int(t: &str) -> Option<i64> {
    let s = t.trim_matches(|c: char| c.is_ascii_whitespace());
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<i64>().ok()
}

fn is_integer_text(t: &str) -> bool {
    let s = t.trim_matches(|c: char| c.is_ascii_whitespace());
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

fn parse_bool(t: &str) -> Option<bool> {
    match t {
        "True" | "TRUE" | "true" => Some(true),
        "False" | "FALSE" | "false" => Some(false),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChunkKind {
    Int,
    Float,
    Bool,
    Object,
}

/// Converts one chunk of raw tokens (None = missing field) as pandas' `_convert_tokens`.
fn convert_chunk(tokens: &[Option<String>]) -> (ChunkKind, Vec<Py>) {
    let is_na = |t: &Option<String>| t.as_deref().is_none_or(|s| NA_STRINGS.contains(&s));
    let any_na = tokens.iter().any(is_na);
    if !any_na {
        let ints: Option<Vec<i64>> = tokens
            .iter()
            .map(|t| parse_int(t.as_deref().unwrap()))
            .collect();
        if let Some(v) = ints {
            return (ChunkKind::Int, v.into_iter().map(Py::Int).collect());
        }
        // integers beyond int64: pandas' uint64 / object fallback keeps the text
        if tokens
            .iter()
            .all(|t| is_integer_text(t.as_deref().unwrap()))
        {
            let vals = tokens.iter().map(|t| Py::Str(t.clone().unwrap())).collect();
            return (ChunkKind::Object, vals);
        }
    }
    let floats: Option<Vec<f64>> = tokens
        .iter()
        .map(|t| {
            if is_na(t) {
                Some(f64::NAN)
            } else {
                parse_double(t.as_deref().unwrap())
            }
        })
        .collect();
    if let Some(v) = floats {
        return (ChunkKind::Float, v.into_iter().map(Py::Float).collect());
    }
    let bools: Option<Vec<Option<bool>>> = tokens
        .iter()
        .map(|t| {
            if is_na(t) {
                Some(None)
            } else {
                parse_bool(t.as_deref().unwrap()).map(Some)
            }
        })
        .collect();
    if let Some(v) = bools {
        // with missing values pandas upcasts to object (True/False/NaN)
        let kind = if any_na {
            ChunkKind::Object
        } else {
            ChunkKind::Bool
        };
        let vals = v
            .into_iter()
            .map(|b| b.map_or(Py::nan(), Py::Bool))
            .collect();
        return (kind, vals);
    }
    let vals = tokens
        .iter()
        .map(|t| {
            if is_na(t) {
                Py::nan()
            } else {
                Py::Str(t.clone().unwrap())
            }
        })
        .collect();
    (ChunkKind::Object, vals)
}

/// Concatenates converted chunks as `_concatenate_chunks` / `np.concatenate` do.
fn concat_chunks(chunks: Vec<(ChunkKind, Vec<Py>)>) -> Col {
    let kinds: Vec<ChunkKind> = chunks.iter().map(|c| c.0).collect();
    let all = |k: ChunkKind| kinds.iter().all(|&x| x == k);
    let has = |k: ChunkKind| kinds.contains(&k);
    // numpy's common type: int64 + float64 -> float64, bool + int64 -> int64, bool + float64
    // -> float64 (checked against pandas 1.4.3: a bool chunk next to an int chunk prints 0/1)
    let kind = if kinds.is_empty() || has(ChunkKind::Object) {
        Kind::Object
    } else if all(ChunkKind::Int) {
        Kind::Int64
    } else if all(ChunkKind::Bool) {
        Kind::Bool
    } else if has(ChunkKind::Float) {
        Kind::Float64
    } else {
        Kind::Int64
    };
    let mut vals = Vec::with_capacity(chunks.iter().map(|c| c.1.len()).sum());
    for (_, v) in chunks {
        vals.extend(v.into_iter().map(|x| match (kind, x) {
            (Kind::Float64, Py::Int(i)) => Py::Float(i as f64),
            (Kind::Float64, Py::Bool(b)) => Py::Float(f64::from(u8::from(b))),
            (Kind::Int64, Py::Bool(b)) => Py::Int(i64::from(b)),
            (_, x) => x,
        }));
    }
    Col { kind, vals }
}

/// Splits one record's text into fields (`"` quoting, doubled quotes).
fn split_fields(record: &str, sep: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = record.chars().peekable();
    let mut at_start = true;
    let mut in_quotes = false;
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        if c == sep {
            fields.push(std::mem::take(&mut field));
            at_start = true;
            continue;
        }
        if c == '"' && at_start {
            in_quotes = true;
            at_start = false;
            continue;
        }
        at_start = false;
        field.push(c);
    }
    fields.push(field);
    fields
}

/// Reads records (physical lines joined while inside quotes), skipping blank lines.
struct Records<R> {
    reader: R,
    line: String,
    sep: char,
}

impl<R: BufRead> Records<R> {
    fn next_record(&mut self) -> io::Result<Option<String>> {
        loop {
            self.line.clear();
            if self.reader.read_line(&mut self.line)? == 0 {
                return Ok(None);
            }
            let mut rec = trim_eol(&self.line).to_owned();
            while quote_open(&rec, self.sep) {
                self.line.clear();
                if self.reader.read_line(&mut self.line)? == 0 {
                    break;
                }
                rec.push('\n');
                rec.push_str(trim_eol(&self.line));
            }
            if !rec.is_empty() {
                return Ok(Some(rec));
            }
        }
    }
}

fn trim_eol(s: &str) -> &str {
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.strip_suffix('\r').unwrap_or(s)
}

fn quote_open(rec: &str, sep: char) -> bool {
    // a field opened with '"' and not yet closed
    if !rec.contains('"') {
        return false;
    }
    let mut in_quotes = false;
    let mut at_start = true;
    let mut chars = rec.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            continue;
        }
        match c {
            '"' if at_start => in_quotes = true,
            c if c == sep => {
                at_start = true;
                continue;
            }
            _ => {}
        }
        at_start = false;
    }
    in_quotes
}

/// Column names with pandas' de-duplication (`x`, `x.1`, ...).
fn mangle(names: Vec<String>) -> Vec<String> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    names
        .into_iter()
        .map(|n| {
            let count = seen.entry(n.clone()).or_insert(0);
            let out = if *count == 0 {
                n.clone()
            } else {
                format!("{n}.{count}")
            };
            *count += 1;
            out
        })
        .collect()
}

/// The requested columns of a table, typed as pandas reads them.
pub struct Table {
    /// All column names (after de-duplication).
    pub names: Vec<String>,
    /// Requested columns by name.
    pub cols: HashMap<String, Col>,
    pub n_rows: usize,
}

impl Table {
    pub fn col(&self, name: &str) -> io::Result<&Col> {
        self.cols
            .get(name)
            .ok_or_else(|| invalid(format!("no column {name:?}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }
}

/// `pd.read_csv(reader, sep=sep, skiprows=skip_lines)` keeping the columns named in `want`
/// (all columns when `want` is None). Positional columns can be requested as `"#0"`, `"#1"`.
pub fn read_table(
    mut reader: impl BufRead,
    sep: char,
    skip_lines: usize,
    want: Option<&[&str]>,
) -> io::Result<Table> {
    let mut line = String::new();
    for _ in 0..skip_lines {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
    }
    let mut records = Records {
        reader,
        line: String::new(),
        sep,
    };
    let header = records
        .next_record()?
        .ok_or_else(|| invalid("No columns to parse from file"))?;
    let names = mangle(split_fields(&header, sep));
    let width = names.len();
    let wanted: Vec<(usize, String)> = match want {
        None => names.iter().cloned().enumerate().collect(),
        Some(w) => {
            let mut v = Vec::new();
            for &n in w {
                if let Some(pos) = n.strip_prefix('#').and_then(|p| p.parse::<usize>().ok()) {
                    if pos < width {
                        v.push((pos, n.to_owned()));
                    }
                } else if let Some(i) = names.iter().position(|x| x == n) {
                    v.push((i, n.to_owned()));
                }
            }
            v
        }
    };
    // buffer_lines: the largest power of two with 2 * lines >= 2**20 // width
    let heuristic = (1usize << 20) / width.max(1);
    let mut chunk_rows = 1usize;
    while chunk_rows * 2 < heuristic {
        chunk_rows *= 2;
    }

    let mut chunks: Vec<Vec<(ChunkKind, Vec<Py>)>> = vec![Vec::new(); wanted.len()];
    let mut buf: Vec<Vec<Option<String>>> = vec![Vec::with_capacity(chunk_rows); wanted.len()];
    let mut n_rows = 0usize;
    let flush = |buf: &mut Vec<Vec<Option<String>>>,
                 chunks: &mut Vec<Vec<(ChunkKind, Vec<Py>)>>| {
        for (b, c) in buf.iter_mut().zip(chunks.iter_mut()) {
            if !b.is_empty() {
                c.push(convert_chunk(b));
                b.clear();
            }
        }
    };
    while let Some(rec) = records.next_record()? {
        let fields = split_fields(&rec, sep);
        if fields.len() > width {
            return Err(invalid(format!(
                "Expected {width} fields in line {}, saw {}",
                skip_lines + n_rows + 2,
                fields.len()
            )));
        }
        let mut fields: Vec<Option<String>> = fields.into_iter().map(Some).collect();
        for (k, (i, _)) in wanted.iter().enumerate() {
            buf[k].push(fields.get_mut(*i).and_then(Option::take));
        }
        n_rows += 1;
        if n_rows.is_multiple_of(chunk_rows) {
            flush(&mut buf, &mut chunks);
        }
    }
    flush(&mut buf, &mut chunks);
    let cols = wanted
        .into_iter()
        .zip(chunks)
        .map(|((_, name), c)| (name, concat_chunks(c)))
        .collect();
    Ok(Table {
        names,
        cols,
        n_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> Table {
        read_table(text.as_bytes(), '\t', 0, None).unwrap()
    }

    #[test]
    fn types_like_pandas() {
        let t = table("a\tb\tc\td\te\n1\t0.000116279069767442\t-\tTrue\t\n2\t2\tx\tFalse\t\n");
        assert_eq!(t.col("a").unwrap().kind, Kind::Int64);
        let b = t.col("b").unwrap();
        assert_eq!(b.kind, Kind::Float64);
        assert_eq!(b.cell(0), "0.0001162790697674"); // pandas' 17-digit parser
        assert_eq!(b.cell(1), "2.0");
        assert_eq!(t.col("c").unwrap().kind, Kind::Object);
        assert_eq!(t.col("d").unwrap().kind, Kind::Bool);
        assert_eq!(t.col("e").unwrap().kind, Kind::Float64); // all missing
    }

    #[test]
    fn chunks_are_typed_separately() {
        // 100 columns -> 8,192-row chunks; a string only in the second chunk leaves the
        // first chunk's ints as ints (pandas 1.4.3, checked)
        let header: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();
        let mut text = header.join("\t") + "\n";
        for r in 0..9000 {
            let first = if r == 8999 { "x" } else { "1" };
            text.push_str(first);
            text.push_str(&"\t1".repeat(99));
            text.push('\n');
        }
        let t = table(&text);
        let c = t.col("c0").unwrap();
        assert_eq!(c.kind, Kind::Object);
        assert_eq!(c.vals[0], Py::Int(1));
        assert_eq!(c.vals[8192], Py::str("1"));
    }

    #[test]
    fn duplicate_names_and_quotes() {
        let t = read_table("x,x,y\n1,\"a,b\",\"q\"\"r\"\n".as_bytes(), ',', 0, None).unwrap();
        assert_eq!(t.names, ["x", "x.1", "y"]);
        assert_eq!(t.col("x.1").unwrap().vals[0], Py::str("a,b"));
        assert_eq!(t.col("y").unwrap().vals[0], Py::str("q\"r"));
    }

    #[test]
    fn out_of_range_numbers_stay_text() {
        let t = table("a\tb\n10000000000000000000\t1e400\n");
        assert_eq!(t.col("a").unwrap().vals[0], Py::str("10000000000000000000"));
        assert_eq!(t.col("b").unwrap().vals[0], Py::str("1e400"));
    }
}
